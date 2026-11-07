#!/usr/bin/env python3
"""Gate 2 confounder analysis — V2 (pooled-null-matched construction).

ROLE: unchanged from the previous version (see the original docstring for
the full paper mapping and design description).

PERF CHANGES in this file only (no statistical changes):
  PERF 1. --max-tasks-per-child (default 4 = the BSOD guard for long
         runs). Set 0 to disable worker recycling: for short smoke runs
         (--n-trials 20) the default was costing more in Windows spawn /
         re-import overhead than the trials themselves.
  PERF 2. --resume (default on). A crashed run now continues from
         <output>.partial.json instead of restarting from zero; seeds
         already in the partial file are skipped and their results kept.
  PERF 3. --cache-dir. Stage A-D per trial (geometry -> full-environment
         truth -> TLE-fitting convention -> honest R -> matched-deficit
         noise) is cached to disk keyed by seed, so re-running with
         different confounder sets, r-scales, or thresholds never re-pays
         the fitting cost. Stage E (the filters) always re-runs.
         Same seeds => same draws => identical streams, so this cannot
         change the statistics.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from constants.constants import (
    F107_BASELINE, QUIET_KP,
    CD_MEAN, CD_SIGMA, CD_MIN, CD_MAX, CD_TAU_S, CD_DRIFT_FRAC,
    REENTRY_ALT,
)
from phase2_filter.config import R_DIAG_ELEMENTS
from phase2_filter.ekf import EKF
from phase2_filter.smoother import fraser_potter_smoother
from phase2_filter.test_statistics import compute_mahalanobis_distance
from propagator.orbital import MeanElements
from propagator.atmosphere import (
    sample_f107_phases, sample_storm_events, f107_at_time, kp_at_time,
)
from propagator.propagator import _step_rates
from phase2_filter.validation import _default_el0

from phase2_filter.convention_consistency_test import (
    generate_tle_like_observations,
    empirical_R_from_obs,
    _build_initial_state,
    _build_initial_covariance,
    SANITY_CEILING,
    R_EMP_CAP_FACTOR,
)

ALL_CONFOUNDERS = [
    "baseline",
    "f107_flat",
    "kp_quiet",
    "env_flat",
    "cd_nominal",
    "r_overconfident",
    "all",
]

CONFOUNDER_SPECS = {
    "baseline":        {"f107": "truth", "kp": "truth", "cd": "truth",   "r": 1.0},
    "f107_flat":       {"f107": "flat",   "kp": "truth", "cd": "truth",   "r": 1.0},
    "kp_quiet":        {"f107": "truth",  "kp": "quiet", "cd": "truth",   "r": 1.0},
    "env_flat":        {"f107": "flat",   "kp": "quiet", "cd": "truth",   "r": 1.0},
    "cd_nominal":      {"f107": "truth",  "kp": "truth", "cd": "nominal", "r": 1.0},
    "r_overconfident": {"f107": "truth",  "kp": "truth", "cd": "truth",   "r": 0.1},
    "all":             {"f107": "flat",   "kp": "quiet", "cd": "nominal", "r": 1.0},
}

CD_NOMINAL = 2.2
COV_GUARD = 1.0e12


def _parse_confounders(s: str) -> List[str]:
    names = [c.strip() for c in s.split(",")]
    for c in names:
        if c not in ALL_CONFOUNDERS:
            raise ValueError(
                f"Unknown confounder '{c}'. Valid: {ALL_CONFOUNDERS}")
    return names


# ---------------------------------------------------------------------------
# Stage B: full-environment truth propagation.
# ---------------------------------------------------------------------------

def propagate_truth_full_environment(
    el0: MeanElements,
    epoch_jd: float,
    duration_s: float,
    dt_s: float,
    area: float = 1.0,
    mass: float = 200.0,
    f107_base: float = F107_BASELINE,
    rng: Optional[np.random.Generator] = None,
    truth_substeps: int = 4,
) -> Tuple[Dict, List, Dict, float]:
    """Known-impulse-free truth under the FULL environment model."""
    if rng is None:
        rng = np.random.default_rng()
    Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))
    Cd = Cd_base
    t = 0.0
    n_steps = int(duration_s / dt_s)
    storms = sample_storm_events(duration_s, rng)
    f107_phases = sample_f107_phases(rng)
    cd_sigma_step = CD_DRIFT_FRAC * Cd_base * math.sqrt(2 * dt_s / CD_TAU_S)
    hist = {"t": [], "a": [], "e": [], "i": [],
            "Omega": [], "omega": [], "M": [], "Cd": []}
    el = MeanElements(a=el0.a, ecc=el0.ecc, inc=el0.inc,
                      raan=el0.raan, argp=el0.argp, M=el0.M)
    for _ in range(n_steps):
        if el.alt_m() < REENTRY_ALT:
            break
        f107 = f107_at_time(t, f107_phases, f_base=f107_base)
        kp = kp_at_time(t, storms)
        n_sub = max(1, truth_substeps)
        dt_sub = dt_s / n_sub
        for _ in range(n_sub):
            d_a, d_ecc, d_inc, d_raan, d_argp, d_M = _step_rates(
                el, Cd, area, mass, epoch_jd, t, f107, kp
            )
            el.a += d_a * dt_sub
            el.ecc = max(0.0, el.ecc + d_ecc * dt_sub)
            el.inc += d_inc * dt_sub
            el.raan = (el.raan + d_raan * dt_sub) % (2 * math.pi)
            el.argp = (el.argp + d_argp * dt_sub) % (2 * math.pi)
            el.M = (el.M + d_M * dt_sub) % (2 * math.pi)
        Cd += (1.0 / CD_TAU_S) * (Cd_base - Cd) * dt_s \
            + cd_sigma_step * rng.normal()
        Cd = float(np.clip(Cd, CD_MIN, CD_MAX))
        t += dt_s
        hist["t"].append(t)
        hist["a"].append(el.a)
        hist["e"].append(el.ecc)
        hist["i"].append(el.inc)
        hist["Omega"].append(el.raan)
        hist["omega"].append(el.argp)
        hist["M"].append(el.M)
        hist["Cd"].append(Cd)
    return hist, storms, f107_phases, Cd_base


# ---------------------------------------------------------------------------
# Stage E: environment-aware standard window filter.
# ---------------------------------------------------------------------------

def _run_window_filter(
    measurements: List[Tuple[float, np.ndarray]],
    epoch_jd: float,
    tau: float,
    q: float,
    edge_trim_s: float,
    R_diag: np.ndarray,
    cd_filter: float,
    f107_mode: str,
    kp_mode: str,
    storms: List,
    f107_phases: Dict,
    p0_bwd_inflate: float = 100.0,
    area: float = 1.0,
    mass: float = 200.0,
) -> Dict:
    """Fwd EKF + bwd EKF + Fraser-Potter smoother -> peak Mahalanobis
    (a-channel only), production 7-day edge trim."""
    R_diag = np.asarray(R_diag, dtype=float)
    el0 = MeanElements(a=measurements[0][1][0],
                       ecc=measurements[0][1][1],
                       inc=measurements[0][1][2],
                       raan=measurements[0][1][3],
                       argp=measurements[0][1][4],
                       M=measurements[0][1][5])
    x0 = _build_initial_state(el0)
    P0 = _build_initial_covariance(el0, R_diag)

    def env(t: float) -> Tuple[float, float]:
        f107 = (F107_BASELINE if f107_mode == "flat"
                else f107_at_time(t, f107_phases, f_base=F107_BASELINE))
        kp = QUIET_KP if kp_mode == "quiet" else kp_at_time(t, storms)
        return f107, kp

    ekf_fwd = EKF(x0, P0, cd_filter, area, mass, epoch_jd,
                  tau=tau, q=q, direction="forward")
    fwd_states, fwd_covs = [], []
    prev_t = 0.0
    for t_rel, z in measurements:
        dt = t_rel - prev_t
        f107, kp = env(t_rel)
        ekf_fwd.predict(dt, f107, kp)
        ekf_fwd.update(z, R_diag)
        fwd_states.append(ekf_fwd.x.copy())
        fwd_covs.append(ekf_fwd.P.copy())
        prev_t = t_rel
    if len(fwd_states) < 5:
        raise RuntimeError("too few forward epochs")

    x0_bwd = fwd_states[-1].copy()
    P0_bwd = fwd_covs[-1].copy() * p0_bwd_inflate
    ekf_bwd = EKF(x0_bwd, P0_bwd, cd_filter, area, mass, epoch_jd,
                  tau=tau, q=q, direction="backward")
    bwd_apriori_states, bwd_apriori_covs = [], []
    prev_t = measurements[-1][0]
    for t_rel, z in reversed(measurements):
        dt = t_rel - prev_t
        if abs(dt) > 1e-12:
            f107, kp = env(t_rel)
            ekf_bwd.predict(dt, f107, kp)
        bwd_apriori_states.append(ekf_bwd.x.copy())
        bwd_apriori_covs.append(ekf_bwd.P.copy())
        ekf_bwd.update(z, R_diag)
        prev_t = t_rel

    for s, P in zip(bwd_apriori_states, bwd_apriori_covs):
        if (not np.all(np.isfinite(s)) or not np.all(np.isfinite(P))
                or float(np.max(np.diag(P))) > COV_GUARD):
            raise RuntimeError("backward pass diverged")

    _, sm_covs = fraser_potter_smoother(
        fwd_states, fwd_covs,
        list(reversed(bwd_apriori_states)),
        list(reversed(bwd_apriori_covs)))

    times = [t for t, _ in measurements]
    dur = times[-1]
    keep = [i for i, t in enumerate(times)
            if edge_trim_s <= t <= dur - edge_trim_s]
    if len(keep) < 5:
        keep = list(range(len(times)))

    bwd_states_rev = list(reversed(bwd_apriori_states))
    bwd_covs_rev = list(reversed(bwd_apriori_covs))
    mask = np.zeros(9, dtype=bool)
    mask[0] = True
    maha = np.asarray(compute_mahalanobis_distance(
        [fwd_states[i] for i in keep], [fwd_covs[i] for i in keep],
        [bwd_states_rev[i] for i in keep], [bwd_covs_rev[i] for i in keep],
        [sm_covs[i] for i in keep], state_mask=mask))

    peak = float(np.max(maha))
    if not np.isfinite(peak) or abs(peak) > SANITY_CEILING:
        raise RuntimeError("non-finite/implausible peak")
    i_pk = int(np.argmax(maha))
    return {"peak": peak,
            "argmax_frac": float(keep[i_pk] / max(1, len(times) - 1))}


# ---------------------------------------------------------------------------
# One trial (module-level for multiprocessing).
# ---------------------------------------------------------------------------

def _run_one_trial(args_tuple) -> Dict:
    (seed, window_days, tau, q, tle_cadence_days, fit_span_days,
     truth_dt_hours, max_nfev, edge_trim_days, r_scale, p0_bwd_inflate,
     fit_rms_max_m, confounders, r_diag_prod, cache_dir) = args_tuple

    # ---- PERF 3: stream cache -------------------------------------
    cpath = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cpath = os.path.join(cache_dir, f"trial_{seed}.npz")

    epoch_jd = 2460000.5
    area, mass = 1.0, 200.0
    duration_s = window_days * 86400.0
    truth_dt_s = truth_dt_hours * 3600.0

    if cpath is not None and os.path.exists(cpath):
        d = np.load(cpath, allow_pickle=True)
        blob = d["blob"]
        storms, f107_phases, geom, diag = (
            blob[0], blob[1], blob[2], blob[3])
        meas_t = d["meas_t"]
        meas_z = d["meas_z"]
        measurements = [(float(meas_t[i]), np.asarray(meas_z[i], dtype=float))
                        for i in range(len(meas_t))]
        cd_base = float(d["cd_base"])
        R_used = np.asarray(d["R_used"], dtype=float)
        R_emp_raw = np.asarray(d["R_emp_raw"], dtype=float)
        fit_rms_m = float(d["fit_rms_m"])
    else:
        rng = np.random.default_rng(seed)

        # Identical geometry draw to the pooled synthetic null.
        el0 = _default_el0(alt_km=800.0, inc_deg=98.0)
        el0 = MeanElements(
            a=el0.a + (rng.uniform(700.0, 1000.0) - 800.0) * 1000.0,
            ecc=rng.uniform(1.0e-5, 1.0e-3),
            inc=math.radians(rng.uniform(50.0, 105.0)),
            raan=rng.uniform(0.0, 2 * math.pi),
            argp=rng.uniform(0.0, 2 * math.pi),
            M=rng.uniform(0.0, 2 * math.pi),
        )
        geom = {"a0_m": float(el0.a), "e0": float(el0.ecc),
                "inc0_deg": float(math.degrees(el0.inc))}

        # Stage B: truth with the full environment realized.
        try:
            truth, storms, f107_phases, cd_base = \
                propagate_truth_full_environment(
                    el0, epoch_jd, duration_s, truth_dt_s,
                    area=area, mass=mass, rng=rng)
        except Exception as e:
            return {"seed": seed, "ok": False, "stage": "truth",
                    "error": f"{type(e).__name__}: {e}", "geometry": geom}

        # Stage C: the pooled null's TLE-fitting convention.
        try:
            obs, diag = generate_tle_like_observations(
                truth, epoch_jd, tle_cadence_days, fit_span_days,
                "sgp4_consistent", True, max_nfev=max_nfev,
                fit_rms_max_m=fit_rms_max_m)
        except Exception as e:
            return {"seed": seed, "ok": False, "stage": "tle-fit",
                    "error": f"{type(e).__name__}: {e}", "geometry": geom}
        if len(obs) < 10:
            return {"seed": seed, "ok": False, "stage": "tle-fit",
                    "error": f"too few obs ({len(obs)}); fit failures "
                             f"{diag.get('n_fit_failures', '?')}/"
                             f"{diag.get('n_fit_attempts', '?')}; QC-rejected "
                             f"{diag.get('n_rejected_fits', '?')}",
                    "geometry": geom}

        # Stage D: honest covariance + matched-deficit noise.
        R_emp_raw = empirical_R_from_obs(obs)
        if R_emp_raw is None:
            R_emp_raw = np.zeros(6)
        R_used = np.minimum(
            np.maximum(R_emp_raw, r_diag_prod),
            r_diag_prod * R_EMP_CAP_FACTOR,
        ) * r_scale
        noise_scale = np.sqrt(np.maximum(R_used - R_emp_raw, 0.0))
        measurements = []
        for t, z, _ in obs:
            zn = z.copy()
            zn += rng.normal(scale=noise_scale)
            measurements.append((t, zn))
        fit_rms_m = float(np.mean([o[2] for o in obs]))

        # ---- PERF 3: save the stream for future scenario re-runs ---
        if cpath is not None:
            try:
                np.savez_compressed(
                    cpath,
                    blob=np.array((storms, f107_phases, geom, diag),
                                  dtype=object),
                    meas_t=np.array([t for t, _ in measurements]),
                    meas_z=np.stack([z for _, z in measurements]),
                    cd_base=float(cd_base),
                    R_used=R_used, R_emp_raw=R_emp_raw,
                    fit_rms_m=fit_rms_m)
            except Exception:
                pass   # cache is best-effort; never fail a trial over it

    # Stage E: every confounder on the same measurements (paired design).
    peaks: Dict[str, Optional[float]] = {}
    errors: Dict[str, Optional[str]] = {}
    for c in confounders:
        spec = CONFOUNDER_SPECS[c]
        r_filter = R_used * spec["r"]
        cd_filter = cd_base if spec["cd"] == "truth" else CD_NOMINAL
        try:
            res = _run_window_filter(
                measurements, epoch_jd, tau, q,
                edge_trim_s=edge_trim_days * 86400.0,
                R_diag=r_filter, cd_filter=cd_filter,
                f107_mode=spec["f107"], kp_mode=spec["kp"],
                storms=storms, f107_phases=f107_phases,
                p0_bwd_inflate=p0_bwd_inflate,
                area=area, mass=mass)
            peaks[c] = res["peak"]
            errors[c] = None
        except Exception as e:
            peaks[c] = None
            errors[c] = f"{type(e).__name__}: {str(e)[:120]}"

    return {
        "seed": seed, "ok": True, "geometry": geom,
        "n_obs": len(measurements),
        "r_used": [float(v) for v in R_used],
        "r_emp_raw": [float(v) for v in R_emp_raw],
        "fit_rms_m": fit_rms_m,
        "cd_truth_base": float(cd_base),
        "peaks": peaks, "errors": errors,
        **{k: v for k, v in diag.items()
           if isinstance(v, (int, float, str, bool, type(None)))},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--confounders", type=str,
                   default="baseline,f107_flat,kp_quiet,env_flat,cd_nominal",
                   help=f"Comma-separated from: {','.join(ALL_CONFOUNDERS)}")
    p.add_argument("--n-trials", type=int, default=2000)
    p.add_argument("--window-days", type=float, default=90.0)
    p.add_argument("--tau", type=float, default=3e4)
    p.add_argument("--q", type=float, default=1e-19)
    p.add_argument("--truth-dt-hours", type=float, default=6.0)
    p.add_argument("--tle-cadence-days", type=float, default=2.0)
    p.add_argument("--fit-span-days", type=float, default=4.0)
    p.add_argument("--max-nfev", type=int, default=2000)
    p.add_argument("--edge-trim-days", type=float, default=7.0)
    p.add_argument("--fit-rms-max-m", type=float, default=5000.0)
    p.add_argument("--r-scale", type=float, default=1.0)
    p.add_argument("--p0-bwd-inflate", type=float, default=100.0)
    p.add_argument("--threshold", type=float, default=417.94)
    p.add_argument("--seed-offset", type=int, default=600000)
    p.add_argument("--n-workers", type=int, default=None)
    # ---- PERF knobs -------------------------------------------------
    p.add_argument("--max-tasks-per-child", type=int, default=4,
                   help="worker recycle count (BSOD guard for long runs). "
                        "0 = never recycle; use 0 for short smoke runs "
                        "where spawn/re-import overhead dominates.")
    p.add_argument("--resume", type=int, default=1,
                   help="1 (default): continue from <output>.partial.json "
                        "if it exists. 0: start fresh.")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="directory for per-seed fitted-stream caches. "
                        "Stage A-D is then computed once per seed forever; "
                        "re-runs with different confounders/r-scale/"
                        "threshold reuse the identical streams.")
    p.add_argument("--output", default="results/gate2_confounders_v2.json")
    args = p.parse_args()

    confounders = _parse_confounders(args.confounders)
    n_workers = args.n_workers or max(1, min(4, mp.cpu_count()))
    r_diag_prod = np.asarray(R_DIAG_ELEMENTS, dtype=float)

    print(f"Gate 2 confounders (V2, pooled-null-matched): {confounders}")
    print(f"  n-trials={args.n_trials}  threshold={args.threshold:.3f}  "
          f"tau={args.tau:.2e}  q={args.q:.2e}  window={args.window_days:.0f}d")
    print(f"  truth-dt={args.truth_dt_hours}h  cadence={args.tle_cadence_days}d  "
          f"fit-span={args.fit_span_days}d  max-nfev={args.max_nfev}")
    print(f"  edge-trim={args.edge_trim_days}d  fit-rms-max={args.fit_rms_max_m:.0f}m  "
          f"r-scale={args.r_scale}  p0-bwd-inflate={args.p0_bwd_inflate}")
    if args.cache_dir:
        print(f"  stream cache: {args.cache_dir}")
    print()

    part_path = args.output + ".partial.json"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---- PERF 2: resume from the partial checkpoint -----------------
    results = []
    done_seeds = set()
    if args.resume and os.path.exists(part_path):
        try:
            with open(part_path) as f:
                prev = json.load(f).get("partial", [])
            results = list(prev)
            done_seeds = {r.get("seed") for r in prev if r.get("ok")}
            print(f"RESUME: reusing {len(done_seeds)} finished trials "
                  f"from {part_path}")
        except Exception as e:
            print(f"Could not read partial checkpoint ({e}); starting fresh")
            results, done_seeds = [], set()

    all_tasks = [
        (args.seed_offset + i, args.window_days, args.tau, args.q,
         args.tle_cadence_days, args.fit_span_days, args.truth_dt_hours,
         args.max_nfev, args.edge_trim_days, args.r_scale,
         args.p0_bwd_inflate, args.fit_rms_max_m, confounders,
         r_diag_prod, args.cache_dir)
        for i in range(args.n_trials)
    ]
    tasks = [t for t in all_tasks if t[0] not in done_seeds]
    if len(tasks) < len(all_tasks):
        print(f"Skipped {len(all_tasks) - len(tasks)} already-done seeds")
    n_tasks = len(tasks)

    t0 = time.time()
    pool_kwargs = {"maxtasksperchild": args.max_tasks_per_child or None}
    if n_tasks:
        with mp.Pool(n_workers, **pool_kwargs) as pool:
            chunk = max(1, n_tasks // (n_workers * 8))
            done = 0
            for r in pool.imap_unordered(_run_one_trial, tasks, chunksize=chunk):
                results.append(r)
                done += 1
                if done % max(1, n_tasks // 20) == 0 or done == n_tasks:
                    elapsed = time.time() - t0
                    eta_min = (n_tasks - done) * (elapsed / max(1, done)) / 60.0
                    print(f"  [{done:>5}/{n_tasks}] {100.0*done/n_tasks:5.1f}%  "
                          f"ETA {eta_min:.1f}m")
                    with open(part_path, "w") as f:
                        json.dump({"config": vars(args), "partial": results}, f)
    else:
        print("Nothing to do (all trials already complete).")

    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    fail_rate = len(failed) / max(1, len(results))
    print(f"\nFinished {len(results)} trials in {(time.time()-t0)/60:.1f} min; "
          f"construction failures: {len(failed)} ({100*fail_rate:.2f}%)")
    if failed:
        from collections import Counter
        for err, n in Counter(r.get("error", "?")[:100]
                               for r in failed).most_common(5):
            print(f"  {n}x {err}")
        if fail_rate > 0.02:
            print("  WARNING: failure rate > 2%. If failures correlate with "
                  "extreme peaks, the reported FP rates are biased LOW; fix "
                  "the failure mode before quoting them.")

    pct_ladder = [50, 90, 99, 99.5, 99.9, 99.95, 99.97, 99.99]
    print(f"\n{'Scenario':<20} {'Nvalid':>7} {'Excl':>6} {'Mean':>9} "
          f"{'p99.9':>9} {'FP':>5} {'FP rate':>9} {'E[design]':>10}")
    print("-" * 78)

    summary = {}
    for c in confounders:
        valid = [r for r in ok if r["peaks"].get(c) is not None]
        excluded = [r for r in ok if r["peaks"].get(c) is None]
        n_valid = len(valid)
        entry = {
            "n_valid_windows": n_valid,
            "n_excluded_divergent": len(excluded),
            "exclusion_rate": len(excluded) / max(1, len(ok)),
            "n_construction_failures": len(failed),
        }
        if n_valid == 0:
            entry.update({"mean_peak": None, "std_peak": None,
                          "percentiles": None, "max_peak": None,
                          "fp_count": 0, "fp_rate": None,
                          "fp_rate_percent": None})
            summary[c] = entry
            print(f"{c:<20} {0:>7} {len(excluded):>6} {'N/A':>9} "
                  f"{'N/A':>9} {0:>5} {'N/A':>9} {'N/A':>10}")
            continue
        peaks = np.array([r["peaks"][c] for r in valid])
        fp_count = int(np.sum(peaks > args.threshold))
        fp_rate = fp_count / n_valid
        entry.update({
            "mean_peak": float(np.mean(peaks)),
            "std_peak": float(np.std(peaks)),
            "percentiles": {str(pc): float(np.percentile(peaks, pc))
                            for pc in pct_ladder},
            "max_peak": float(np.max(peaks)),
            "fp_count": fp_count,
            "fp_rate": float(fp_rate),
            "fp_rate_percent": float(100.0 * fp_rate),
            "expected_at_design": float(n_valid * 0.001),
            "mean_r_used": [float(v) for v in
                            np.mean([r["r_used"] for r in valid], axis=0)],
        })
        summary[c] = entry
        p999 = entry["percentiles"]["99.9"]
        print(f"{c:<20} {n_valid:>7} {len(excluded):>6} "
              f"{entry['mean_peak']:>9.3f} {p999:>9.3f} {fp_count:>5} "
              f"{100*fp_rate:>8.4f}% {entry['expected_at_design']:>10.2f}")

    print(f"\nThreshold (design eta): {args.threshold:.3f}")
    if "baseline" in summary and summary["baseline"].get("fp_rate") is not None:
        print(f"Consistency check: baseline FP rate "
              f"{summary['baseline']['fp_rate_percent']:.4f}% vs the pooled "
              f"null's 0.1% design rate; each confounder's excess over "
              f"baseline is that confounder's isolated effect (paired trials).")

    output_data = {
        "config": {
            **vars(args),
            "confounders": confounders,
            "cov_guard": COV_GUARD,
        },
        "summary": summary,
        "trials": ok,
        "failed": failed,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    if os.path.exists(part_path):
        os.remove(part_path)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()