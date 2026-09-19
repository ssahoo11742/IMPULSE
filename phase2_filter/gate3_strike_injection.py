#!/usr/bin/env python3
"""Gate 3: Fleet-scale strike injection for detection efficiency curve ε(Δv).

For each object in a stratified fleet sample, injects a single debris strike
at mid-duration across a grid of Δv magnitudes. Runs the filter with the
degraded "all" confounder model (flat F10.7, quiet Kp, nominal Cd=2.2) and
compares peak Mahalanobis distance to the Gate 1 threshold.

Produces:
  ε(Δv) = fraction of strikes detected at each Δv
  DV_50  = Δv where ε = 50%
  DV_90  = Δv where ε = 90%

Usage:
    python -m phase2_filter.gate3_strike_injection \
        --tle-file TLE/real_tles.tle \
        --meta TLE/real_tles.tle.meta.json \
        --durations TLE/real_tles.tle.durations.json \
        --stratified-sample 1000 \
        --threshold 13.258 \
        --n-workers 9 \
        --output results/gate3_efficiency.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from constants.constants import MU, R_EARTH, F107_BASELINE, QUIET_KP
from constants.constants import CD_MEAN, CD_SIGMA, CD_MIN, CD_MAX, CD_TAU_S, CD_DRIFT_FRAC, REENTRY_ALT
from phase2_filter.config import DEFAULT_TAU_S, DEFAULT_Q, R_DIAG_ELEMENTS
from phase2_filter.ekf import EKF
from phase2_filter.smoother import fraser_potter_smoother
from phase2_filter.test_statistics import compute_mahalanobis_distance
from phase2_filter.dynamics import hk_from_e_argp
from propagator.orbital import MeanElements, mean_to_true_anomaly
from propagator.atmosphere import sample_f107_phases, sample_storm_events, f107_at_time, kp_at_time
from propagator.propagator import _step_rates
from propagator.debris_impacts import gauss_vop
from TLE.tle_io import parse_tle, load_tles_from_file, tle_to_elements

SANITY_CEILING = 1.0e4

# ---------------------------------------------------------------------------
# Object loading & stratified sampling (from gate2_confounders)
# ---------------------------------------------------------------------------

def stratified_sample(objects_to_process, n_target=500, n_inc_bins=6, n_sma_bins=6, seed=42):
    rng_local = np.random.default_rng(seed)
    incs = np.array([o["inclination"] for o in objects_to_process])
    smas = np.array([0.5 * (o["tle_data"]["periapsis"] + o["tle_data"]["apoapsis"])
                      for o in objects_to_process])
    inc_edges = np.linspace(incs.min(), incs.max() + 1e-6, n_inc_bins + 1)
    sma_edges = np.linspace(smas.min(), smas.max() + 1e-6, n_sma_bins + 1)
    buckets = {}
    for idx, obj in enumerate(objects_to_process):
        ib = min(np.searchsorted(inc_edges, incs[idx], side="right") - 1, n_inc_bins - 1)
        sb = min(np.searchsorted(sma_edges, smas[idx], side="right") - 1, n_sma_bins - 1)
        buckets.setdefault((ib, sb), []).append(idx)
    selected = []
    per_bucket = max(1, n_target // len(buckets))
    for key, idxs in buckets.items():
        rng_local.shuffle(idxs)
        selected.extend(idxs[:per_bucket])
    if len(selected) < n_target:
        remaining = [i for i in range(len(objects_to_process)) if i not in set(selected)]
        rng_local.shuffle(remaining)
        selected.extend(remaining[:n_target - len(selected)])
    selected = selected[:n_target]
    print(f"Stratified sample: {len(selected)} objects across {len(buckets)} bins")
    return [objects_to_process[i] for i in selected]


def load_objects(tle_file: str, meta_file: str, durations_file: str,
                 max_duration_days: float = 7300.0, window_days: float = 90.0):
    tle_tuples = load_tles_from_file(tle_file)
    with open(meta_file) as f:
        meta_records = json.load(f)
    tle_map = {}
    for name, line1, line2 in tle_tuples:
        tle_dict = parse_tle(name, line1, line2)
        tle_map[tle_dict.get("norad_id", "unknown")] = tle_dict
    with open(durations_file) as f:
        durations_map = json.load(f)
    objects_to_process = []
    for rec in meta_records:
        norad_id = str(rec.get("NORAD_CAT_ID", ""))
        if norad_id not in tle_map or norad_id not in durations_map:
            continue
        duration_days = durations_map[norad_id].get("duration_days", 0)
        duration_days = min(duration_days, max_duration_days)
        if duration_days < window_days:
            continue
        tle_data = tle_map[norad_id]
        tle_data["inclination"] = float(rec.get("INCLINATION", 0))
        tle_data["periapsis"] = float(rec.get("PERIAPSIS", 0))
        tle_data["apoapsis"] = float(rec.get("APOAPSIS", 0))
        objects_to_process.append({
            "norad_id": norad_id,
            "tle_data": tle_data,
            "duration_days": duration_days,
            "inclination": tle_data["inclination"],
            "sma_km": 0.5 * (tle_data["periapsis"] + tle_data["apoapsis"]),
        })
    return objects_to_process


# ---------------------------------------------------------------------------
# Truth propagation with strike (adapted from validation.py)
# ---------------------------------------------------------------------------

def _propagate_truth_with_strike(
    el0: MeanElements,
    epoch_jd: float,
    duration_s: float,
    dt_s: float,
    area: float,
    mass: float,
    strike_time: float,
    strike_dv_mag: float,
    rng: np.random.Generator,
    Cd_base: Optional[float] = None,
    f107_base: float = F107_BASELINE,
) -> Dict:
    """Propagate truth trajectory with a single strike injected."""
    if Cd_base is None:
        Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))

    el = el0.copy()
    Cd = Cd_base
    t = 0.0
    n_steps = int(duration_s / dt_s)

    storms = sample_storm_events(duration_s, rng)
    f107_phases = sample_f107_phases(rng)
    cd_sigma_step = CD_DRIFT_FRAC * Cd_base * math.sqrt(2 * dt_s / CD_TAU_S)

    hist = {"t": [], "a": [], "e": [], "i": [],
            "Omega": [], "omega": [], "M": [], "Cd": []}

    for _ in range(n_steps):
        if el.alt_m() < REENTRY_ALT:
            break

        f107 = f107_at_time(t, f107_phases, f_base=f107_base)
        kp = kp_at_time(t, storms)

        d_a, d_ecc, d_inc, d_raan, d_argp, d_M = _step_rates(el, Cd, area, mass, epoch_jd, t, f107, kp)

        el.a += d_a * dt_s
        el.ecc = max(0.0, el.ecc + d_ecc * dt_s)
        el.inc += d_inc * dt_s
        el.raan = (el.raan + d_raan * dt_s) % (2 * math.pi)
        el.argp = (el.argp + d_argp * dt_s) % (2 * math.pi)
        el.M = (el.M + d_M * dt_s) % (2 * math.pi)

        # --- single strike injection ---
        if t <= strike_time < t + dt_s:
            dv_rsw = np.array([0.0, -strike_dv_mag, 0.0])
            nu, _ = mean_to_true_anomaly(el.M, el.ecc)
            da, de, di, dOm, darg = gauss_vop(el.a, el.ecc, el.inc, el.argp, nu, dv_rsw)
            el.a += da
            el.ecc = max(0.0, el.ecc + de)
            el.inc += di
            el.raan = (el.raan + dOm) % (2 * math.pi)
            el.argp = (el.argp + darg) % (2 * math.pi)

        Cd += (1.0 / CD_TAU_S) * (Cd_base - Cd) * dt_s + cd_sigma_step * rng.normal()
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

    return hist


def _generate_measurements(truth_hist: Dict, rng: np.random.Generator) -> List[Tuple[float, np.ndarray]]:
    n = len(truth_hist["t"])
    meas = []
    for i in range(n):
        z = np.array([
            truth_hist["a"][i], truth_hist["e"][i], truth_hist["i"][i],
            truth_hist["Omega"][i], truth_hist["omega"][i], truth_hist["M"][i]
        ])
        z += rng.normal(scale=np.sqrt(R_DIAG_ELEMENTS))
        meas.append((truth_hist["t"][i], z))
    return meas


# ---------------------------------------------------------------------------
# Filter with degraded "all" confounder model
# ---------------------------------------------------------------------------

def _build_initial_state(el: MeanElements) -> np.ndarray:
    h, k = hk_from_e_argp(el.ecc, el.argp)
    x0 = np.zeros(9)
    x0[0] = el.a
    x0[1] = h
    x0[2] = k
    x0[3] = el.inc
    x0[4] = el.raan
    x0[5] = el.M
    return x0


def _build_initial_covariance(el: MeanElements) -> np.ndarray:
    e, argp = el.ecc, el.argp
    R_h = math.sin(argp)**2 * R_DIAG_ELEMENTS[1] + (e * math.cos(argp))**2 * R_DIAG_ELEMENTS[4]
    R_k = math.cos(argp)**2 * R_DIAG_ELEMENTS[1] + (e * math.sin(argp))**2 * R_DIAG_ELEMENTS[4]
    P0 = np.diag([
        R_DIAG_ELEMENTS[0], R_h, R_k, R_DIAG_ELEMENTS[2], R_DIAG_ELEMENTS[3], R_DIAG_ELEMENTS[5],
        1.0e-6, 1.0e-6, 1.0e-6
    ])
    return P0


def _run_degraded_filter(
    measurements: List[Tuple[float, np.ndarray]],
    el0: MeanElements,
    Cd_filter: float,
    area: float,
    mass: float,
    epoch_jd: float,
    tau: float,
    q: float,
) -> Dict:
    """Run forward+backward+smoother with degraded (all-confounder) settings."""
    x0 = _build_initial_state(el0)
    P0 = _build_initial_covariance(el0)

    # Forward
    ekf_fwd = EKF(x0, P0, Cd_filter, area, mass, epoch_jd, tau=tau, q=q, direction="forward")
    fwd_states, fwd_covs = [], []
    prev_t = 0.0
    for t, z in measurements:
        dt = t - prev_t
        ekf_fwd.predict(dt, F107_BASELINE, QUIET_KP)
        ekf_fwd.update(z, R_DIAG_ELEMENTS)
        fwd_states.append(ekf_fwd.x.copy())
        fwd_covs.append(ekf_fwd.P.copy())
        prev_t = t

    # Backward
    x0_bwd = fwd_states[-1].copy()
    P0_bwd = fwd_covs[-1].copy() * 100.0
    ekf_bwd = EKF(x0_bwd, P0_bwd, Cd_filter, area, mass, epoch_jd, tau=tau, q=q, direction="backward")
    bwd_apriori_states, bwd_apriori_covs = [], []
    prev_t = measurements[-1][0]
    for t, z in reversed(measurements):
        dt = t - prev_t
        if abs(dt) > 1e-12:
            ekf_bwd.predict(dt, F107_BASELINE, QUIET_KP)
        bwd_apriori_states.append(ekf_bwd.x.copy())
        bwd_apriori_covs.append(ekf_bwd.P.copy())
        ekf_bwd.update(z, R_DIAG_ELEMENTS)
        prev_t = t

    # Smoother
    sm_states, sm_covs = fraser_potter_smoother(
        fwd_states, fwd_covs,
        list(reversed(bwd_apriori_states)),
        list(reversed(bwd_apriori_covs))
    )

    return {
        "fwd_states": fwd_states,
        "fwd_covs": fwd_covs,
        "bwd_states": list(reversed(bwd_apriori_states)),
        "bwd_covs": list(reversed(bwd_apriori_covs)),
        "sm_states": sm_states,
        "sm_covs": sm_covs,
    }


# ---------------------------------------------------------------------------
# Per-object strike sweep
# ---------------------------------------------------------------------------

def run_strike_sweep_on_object(
    tle_data: Dict,
    duration_days: float,
    tau: float,
    q: float,
    dv_grid: np.ndarray,
    threshold: float,
    seed: int,
) -> Dict:
    """For one object, inject strikes at each Δv and record detection."""
    try:
        rng = np.random.default_rng(seed)
        epoch_jd = tle_data["epoch_jd"]
        dt_s = 86400.0
        area, mass = 1.0, 200.0
        Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))
        Cd_filter = 2.2  # degraded: population mean

        duration_s = duration_days * 86400.0
        strike_time = duration_s / 2.0

        el0 = tle_to_elements(tle_data)

        results = {}
        mask = np.zeros(9, dtype=bool)
        mask[0] = True  # restrict to 'a'

        for dv in dv_grid:
            try:
                # Propagate truth with this Δv
                truth_hist = _propagate_truth_with_strike(
                    el0, epoch_jd, duration_s, dt_s, area, mass,
                    strike_time, float(dv), rng, Cd_base=Cd_base
                )
                meas = _generate_measurements(truth_hist, rng)

                # Run degraded filter
                filt = _run_degraded_filter(meas, el0, Cd_filter, area, mass, epoch_jd, tau, q)

                # Peak Mahalanobis around strike
                maha = compute_mahalanobis_distance(
                    filt["fwd_states"], filt["fwd_covs"],
                    filt["bwd_states"], filt["bwd_covs"],
                    filt["sm_covs"], state_mask=mask
                )
                times = np.array([t for t, _ in meas])
                window = (times >= strike_time - 86400) & (times <= strike_time + 86400)
                peak = float(maha[window].max()) if window.any() else float(maha.max())

                if not np.isfinite(peak) or abs(peak) > SANITY_CEILING:
                    results[float(dv)] = {"detected": False, "peak_maha": None, "error": "diverged"}
                else:
                    results[float(dv)] = {"detected": peak > threshold, "peak_maha": peak}
            except Exception as e:
                results[float(dv)] = {"detected": False, "peak_maha": None, "error": str(e)}

        return {
            "norad_id": tle_data.get("norad_id", "unknown"),
            "success": True,
            "results": results,
        }
    except Exception as e:
        return {
            "norad_id": tle_data.get("norad_id", "unknown"),
            "success": False,
            "error": str(e),
            "results": {},
        }


def process_one_wrapper(args):
    tle_data, duration_days, tau, q, dv_grid, threshold, seed = args
    return run_strike_sweep_on_object(tle_data, duration_days, tau, q, dv_grid, threshold, seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gate 3: Fleet strike injection for ε(Δv)")
    parser.add_argument("--tle-file", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--durations", required=True)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU_S)
    parser.add_argument("--q", type=float, default=DEFAULT_Q)
    parser.add_argument("--duration-days", type=float, default=90.0,
                        help="Duration for each strike test [days]")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Gate 1 detection threshold (Mahalanobis)")
    parser.add_argument("--dv-min", type=float, default=1.0e-4)
    parser.add_argument("--dv-max", type=float, default=0.5)
    parser.add_argument("--n-dv", type=int, default=14,
                        help="Number of Δv points in log-spaced grid")
    parser.add_argument("--stratified-sample", type=int, default=1000)
    parser.add_argument("--n-inc-bins", type=int, default=6)
    parser.add_argument("--n-sma-bins", type=int, default=6)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=400000)
    parser.add_argument("--output", default="results/gate3_efficiency.json")
    args = parser.parse_args()

    dv_grid = np.logspace(np.log10(args.dv_min), np.log10(args.dv_max), args.n_dv)
    print(f"Gate 3: Fleet strike injection")
    print(f"  Threshold:     {args.threshold:.3f}")
    print(f"  tau={args.tau:.3e}, q={args.q:.3e}")
    print(f"  Δv grid:       {args.dv_min:.2e} to {args.dv_max:.2e} m/s ({args.n_dv} points)")
    print(f"  Duration:      {args.duration_days:.0f} days per object")
    print()

    # Load objects
    objects_to_process = load_objects(
        args.tle_file, args.meta, args.durations,
        max_duration_days=args.duration_days,
        window_days=args.duration_days
    )
    objects_to_process = stratified_sample(
        objects_to_process, n_target=args.stratified_sample,
        n_inc_bins=args.n_inc_bins, n_sma_bins=args.n_sma_bins
    )

    print(f"Processing {len(objects_to_process)} objects...")

    tasks = [
        (
            obj["tle_data"], args.duration_days,
            args.tau, args.q, dv_grid, args.threshold,
            args.seed_offset + idx
        )
        for idx, obj in enumerate(objects_to_process)
    ]

    t0 = time.time()
    all_results = []

    with mp.Pool(args.n_workers) as pool:
        iter_results = pool.imap_unordered(
            process_one_wrapper, tasks,
            chunksize=max(1, len(tasks) // (args.n_workers * 4))
        )
        for i, result in enumerate(iter_results, 1):
            all_results.append(result)
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                pct = 100.0 * i / len(tasks)
                elapsed = time.time() - t0
                eta = (len(tasks) - i) * (elapsed / i) if i > 0 else 0
                print(f"  [{i:>4}/{len(tasks)}] {pct:>5.1f}% | ETA: {eta/60:.1f}m")

    elapsed_total = time.time() - t0
    successful = [r for r in all_results if r["success"]]
    print(f"\nFinished {len(all_results)} objects in {elapsed_total/60:.1f} min")
    print(f"Successful: {len(successful)}, Failed: {len(all_results) - len(successful)}")

    # Aggregate ε(Δv)
    detections = {float(dv): [] for dv in dv_grid}
    peaks = {float(dv): [] for dv in dv_grid}

    for r in successful:
        for dv_str, res in r["results"].items():
            dv = float(dv_str)
            if res.get("peak_maha") is not None:
                detections[dv].append(1 if res["detected"] else 0)
                peaks[dv].append(res["peak_maha"])

    print(f"\n{'='*60}")
    print("DETECTION EFFICIENCY ε(Δv)")
    print(f"{'='*60}")
    print(f"{'Δv (m/s)':>12} {'N trials':>10} {'ε(Δv)':>10} {'mean peak':>12} {'std peak':>12}")
    print(f"{'-'*60}")

    efficiency = {}
    for dv in dv_grid:
        dv_f = float(dv)
        n = len(detections[dv_f])
        if n == 0:
            print(f"{dv_f:12.5f} {0:>10} {'N/A':>10} {'N/A':>12} {'N/A':>12}")
            efficiency[dv_f] = {"n": 0, "epsilon": None, "mean_peak": None, "std_peak": None}
            continue
        eps = np.mean(detections[dv_f])
        mean_peak = np.mean(peaks[dv_f]) if peaks[dv_f] else None
        std_peak = np.std(peaks[dv_f]) if peaks[dv_f] else None
        print(f"{dv_f:12.5f} {n:>10} {eps:>10.4f} {mean_peak:>12.3f} {std_peak:>12.3f}")
        efficiency[dv_f] = {
            "n": n, "epsilon": float(eps),
            "mean_peak": float(mean_peak) if mean_peak is not None else None,
            "std_peak": float(std_peak) if std_peak is not None else None,
        }

    # Interpolate DV_50 and DV_90
    dvs = np.array([float(dv) for dv in dv_grid])
    eps_arr = np.array([efficiency[float(dv)]["epsilon"] for dv in dv_grid])
    valid = ~np.isnan(eps_arr)

    if valid.any():
        dvs_v = dvs[valid]
        eps_v = eps_arr[valid]
        sort_idx = np.argsort(eps_v)
        dvs_v = dvs_v[sort_idx]
        eps_v = eps_v[sort_idx]

        dv50 = float(np.interp(0.5, eps_v, dvs_v)) if eps_v.min() <= 0.5 <= eps_v.max() else None
        dv90 = float(np.interp(0.9, eps_v, dvs_v)) if eps_v.min() <= 0.9 <= eps_v.max() else None

        print(f"\n{'='*60}")
        if dv50 is not None:
            print(f"DV_50 (50% detection) = {dv50:.5f} m/s")
        else:
            print("DV_50: sweep did not bracket 50% — widen Δv range")
        if dv90 is not None:
            print(f"DV_90 (90% detection) = {dv90:.5f} m/s")
        else:
            print("DV_90: sweep did not bracket 90% — widen Δv range")

    # Save
    output_data = {
        "config": {
            "tau": args.tau, "q": args.q,
            "duration_days": args.duration_days,
            "threshold": args.threshold,
            "dv_min": args.dv_min, "dv_max": args.dv_max, "n_dv": args.n_dv,
            "n_objects": len(objects_to_process),
            "n_successful": len(successful),
        },
        "efficiency": efficiency,
        "dv50": dv50 if 'dv50' in dir() else None,
        "dv90": dv90 if 'dv90' in dir() else None,
        "per_object_results": all_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()