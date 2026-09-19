#!/usr/bin/env python3
"""Gate 2 confounder analysis for DRIFTS.

Runs the fleet null test with controlled model mismatches between
truth propagation (always full model) and filter estimation (degraded
model depending on confounder scenario).

Individual confounder runs diagnose which mismatches are dangerous.
The combined run gives the extrinsic false-positive rate r_extrinsic.

Truth always gets:
  - Time-varying F10.7 (annual + Carrington cycles)
  - Sampled geomagnetic storm events
  - Drifting drag coefficient (FOGM around object's true Cd_base)
  - Full SRP and third-body perturbations
  - Actual measurement noise R_DIAG_ELEMENTS

Filter gets a degraded version per confounder:
  baseline       : matched to truth (reproduces Gate 1)
  f107_flat      : constant F107_BASELINE (no solar cycle knowledge)
  kp_quiet       : constant QUIET_KP (no storm knowledge)
  env_flat       : both flat F10.7 and quiet Kp (no space weather at all)
  cd_nominal     : fixed Cd=2.2 (population mean, not object's true Cd)
  r_overconfident: R_DIAG_ELEMENTS / 10 (thinks TLEs are 10x better)
  all            : combined realistic mismatch (env_flat + cd_nominal)
                   Does NOT include r_overconfident (scaling artifact)

Usage:
    # Individual confounders (diagnostic)
    python -m phase2_filter.gate2_confounders \\
        --tle-file TLE/real_tles.tle \\
        --meta TLE/real_tles.tle.meta.json \\
        --durations TLE/real_tles.tle.durations.json \\
        --stratified-sample 100 \\
        --confounders baseline,f107_flat,kp_quiet,env_flat,cd_nominal,r_overconfident \\
        --n-workers 8 --threshold 13.258 \\
        --output results/gate2_confounders.json

    # Combined run (the actual r_extrinsic)
    python -m phase2_filter.gate2_confounders \\
        --tle-file TLE/real_tles.tle \\
        --meta TLE/real_tles.tle.meta.json \\
        --durations TLE/real_tles.tle.durations.json \\
        --stratified-sample 1000 \\
        --confounders all \\
        --n-workers 9 --threshold 13.258 \\
        --output results/gate2_all.json
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
from phase2_filter.config import DEFAULT_TAU_S, DEFAULT_Q, R_DIAG_ELEMENTS
from phase2_filter.ekf import EKF
from phase2_filter.smoother import fraser_potter_smoother
from phase2_filter.test_statistics import compute_mahalanobis_distance
from phase2_filter.dynamics import hk_from_e_argp
from propagator.orbital import MeanElements
from propagator.atmosphere import sample_f107_phases, sample_storm_events, f107_at_time, kp_at_time
from propagator.propagator import _step_rates
from TLE.tle_io import parse_tle, load_tles_from_file, tle_to_elements
from constants.constants import (
    CD_MEAN, CD_SIGMA, CD_MIN, CD_MAX, CD_TAU_S, CD_DRIFT_FRAC,
    REENTRY_ALT, F107_BASELINE,
)

SANITY_CEILING = 1.0e4

# ---------------------------------------------------------------------------
# Confounder registry
# ---------------------------------------------------------------------------
ALL_CONFOUNDERS = [
    "baseline",
    "f107_flat",
    "kp_quiet",
    "env_flat",
    "cd_nominal",
    "r_overconfident",
    "all",
]


def _parse_confounders(s: str) -> List[str]:
    names = [c.strip() for c in s.split(",")]
    for c in names:
        if c not in ALL_CONFOUNDERS:
            raise ValueError(f"Unknown confounder '{c}'. Valid: {ALL_CONFOUNDERS}")
    return names


# ---------------------------------------------------------------------------
# Object loading & stratified sampling (copied from fleet_null_test.py)
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
# Truth propagation (full model — identical for all confounders)
# ---------------------------------------------------------------------------
def propagate_truth_from_tle(
    tle: Dict, epoch_jd: float, duration_s: float, dt_s: float,
    area: float = 1.0, mass: float = 200.0, f107_base: float = F107_BASELINE,
    Cd_base: Optional[float] = None, rng: Optional[np.random.Generator] = None,
    vary_cd: bool = True, truth_substeps: int = 4,
) -> Tuple[Dict, List, Dict, float]:
    if rng is None:
        rng = np.random.default_rng()
    if Cd_base is None:
        Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))
    el = tle_to_elements(tle)
    Cd = Cd_base
    t = 0.0
    n_steps = int(duration_s / dt_s)
    storms = sample_storm_events(duration_s, rng)
    f107_phases = sample_f107_phases(rng)
    cd_sigma_step = CD_DRIFT_FRAC * Cd_base * math.sqrt(2 * dt_s / CD_TAU_S)
    hist = {"t": [], "a": [], "e": [], "i": [], "Omega": [], "omega": [], "M": [], "Cd": []}
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
        if vary_cd:
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
    return hist, storms, f107_phases, Cd_base


def generate_noisy_measurements_from_truth(
    truth_hist: Dict, R_diag: np.ndarray = R_DIAG_ELEMENTS,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, np.ndarray]]:
    if rng is None:
        rng = np.random.default_rng()
    n = len(truth_hist["t"])
    meas = []
    for i in range(n):
        z = np.array([
            truth_hist["a"][i], truth_hist["e"][i], truth_hist["i"][i],
            truth_hist["Omega"][i], truth_hist["omega"][i], truth_hist["M"][i]
        ])
        z += rng.normal(scale=np.sqrt(R_diag))
        meas.append((truth_hist["t"][i], z))
    return meas


# ---------------------------------------------------------------------------
# Filter runner with confounder support
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
    R_diag = R_DIAG_ELEMENTS
    e, argp = el.ecc, el.argp
    R_h = math.sin(argp)**2 * R_diag[1] + (e * math.cos(argp))**2 * R_diag[4]
    R_k = math.cos(argp)**2 * R_diag[1] + (e * math.sin(argp))**2 * R_diag[4]
    diag = [
        R_diag[0], R_h, R_k, R_diag[2], R_diag[3], R_diag[5],
        1.0e-6, 1.0e-6, 1.0e-6
    ]
    return np.diag(diag)


def _run_one_window_confounder(
    window_measurements: List[Tuple[float, np.ndarray]],
    el_window_start: MeanElements,
    epoch_jd: float,
    tau: float,
    q: float,
    area: float,
    mass: float,
    storms: List,
    f107_phases: Dict,
    base_t_offset: float,
    rng: np.random.Generator,
    Cd_truth_base: float,
    confounder: str,
) -> Dict:
    """Run forward+backward+smoother with a specific confounder mismatch.

    Truth always used full model (time-varying F10.7, storms, drifting Cd).
    Filter gets degraded version based on confounder.
    """
    # Determine filter parameters based on confounder
    cd_filter = Cd_truth_base
    r_filter = R_DIAG_ELEMENTS.copy()

    if confounder in ("cd_nominal", "all"):
        cd_filter = 2.2  # population mean, not object's true Cd

    if confounder == "r_overconfident":
        r_filter = R_DIAG_ELEMENTS * 0.1  # overconfident by 10x
    # Note: "all" does NOT include R overconfidence — that's a scaling artifact,
    # not a realistic model mismatch. "all" = env_flat + cd_nominal only.

    x0 = _build_initial_state(el_window_start)
    P0 = _build_initial_covariance(el_window_start)

    # --- Forward pass ---
    ekf_fwd = EKF(x0, P0, cd_filter, area, mass, epoch_jd, tau=tau, q=q, direction="forward")
    fwd_states, fwd_covs = [], []
    prev_t = 0.0

    for t_rel, z in window_measurements:
        dt = t_rel - prev_t
        abs_t = base_t_offset + prev_t

        # Confounder-specific environment for filter
        if confounder in ("f107_flat", "env_flat", "all"):
            f107 = F107_BASELINE
        else:
            f107 = f107_at_time(abs_t, f107_phases, f_base=F107_BASELINE)

        if confounder in ("kp_quiet", "env_flat", "all"):
            kp = QUIET_KP
        else:
            kp = kp_at_time(abs_t, storms)

        ekf_fwd.predict(dt, f107, kp)
        ekf_fwd.update(z, r_filter)
        fwd_states.append(ekf_fwd.x.copy())
        fwd_covs.append(ekf_fwd.P.copy())
        prev_t = t_rel

    # --- Backward pass ---
    x0_bwd = fwd_states[-1].copy()
    P0_bwd = fwd_covs[-1].copy() * 100.0

    ekf_bwd = EKF(x0_bwd, P0_bwd, cd_filter, area, mass, epoch_jd, tau=tau, q=q, direction="backward")
    bwd_apriori_states, bwd_apriori_covs = [], []
    prev_t = window_measurements[-1][0]

    for t_rel, z in reversed(window_measurements):
        dt = t_rel - prev_t
        if abs(dt) > 1e-12:
            abs_t = base_t_offset + prev_t
            if confounder in ("f107_flat", "env_flat", "all"):
                f107 = F107_BASELINE
            else:
                f107 = f107_at_time(abs_t, f107_phases, f_base=F107_BASELINE)
            if confounder in ("kp_quiet", "env_flat", "all"):
                kp = QUIET_KP
            else:
                kp = kp_at_time(abs_t, storms)
            ekf_bwd.predict(dt, f107, kp)

        bwd_apriori_states.append(ekf_bwd.x.copy())
        bwd_apriori_covs.append(ekf_bwd.P.copy())
        ekf_bwd.update(z, r_filter)
        prev_t = t_rel

    # --- Smoother ---
    sm_states, sm_covs = fraser_potter_smoother(
        fwd_states, fwd_covs,
        list(reversed(bwd_apriori_states)),
        list(reversed(bwd_apriori_covs))
    )

    # --- Mahalanobis distance (full window, matching Gate 1) ---
    mask = np.zeros(9, dtype=bool)
    mask[0] = True  # restrict to 'a' index

    maha = compute_mahalanobis_distance(
        fwd_states, fwd_covs,
        list(reversed(bwd_apriori_states)),
        list(reversed(bwd_apriori_covs)),
        sm_covs,
        state_mask=mask
    )

    peak_maha = float(np.max(maha))
    peak_idx = int(np.argmax(maha))
    peak_time_rel = window_measurements[peak_idx][0]

    if not np.isfinite(peak_maha) or abs(peak_maha) > SANITY_CEILING:
        raise RuntimeError(f"peak_maha={peak_maha:.3e} diverged")

    return {"peak_maha": peak_maha, "peak_time_rel_s": peak_time_rel}


def run_confounders_on_object(
    tle_data: Dict,
    duration_days: float,
    tau: float,
    q: float,
    window_days: float,
    seed: int,
    confounders: List[str],
) -> Dict:
    """Process a single object: propagate truth once, then run filter
    for each confounder on the same measurements."""
    try:
        rng = np.random.default_rng(seed)
        epoch_jd = tle_data["epoch_jd"]
        dt_s = 86400.0
        area, mass = 1.0, 200.0

        n_windows = int(duration_days // window_days)
        if n_windows < 1:
            return {
                "norad_id": tle_data.get("norad_id", "unknown"),
                "n_windows": 0,
                "success": False,
                "error": f"duration {duration_days:.0f} < window {window_days:.0f}",
                "confounder_peaks": {},
            }

        duration_days_used = n_windows * window_days
        duration_s = duration_days_used * 86400.0

        # Propagate truth ONCE (full model)
        truth_hist, storms, f107_phases, Cd_truth_base = propagate_truth_from_tle(
            tle_data, epoch_jd, duration_s, dt_s,
            area=area, mass=mass, rng=rng
        )

        # Generate noisy measurements ONCE
        measurements = generate_noisy_measurements_from_truth(truth_hist, rng=rng)

        # Initialize per-confounder peak storage
        confounder_peaks = {c: [] for c in confounders}

        # Process each window
        for w in range(n_windows):
            w_start_s = w * window_days * 86400.0
            w_end_s = (w + 1) * window_days * 86400.0

            idx_start = next((i for i, (t, _) in enumerate(measurements) if t >= w_start_s), None)
            idx_end = next((i for i, (t, _) in enumerate(measurements) if t >= w_end_s), len(measurements))

            if idx_start is None or idx_end - idx_start < 2:
                for c in confounders:
                    confounder_peaks[c].append(None)
                continue

            window_meas_abs = measurements[idx_start:idx_end]
            window_meas_rel = [(t - w_start_s, z) for t, z in window_meas_abs]

            idx_truth = max(0, idx_start - 1) if idx_start > 0 else 0
            el_start = MeanElements(
                a=truth_hist["a"][idx_truth],
                ecc=truth_hist["e"][idx_truth],
                inc=truth_hist["i"][idx_truth],
                raan=truth_hist["Omega"][idx_truth],
                argp=truth_hist["omega"][idx_truth],
                M=truth_hist["M"][idx_truth]
            )

            # Run each confounder on the same window
            for c in confounders:
                try:
                    result = _run_one_window_confounder(
                        window_meas_rel, el_start, epoch_jd, tau, q,
                        area, mass, storms, f107_phases, w_start_s,
                        rng, Cd_truth_base, c
                    )
                    confounder_peaks[c].append(result["peak_maha"])
                except Exception as e:
                    confounder_peaks[c].append(None)

        return {
            "norad_id": tle_data.get("norad_id", "unknown"),
            "n_windows": n_windows,
            "duration_days_used": duration_days_used,
            "success": True,
            "error": None,
            "confounder_peaks": confounder_peaks,
        }

    except Exception as e:
        return {
            "norad_id": tle_data.get("norad_id", "unknown"),
            "n_windows": 0,
            "duration_days_used": None,
            "success": False,
            "error": str(e),
            "confounder_peaks": {},
        }


def process_one_wrapper(args):
    tle_data, duration_days, tau, q, window_days, seed_offset, idx, confounders = args
    return run_confounders_on_object(
        tle_data, duration_days, tau, q, window_days, seed_offset + idx, confounders
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gate 2 confounder analysis")
    parser.add_argument("--tle-file", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--durations", required=True)
    parser.add_argument("--tau", type=float, default=3e4)
    parser.add_argument("--q", type=float, default=1e-19)
    parser.add_argument("--window-days", type=float, default=90.0)
    parser.add_argument("--max-duration-days", type=float, default=7300.0)
    parser.add_argument("--stratified-sample", type=int, default=None)
    parser.add_argument("--n-inc-bins", type=int, default=6)
    parser.add_argument("--n-sma-bins", type=int, default=6)
    parser.add_argument("--confounders", type=str, default="baseline,f107_flat,kp_quiet,env_flat,cd_nominal,r_overconfident",
                        help=f"Comma-separated list from: {','.join(ALL_CONFOUNDERS)}")
    parser.add_argument("--threshold", type=float, default=13.258,
                        help="Gate 1 null threshold (99.97th percentile)")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=300000)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--output", default="results/gate2_confounders.json")
    args = parser.parse_args()

    confounders = _parse_confounders(args.confounders)
    print(f"Gate 2 confounder analysis")
    print(f"  Confounders: {confounders}")
    print(f"  Threshold:   {args.threshold:.3f}")
    print(f"  tau={args.tau:.3e}, q={args.q:.3e}, window={args.window_days:.0f}d")
    print()

    # Load objects
    objects_to_process = load_objects(
        args.tle_file, args.meta, args.durations,
        max_duration_days=args.max_duration_days,
        window_days=args.window_days
    )

    if args.stratified_sample:
        objects_to_process = stratified_sample(
            objects_to_process, n_target=args.stratified_sample,
            n_inc_bins=args.n_inc_bins, n_sma_bins=args.n_sma_bins
        )
    elif args.max_objects:
        objects_to_process = objects_to_process[:args.max_objects]

    print(f"Processing {len(objects_to_process)} objects...")

    # Build tasks
    tasks = [
        (
            obj["tle_data"], obj["duration_days"],
            args.tau, args.q, args.window_days,
            args.seed_offset, idx, confounders
        )
        for idx, obj in enumerate(objects_to_process)
    ]

    total_windows_expected = sum(int(obj["duration_days"] // args.window_days) for obj in objects_to_process)
    print(f"Total expected windows: {total_windows_expected}")
    print(f"Using {args.n_workers} workers...")
    print()

    results = []
    t0 = time.time()

    with mp.Pool(args.n_workers) as pool:
        iter_results = pool.imap_unordered(
            process_one_wrapper, tasks,
            chunksize=max(1, len(tasks) // (args.n_workers * 4))
        )
        for i, result in enumerate(iter_results, 1):
            results.append(result)
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                pct = 100.0 * i / len(tasks)
                elapsed = time.time() - t0
                eta = (len(tasks) - i) * (elapsed / i) if i > 0 else 0
                print(f"  [{i:>4}/{len(tasks)} objects] {pct:>5.1f}% | ETA: {eta/60:.1f}m")

    elapsed_total = time.time() - t0
    print(f"\nFinished {len(results)} objects in {elapsed_total/60:.1f} min")

    # Aggregate results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    print(f"Successful: {len(successful)}, Failed: {len(failed)}")

    # Collect peaks per confounder
    all_peaks = {c: [] for c in confounders}
    for r in successful:
        peaks_dict = r.get("confounder_peaks", {})
        for c in confounders:
            if c in peaks_dict:
                all_peaks[c].extend([p for p in peaks_dict[c] if p is not None])

    # Summary table
    print(f"\n{'='*70}")
    print("GATE 2 CONFOUNTER FALSE-POSITIVE RATES")
    print(f"{'='*70}")
    print(f"{'Scenario':<25} {'N windows':>10} {'Mean':>8} {'Std':>8} {'99.97%':>8} {'FP rate':>10}")
    print(f"{'-'*70}")

    summary = {}
    for c in confounders:
        peaks = np.array(all_peaks[c])
        n = len(peaks)
        if n == 0:
            print(f"{c:<25} {0:>10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            summary[c] = {"n": 0, "mean": None, "std": None, "pct9997": None, "fp_rate": None}
            continue
        mean = float(np.mean(peaks))
        std = float(np.std(peaks))
        pct9997 = float(np.percentile(peaks, 99.97))
        fp_rate = float(np.mean(peaks > args.threshold))
        print(f"{c:<25} {n:>10} {mean:>8.3f} {std:>8.3f} {pct9997:>8.3f} {fp_rate*100:>9.4f}%")
        summary[c] = {
            "n": n, "mean": mean, "std": std,
            "pct9997": pct9997, "fp_rate": fp_rate,
            "fp_rate_percent": fp_rate * 100.0,
        }

    print(f"{'-'*70}")
    print(f"\nThreshold: {args.threshold:.3f} (Gate 1 99.97th percentile)")
    print(f"FP rate = fraction of null windows exceeding threshold")
    print(f"\nInterpretation:")
    print(f"  - baseline should match Gate 1 (~0.03% FP rate)")
    print(f"  - Individual confounders show which mismatches matter")
    print(f"  - 'all' gives the combined extrinsic rate r_extrinsic")

    # Save results
    output_data = {
        "config": {
            "tau": args.tau, "q": args.q,
            "window_days": args.window_days,
            "threshold": args.threshold,
            "n_objects": len(objects_to_process),
            "n_successful": len(successful),
            "confounders": confounders,
        },
        "summary": summary,
        "results": successful,
        "failed": failed,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
    
