#!/usr/bin/env python3
"""
Process a fleet of TLEs through the EKF + smoother pipeline to validate
filter behavior on real orbits. This is a VALIDATION TEST only — it uses
synthetic data (one TLE per object propagated forward with added noise).

PURPOSE:
    - Test filter stability on real TLE-derived orbits (varying inclinations,
      altitudes, eccentricities)
    - Generate a null distribution from real orbits for comparison to the
      fixed-orbit null calibration (mean=3.17, std=1.36)
    - Identify orbit-dependent filter performance

WHAT THIS SCRIPT DOES:
    1. Takes ONE TLE per object (the most recent)
    2. Propagates it forward cleanly for its clipped duration (no debris)
    3. Adds synthetic Gaussian noise (using R_DIAG_ELEMENTS)
    4. Splits into 90-day windows
    5. Runs EKF + smoother on each window independently
    6. Reports peak Mahalanobis distribution across all windows

WHAT THIS SCRIPT DOES NOT DO:
    - Does NOT detect real debris (uses synthetic data)
    - Does NOT use full TLE histories (only one TLE per object)
    - Does NOT replace the null calibration (threshold is already known)

Usage:
    python -m phase2_filter.process_fleet \\
        --tle-file TLE/real_tles.tle \\
        --meta TLE/real_tles.tle.meta.json \\
        --durations TLE/real_tles.tle.durations.json \\
        --tau 3e4 --q 1e-19 --window-days 90 --max-duration-days 7300 \\
        --output results/fleet_validation.json \\
        --n-workers 8
"""

import argparse
import json
import math
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np

from constants.constants import MU, R_EARTH
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
import time
SANITY_CEILING = 1.0e4

# Reference null calibration values (from calibrate_null_threshold.py)
REF_NULL_MEAN = 3.174
REF_NULL_STD = 1.356
REF_NULL_99PCT = 6.830
REF_NULL_9997PCT = 8.843

def stratified_sample(objects_to_process, n_target=500, n_inc_bins=6, n_sma_bins=6, seed=42):
    """Pick a subset that spreads across (inclination, altitude) bins instead
    of taking objects in file order, which can badly under-represent the
    fleet's dominant populations (e.g. SSO)."""
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

    n_buckets = len(buckets)
    per_bucket = max(1, n_target // n_buckets)

    selected = []
    for key, idxs in buckets.items():
        rng_local.shuffle(idxs)
        selected.extend(idxs[:per_bucket])

    if len(selected) < n_target:
        remaining = [i for i in range(len(objects_to_process)) if i not in set(selected)]
        rng_local.shuffle(remaining)
        selected.extend(remaining[:n_target - len(selected)])

    selected = selected[:n_target]
    print(f"Stratified sample: {len(selected)} objects across {n_buckets} "
          f"(inc x altitude) bins ({n_inc_bins}x{n_sma_bins} grid)")
    return [objects_to_process[i] for i in selected]

def propagate_truth_from_tle(
    tle: Dict, epoch_jd: float, duration_s: float, dt_s: float,
    area: float = 1.0, mass: float = 200.0, f107_base: float = F107_BASELINE,
    Cd_base: Optional[float] = None, rng: Optional[np.random.Generator] = None,
    vary_cd: bool = True, truth_substeps: int = 4,
) -> Tuple[Dict, List, Dict, float]:
    """
    Propagate a truth trajectory from a TLE with NO debris forcing.
    
    Returns:
        hist: Dictionary of time histories [t, a, e, i, Omega, omega, M, Cd]
        storms: List of storm events
        f107_phases: Solar flux phases
    """
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
    truth_hist: Dict, 
    R_diag: np.ndarray = R_DIAG_ELEMENTS,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, np.ndarray]]:
    """
    Add Gaussian noise to truth elements to simulate TLE measurements.
    
    R_diag values are VARIANCES (σ²). The noise standard deviation is sqrt(R_diag).
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n = len(truth_hist["t"])
    meas = []
    
    for i in range(n):
        z = np.array([
            truth_hist["a"][i],
            truth_hist["e"][i],
            truth_hist["i"][i],
            truth_hist["Omega"][i],
            truth_hist["omega"][i],
            truth_hist["M"][i]
        ])
        z += rng.normal(scale=np.sqrt(R_diag))
        meas.append((truth_hist["t"][i], z))
        
    return meas


def _build_initial_state(el: MeanElements) -> np.ndarray:
    """Build 9-state initial vector [a, h, k, i, Omega, M, 0, 0, 0]."""
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
    """Build diagonal initial covariance in (a, h, k, i, Omega, M) space."""
    R_diag = R_DIAG_ELEMENTS
    e, argp = el.ecc, el.argp
    
    # Transform covariance from (e, argp) to (h, k)
    R_h = math.sin(argp)**2 * R_diag[1] + (e * math.cos(argp))**2 * R_diag[4]
    R_k = math.cos(argp)**2 * R_diag[1] + (e * math.sin(argp))**2 * R_diag[4]
    
    diag = [
        R_diag[0],   # a
        R_h,         # h
        R_k,         # k
        R_diag[2],   # i
        R_diag[3],   # Omega
        R_diag[5],   # M
        1.0e-6,      # w_R (small initial uncertainty)
        1.0e-6,      # w_S
        1.0e-6,      # w_W
    ]
    return np.diag(diag)


def _run_one_window(
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
    Cd_base: float,
) -> Dict:
    """
    Run forward+backward+smoother on ONE 90-day window.
    
    Args:
        window_measurements: List of (t_rel, z) where t_rel is seconds from window start
        el_window_start: True mean elements at window start
        base_t_offset: Absolute time at window start (seconds from epoch)
    
    Returns:
        Dict with peak_maha and peak_time_rel_s
    """
    x0 = _build_initial_state(el_window_start)
    P0 = _build_initial_covariance(el_window_start)
    

    # --- Forward pass ---
    ekf_fwd = EKF(x0, P0, Cd_base, area, mass, epoch_jd, tau=tau, q=q, direction="forward")
    fwd_states, fwd_covs = [], []
    prev_t = 0.0
    
    for t_rel, z in window_measurements:
        dt = t_rel - prev_t
        abs_t = base_t_offset + prev_t
        f107 = f107_at_time(abs_t, f107_phases, f_base=F107_BASELINE)
        kp = kp_at_time(abs_t, storms)
        
        ekf_fwd.predict(dt, f107, kp)
        ekf_fwd.update(z, R_DIAG_ELEMENTS)
        
        fwd_states.append(ekf_fwd.x.copy())
        fwd_covs.append(ekf_fwd.P.copy())
        prev_t = t_rel

    # --- Backward pass ---
    x0_bwd = fwd_states[-1].copy()
    P0_bwd = fwd_covs[-1].copy() * 100.0
    
    ekf_bwd = EKF(x0_bwd, P0_bwd, Cd_base, area, mass, epoch_jd, tau=tau, q=q, direction="backward")
    bwd_apriori_states, bwd_apriori_covs = [], []
    prev_t = window_measurements[-1][0]
    
    for t_rel, z in reversed(window_measurements):
        dt = t_rel - prev_t
        if abs(dt) > 1e-12:
            abs_t = base_t_offset + prev_t
            f107 = f107_at_time(abs_t, f107_phases, f_base=F107_BASELINE)
            kp = kp_at_time(abs_t, storms)
            ekf_bwd.predict(dt, f107, kp)
            
        bwd_apriori_states.append(ekf_bwd.x.copy())
        bwd_apriori_covs.append(ekf_bwd.P.copy())
        ekf_bwd.update(z, R_DIAG_ELEMENTS)
        prev_t = t_rel

    # --- Smoother ---
    sm_states, sm_covs = fraser_potter_smoother(
        fwd_states, 
        fwd_covs,
        list(reversed(bwd_apriori_states)),
        list(reversed(bwd_apriori_covs))
    )

    # --- Mahalanobis distance (restricted to 'a') ---
    mask = np.zeros(9, dtype=bool)
    mask[0] = True
    
    maha = compute_mahalanobis_distance(
        fwd_states,
        fwd_covs,
        list(reversed(bwd_apriori_states)),
        list(reversed(bwd_apriori_covs)),
        sm_covs,
        state_mask=mask
    )

    peak_maha = float(np.max(maha))
    peak_idx = int(np.argmax(maha))
    peak_time_rel = window_measurements[peak_idx][0]

    # Sanity check
    if not np.isfinite(peak_maha) or abs(peak_maha) > SANITY_CEILING:
        raise RuntimeError(
            f"peak_maha={peak_maha:.3e} is non-finite or unphysically large "
            f"- filter diverged in this window"
        )

    return {"peak_maha": peak_maha, "peak_time_rel_s": peak_time_rel}


def run_filter_on_object(
    tle_data: Dict,
    duration_days: float,
    tau: float,
    q: float,
    window_days: float,
    seed: int,
) -> Dict:
    """
    Process a single object: propagate truth, generate measurements, window,
    run EKF on each window.
    """
    try:
        rng = np.random.default_rng(seed)
        epoch_jd = tle_data["epoch_jd"]
        dt_s = 86400.0
        area, mass = 1.0, 200.0

        n_windows = int(duration_days // window_days)
        if n_windows < 1:
            return {
                "norad_id": tle_data.get("norad_id", "unknown"),
                "windows": [],
                "n_windows": 0,
                "duration_days_used": 0.0,
                "success": False,
                "error": f"duration_days={duration_days:.0f} < window_days={window_days:.0f}"
            }

        duration_days_used = n_windows * window_days
        duration_s = duration_days_used * 86400.0

        # Propagate truth (no debris)
        truth_hist, storms, f107_phases, Cd_truth_base = propagate_truth_from_tle(
            tle_data, epoch_jd, duration_s, dt_s,
            area=area, mass=mass, rng=rng
        )
        
        # Generate noisy measurements
        measurements = generate_noisy_measurements_from_truth(truth_hist, rng=rng)

        # Process each window
        window_results = []
        for w in range(n_windows):
            w_start_s = w * window_days * 86400.0
            w_end_s = (w + 1) * window_days * 86400.0

            idx_start = next((i for i, (t, _) in enumerate(measurements) if t >= w_start_s), None)
            idx_end = next((i for i, (t, _) in enumerate(measurements) if t >= w_end_s), len(measurements))
            
            if idx_start is None or idx_end - idx_start < 2:
                continue

            window_meas_abs = measurements[idx_start:idx_end]
            window_meas_rel = [(t - w_start_s, z) for t, z in window_meas_abs]

            # True state at window start
            idx_truth = max(0, idx_start - 1) if idx_start > 0 else 0
            el_start = MeanElements(
                a=truth_hist["a"][idx_truth],
                ecc=truth_hist["e"][idx_truth],
                inc=truth_hist["i"][idx_truth],
                raan=truth_hist["Omega"][idx_truth],
                argp=truth_hist["omega"][idx_truth],
                M=truth_hist["M"][idx_truth]
            )

            try:
                result = _run_one_window(
                    window_meas_rel,
                    el_start,
                    epoch_jd,
                    tau,
                    q,
                    area,
                    mass,
                    storms,
                    f107_phases,
                    w_start_s,
                    rng,
                    Cd_truth_base,
                )
                result["window_index"] = w
                result["window_start_day"] = w * window_days
                window_results.append(result)
            except Exception as e:
                window_results.append({
                    "window_index": w,
                    "window_start_day": w * window_days,
                    "peak_maha": None,
                    "error": str(e)
                })

        return {
            "norad_id": tle_data.get("norad_id", "unknown"),
            "windows": window_results,
            "n_windows": n_windows,
            "duration_days_used": duration_days_used,
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "norad_id": tle_data.get("norad_id", "unknown"),
            "windows": [],
            "n_windows": 0,
            "duration_days_used": None,
            "success": False,
            "error": str(e)
        }


def process_one_wrapper(args):
    """Multiprocessing wrapper."""
    tle_data, duration_days, tau, q, window_days, seed_offset, idx = args
    return run_filter_on_object(
        tle_data, duration_days, tau, q, window_days, seed_offset + idx
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process a fleet of TLEs (windowed validation test)"
    )
    
    parser.add_argument("--stratified-sample", type=int, default=None,
                        help="Instead of --max-objects, pick this many objects "
                             "spread across inclination/altitude bins")
    parser.add_argument("--n-inc-bins", type=int, default=6)
    parser.add_argument("--n-sma-bins", type=int, default=6)
    
    parser.add_argument("--tle-file", required=True, help="Path to TLE file")
    parser.add_argument("--meta", required=True, help="Path to .meta.json")
    parser.add_argument("--durations", required=True, help="Path to durations.json")
    parser.add_argument("--tau", type=float, default=3e4, help="FOGM time constant [s]")
    parser.add_argument("--q", type=float, default=1e-19, help="Process noise spectral density")
    parser.add_argument("--window-days", type=float, default=90.0, 
                        help="Window length [days] - must match calibration")
    parser.add_argument("--max-duration-days", type=float, default=7300.0,
                        help="Cap total history [days] (e.g., 7300 = 20 years)")
    parser.add_argument("--output", default="results/fleet_validation.json",
                        help="Output JSON file path")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--seed-offset", type=int, default=100000,
                        help="Base seed for RNG")
    parser.add_argument("--max-objects", type=int, default=None,
                        help="Limit number of objects (for testing)")
    parser.add_argument("--reference-null-mean", type=float, default=REF_NULL_MEAN,
                        help="Reference null mean for comparison")
    parser.add_argument("--reference-null-std", type=float, default=REF_NULL_STD,
                        help="Reference null std for comparison")
    parser.add_argument("--reference-threshold", type=float, default=REF_NULL_9997PCT,
                        help="Reference null threshold (99.97th percentile)")
    args = parser.parse_args()

    # --- Load data ---
    print(f"Loading TLEs from {args.tle_file}...")
    tle_tuples = load_tles_from_file(args.tle_file)
    print(f"Loaded {len(tle_tuples)} TLEs")

    with open(args.meta) as f:
        meta_records = json.load(f)

    tle_map = {}
    for name, line1, line2 in tle_tuples:
        tle_dict = parse_tle(name, line1, line2)
        tle_map[tle_dict.get("norad_id", "unknown")] = tle_dict
    print(f"Parsed {len(tle_map)} TLEs with NORAD IDs")

    with open(args.durations) as f:
        durations_map = json.load(f)

    # --- Build object list ---
    objects_to_process = []
    for rec in meta_records:
        norad_id = str(rec.get("NORAD_CAT_ID", ""))
        if norad_id not in tle_map or norad_id not in durations_map:
            continue

        duration_days = durations_map[norad_id].get("duration_days", 0)
        if args.max_duration_days:
            duration_days = min(duration_days, args.max_duration_days)

        if duration_days < args.window_days:
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

    if args.stratified_sample:
        objects_to_process = stratified_sample(
            objects_to_process, n_target=args.stratified_sample,
            n_inc_bins=args.n_inc_bins, n_sma_bins=args.n_sma_bins)
    elif args.max_objects:
        objects_to_process = objects_to_process[:args.max_objects]

    print(f"Processing {len(objects_to_process)} objects, window_days={args.window_days:.0f}...")

    # --- Process in parallel ---
    tasks = [
        (
            obj["tle_data"],
            obj["duration_days"],
            args.tau,
            args.q,
            args.window_days,
            args.seed_offset,
            idx
        )
        for idx, obj in enumerate(objects_to_process)
    ]

    # Pre-calculate total expected windows for true % progress
    total_windows_expected = sum(
        int(obj["duration_days"] // args.window_days)
        for obj in objects_to_process
    )
    report_interval = max(1, int(total_windows_expected * 0.05))  # every 5%
    if total_windows_expected == 0:
        report_interval = 1

    print(f"Using {args.n_workers} workers...")
    print(f"Total expected windows: {total_windows_expected} "
          f"(will report every ~{report_interval} windows)")

    results = []
    windows_done = 0
    next_report = report_interval
    t0 = time.time()

    with mp.Pool(args.n_workers) as pool:
        # imap_unordered lets us tally progress as each object finishes
        iter_results = pool.imap_unordered(
            process_one_wrapper,
            tasks,
            chunksize=max(1, len(tasks) // (args.n_workers * 4))
        )

        for result in iter_results:
            results.append(result)
            windows_done += result.get("n_windows", 0)

            if windows_done >= next_report or windows_done >= total_windows_expected:
                pct = 100.0 * windows_done / total_windows_expected if total_windows_expected else 100.0
                elapsed = time.time() - t0
                rate = windows_done / elapsed if elapsed > 0 else 0.0
                remaining = total_windows_expected - windows_done
                eta_s = remaining / rate if rate > 0 else 0.0

                if eta_s < 120:
                    eta_str = f"{eta_s:.0f}s"
                elif eta_s < 7200:
                    eta_str = f"{eta_s/60:.1f}m"
                else:
                    eta_str = f"{eta_s/3600:.1f}h"

                print(f"  [{windows_done:>6}/{total_windows_expected:<6} windows] "
                      f"{pct:>5.1f}% | {len(results):>5}/{len(tasks)} objects | "
                      f"ETA: {eta_str}")

                # advance to next 5% milestone
                while next_report <= windows_done:
                    next_report += report_interval

    print(f"Finished all {len(results)} objects ({windows_done} windows) in "
          f"{(time.time()-t0)/60:.1f} min")

    # --- Analyze results ---
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    all_windows = [w for r in successful for w in r["windows"]]
    total_windows = len(all_windows)
    ok_windows = [w for w in all_windows if w.get("peak_maha") is not None]
    bad_windows = total_windows - len(ok_windows)

    print(f"\nCompleted: {len(successful)} objects succeeded, {len(failed)} objects failed entirely")
    print(f"Total windows analyzed: {total_windows} ({len(ok_windows)} clean, {bad_windows} diverged/skipped)")

    if failed:
        print("\nFirst few object-level failures:")
        for f in failed[:5]:
            print(f"  {f['norad_id']}: {f['error']}")

    # --- Save results ---
    output_data = {
        "config": {
            "tau": args.tau,
            "q": args.q,
            "window_days": args.window_days,
            "max_duration_days": args.max_duration_days,
            "n_objects_processed": len(objects_to_process),
            "n_objects_successful": len(successful),
            "total_windows": total_windows,
        },
        "results": successful,
        "failed": failed,
    }

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved results to {args.output}")

    # --- Compare to reference null ---
    if ok_windows:
        peaks = np.array([w["peak_maha"] for w in ok_windows])
        print(f"\n{'='*60}")
        print("WINDOW-LEVEL PEAK MAHALANOBIS DISTRIBUTION")
        print(f"{'='*60}")
        print(f"  Mean:  {peaks.mean():.3f}")
        print(f"  Std:   {peaks.std():.3f}")
        print(f"  Min:   {peaks.min():.3f}")
        print(f"  Max:   {peaks.max():.3f}")
        print(f"  95%:   {np.percentile(peaks, 95):.3f}")
        print(f"  99%:   {np.percentile(peaks, 99):.3f}")
        print(f"  99.9%: {np.percentile(peaks, 99.9):.3f}")

        print(f"\n{'='*60}")
        print("COMPARISON TO REFERENCE NULL CALIBRATION")
        print(f"{'='*60}")
        print(f"  Reference null mean:     {args.reference_null_mean:.3f}")
        print(f"  This run mean:           {peaks.mean():.3f}")
        print(f"  Difference:              {peaks.mean() - args.reference_null_mean:.3f} "
              f"({100*(peaks.mean() - args.reference_null_mean)/args.reference_null_mean:.1f}%)")
        print(f"  Reference null std:      {args.reference_null_std:.3f}")
        print(f"  This run std:            {peaks.std():.3f}")
        print(f"  Reference 99.97th pct:   {args.reference_threshold:.3f}")
        print(f"  This run 99.97th pct:    {np.percentile(peaks, 99.97):.3f}")
        
        if abs(peaks.mean() - args.reference_null_mean) > 0.5:
            print("\n⚠️  WARNING: This run's mean differs significantly from reference.")
            print("   This suggests orbit-dependent filter behavior or a setup mismatch.")
            print("   Investigate by grouping results by inclination/altitude.")
        else:
            print("\n✅ This run's distribution matches the reference null calibration.")
            print("   The filter behaves consistently across different orbits.")

        # --- Group by inclination ---
        print(f"\n{'='*60}")
        print("DISTRIBUTION BY INCLINATION BIN")
        print(f"{'='*60}")
        
        inc_bins = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]
        for lo, hi in inc_bins:
            peaks_in_bin = []
            for r in successful:
                norad_id = r["norad_id"]
                # Find the object's inclination
                obj = next((o for o in objects_to_process if o["norad_id"] == norad_id), None)
                if obj is None:
                    continue
                inc = obj["inclination"]
                if lo <= inc < hi:
                    for w in r["windows"]:
                        if w.get("peak_maha") is not None:
                            peaks_in_bin.append(w["peak_maha"])
            
            if peaks_in_bin:
                print(f"  {lo:3d}-{hi:3d}°: n={len(peaks_in_bin):4d}, mean={np.mean(peaks_in_bin):.3f}, std={np.std(peaks_in_bin):.3f}")
            else:
                print(f"  {lo:3d}-{hi:3d}°: no objects")
            print(f"\n{'='*60}")
            
        print("DISTRIBUTION BY ALTITUDE BIN")
        print(f"{'='*60}")

        sma_bins = [(700, 800), (800, 900), (900, 1000), (1000, 1100)]
        for lo, hi in sma_bins:
            peaks_in_bin = []
            for r in successful:
                norad_id = r["norad_id"]
                obj = next((o for o in objects_to_process if o["norad_id"] == norad_id), None)
                if obj is None:
                    continue
                if lo <= obj["sma_km"] < hi:
                    for w in r["windows"]:
                        if w.get("peak_maha") is not None:
                            peaks_in_bin.append(w["peak_maha"])
            if peaks_in_bin:
                print(f"  {lo:4d}-{hi:<4d}km: n={len(peaks_in_bin):5d}, "
                      f"mean={np.mean(peaks_in_bin):.3f}, std={np.std(peaks_in_bin):.3f}")
            else:
                print(f"  {lo:4d}-{hi:<4d}km: no objects")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print("This is a VALIDATION TEST using SYNTHETIC data (one TLE per object).")
    print("It does NOT detect real debris.")
    print("")
    print("Purpose:")
    print("  1. Verify filter stability on real orbits")
    print("  2. Identify orbit-dependent performance")
    print("  3. Compare to the fixed-orbit null calibration")
    print("")
    print("Next steps:")
    print("  1. If mean ≈ 3.17: Filter is stable across orbits")
    print("  2. If mean ≠ 3.17: Investigate orbit-dependent behavior")
    print("  3. Then: Build real data pipeline with full TLE histories")


if __name__ == "__main__":
    main()