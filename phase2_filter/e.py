#!/usr/bin/env python3
"""
Process a FIXED-ORBIT fleet through the EKF + smoother pipeline.

This is a DIRECT COMPANION to calibrate_null_threshold.py — every object
uses the SAME orbital elements (SMA, INC, etc.) so that the only variation
across the fleet comes from:
    - Random seed (different noise realizations)
    - Random Cd drift realizations
    - Random F10.7 / storm sampling

PURPOSE:
    - Apples-to-apples comparison with the null calibration from
      calibrate_null_threshold.py (which also uses a single fixed orbit)
    - Isolate whether differences between the real-TLE fleet and the null
      calibration are due to orbit diversity or filter behaviour
    - Validate that process_fleet gives the same distribution as
      calibrate_null_threshold when the orbit is identical

WHAT THIS SCRIPT DOES:
    1. Uses ONE fixed set of Keplerian elements for ALL objects
    2. Propagates each object forward cleanly for the requested duration
    3. Adds synthetic Gaussian noise (using R_DIAG_ELEMENTS)
    4. Splits into 90-day windows
    5. Runs EKF + smoother on each window independently
    6. Reports peak Mahalanobis distribution across all windows

WHAT THIS SCRIPT DOES NOT DO:
    - Does NOT read real TLEs
    - Does NOT vary SMA or INC across the fleet
    - Does NOT detect real debris (uses synthetic data)

Usage:
    python -m phase2_filter.process_fleet_fixed_orbit \
        --fixed-sma-m 7000000 \
        --fixed-inc-deg 98.0 \
        --fixed-ecc 0.001 \
        --fixed-raan-deg 0.0 \
        --fixed-argp-deg 0.0 \
        --fixed-m-deg 0.0 \
        --epoch-jd 2460000.5 \
        --n-objects 500 \
        --duration-days 90 \
        --tau 3e4 --q 1e-19 --window-days 90 \
        --output results/fleet_fixed_orbit.json \
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


def propagate_truth_fixed_orbit(
    el: MeanElements,
    epoch_jd: float,
    duration_s: float,
    dt_s: float,
    area: float = 1.0,
    mass: float = 200.0,
    f107_base: float = F107_BASELINE,
    Cd_base: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    vary_cd: bool = True,
    truth_substeps: int = 4,
) -> Tuple[Dict, List, Dict, float]:
    """
    Propagate a truth trajectory from FIXED orbital elements with NO debris forcing.

    This is identical to propagate_truth_from_tle except the initial MeanElements
    are passed directly instead of being parsed from a TLE dict.

    Returns:
        hist: Dictionary of time histories [t, a, e, i, Omega, omega, M, Cd]
        storms: List of storm events
        f107_phases: Solar flux phases
        Cd_base: The base Cd value used
    """
    if rng is None:
        rng = np.random.default_rng()
    if Cd_base is None:
        Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))

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


def run_filter_on_object_fixed_orbit(
    fixed_elements: MeanElements,
    epoch_jd: float,
    duration_days: float,
    tau: float,
    q: float,
    window_days: float,
    seed: int,
) -> Dict:
    """
    Process a single object with FIXED orbital elements.

    The only differences between objects are the random seed (noise & Cd drift).
    """
    try:
        rng = np.random.default_rng(seed)
        dt_s = 86400.0
        area, mass = 1.0, 200.0

        n_windows = int(duration_days // window_days)
        if n_windows < 1:
            return {
                "object_id": f"fixed_orbit_seed_{seed}",
                "windows": [],
                "n_windows": 0,
                "duration_days_used": 0.0,
                "success": False,
                "error": f"duration_days={duration_days:.0f} < window_days={window_days:.0f}"
            }

        duration_days_used = n_windows * window_days
        duration_s = duration_days_used * 86400.0

        # Propagate truth (no debris) — COPY the elements so each object is independent
        el = MeanElements(
            a=fixed_elements.a,
            ecc=fixed_elements.ecc,
            inc=fixed_elements.inc,
            raan=fixed_elements.raan,
            argp=fixed_elements.argp,
            M=fixed_elements.M
        )

        truth_hist, storms, f107_phases, Cd_truth_base = propagate_truth_fixed_orbit(
            el, epoch_jd, duration_s, dt_s,
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
            "object_id": f"fixed_orbit_seed_{seed}",
            "windows": window_results,
            "n_windows": n_windows,
            "duration_days_used": duration_days_used,
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "object_id": f"fixed_orbit_seed_{seed}",
            "windows": [],
            "n_windows": 0,
            "duration_days_used": None,
            "success": False,
            "error": str(e)
        }


def process_one_wrapper(args):
    """Multiprocessing wrapper."""
    fixed_elements, epoch_jd, duration_days, tau, q, window_days, seed = args
    return run_filter_on_object_fixed_orbit(
        fixed_elements, epoch_jd, duration_days, tau, q, window_days, seed
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process a fixed-orbit fleet (apples-to-apples with null calibration)"
    )

    # --- Fixed orbit parameters (must match calibrate_null_threshold.py / validation.py) ---
    parser.add_argument("--fixed-sma-m", type=float, required=True,
                        help="Fixed semi-major axis [m]. "
                             "Set this to the SAME value used by run_impulse_test in validation.py")
    parser.add_argument("--fixed-inc-deg", type=float, required=True,
                        help="Fixed inclination [deg]. "
                             "Set this to the SAME value used by run_impulse_test in validation.py")
    parser.add_argument("--fixed-ecc", type=float, default=0.001,
                        help="Fixed eccentricity [default: 0.001]")
    parser.add_argument("--fixed-raan-deg", type=float, default=0.0,
                        help="Fixed RAAN [deg, default: 0.0]")
    parser.add_argument("--fixed-argp-deg", type=float, default=0.0,
                        help="Fixed argument of periapsis [deg, default: 0.0]")
    parser.add_argument("--fixed-m-deg", type=float, default=0.0,
                        help="Fixed mean anomaly [deg, default: 0.0]")
    parser.add_argument("--epoch-jd", type=float, default=2460000.5,
                        help="Fixed Julian Date epoch [default: 2460000.5]")

    # --- Fleet & processing parameters ---
    parser.add_argument("--n-objects", type=int, default=500,
                        help="Number of objects to simulate (each gets a unique seed)")
    parser.add_argument("--tau", type=float, default=3e4, help="FOGM time constant [s]")
    parser.add_argument("--q", type=float, default=1e-19, help="Process noise spectral density")
    parser.add_argument("--window-days", type=float, default=90.0,
                        help="Window length [days] - must match calibration")
    parser.add_argument("--max-duration-days", type=float, default=7300.0,
                        help="Cap total history [days] (e.g., 7300 = 20 years)")
    parser.add_argument("--output", default="results/fleet_fixed_orbit.json",
                        help="Output JSON file path")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--seed-offset", type=int, default=100000,
                        help="Base seed for RNG")
    parser.add_argument("--reference-null-mean", type=float, default=REF_NULL_MEAN,
                        help="Reference null mean for comparison")
    parser.add_argument("--reference-null-std", type=float, default=REF_NULL_STD,
                        help="Reference null std for comparison")
    parser.add_argument("--reference-threshold", type=float, default=REF_NULL_9997PCT,
                        help="Reference null threshold (99.97th percentile)")
    args = parser.parse_args()

    # --- Build fixed orbital elements ---
    fixed_elements = MeanElements(
        a=args.fixed_sma_m,
        ecc=args.fixed_ecc,
        inc=math.radians(args.fixed_inc_deg),
        raan=math.radians(args.fixed_raan_deg),
        argp=math.radians(args.fixed_argp_deg),
        M=math.radians(args.fixed_m_deg),
    )

    print(f"Fixed orbit: a={args.fixed_sma_m:.0f} m, "
          f"e={args.fixed_ecc}, i={args.fixed_inc_deg} deg, "
          f"RAAN={args.fixed_raan_deg} deg, argp={args.fixed_argp_deg} deg, "
          f"M={args.fixed_m_deg} deg")
    print(f"Epoch JD: {args.epoch_jd}")
    print(f"Processing {args.n_objects} objects, window_days={args.window_days:.0f}...")

    # --- Build task list ---
    tasks = [
        (fixed_elements, args.epoch_jd, args.max_duration_days, args.tau, args.q,
         args.window_days, args.seed_offset + i)
        for i in range(args.n_objects)
    ]

    # Pre-calculate total expected windows for true % progress
    total_windows_expected = sum(
        int(args.max_duration_days // args.window_days)
        for _ in range(args.n_objects)
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
                      f"{pct:>5.1f}% | {len(results):>5}/{args.n_objects} objects | "
                      f"ETA: {eta_str}")

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
            print(f"  {f['object_id']}: {f['error']}")

    # --- Save results ---
    output_data = {
        "config": {
            "tau": args.tau,
            "q": args.q,
            "window_days": args.window_days,
            "max_duration_days": args.max_duration_days,
            "n_objects_processed": args.n_objects,
            "n_objects_successful": len(successful),
            "total_windows": total_windows,
            "fixed_orbit": {
                "sma_m": args.fixed_sma_m,
                "ecc": args.fixed_ecc,
                "inc_deg": args.fixed_inc_deg,
                "raan_deg": args.fixed_raan_deg,
                "argp_deg": args.fixed_argp_deg,
                "m_deg": args.fixed_m_deg,
                "epoch_jd": args.epoch_jd,
            },
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
        print("\n" + ":" * 60)
        print("WINDOW-LEVEL PEAK MAHALANOBIS DISTRIBUTION")
        print(":" * 60)
        print(f"  Mean:  {peaks.mean():.3f}")
        print(f"  Std:   {peaks.std():.3f}")
        print(f"  Min:   {peaks.min():.3f}")
        print(f"  Max:   {peaks.max():.3f}")
        print(f"  95%:   {np.percentile(peaks, 95):.3f}")
        print(f"  99%:   {np.percentile(peaks, 99):.3f}")
        print(f"  99.9%: {np.percentile(peaks, 99.9):.3f}")

        print("\n" + ":" * 60)
        print("COMPARISON TO REFERENCE NULL CALIBRATION")
        print(":" * 60)
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
            print("   Check that --fixed-sma-m and --fixed-inc-deg match validation.py.")
            print("   If they DO match, the difference is due to something other than orbit.")
        else:
            print("\n✅ This run's distribution matches the reference null calibration.")
            print("   The filter behaves consistently with the fixed-orbit null case.")

    print("\n" + ":" * 60)
    print("INTERPRETATION")
    print(":" * 60)
    print("This is a FIXED-ORBIT VALIDATION TEST using SYNTHETIC data.")
    print("Every object has IDENTICAL SMA and INC — only the noise & Cd drift vary.")
    print("")
    print("Purpose:")
    print("  1. Direct comparison to calibrate_null_threshold.py")
    print("  2. Isolate orbit-dependent effects from filter-intrinsic effects")
    print("  3. If this matches calibrate_null but real-TLE fleet does NOT,")
    print("     the difference is orbit-dependent (altitude / inclination).")
    print("")
    print("Next steps:")
    print("  1. Run this with the SAME (tau, q, window-days) as the real-TLE fleet")
    print("  2. Compare distributions — if they differ, the effect is orbit-driven")
    print("  3. Then: Decide whether to use orbit-dependent thresholds")


if __name__ == "__main__":
    main()