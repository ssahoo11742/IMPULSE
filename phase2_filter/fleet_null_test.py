"""Fleet validation: run EKF+smoother on real-orbit TLEs with synthetic null data.

Takes one TLE per object, propagates it clean (no debris), adds measurement
noise, windows into 90-day chunks, and reports the peak Mahalanobis
distribution. Validation only — does not detect real debris.

Usage:
    python -m phase2_filter.process_fleet \
        --tle-file TLE/real_tles.tle \
        --meta TLE/real_tles.tle.meta.json \
        --durations TLE/real_tles.tle.durations.json \
        --tau 3e4 --q 1e-19 --window-days 90 --max-duration-days 7300 \
        --output results/fleet_validation.json \
        --n-workers 8
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
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

SANITY_CEILING = 1.0e4

REF_NULL_MEAN = 3.174
REF_NULL_STD = 1.356
REF_NULL_99PCT = 6.830
REF_NULL_9997PCT = 8.843


def stratified_sample(objects_to_process, n_target=500, n_inc_bins=6, n_sma_bins=6, seed=42):
    """Sample across (inclination, altitude) bins instead of file order."""
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
    print(f"Stratified sample: {len(selected)} objects across {n_buckets} bins "
          f"({n_inc_bins}x{n_sma_bins})")
    return [objects_to_process[i] for i in selected]


def propagate_truth_from_tle(
    tle: Dict, epoch_jd: float, duration_s: float, dt_s: float,
    area: float = 1.0, mass: float = 200.0, f107_base: float = F107_BASELINE,
    Cd_base: Optional[float] = None, rng: Optional[np.random.Generator] = None,
    vary_cd: bool = True, truth_substeps: int = 4,
) -> Tuple[Dict, List, Dict, float]:
    """Propagate truth from a TLE with no debris forcing."""
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
    """Add Gaussian noise to truth elements (R_diag are variances)."""
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
        1.0e-6, 1.0e-6, 1.0e-6,
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
    """Forward + backward + smoother on one window. Returns peak_maha."""
    x0 = _build_initial_state(el_window_start)
    P0 = _build_initial_covariance(el_window_start)

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

    sm_states, sm_covs = fraser_potter_smoother(
        fwd_states, fwd_covs,
        list(reversed(bwd_apriori_states)),
        list(reversed(bwd_apriori_covs))
    )

    mask = np.zeros(9, dtype=bool)
    mask[0] = True

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


def run_filter_on_object(
    tle_data: Dict,
    duration_days: float,
    tau: float,
    q: float,
    window_days: float,
    seed: int,
) -> Dict:
    """Propagate truth, generate measurements, window, run EKF per window."""
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

        truth_hist, storms, f107_phases, Cd_truth_base = propagate_truth_from_tle(
            tle_data, epoch_jd, duration_s, dt_s,
            area=area, mass=mass, rng=rng
        )

        measurements = generate_noisy_measurements_from_truth(truth_hist, rng=rng)

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
                    window_meas_rel, el_start, epoch_jd, tau, q,
                    area, mass, storms, f107_phases, w_start_s, rng, Cd_truth_base,
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
    tle_data, duration_days, tau, q, window_days, seed_offset, idx = args
    return run_filter_on_object(
        tle_data, duration_days, tau, q, window_days, seed_offset + idx
    )


def main():
    parser = argparse.ArgumentParser(description="Fleet TLE validation (windowed null test)")

    parser.add_argument("--stratified-sample", type=int, default=None,
                        help="Sample this many objects across inc/alt bins")
    parser.add_argument("--n-inc-bins", type=int, default=6)
    parser.add_argument("--n-sma-bins", type=int, default=6)
    parser.add_argument("--tle-file", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--durations", required=True)
    parser.add_argument("--tau", type=float, default=3e4)
    parser.add_argument("--q", type=float, default=1e-19)
    parser.add_argument("--window-days", type=float, default=90.0)
    parser.add_argument("--max-duration-days", type=float, default=7300.0)
    parser.add_argument("--output", default="results/fleet_validation.json")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=100000)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--reference-null-mean", type=float, default=REF_NULL_MEAN)
    parser.add_argument("--reference-null-std", type=float, default=REF_NULL_STD)
    parser.add_argument("--reference-threshold", type=float, default=REF_NULL_9997PCT)
    args = parser.parse_args()

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

    tasks = [
        (obj["tle_data"], obj["duration_days"], args.tau, args.q,
         args.window_days, args.seed_offset, idx)
        for idx, obj in enumerate(objects_to_process)
    ]

    total_windows_expected = sum(
        int(obj["duration_days"] // args.window_days)
        for obj in objects_to_process
    )
    report_interval = max(1, int(total_windows_expected * 0.05))
    if total_windows_expected == 0:
        report_interval = 1

    print(f"Using {args.n_workers} workers, {total_windows_expected} expected windows")

    results = []
    windows_done = 0
    next_report = report_interval
    t0 = time.time()

    with mp.Pool(args.n_workers) as pool:
        iter_results = pool.imap_unordered(
            process_one_wrapper, tasks,
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
                      f"{pct:>5.1f}% | {len(results):>5}/{len(tasks)} objects | ETA: {eta_str}")

                while next_report <= windows_done:
                    next_report += report_interval

    print(f"Finished {len(results)} objects ({windows_done} windows) in "
          f"{(time.time()-t0)/60:.1f} min")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    all_windows = [w for r in successful for w in r["windows"]]
    total_windows = len(all_windows)
    ok_windows = [w for w in all_windows if w.get("peak_maha") is not None]
    bad_windows = total_windows - len(ok_windows)

    print(f"\nCompleted: {len(successful)} ok, {len(failed)} failed")
    print(f"Windows: {total_windows} total ({len(ok_windows)} clean, {bad_windows} diverged)")

    if failed:
        print("First few failures:")
        for f in failed[:5]:
            print(f"  {f['norad_id']}: {f['error']}")

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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved to {args.output}")

    if ok_windows:
        peaks = np.array([w["peak_maha"] for w in ok_windows])
        print(f"\nPeak Mahalanobis:")
        print(f"  mean={peaks.mean():.3f}  std={peaks.std():.3f}")
        print(f"  min={peaks.min():.3f}  max={peaks.max():.3f}")
        print(f"  95%={np.percentile(peaks, 95):.3f}  "
              f"99%={np.percentile(peaks, 99):.3f}  "
              f"99.9%={np.percentile(peaks, 99.9):.3f}")

        print(f"\nvs reference null:")
        print(f"  ref mean={args.reference_null_mean:.3f}  this={peaks.mean():.3f}  "
              f"Δ={peaks.mean() - args.reference_null_mean:.3f}")
        print(f"  ref std={args.reference_null_std:.3f}  this={peaks.std():.3f}")
        print(f"  ref 99.97%={args.reference_threshold:.3f}  "
              f"this={np.percentile(peaks, 99.97):.3f}")

        print("\nBy inclination:")
        inc_bins = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]
        for lo, hi in inc_bins:
            peaks_in_bin = []
            for r in successful:
                obj = next((o for o in objects_to_process if o["norad_id"] == r["norad_id"]), None)
                if obj is None:
                    continue
                if lo <= obj["inclination"] < hi:
                    for w in r["windows"]:
                        if w.get("peak_maha") is not None:
                            peaks_in_bin.append(w["peak_maha"])
            if peaks_in_bin:
                print(f"  {lo:3d}-{hi:3d}°: n={len(peaks_in_bin):4d}  "
                      f"mean={np.mean(peaks_in_bin):.3f}  std={np.std(peaks_in_bin):.3f}")
            else:
                print(f"  {lo:3d}-{hi:3d}°: no objects")

        print("\nBy altitude:")
        sma_bins = [(700, 800), (800, 900), (900, 1000), (1000, 1100)]
        for lo, hi in sma_bins:
            peaks_in_bin = []
            for r in successful:
                obj = next((o for o in objects_to_process if o["norad_id"] == r["norad_id"]), None)
                if obj is None:
                    continue
                if lo <= obj["sma_km"] < hi:
                    for w in r["windows"]:
                        if w.get("peak_maha") is not None:
                            peaks_in_bin.append(w["peak_maha"])
            if peaks_in_bin:
                print(f"  {lo:4d}-{hi:<4d}km: n={len(peaks_in_bin):5d}  "
                      f"mean={np.mean(peaks_in_bin):.3f}  std={np.std(peaks_in_bin):.3f}")
            else:
                print(f"  {lo:4d}-{hi:<4d}km: no objects")


if __name__ == "__main__":
    main()