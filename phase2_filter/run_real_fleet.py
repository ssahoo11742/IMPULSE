#!/usr/bin/env python3
"""Run real fleet through degraded filter with quality cuts.

Quality cuts applied per window:
  - n_tles >= 15
  - max_gap_days <= 14
  - abs(alt_drop_km) <= 10
  - Skip w=0 if object has >90 days of data (poor initial TLE bias)
  - EKF divergence guard (skip window if covariance explodes)

Usage:
    python -m phase2_filter.run_real_fleet \
        --tle-history-dir TLE/histories/ \
        --meta TLE/real_tles.tle.meta.json \
        --durations TLE/real_tles.tle.durations_2009_2026.json \
        --threshold 13.258 --tau 3e4 --q 1e-19 \
        --window-days 90 --n-workers 4 \
        --output results/real_detections.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from constants.constants import MU, R_EARTH, F107_BASELINE, QUIET_KP
from phase2_filter.config import R_DIAG_ELEMENTS
from phase2_filter.ekf import EKF
from phase2_filter.smoother import fraser_potter_smoother
from phase2_filter.test_statistics import compute_mahalanobis_distance
from phase2_filter.dynamics import hk_from_e_argp
from propagator.orbital import MeanElements
from TLE.tle_io import parse_tle, load_tles_from_file

SANITY_CEILING = 1.0e4
MAX_TLES_PER_FILE = 500


def load_tles_for_object(fpath: str) -> List[Dict]:
    """Load TLEs from a single .tle file, capped at MAX_TLES_PER_FILE."""
    tles = []
    try:
        raw = load_tles_from_file(fpath)
        for name, l1, l2 in raw:
            try:
                tle = parse_tle(name, l1, l2)
                if tle.get("epoch_jd", 0) > 0:
                    tles.append(tle)
            except Exception:
                pass
            if len(tles) >= MAX_TLES_PER_FILE:
                break
    except Exception as e:
        print(f"  Warning: could not read {fpath}: {e}")
    return tles


def tle_to_measurement(tle: Dict) -> Optional[np.ndarray]:
    a = tle.get("a")
    if a is None or a <= 0:
        n_rev = tle.get("n_revday", 0.0)
        if n_rev <= 0:
            return None
        n_rad_s = n_rev * 2 * math.pi / 86400.0
        a = (MU / (n_rad_s ** 2)) ** (1.0 / 3.0)
    e = tle.get("ecc", 0.0)
    i = tle.get("inc", 0.0)
    raan = tle.get("raan", 0.0)
    argp = tle.get("argp", 0.0)
    M = tle.get("M", 0.0)
    return np.array([a, e, i, raan, argp, M])


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


def process_object_real(
    norad_id: str,
    tle_list: List[Dict],
    duration_days: float,
    tau: float,
    q: float,
    window_days: float,
    threshold: float,
) -> Dict:
    try:
        tle_list = sorted(tle_list, key=lambda t: t.get("epoch_jd", 0))
        measurements = []
        epoch0 = tle_list[0].get("epoch_jd", 0)
        for tle in tle_list:
            epoch_jd = tle.get("epoch_jd", 0)
            if epoch_jd == 0:
                continue
            meas = tle_to_measurement(tle)
            if meas is None:
                continue
            t_rel = (epoch_jd - epoch0) * 86400.0
            measurements.append((t_rel, meas))

        if len(measurements) < 5:
            return {
                "norad_id": norad_id, "success": False,
                "error": f"too few measurements ({len(measurements)})",
                "n_windows": 0, "n_valid_windows": 0, "peaks": [], "detected": 0,
            }

        window_s = window_days * 86400.0
        max_t = measurements[-1][0]
        n_windows = int(max_t // window_s) + 1

        peaks = []
        detected = 0
        n_valid_windows = 0
        cd_filter = 2.2
        area, mass = 1.0, 200.0
        mask = np.zeros(9, dtype=bool)
        mask[0] = True
        total_data_days = max_t / 86400.0

        for w in range(n_windows):
            w_start = w * window_s
            w_end = (w + 1) * window_s
            window_meas = [(t, z) for t, z in measurements if w_start <= t < w_end]

            # --- QUALITY CUTS ---
            if len(window_meas) < 15:
                continue

            times = [t for t, _ in window_meas]
            gaps = [(times[i+1] - times[i]) / 86400.0 for i in range(len(times) - 1)]
            if gaps and max(gaps) > 14.0:
                continue

            a_start = float(window_meas[0][1][0])
            a_end = float(window_meas[-1][1][0])
            alt_drop = (a_start - a_end) / 1000.0
            if abs(alt_drop) > 10.0:
                continue

            # Skip window 0 if object has enough later data (poor initial TLE bias)
            if w == 0 and total_data_days > 90.0:
                continue
            # --- END QUALITY CUTS ---

            n_valid_windows += 1

            el0 = MeanElements(
                a=a_start,
                ecc=float(window_meas[0][1][1]),
                inc=float(window_meas[0][1][2]),
                raan=float(window_meas[0][1][3]),
                argp=float(window_meas[0][1][4]),
                M=float(window_meas[0][1][5]),
            )
            window_meas_rel = [(t - w_start, z) for t, z in window_meas]

            try:
                # Forward
                x0 = _build_initial_state(el0)
                P0 = _build_initial_covariance(el0)
                ekf_fwd = EKF(x0, P0, cd_filter, area, mass, epoch0, tau=tau, q=q, direction="forward")
                fwd_states, fwd_covs = [], []
                prev_t = 0.0
                for t_rel, z in window_meas_rel:
                    dt = t_rel - prev_t
                    ekf_fwd.predict(dt, F107_BASELINE, QUIET_KP)
                    ekf_fwd.update(z, R_DIAG_ELEMENTS)
                    fwd_states.append(ekf_fwd.x.copy())
                    fwd_covs.append(ekf_fwd.P.copy())
                    prev_t = t_rel

                if len(fwd_states) < 2:
                    continue

                # Backward
                x0_bwd = fwd_states[-1].copy()
                P0_bwd = fwd_covs[-1].copy() * 100.0
                ekf_bwd = EKF(x0_bwd, P0_bwd, cd_filter, area, mass, epoch0, tau=tau, q=q, direction="backward")
                bwd_apriori_states, bwd_apriori_covs = [], []
                prev_t = window_meas_rel[-1][0]
                for t_rel, z in reversed(window_meas_rel):
                    dt = t_rel - prev_t
                    if abs(dt) > 1e-12:
                        ekf_bwd.predict(dt, F107_BASELINE, QUIET_KP)
                    bwd_apriori_states.append(ekf_bwd.x.copy())
                    bwd_apriori_covs.append(ekf_bwd.P.copy())
                    ekf_bwd.update(z, R_DIAG_ELEMENTS)
                    prev_t = t_rel

                # Smoother
                sm_states, sm_covs = fraser_potter_smoother(
                    fwd_states, fwd_covs,
                    list(reversed(bwd_apriori_states)),
                    list(reversed(bwd_apriori_covs))
                )

                maha = compute_mahalanobis_distance(
                    fwd_states, fwd_covs,
                    list(reversed(bwd_apriori_states)),
                    list(reversed(bwd_apriori_covs)),
                    sm_covs, state_mask=mask
                )

                peak = float(np.max(maha))
                if not np.isfinite(peak) or abs(peak) > SANITY_CEILING:
                    continue

                peaks.append(peak)
                if peak > threshold:
                    detected += 1

            except RuntimeError as e:
                if "diverged" in str(e).lower():
                    continue
                raise

        return {
            "norad_id": norad_id,
            "success": True,
            "n_windows": n_windows,
            "n_valid_windows": n_valid_windows,
            "peaks": peaks,
            "detected": detected,
            "mean_peak": float(np.mean(peaks)) if peaks else None,
            "std_peak": float(np.std(peaks)) if peaks else None,
            "max_peak": float(np.max(peaks)) if peaks else None,
        }

    except Exception as e:
        return {
            "norad_id": norad_id, "success": False, "error": str(e),
            "n_windows": 0, "n_valid_windows": 0, "peaks": [], "detected": 0,
        }


def process_one_wrapper(args):
    """Worker wrapper that loads TLEs from file path instead of receiving them pre-loaded."""
    norad_id, fpath, duration_days, tau, q, window_days, threshold = args
    tle_list = load_tles_for_object(fpath)
    return process_object_real(norad_id, tle_list, duration_days, tau, q, window_days, threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tle-history-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--durations", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--tau", type=float, default=3e4)
    parser.add_argument("--q", type=float, default=1e-19)
    parser.add_argument("--window-days", type=float, default=90.0)
    parser.add_argument("--max-duration-days", type=float, default=7300.0)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--output", default="results/real_detections.json")
    args = parser.parse_args()

    with open(args.durations) as f:
        durations_map = json.load(f)

    with open(args.meta) as f:
        meta_records = json.load(f)
    norad_ids = {str(r.get("NORAD_CAT_ID", "")) for r in meta_records}

    # Build task list: pass file paths, not TLE data, to avoid MemoryError on Windows
    objects = []
    for fname in sorted(os.listdir(args.tle_history_dir)):
        if not fname.endswith(".tle"):
            continue
        nid = fname.replace(".tle", "")
        if nid not in norad_ids:
            continue
        fpath = os.path.join(args.tle_history_dir, fname)
        dur = durations_map.get(nid, {}).get("duration_days", 0)
        dur = min(dur, args.max_duration_days)
        if dur < args.window_days:
            continue
        objects.append((nid, fpath, dur, args.tau, args.q, args.window_days, args.threshold))

    print(f"Processing {len(objects)} objects with sufficient duration...")

    t0 = time.time()
    results = []
    with mp.Pool(args.n_workers, maxtasksperchild=10) as pool:
        iter_results = pool.imap_unordered(
            process_one_wrapper, objects,
            chunksize=max(1, len(objects) // (args.n_workers * 4))
        )
        for i, result in enumerate(iter_results, 1):
            results.append(result)
            if i % max(1, len(objects) // 20) == 0 or i == len(objects):
                pct = 100.0 * i / len(objects)
                elapsed = time.time() - t0
                eta = (len(objects) - i) * (elapsed / i) if i > 0 else 0
                print(f"  [{i:>4}/{len(objects)}] {pct:>5.1f}% | ETA: {eta/60:.1f}m")

    elapsed_total = time.time() - t0
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    print(f"\nFinished {len(results)} objects in {elapsed_total/60:.1f} min")
    print(f"Successful: {len(successful)}, Failed: {len(failed)}")

    if failed:
        errors = Counter(r.get("error", "unknown")[:120] for r in failed)
        print("\nTop failure reasons:")
        for err, count in errors.most_common(10):
            print(f"  {count:>4}x: {err}")

    all_peaks = []
    total_windows = 0
    total_valid_windows = 0
    total_detected = 0
    for r in successful:
        all_peaks.extend(r["peaks"])
        total_windows += r["n_windows"]
        total_valid_windows += r["n_valid_windows"]
        total_detected += r["detected"]

    all_peaks_arr = np.array(all_peaks)
    n_total = len(all_peaks_arr)

    print(f"\n{'='*60}")
    print("REAL FLEET DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Total windows (raw):     {total_windows}")
    print(f"Total valid windows:     {total_valid_windows}")
    print(f"Valid fraction:          {100*total_valid_windows/total_windows:.1f}%" if total_windows > 0 else "N/A")
    print(f"Total peaks computed:    {n_total}")

    if n_total > 0:
        print(f"Mean peak Mahalanobis:   {np.mean(all_peaks_arr):.3f}")
        print(f"Std peak Mahalanobis:    {np.std(all_peaks_arr):.3f}")
        print(f"99.97th percentile:      {np.percentile(all_peaks_arr, 99.97):.3f}")
        print(f"\nThreshold: {args.threshold:.3f}")
        print(f"Raw detections (>thr):   {total_detected}")
        print(f"Raw trigger rate:        {100*total_detected/total_valid_windows:.2f}%" if total_valid_windows > 0 else "N/A")
    else:
        print("\nWARNING: no peaks computed")

    output_data = {
        "config": {
            "threshold": args.threshold, "tau": args.tau, "q": args.q,
            "window_days": args.window_days, "n_objects": len(objects),
            "n_successful": len(successful), "total_windows": total_windows,
            "total_valid_windows": total_valid_windows,
        },
        "summary": {
            "total_peaks": n_total,
            "mean_peak": float(np.mean(all_peaks_arr)) if n_total > 0 else None,
            "std_peak": float(np.std(all_peaks_arr)) if n_total > 0 else None,
            "pct9997": float(np.percentile(all_peaks_arr, 99.97)) if n_total > 0 else None,
            "raw_detections": total_detected,
            "raw_trigger_rate": float(total_detected / total_valid_windows) if total_valid_windows > 0 else 0.0,
        },
        "per_object": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()