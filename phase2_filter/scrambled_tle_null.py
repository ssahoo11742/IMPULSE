#!/usr/bin/env python3
"""Scrambled-TLE null test: background rate on real data with signal destroyed.

Fits a spline to each object's TLE history, shuffles residuals in time,
rebuilds fake TLEs, and runs the degraded filter. Preserves cadence and
noise distribution while removing coherent physical signals.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import UnivariateSpline

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
    return np.diag([
        R_DIAG_ELEMENTS[0], R_h, R_k, R_DIAG_ELEMENTS[2], R_DIAG_ELEMENTS[3], R_DIAG_ELEMENTS[5],
        1.0e-6, 1.0e-6, 1.0e-6
    ])


def scramble_object(
    norad_id: str,
    fpath: str,
    duration_days: float,
    tau: float,
    q: float,
    window_days: float,
    threshold: float,
    seed: int,
) -> Dict:
    try:
        rng = np.random.default_rng(seed)
        tle_list = load_tles_for_object(fpath)
        if not tle_list:
            return {"norad_id": norad_id, "success": False, "error": "no TLEs loaded",
                    "n_valid_windows": 0, "peaks": [], "detected": 0}

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
            return {"norad_id": norad_id, "success": False, "error": "too few measurements",
                    "n_valid_windows": 0, "peaks": [], "detected": 0}

        t_arr = np.array([t for t, _ in measurements])
        z_arr = np.array([z for _, z in measurements])

        if len(t_arr) < 10:
            return {"norad_id": norad_id, "success": False, "error": "too few points for spline",
                    "n_valid_windows": 0, "peaks": [], "detected": 0}

        smooth_vals = np.zeros_like(z_arr)
        for dim in range(6):
            try:
                s = 0.1 * len(t_arr)
                spl = UnivariateSpline(t_arr, z_arr[:, dim], s=s, k=3)
                smooth_vals[:, dim] = spl(t_arr)
            except Exception:
                smooth_vals[:, dim] = np.interp(t_arr, t_arr, z_arr[:, dim])

        residuals = z_arr - smooth_vals
        perm = rng.permutation(len(residuals))
        z_scrambled = smooth_vals + residuals[perm]
        scrambled_meas = [(t_arr[i], z_scrambled[i]) for i in range(len(t_arr))]

        window_s = window_days * 86400.0
        max_t = scrambled_meas[-1][0]
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
            window_meas = [(t, z) for t, z in scrambled_meas if w_start <= t < w_end]

            if len(window_meas) < 15:
                continue
            times = [t for t, _ in window_meas]
            gaps = [(times[i+1] - times[i]) / 86400.0 for i in range(len(times) - 1)]
            if gaps and max(gaps) > 14.0:
                continue
            a_start = float(window_meas[0][1][0])
            a_end = float(window_meas[-1][1][0])
            if abs((a_start - a_end) / 1000.0) > 10.0:
                continue
            if w == 0 and total_data_days > 90.0:
                continue

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
            "norad_id": norad_id, "success": True,
            "n_valid_windows": n_valid_windows,
            "peaks": peaks, "detected": detected,
            "mean_peak": float(np.mean(peaks)) if peaks else None,
            "max_peak": float(np.max(peaks)) if peaks else None,
        }

    except Exception as e:
        return {"norad_id": norad_id, "success": False, "error": str(e),
                "n_valid_windows": 0, "peaks": [], "detected": 0}


def process_one_wrapper(args):
    return scramble_object(*args)


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
    parser.add_argument("--seed-offset", type=int, default=700000)
    parser.add_argument("--output", default="results/scrambled_null.json")
    args = parser.parse_args()

    with open(args.durations) as f:
        durations_map = json.load(f)

    with open(args.meta) as f:
        meta_records = json.load(f)
    norad_ids = {str(r.get("NORAD_CAT_ID", "")) for r in meta_records}

    objects = []
    for idx, fname in enumerate(sorted(os.listdir(args.tle_history_dir))):
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
        objects.append((nid, fpath, dur, args.tau, args.q, args.window_days,
                        args.threshold, args.seed_offset + idx))

    print(f"Processing {len(objects)} objects...")

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
                print(f"  [{i:>4}/{len(objects)}] {pct:>5.1f}%  ETA {eta/60:.1f}m")

    successful = [r for r in results if r["success"]]
    print(f"\nFinished. Successful: {len(successful)}, Failed: {len(results) - len(successful)}")

    all_peaks = []
    total_valid_windows = 0
    total_detected = 0
    for r in successful:
        all_peaks.extend(r["peaks"])
        total_valid_windows += r["n_valid_windows"]
        total_detected += r["detected"]

    all_peaks_arr = np.array(all_peaks)
    n_total = len(all_peaks_arr)

    print(f"\nValid windows: {total_valid_windows}")
    print(f"Peaks: {n_total}")

    if n_total > 0:
        print(f"Mean peak: {np.mean(all_peaks_arr):.3f}  std: {np.std(all_peaks_arr):.3f}")
        print(f"99.97th:   {np.percentile(all_peaks_arr, 99.97):.3f}")
        print(f"Threshold: {args.threshold:.3f}")
        print(f"Triggers:  {total_detected}")
        r_scrambled = total_detected / total_valid_windows if total_valid_windows > 0 else 0.0
        print(f"Scrambled FP rate: {r_scrambled*100:.4f}%")
    else:
        print("WARNING: no peaks computed")

    output_data = {
        "config": {
            "threshold": args.threshold, "tau": args.tau, "q": args.q,
            "window_days": args.window_days, "n_objects": len(objects),
            "n_successful": len(successful), "total_valid_windows": total_valid_windows,
        },
        "summary": {
            "total_peaks": n_total,
            "mean_peak": float(np.mean(all_peaks_arr)) if n_total > 0 else None,
            "std_peak": float(np.std(all_peaks_arr)) if n_total > 0 else None,
            "pct9997": float(np.percentile(all_peaks_arr, 99.97)) if n_total > 0 else None,
            "triggers": total_detected,
            "scrambled_fp_rate": float(total_detected / total_valid_windows) if total_valid_windows > 0 else 0.0,
        },
        "per_object": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()