#!/usr/bin/env python3
"""Diagnose extreme Mahalanobis peaks in real fleet data.

Processes real TLE histories and saves per-window metadata including:
- TLE count and gap statistics
- Altitude at window start/end and drop
- Peak Mahalanobis and trigger status

Usage:
    python -m phase2_filter.diagnose_extreme_peaks \
        --tle-history-dir TLE/histories/ \
        --meta TLE/real_tles.tle.meta.json \
        --durations TLE/real_tles.tle.durations_2009_2026.json \
        --threshold 13.258 \
        --n-workers 9 \
        --output results/extreme_peak_diagnosis.json
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
from TLE.tle_io import parse_tle, load_tles_from_file

SANITY_CEILING = 1.0e4
MAX_TLES_PER_FILE = 500


def load_tle_histories_streaming(tle_dir: str, meta_file: str):
    with open(meta_file) as f:
        meta_records = json.load(f)
    norad_ids = {str(r.get("NORAD_CAT_ID", "")) for r in meta_records}

    histories = {}
    for fname in sorted(os.listdir(tle_dir)):
        if not fname.endswith(".tle"):
            continue
        nid_from_fname = fname.replace(".tle", "")
        if nid_from_fname not in norad_ids:
            continue

        fpath = os.path.join(tle_dir, fname)
        tles = []
        try:
            raw_tles = load_tles_from_file(fpath)
            for name, l1, l2 in raw_tles:
                try:
                    tle = parse_tle(name, l1, l2)
                    if tle.get("epoch_jd", 0) > 0:
                        tles.append(tle)
                except Exception:
                    pass
                if len(tles) >= MAX_TLES_PER_FILE:
                    break
        except Exception as e:
            print(f"  Warning: could not read {fname}: {e}")
            continue

        if tles:
            histories[nid_from_fname] = tles

    return histories


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


def diagnose_object(
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
        a0 = tle_list[0].get("a", 0)
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
                "norad_id": norad_id,
                "success": False,
                "error": f"too few measurements ({len(measurements)})",
                "windows": [],
            }

        window_s = window_days * 86400.0
        max_t = measurements[-1][0]
        n_windows = int(max_t // window_s) + 1

        cd_filter = 2.2
        area, mass = 1.0, 200.0
        mask = np.zeros(9, dtype=bool)
        mask[0] = True

        windows = []
        for w in range(n_windows):
            w_start = w * window_s
            w_end = (w + 1) * window_s
            window_meas = [(t, z) for t, z in measurements if w_start <= t < w_end]
            if len(window_meas) < 3:
                continue

            a_start = float(window_meas[0][1][0])
            a_end = float(window_meas[-1][1][0])
            alt_start = (a_start - R_EARTH) / 1000.0
            alt_end = (a_end - R_EARTH) / 1000.0
            alt_drop = alt_start - alt_end

            times = [t for t, _ in window_meas]
            gaps = [(times[i+1] - times[i])/86400.0 for i in range(len(times)-1)]
            max_gap = max(gaps) if gaps else 0.0

            el0 = MeanElements(
                a=a_start,
                ecc=float(window_meas[0][1][1]),
                inc=float(window_meas[0][1][2]),
                raan=float(window_meas[0][1][3]),
                argp=float(window_meas[0][1][4]),
                M=float(window_meas[0][1][5]),
            )

            window_meas_rel = [(t - w_start, z) for t, z in window_meas]

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
                peak = None

            windows.append({
                "w": w,
                "n_tles": len(window_meas),
                "t_start_days": w_start / 86400.0,
                "t_end_days": w_end / 86400.0,
                "a_start_m": a_start,
                "a_end_m": a_end,
                "alt_start_km": alt_start,
                "alt_end_km": alt_end,
                "alt_drop_km": alt_drop,
                "max_gap_days": max_gap,
                "peak_maha": peak,
                "triggered": peak is not None and peak > threshold,
            })

        return {
            "norad_id": norad_id,
            "success": True,
            "n_windows": len(windows),
            "windows": windows,
        }

    except Exception as e:
        return {
            "norad_id": norad_id,
            "success": False,
            "error": str(e),
            "windows": [],
        }


def process_one_wrapper(args):
    return diagnose_object(*args)


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
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--output", default="results/extreme_peak_diagnosis.json")
    args = parser.parse_args()

    with open(args.durations) as f:
        durations_map = json.load(f)

    print("Loading TLE histories...")
    histories = load_tle_histories_streaming(args.tle_history_dir, args.meta)
    print(f"Loaded {len(histories)} objects")

    objects = []
    for nid, tles in histories.items():
        dur = durations_map.get(nid, {}).get("duration_days", 0)
        dur = min(dur, args.max_duration_days)
        if dur < args.window_days:
            continue
        objects.append((nid, tles, dur, args.tau, args.q, args.window_days, args.threshold))

    print(f"Processing {len(objects)} objects...")

    t0 = time.time()
    results = []
    with mp.Pool(args.n_workers) as pool:
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

    successful = [r for r in results if r["success"]]
    print(f"\nFinished. Successful: {len(successful)}, Failed: {len(results) - len(successful)}")

    # Summary of extremes
    all_windows = []
    for r in successful:
        for w in r["windows"]:
            w["norad_id"] = r["norad_id"]
            all_windows.append(w)

    all_windows.sort(key=lambda x: x.get("peak_maha", 0) or 0, reverse=True)

    print(f"\n{'='*70}")
    print("TOP 20 EXTREME WINDOWS")
    print(f"{'='*70}")
    print(f"{'NORAD':>8} {'W':>4} {'Peak':>10} {'Alt0':>8} {'Alt1':>8} {'Drop':>8} {'N_TLE':>6} {'MaxGap':>8}")
    print(f"{'-'*70}")
    for w in all_windows[:20]:
        print(f"{w['norad_id']:>8} {w['w']:>4} {w['peak_maha']:>10.1f} "
              f"{w['alt_start_km']:>8.1f} {w['alt_end_km']:>8.1f} {w['alt_drop_km']:>8.1f} "
              f"{w['n_tles']:>6} {w['max_gap_days']:>8.1f}")

    print(f"\n{'='*70}")
    print("DECAY CORRELATION")
    print(f"{'='*70}")
    triggered = [w for w in all_windows if w.get("triggered")]
    print(f"Total windows:        {len(all_windows)}")
    print(f"Triggered windows:    {len(triggered)}")
    print(f"Trigger rate:         {100*len(triggered)/len(all_windows):.2f}%")
    print()
    print(f"Triggered windows with alt_drop > 10 km:   {sum(1 for w in triggered if w['alt_drop_km'] > 10)}")
    print(f"Triggered windows with alt_drop > 50 km:   {sum(1 for w in triggered if w['alt_drop_km'] > 50)}")
    print(f"Triggered windows with alt_drop > 100 km:  {sum(1 for w in triggered if w['alt_drop_km'] > 100)}")
    print(f"Triggered windows with max_gap > 14 days:  {sum(1 for w in triggered if w['max_gap_days'] > 14)}")
    print(f"Triggered windows with n_tles < 10:        {sum(1 for w in triggered if w['n_tles'] < 10)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": {"threshold": args.threshold, "window_days": args.window_days},
            "per_object": results,
            "top_extreme": all_windows[:100],
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()