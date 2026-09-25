#!/usr/bin/env python3
"""Real-TLE strike injection: calibrate ε(Δv) on actual TLE histories.

Perturbs real measurements via Gauss-VOP at a random epoch, then runs the
degraded filter. Accounts for real cadence, noise, and data gaps.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

import numpy as np

from constants.constants import MU, R_EARTH, F107_BASELINE, QUIET_KP
from phase2_filter.config import R_DIAG_ELEMENTS
from phase2_filter.ekf import EKF
from phase2_filter.smoother import fraser_potter_smoother
from phase2_filter.test_statistics import compute_mahalanobis_distance
from phase2_filter.dynamics import hk_from_e_argp
from propagator.orbital import MeanElements, mean_to_true_anomaly
from propagator.debris_impacts import gauss_vop
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


def inject_strike_into_measurements(
    measurements: List,
    strike_time: float,
    dv_mag: float,
) -> List:
    """Perturb measurements at/after strike_time via Gauss-VOP."""
    idx = 0
    for i, (t, z) in enumerate(measurements):
        if t > strike_time:
            idx = i
            break
    else:
        idx = len(measurements) - 1

    if idx == 0:
        t0, z0 = measurements[0]
    else:
        t0, z0 = measurements[idx - 1]
        t1, z1 = measurements[idx]
        frac = (strike_time - t0) / (t1 - t0) if t1 != t0 else 0.0
        z0 = z0 + frac * (z1 - z0)

    a, e, i, raan, argp, M = z0
    nu, _ = mean_to_true_anomaly(M, e)

    dv_rsw = np.array([0.0, -float(dv_mag), 0.0])
    da, de, di, dOm, darg = gauss_vop(a, e, i, argp, nu, dv_rsw)

    perturbed = []
    for t, z in measurements:
        if t >= strike_time:
            z_new = z.copy()
            z_new[0] += da
            z_new[1] = max(0.0, z_new[1] + de)
            z_new[2] += di
            z_new[3] = (z_new[3] + dOm) % (2 * math.pi)
            z_new[4] = (z_new[4] + darg) % (2 * math.pi)
            perturbed.append((t, z_new))
        else:
            perturbed.append((t, z.copy()))

    return perturbed


def run_strike_sweep_on_object(
    norad_id: str,
    fpath: str,
    duration_days: float,
    tau: float,
    q: float,
    window_days: float,
    threshold: float,
    dv_grid: np.ndarray,
    seed: int,
) -> Dict:
    try:
        rng = np.random.default_rng(seed)
        tle_list = load_tles_for_object(fpath)
        if not tle_list:
            return {"norad_id": norad_id, "success": False, "error": "no TLEs loaded", "results": {}}

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

        if len(measurements) < 15:
            return {"norad_id": norad_id, "success": False,
                    "error": f"too few measurements ({len(measurements)})", "results": {}}

        max_t = measurements[-1][0]
        total_data_days = max_t / 86400.0

        window_s = window_days * 86400.0
        n_windows = int(max_t // window_s)
        if n_windows < 1:
            return {"norad_id": norad_id, "success": False, "error": "no windows", "results": {}}

        valid_windows = []
        for w in range(n_windows):
            w_start = w * window_s
            w_end = (w + 1) * window_s
            w_meas = [(t, z) for t, z in measurements if w_start <= t < w_end]
            if len(w_meas) < 15:
                continue
            times = [t for t, _ in w_meas]
            gaps = [(times[i+1] - times[i]) / 86400.0 for i in range(len(times) - 1)]
            if gaps and max(gaps) > 14.0:
                continue
            a_s = float(w_meas[0][1][0])
            a_e = float(w_meas[-1][1][0])
            if abs((a_s - a_e) / 1000.0) > 10.0:
                continue
            if w == 0 and total_data_days > 90.0:
                continue
            valid_windows.append(w)

        if not valid_windows:
            return {"norad_id": norad_id, "success": False, "error": "no valid windows", "results": {}}

        w_inj = rng.choice(valid_windows)
        w_start = w_inj * window_s
        w_end = (w_inj + 1) * window_s
        strike_time = (w_start + w_end) / 2.0

        results = {}
        cd_filter = 2.2
        area, mass = 1.0, 200.0
        mask = np.zeros(9, dtype=bool)
        mask[0] = True

        for dv in dv_grid:
            dv_f = float(dv)
            try:
                perturbed = inject_strike_into_measurements(measurements, strike_time, dv_f)
                window_meas = [(t, z) for t, z in perturbed if w_start <= t < w_end]
                if len(window_meas) < 3:
                    results[dv_f] = {"detected": False, "peak_maha": None, "error": "too few post-injection"}
                    continue

                el0 = MeanElements(
                    a=float(window_meas[0][1][0]),
                    ecc=float(window_meas[0][1][1]),
                    inc=float(window_meas[0][1][2]),
                    raan=float(window_meas[0][1][3]),
                    argp=float(window_meas[0][1][4]),
                    M=float(window_meas[0][1][5]),
                )
                window_meas_rel = [(t - w_start, z) for t, z in window_meas]

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
                    results[dv_f] = {"detected": False, "peak_maha": None, "error": "fwd too short"}
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

                times = np.array([t for t, _ in window_meas_rel])
                window_mask = (times >= (strike_time - w_start - 86400)) & (times <= (strike_time - w_start + 86400))
                peak = float(maha[window_mask].max()) if window_mask.any() else float(maha.max())

                if not np.isfinite(peak) or abs(peak) > SANITY_CEILING:
                    results[dv_f] = {"detected": False, "peak_maha": None, "error": "peak diverged"}
                    continue

                results[dv_f] = {"detected": peak > threshold, "peak_maha": peak}

            except RuntimeError as e:
                if "diverged" in str(e).lower():
                    results[dv_f] = {"detected": False, "peak_maha": None, "error": "ekf diverged"}
                    continue
                raise
            except Exception as e:
                results[dv_f] = {"detected": False, "peak_maha": None, "error": str(e)}

        return {
            "norad_id": norad_id, "success": True,
            "w_injected": int(w_inj), "strike_time_rel_s": float(strike_time),
            "results": results,
        }

    except Exception as e:
        return {"norad_id": norad_id, "success": False, "error": str(e), "results": {}}


def process_one_wrapper(args):
    return run_strike_sweep_on_object(*args)


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
    parser.add_argument("--dv-min", type=float, default=1.0e-4)
    parser.add_argument("--dv-max", type=float, default=0.5)
    parser.add_argument("--n-dv", type=int, default=14)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=800000)
    parser.add_argument("--output", default="results/real_tle_efficiency.json")
    args = parser.parse_args()

    dv_grid = np.logspace(np.log10(args.dv_min), np.log10(args.dv_max), args.n_dv)
    print(f"Real-TLE strike injection")
    print(f"  threshold={args.threshold:.3f}  tau={args.tau:.3e}  q={args.q:.3e}")
    print(f"  Δv {args.dv_min:.2e} → {args.dv_max:.2e} ({args.n_dv} pts)")
    print()

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
                        args.threshold, dv_grid, args.seed_offset + idx))

    print(f"Processing {len(objects)} objects...")

    t0 = time.time()
    all_results = []
    with mp.Pool(args.n_workers, maxtasksperchild=10) as pool:
        iter_results = pool.imap_unordered(
            process_one_wrapper, objects,
            chunksize=max(1, len(objects) // (args.n_workers * 4))
        )
        for i, result in enumerate(iter_results, 1):
            all_results.append(result)
            if i % max(1, len(objects) // 20) == 0 or i == len(objects):
                pct = 100.0 * i / len(objects)
                elapsed = time.time() - t0
                eta = (len(objects) - i) * (elapsed / i) if i > 0 else 0
                print(f"  [{i:>4}/{len(objects)}] {pct:>5.1f}%  ETA {eta/60:.1f}m")

    successful = [r for r in all_results if r["success"]]
    print(f"\nFinished. Successful: {len(successful)}, Failed: {len(all_results) - len(successful)}")

    detections = {float(dv): [] for dv in dv_grid}
    peaks = {float(dv): [] for dv in dv_grid}

    for r in successful:
        for dv_str, res in r["results"].items():
            dv = float(dv_str)
            if res.get("peak_maha") is not None:
                detections[dv].append(1 if res["detected"] else 0)
                peaks[dv].append(res["peak_maha"])

    print(f"\n{'Δv (m/s)':>12} {'N':>8} {'ε':>8} {'mean peak':>12} {'std peak':>12}")
    print("-" * 55)

    efficiency = {}
    for dv in dv_grid:
        dv_f = float(dv)
        n = len(detections[dv_f])
        if n == 0:
            print(f"{dv_f:12.5f} {0:>8} {'N/A':>8} {'N/A':>12} {'N/A':>12}")
            efficiency[dv_f] = {"n": 0, "epsilon": None, "mean_peak": None, "std_peak": None}
            continue
        eps = np.mean(detections[dv_f])
        mean_peak = np.mean(peaks[dv_f]) if peaks[dv_f] else None
        std_peak = np.std(peaks[dv_f]) if peaks[dv_f] else None
        print(f"{dv_f:12.5f} {n:>8} {eps:>8.4f} {mean_peak:>12.3f} {std_peak:>12.3f}")
        efficiency[dv_f] = {
            "n": n, "epsilon": float(eps),
            "mean_peak": float(mean_peak) if mean_peak is not None else None,
            "std_peak": float(std_peak) if std_peak is not None else None,
        }

    dvs = np.array([float(dv) for dv in dv_grid])
    eps_arr = np.array([efficiency[float(dv)]["epsilon"] for dv in dv_grid])
    valid = ~np.isnan(eps_arr.astype(float))
    dv50 = dv90 = None
    if valid.any():
        dvs_v = dvs[valid]
        eps_v = eps_arr[valid].astype(float)
        sort_idx = np.argsort(eps_v)
        dvs_v = dvs_v[sort_idx]
        eps_v = eps_v[sort_idx]
        if eps_v.min() <= 0.5 <= eps_v.max():
            dv50 = float(np.interp(0.5, eps_v, dvs_v))
        if eps_v.min() <= 0.9 <= eps_v.max():
            dv90 = float(np.interp(0.9, eps_v, dvs_v))

    print()
    print(f"DV_50 = {dv50:.5f} m/s" if dv50 else "DV_50: not bracketed")
    print(f"DV_90 = {dv90:.5f} m/s" if dv90 else "DV_90: not bracketed")

    output_data = {
        "config": {
            "tau": args.tau, "q": args.q, "window_days": args.window_days,
            "threshold": args.threshold, "dv_min": args.dv_min, "dv_max": args.dv_max,
            "n_dv": args.n_dv, "n_objects": len(objects), "n_successful": len(successful),
        },
        "efficiency": efficiency,
        "dv50": dv50, "dv90": dv90,
        "per_object_results": all_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()