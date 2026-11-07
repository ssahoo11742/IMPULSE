#!/usr/bin/env python3
"""Real-TLE strike injection: calibrate epsilon(dv) on actual TLE histories. V2.

(Same physics / paper mapping as the previous version; see its docstring.
This file only changes PERFORMANCE, never the statistics:)

  PERF 1. STRIKE OFFSETS ARE LINEAR IN dv. The Gauss variational equations
         are linear in the impulse, so the element offsets for ANY dv are
         just (offsets at dv=1 m/s) * dv. Previously every dv re-ran the
         interpolation + mean_to_true_anomaly + gauss_vop and re-copied the
         FULL measurement stream (up to 500 arrays) per dv -> 9x redundant
         work per object. Now: solve once, scale per dv, and only ever
         touch the in-window measurements.

  PERF 2. THE WINDOW IS BUILT ONCE. Previously window_meas was rebuilt by
         scanning the full perturbed stream for each of the 9 dv points.
         Now the window slice is computed once and split into pre-strike /
         post-strike parts; each dv only rewrites the post-strike part.

  PERF 3. --n-workers defaults to ALL logical CPUs (was 4).

Everything else --- window selection, honest-R, the paired design, the
exclusion census, the output format --- is byte-for-byte the same logic.
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

from constants.constants import MU
from propagator.orbital import mean_to_true_anomaly
from propagator.debris_impacts import gauss_vop
from TLE.tle_io import parse_tle, load_tles_from_file
from phase2_filter.config import R_DIAG_ELEMENTS
from phase2_filter.convention_consistency_test import (
    R_EMP_CAP_FACTOR,
    empirical_R_from_obs,
    run_filter_window,
)

MAX_TLES_PER_FILE = 500
EXCLUSION_WARN_FRAC = 0.02
DEFAULT_DV_GRID = "0.0001,0.02,0.05,0.1,0.2,0.4,0.8,1.6,3.2"


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


def honest_r_from_stream(measurements: List,
                         r_from_fit: bool,
                         r_scale: float) -> np.ndarray:
    """Per-object honest measurement covariance, fleet-scan-consistent."""
    R_diag = np.asarray(R_DIAG_ELEMENTS, dtype=float) * r_scale
    r_used = R_diag.copy()
    if r_from_fit:
        obs = [(t, z, None) for (t, z) in measurements]
        R_emp = empirical_R_from_obs(obs)
        if R_emp is not None:
            r_used = np.maximum(np.asarray(R_emp, dtype=float), R_diag)
    r_used = np.minimum(
        r_used,
        np.asarray(R_DIAG_ELEMENTS, dtype=float)
        * R_EMP_CAP_FACTOR * max(1.0, r_scale),
    )
    return r_used


def strike_offsets_per_unit_dv(measurements: List, strike_time: float):
    """Solve the Gauss-VOP offsets ONCE for a unit (-S) impulse and return
    the per-dv-m/s element offsets (da, de, di, dOm, darg).

    gauss_vop is linear in the impulse vector, so offsets(dv) =
    offsets(1 m/s) * dv. This replaces the per-dv interpolation +
    mean_to_true_anomaly + gauss_vop solve (9x per object before).

    The strike state is interpolated from the FULL stream exactly as the
    original inject_strike_into_measurements did --- identical numerics.
    """
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

    dv_unit = np.array([0.0, -1.0, 0.0])
    da, de, di, dOm, darg = gauss_vop(a, e, i, argp, nu, dv_unit)
    return (float(da), float(de), float(di), float(dOm), float(darg))


def apply_unit_offsets(z: np.ndarray, off, dv: float) -> np.ndarray:
    """Perturb one measurement by (unit offsets) * dv. Same channel set
    as the original: a, e (floored at 0), i, raan, argp; M untouched."""
    da, de, di, dOm, darg = off
    z_new = z.copy()
    z_new[0] += da * dv
    z_new[1] = max(0.0, z_new[1] + de * dv)
    z_new[2] += di * dv
    z_new[3] = (z_new[3] + dOm * dv) % (2 * math.pi)
    z_new[4] = (z_new[4] + darg * dv) % (2 * math.pi)
    return z_new


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
    r_from_fit: bool,
    r_scale: float,
    edge_trim_days: float,
    p0_bwd_inflate: float,
) -> Dict:
    try:
        rng = np.random.default_rng(seed)
        tle_list = load_tles_for_object(fpath)
        if not tle_list:
            return {"norad_id": norad_id, "success": False,
                    "error": "no TLEs loaded", "results": {}}

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
                    "error": f"too few measurements ({len(measurements)})",
                    "results": {}}

        max_t = measurements[-1][0]
        total_data_days = max_t / 86400.0

        window_s = window_days * 86400.0
        n_windows = int(max_t // window_s)
        if n_windows < 1:
            return {"norad_id": norad_id, "success": False,
                    "error": "no windows", "results": {}}

        # Window validity: identical cuts to the fleet scan.
        valid_windows = []
        for w in range(n_windows):
            w_start = w * window_s
            w_end = (w + 1) * window_s
            w_meas = [(t, z) for t, z in measurements if w_start <= t < w_end]
            if len(w_meas) < 15:
                continue
            times = [t for t, _ in w_meas]
            gaps = [(times[i + 1] - times[i]) / 86400.0
                    for i in range(len(times) - 1)]
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
            return {"norad_id": norad_id, "success": False,
                    "error": "no valid windows", "results": {}}

        # Honest per-object covariance from the UNPERTURBED stream, once,
        # shared by every dv on this object (paired across dv).
        r_used = honest_r_from_stream(measurements, r_from_fit, r_scale)
        r_ratio = [float(v) for v in
                   (r_used / np.asarray(R_DIAG_ELEMENTS, dtype=float))]

        # Pick a window whose data bracket the strike epoch.
        order = rng.permutation(valid_windows)
        w_inj = None
        for w in order:
            w_start = w * window_s
            w_end = (w + 1) * window_s
            strike_time = (w_start + w_end) / 2.0
            n_after = sum(1 for t, _ in measurements
                          if strike_time <= t < w_end)
            if n_after >= 3:
                w_inj = int(w)
                break
        if w_inj is None:
            return {"norad_id": norad_id, "success": False,
                    "error": "no window brackets the midpoint",
                    "results": {}}
        w_start = w_inj * window_s
        w_end = (w_inj + 1) * window_s
        strike_time = (w_start + w_end) / 2.0
        edge_trim_s = edge_trim_days * 86400.0

        # ---- PERF: solve the strike ONCE, build the window ONCE --------
        off = strike_offsets_per_unit_dv(measurements, strike_time)

        # Unperturbed window slice, relative times, split at the strike.
        window_pre = []   # (t_rel, z): before the strike, shared by all dv
        window_post = []  # (t_rel, z): at/after the strike, perturbed per dv
        for t, z in measurements:
            if w_start <= t < w_end:
                if t >= strike_time:
                    window_post.append((t - w_start, z))
                else:
                    window_pre.append((t - w_start, z))
        n_window = len(window_pre) + len(window_post)

        # Precompute the per-dv perturbed post-strike parts once each.
        post_by_dv = {}
        for dv in dv_grid:
            dv_f = float(dv)
            post_by_dv[dv_f] = [
                (t, apply_unit_offsets(z, off, dv_f)) for t, z in window_post
            ]

        results = {}
        for dv in dv_grid:
            dv_f = float(dv)
            try:
                if n_window < 5:
                    results[dv_f] = {
                        "detected": False, "peak_maha": None,
                        "excluded": "too_few_window_measurements",
                    }
                    continue

                window_meas_rel = window_pre + post_by_dv[dv_f]
                window_meas_rel.sort(key=lambda x: x[0])

                try:
                    fw = run_filter_window(
                        window_meas_rel, epoch0, tau, q,
                        edge_trim_s=edge_trim_s,
                        R_diag=r_used,
                        p0_bwd_inflate=p0_bwd_inflate,
                    )
                except RuntimeError as e:
                    if "diverged" in str(e).lower():
                        results[dv_f] = {
                            "detected": False, "peak_maha": None,
                            "excluded": "ekf_diverged",
                        }
                        continue
                    raise

                if fw is None:
                    results[dv_f] = {
                        "detected": False, "peak_maha": None,
                        "excluded": "diverged_or_sanity_censored",
                    }
                    continue

                peak = float(fw["peak"])
                results[dv_f] = {
                    "detected": peak > threshold,
                    "peak_maha": peak,
                    "argmax_t_s": fw["argmax_t_s"],
                    "n_epochs_used": fw["n_epochs_used"],
                    "excluded": None,
                }

            except Exception as e:
                results[dv_f] = {
                    "detected": False, "peak_maha": None,
                    "excluded": f"error: {e}",
                }

        return {
            "norad_id": norad_id, "success": True,
            "w_injected": int(w_inj), "strike_time_rel_s": float(strike_time),
            "r_ratio": r_ratio, "results": results,
        }

    except Exception as e:
        return {"norad_id": norad_id, "success": False,
                "error": str(e), "results": {}}


def process_one_wrapper(args):
    return run_strike_sweep_on_object(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tle-history-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--durations", required=True)
    parser.add_argument("--threshold", type=float, default=417.94)
    parser.add_argument("--tau", type=float, default=3e4)
    parser.add_argument("--q", type=float, default=1e-19)
    parser.add_argument("--window-days", type=float, default=90.0)
    parser.add_argument("--max-duration-days", type=float, default=7300.0)
    parser.add_argument("--dv-grid", default=DEFAULT_DV_GRID)
    parser.add_argument("--dv-min", type=float, default=1.0e-4)
    parser.add_argument("--dv-max", type=float, default=0.5)
    parser.add_argument("--n-dv", type=int, default=14)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--r-from-fit", type=int, default=1)
    parser.add_argument("--r-scale", type=float, default=1.0)
    parser.add_argument("--edge-trim-days", type=float, default=7.0)
    parser.add_argument("--p0-bwd-inflate", type=float, default=100.0)
    # PERF 3: default to all logical CPUs instead of 4.
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed-offset", type=int, default=800000)
    parser.add_argument("--output", default="results/real_tle_efficiency.json")
    args = parser.parse_args()

    if args.dv_grid.strip():
        dv_grid = np.array(sorted(
            {float(x) for x in args.dv_grid.split(",") if x.strip()}))
    else:
        dv_grid = np.logspace(np.log10(args.dv_min),
                              np.log10(args.dv_max), args.n_dv)

    print("Real-TLE strike injection (V2, production-consistent)")
    print(f"  threshold={args.threshold:.3f}  tau={args.tau:.3e}  "
          f"q={args.q:.3e}")
    print(f"  honest-R={'on' if args.r_from_fit else 'OFF (ablation)'}"
          f"  r-scale={args.r_scale}  edge-trim={args.edge_trim_days}d")
    print(f"  dv grid: {', '.join(f'{v:g}' for v in dv_grid)}")
    print(f"  workers: {args.n_workers}")
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
                        args.threshold, dv_grid, args.seed_offset + idx,
                        bool(args.r_from_fit), args.r_scale,
                        args.edge_trim_days, args.p0_bwd_inflate))

    if args.max_objects and args.max_objects > 0 \
            and len(objects) > args.max_objects:
        sub_rng = np.random.default_rng(args.seed_offset)
        keep = np.sort(sub_rng.choice(len(objects),
                                      size=args.max_objects, replace=False))
        objects = [objects[i] for i in keep]
        print(f"Seeded subsample: {len(objects)} of eligible objects")

    print(f"Processing {len(objects)} objects...")
    if not objects:
        return

    t0 = time.time()
    all_results = []
    checkpoint_every = max(1, len(objects) // 20)
    with mp.Pool(args.n_workers, maxtasksperchild=10) as pool:
        iter_results = pool.imap_unordered(
            process_one_wrapper, objects,
            chunksize=max(1, len(objects) // (args.n_workers * 4)))
        for i, result in enumerate(iter_results, 1):
            all_results.append(result)
            if i % checkpoint_every == 0 or i == len(objects):
                pct = 100.0 * i / len(objects)
                elapsed = time.time() - t0
                eta = (len(objects) - i) * (elapsed / i) if i > 0 else 0
                print(f"  [{i:>4}/{len(objects)}] {pct:>5.1f}%  "
                      f"ETA {eta / 60:.1f}m")
                os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                with open(args.output + ".partial.json", "w") as f:
                    json.dump({"config": vars(args) | {
                        "dv_grid": [float(v) for v in dv_grid],
                        "n_eligible_objects": len(objects)},
                        "per_object_results": all_results}, f)

    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    print(f"\nFinished. Successful: {len(successful)}, "
          f"Failed: {len(all_results) - len(successful)}")
    if failed:
        reasons = Counter(str(r.get("error"))[:60] for r in failed)
        for reason, n in reasons.most_common(5):
            print(f"    {n:>5}  {reason}")

    # ---- per-dv aggregation with exclusion census -------------------
    efficiency = {}
    for dv in dv_grid:
        dv_f = float(dv)
        n_attempted = n_scored = n_det = 0
        excluded_reasons = Counter()
        peaks = []
        for r in successful:
            res = r["results"].get(dv_f)
            if res is None:
                continue
            n_attempted += 1
            if res.get("excluded") is not None:
                excluded_reasons[res["excluded"]] += 1
                continue
            n_scored += 1
            if res["detected"]:
                n_det += 1
            if res.get("peak_maha") is not None:
                peaks.append(res["peak_maha"])
        n_excluded = sum(excluded_reasons.values())
        excl_frac = (n_excluded / n_attempted) if n_attempted else 0.0
        if n_attempted and excl_frac > EXCLUSION_WARN_FRAC:
            print(f"  WARNING: dv={dv_f:g} censoring {excl_frac:.1%} "
                  f"(> {EXCLUSION_WARN_FRAC:.0%}) --- exclusions: "
                  f"{dict(excluded_reasons)}")
        efficiency[dv_f] = {
            "n_attempted": n_attempted,
            "n_scored": n_scored,
            "n_excluded": n_excluded,
            "exclusion_reasons": dict(excluded_reasons),
            "n_detected": n_det,
            "epsilon": (n_det / n_scored) if n_scored else None,
            "mean_peak": float(np.mean(peaks)) if peaks else None,
            "std_peak": float(np.std(peaks)) if peaks else None,
        }

    print(f"\n{'dv (m/s)':>12} {'attempt':>8} {'scored':>8} {'excl%':>7} "
          f"{'eps':>8} {'mean peak':>12} {'std peak':>12}")
    print("-" * 70)
    for dv in dv_grid:
        e = efficiency[float(dv)]
        excl_pct = (100.0 * e["n_excluded"] / e["n_attempted"]) \
            if e["n_attempted"] else float("nan")
        eps_str = f"{e['epsilon']:.4f}" if e["epsilon"] is not None else "N/A"
        mp_str = f"{e['mean_peak']:.3f}" if e["mean_peak"] is not None else "N/A"
        sp_str = f"{e['std_peak']:.3f}" if e["std_peak"] is not None else "N/A"
        print(f"{float(dv):12.5g} {e['n_attempted']:>8} {e['n_scored']:>8} "
              f"{excl_pct:>6.1f}% {eps_str:>8} {mp_str:>12} {sp_str:>12}")

    # ---- thresholds and the negligible-dv floor ---------------------
    dv50 = dv90 = None
    pts = [(float(dv), efficiency[float(dv)]["epsilon"])
           for dv in dv_grid if efficiency[float(dv)]["epsilon"] is not None]
    if len(pts) >= 2:
        eps_v = np.array([p[1] for p in pts])
        dvs_v = np.array([p[0] for p in pts])
        order = np.argsort(eps_v)
        eps_v, dvs_v = eps_v[order], dvs_v[order]
        if eps_v.min() <= 0.5 <= eps_v.max():
            dv50 = float(np.interp(0.5, eps_v, dvs_v))
        if eps_v.min() <= 0.9 <= eps_v.max():
            dv90 = float(np.interp(0.9, eps_v, dvs_v))

    dv_min_f = float(np.min(dv_grid))
    floor = efficiency[dv_min_f]
    negligible_dv_floor = {
        "dv": dv_min_f,
        "epsilon": floor["epsilon"],
        "n_scored": floor["n_scored"],
        "note": "in-window background trigger rate at negligible dv",
    }

    print()
    print(f"DV_50 = {dv50:.5f} m/s" if dv50 else "DV_50: not bracketed")
    print(f"DV_90 = {dv90:.5f} m/s" if dv90 else "DV_90: not bracketed")
    if floor["epsilon"] is not None:
        print(f"Negligible-dv floor: {floor['epsilon']:.4%} "
              f"at dv={dv_min_f:g} m/s ({floor['n_scored']} scored)")

    output_data = {
        "config": {
            "tau": args.tau, "q": args.q, "window_days": args.window_days,
            "threshold": args.threshold,
            "dv_grid": [float(v) for v in dv_grid],
            "r_from_fit": bool(args.r_from_fit), "r_scale": args.r_scale,
            "edge_trim_days": args.edge_trim_days,
            "p0_bwd_inflate": args.p0_bwd_inflate,
            "max_objects": args.max_objects,
            "n_eligible_objects": len(objects),
            "n_successful": len(successful),
            "seed_offset": args.seed_offset,
        },
        "efficiency": efficiency,
        "dv50": dv50, "dv90": dv90,
        "negligible_dv_floor": negligible_dv_floor,
        "per_object_results": all_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()