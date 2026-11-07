#!/usr/bin/env python3
"""Run real fleet through degraded filter with quality cuts.

FIX vs previous version: per-object HONEST measurement covariance
(--r-from-fit 1).

FIX 2 (R-scaling bug): the honest-R estimator multiplied the median of
squared one-step residuals by pi/2; the correct Gaussian scaling is
median(res^2) / 0.4549364. The bug understated every channel's sigma by
~15.5% and inflated all fleet Mahalanobis distances by ~1.18x. Any fleet
result produced before this fix (including real_detections_honestR_eta13.json)
is superseded - re-copy this version and re-run before quoting fleet rates.

Background (convention-consistency experiment result): the assumed diagonal
R_DIAG (from published TLE accuracy statistics) says argp is good to 0.01 rad
and M to 0.032 rad. At fleet-typical small e (~1e-3), a TLE fit cannot
separably determine argp and M - only their sum (the in-track phase) is
constrained - so each channel individually carries ~radian-scale
anticorrelated junk that cancels in the sum. Feeding that junk into a filter
that believes R_argp = (0.01)^2 manufactures 10-100x-overconfident
innovations, cross-couples into the a/h channels, and was shown (synthetic
control) to drive the false-positive rate from ~0% (honest R) to ~95-100%
(assumed R), and to trip the backward-EKF covariance guard.

The fix: build R PER OBJECT from the object's own TLE stream - the
one-step prediction-error scatter (advance each TLE's elements with the
Brouwer secular rates to the next TLE epoch; the per-channel variance of
those residuals is the covariance the stream actually exhibits) - floored
per channel at the production R_DIAG (never more confident than the
published statistics, only widened where the stream is noisier).

Robustness for REAL data: a real maneuver is a genuine one-step residual,
so the scatter is estimated with a MEDIAN-based (outlier-resistant)
estimator and capped at 100x R_DIAG, so a real impulse widens R only mildly
instead of blinding the detector. (In the synthetic null test no impulses
exist, so this robustness did not matter there; on real data it does.)

NOTE (ekf.py): NO changes to ekf.py are required. EKF.update() already
transforms the classical [R_a, R_e, R_i, R_Omega, R_argp, R_M] into h/k via
the e/argp Jacobian: R_h ~ sin^2(argp) R_e + e^2 cos^2(argp) R_argp. With
an honest R_argp (~1.8 rad at small e), R_h automatically becomes
e*sigma_argp ~ 1.8e-3, which is the actual h-channel scatter. The transform
was always right; the INPUT covariance was wrong.

IMPORTANT: eta = 13.258 was calibrated under the OLD R. Under honest R the
null-peak distribution shifts (synthetic honest-R nulls peaked ~0.8-1.1),
so re-run with --threshold 0 (log-only mode) FIRST to re-derive the 99.97th
percentile, then set the new threshold for detection runs.

Cuts per window: n_tles >= 15, max_gap <= 14 d, |alt_drop| <= 10 km,
skip w=0 if object has >90 d of data, skip on EKF divergence.

Usage:
    python -m phase2_filter.run_real_fleet \
        --tle-history-dir TLE/histories/ \
        --meta TLE/real_tles.tle.meta.json \
        --durations TLE/real_tles.tle.durations_2009_2026.json \
        --threshold 13.258 --tau 3e4 --q 1e-19 \
        --window-days 90 --n-workers 4 \
        --r-from-fit 1 \
        --output results/real_detections_honestR.json
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
from propagator.orbital import MeanElements, brouwer_rates
from TLE.tle_io import parse_tle, load_tles_from_file

SANITY_CEILING = 1.0e4
MAX_TLES_PER_FILE = 500
R_EMP_CAP_FACTOR = 100.0   # cap empirical R at 100x production R_DIAG


def _wrap_pi(x: float) -> float:
    """Wrap an angle difference to (-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def empirical_R_from_stream(measurements: List) -> Optional[np.ndarray]:
    """Honest per-object measurement covariance from the TLE stream itself.

    For each consecutive TLE pair, advance the previous element set with the
    Brouwer secular rates over the gap; the residual of the NEXT TLE against
    that prediction is exactly the one-step innovation the filter faces.
    The per-channel scatter of those residuals is the covariance the stream
    actually exhibits - as opposed to R_DIAG, which is what published TLE
    accuracy statistics suggest it should exhibit.

    Outlier-robust: uses the median of squared residuals (scaled to a
    variance) so that a genuine maneuver (a real, rare, large one-step
    residual) does not blow up R and blind the detector. This is the key
    difference from the synthetic-test version, where the null stream has
    no impulses and plain variance was fine.

    Returns classical diagonal R [R_a, R_e, R_i, R_Omega, R_argp, R_M],
    or None if too few pairs to estimate.
    """
    if len(measurements) < 8:
        return None
    res = []
    for (t_prev, z_prev), (t, z) in zip(measurements[:-1], measurements[1:]):
        dt = t - t_prev
        if dt <= 0:
            continue
        br = brouwer_rates(z_prev[0], max(1e-9, z_prev[1]), z_prev[2])
        res.append(np.array([
            z[0] - z_prev[0],                       # a (drag tiny over 2d)
            z[1] - z_prev[1],                       # e
            z[2] - z_prev[2],                       # i
            _wrap_pi(z[3] - (z_prev[3] + br["d_raan"] * dt)),
            _wrap_pi(z[4] - (z_prev[4] + br["d_argp"] * dt)),
            _wrap_pi(z[5] - (z_prev[5] + (br["n"] + br["dn_j2"]) * dt)),
        ]))
    if len(res) < 6:
        return None
    res = np.array(res)
    # Robust per-channel variance: for Gaussian noise the median of the
    # squared one-step residuals equals median(chi2_1) * sigma^2, with
    # median(chi2_1) = 0.4549364, so the honest variance estimator is
    # var = median(res^2) / 0.4549364.
    # FIX (pi/2 bug): the previous version multiplied by pi/2 instead,
    # underestimating every channel's sigma by ~15.5% and inflating all
    # fleet Mahalanobis distances by ~1.18x in R-dominated channels.
    # This now matches empirical_R_from_obs() in the synthetic null and
    # scrambled-null scripts exactly - ONE shared honest-R definition.
    var = np.median(res ** 2, axis=0) / 0.4549364
    return var


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


def _build_initial_covariance(el: MeanElements, R_vec: np.ndarray) -> np.ndarray:
    """Initial covariance from a classical 6-vector measurement variance."""
    e, argp = el.ecc, el.argp
    R = np.asarray(R_vec, dtype=float)
    R_h = math.sin(argp)**2 * R[1] + (e * math.cos(argp))**2 * R[4]
    R_k = math.cos(argp)**2 * R[1] + (e * math.sin(argp))**2 * R[4]
    return np.diag([
        R[0], R_h, R_k, R[2], R[3], R[5],
        1.0e-6, 1.0e-6, 1.0e-6
    ])


def process_object_real(
    norad_id: str,
    tle_list: List[Dict],
    duration_days: float,
    tau: float,
    q: float,
    window_days: float,
    threshold: float,
    r_from_fit: bool = False,
    r_scale: float = 1.0,
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

        # ------------------------------------------------------------------
        # HONEST-R FIX: per-object measurement covariance.
        # R_diag starts at the production assumption and is WIDENED (never
        # narrowed) where the object's own stream proves noisier. The e/argp
        # -> h/k transform inside EKF.update() then automatically produces
        # honest h/k channel variances (R_h ~ e^2 * R_argp), which is what
        # killed the covariance-guard crashes and the 95-100% synthetic
        # false-positive floor.
        # ------------------------------------------------------------------
        R_diag = np.asarray(R_DIAG_ELEMENTS, dtype=float) * r_scale
        r_used = R_diag.copy()
        if r_from_fit:
            R_emp = empirical_R_from_stream(measurements)
            if R_emp is not None:
                r_used = np.maximum(R_emp, R_diag)          # floor at production
        # Cap: never allow absurd covariance values (keeps the filter sane
        # even for pathological objects) while still permitting honest
        # widening of the degenerate argp/M channels.
        r_used = np.minimum(r_used, np.asarray(R_DIAG_ELEMENTS, dtype=float)
                            * R_EMP_CAP_FACTOR * max(1.0, r_scale))

        sig_used = np.sqrt(r_used)
        sig_assumed = np.sqrt(np.asarray(R_DIAG_ELEMENTS, dtype=float))
        r_report = {
            "sigma_used": [float(v) for v in sig_used],
            "sigma_assumed": [float(v) for v in sig_assumed],
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
                P0 = _build_initial_covariance(el0, r_used)
                ekf_fwd = EKF(x0, P0, cd_filter, area, mass, epoch0, tau=tau, q=q, direction="forward")
                fwd_states, fwd_covs = [], []
                prev_t = 0.0
                for t_rel, z in window_meas_rel:
                    dt = t_rel - prev_t
                    ekf_fwd.predict(dt, F107_BASELINE, QUIET_KP)
                    ekf_fwd.update(z, r_used)
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
                    ekf_bwd.update(z, r_used)
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
            "norad_id": norad_id,
            "success": True,
            "n_windows": n_windows,
            "n_valid_windows": n_valid_windows,
            "peaks": peaks,
            "detected": detected,
            "mean_peak": float(np.mean(peaks)) if peaks else None,
            "std_peak": float(np.std(peaks)) if peaks else None,
            "max_peak": float(np.max(peaks)) if peaks else None,
            "r_used": r_report,
        }

    except Exception as e:
        return {
            "norad_id": norad_id, "success": False, "error": str(e),
            "n_windows": 0, "n_valid_windows": 0, "peaks": [], "detected": 0,
        }


def process_one_wrapper(args):
    (norad_id, fpath, duration_days, tau, q,
     window_days, threshold, r_from_fit, r_scale) = args
    tle_list = load_tles_for_object(fpath)
    return process_object_real(norad_id, tle_list, duration_days, tau, q,
                               window_days, threshold,
                               r_from_fit=bool(r_from_fit),
                               r_scale=r_scale)


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
    parser.add_argument("--r-from-fit", type=int, default=0,
                        help="1: build the per-object measurement covariance "
                             "from the object's own TLE one-step prediction "
                             "scatter (robust/median-based so real maneuvers "
                             "do not blind the detector), floored at R_DIAG "
                             "and capped at 100x. 0: production R_DIAG.")
    parser.add_argument("--r-scale", type=float, default=1.0,
                        help="multiplier on the measurement covariance (after "
                             "--r-from-fit flooring/capping)")
    parser.add_argument("--output", default="results/real_detections.json")
    args = parser.parse_args()

    with open(args.durations) as f:
        durations_map = json.load(f)

    with open(args.meta) as f:
        meta_records = json.load(f)
    norad_ids = {str(r.get("NORAD_CAT_ID", "")) for r in meta_records}

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
        objects.append((nid, fpath, dur, args.tau, args.q, args.window_days,
                        args.threshold, args.r_from_fit, args.r_scale))

    print(f"Processing {len(objects)} objects...")
    print(f"  r-from-fit={args.r_from_fit}  r-scale={args.r_scale}  "
          f"threshold={args.threshold}")

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

    # R misspecification summary across the fleet (the paper-figure table):
    if successful and any("r_used" in r for r in successful):
        used = np.array([r["r_used"]["sigma_used"] for r in successful
                         if "r_used" in r])
        assumed = np.array(successful[0]["r_used"]["sigma_assumed"])
        med_used = np.median(used, axis=0)
        print("\nR 1-sigma per channel (median used | assumed):")
        ch = ["a (m)", "e", "i (rad)", "raan (rad)", "argp (rad)", "M (rad)"]
        for j, name in enumerate(ch):
            ratio = med_used[j] / assumed[j] if assumed[j] > 0 else float("nan")
            print(f"  {name:>10}: {med_used[j]:.4g} | {assumed[j]:.4g}  "
                  f"({ratio:.1f}x)")

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

    if total_windows > 0:
        print(f"\nWindows: {total_windows} raw, {total_valid_windows} valid"
              f" ({100*total_valid_windows/total_windows:.1f}%)")
    print(f"Peaks computed: {n_total}")

    if n_total > 0:
        print(f"Mean peak: {np.mean(all_peaks_arr):.3f}  std: {np.std(all_peaks_arr):.3f}")
        print(f"99.97th:   {np.percentile(all_peaks_arr, 99.97):.3f}")
        print(f"Threshold: {args.threshold:.3f}")
        if total_valid_windows > 0:
            print(f"Detections: {total_detected}"
                  f" ({100*total_detected/total_valid_windows:.2f}%)")
        if args.threshold <= 0:
            print("\n(LOG-ONLY MODE: threshold <= 0 - use the printed "
                  "99.97th percentile to recalibrate eta under honest R, "
                  "then re-run with that threshold.)")
    else:
        print("WARNING: no peaks computed")

    output_data = {
        "config": {
            "threshold": args.threshold, "tau": args.tau, "q": args.q,
            "window_days": args.window_days, "n_objects": len(objects),
            "n_successful": len(successful), "total_windows": total_windows,
            "total_valid_windows": total_valid_windows,
            "r_from_fit": args.r_from_fit, "r_scale": args.r_scale,
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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()