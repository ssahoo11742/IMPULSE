#!/usr/bin/env python3
"""Scrambled-TLE null test: background rate on real data with signal destroyed.

TWO NULL CONSTRUCTIONS (--scramble-mode):

  spline (default, V1-comparable):
      Fits a spline to each object's TLE history, permutes the residuals in
      time (row-wise, so cross-channel correlation is preserved), rebuilds
      fake TLEs. This is the V1 construction that produced the 19.46% floor.
      KNOWN DISTORTIONS (documented, kept for comparability):
        * The smoothing parameter s = 0.1*n is UNIT-DEPENDENT. For the a
          channel (metres) it is unattainably tight -> near-interpolating
          spline -> residuals smaller than the channel's real noise ->
          null TOO CLEAN in a (FITPACK "s too small" warnings).
        * The M channel is a ~96-min sawtooth sampled at multi-day spacing:
          the spline fits noise, residuals ~ Uniform(-pi, pi), RMS ~
          pi/sqrt(3) ~ 1.81 rad, which gets INJECTED as scrambled M noise -
          ~38 sigma against the honest sigma_M ~ 0.05 rad. raan/argp wrap
          within multi-year histories too (same pathology).
      => spline-mode floors are conservative (null harder than reality);
         DO NOT calibrate eta on spline mode.

  residual (V2 eta-calibration null):
      Permutes the ONE-STEP SECULAR-PREDICTION RESIDUALS - the exact
      quantities empirical_R_from_stream measures and the exact innovations
      the filter faces. Rebuilds the stream as a chain:
          z_scrambled(t_{k+1}) = secular_predict(z_scrambled(t_k), dt)
                                 + permuted_residual[k]
      with angles wrapped and e clamped. By construction the injected noise
      scale per channel EQUALS the empirical per-channel scatter, so the
      null is self-consistent with the honest per-object R; no smoothing
      parameter, no unit dependence, no wrap artifacts. The null hypothesis
      it embodies: "secular dynamics + the stream's own step noise, with
      temporal coherence destroyed" - which is exactly what an FP floor and
      an eta calibration require.

HONEST-R (--r-from-fit 1):

  R_emp = per-channel scatter of the one-step prediction residuals of the
  ORIGINAL stream (median-based /0.4549 so real maneuvers do not poison R),
  floored per channel at R_DIAG, capped at 100x. Applied to update(),
  backward update, and initial covariance. The fleet run showed the real
  systematic floor at eta=13.258 falls 51.55% -> 18.75% under this fix.

ETA RECALIBRATION (why residual mode exists):
  with ~15,800 null windows the 99.97th percentile has ~5 expected
  exceedances - statistically supported. The 99.97th percentile of the
  RESIDUAL-mode honest-R null IS the defensible eta_new. The synthetic null
  (1317 censored trials) and the spline null (M-wrap inflated) are both
  unsuitable for eta.

Recommended V2 runs (matched seeds):
  1. spline + honest R   -> conservative floor, V1-comparable  [may already be running]
  2. residual + honest R -> the FP floor + eta_new             [the calibration run]
  3. residual + prod R    -> optional: isolates the R-fix effect within the
                            residual construction (2x2 with V1's spline+prod)

Usage:
    python -m phase2_filter.scrambled_tle_null \
        --tle-history-dir TLE/histories/ \
        --meta TLE/real_tles.tle.meta.json \
        --durations TLE/real_tles.tle.durations_2009_2026.json \
        --threshold 13.258 --tau 3e4 --q 1e-19 \
        --window-days 90 --n-workers 4 \
        --r-from-fit 1 --scramble-mode residual \
        --output results/scrambled_null_residual_honestR.json
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
from propagator.orbital import MeanElements, brouwer_rates
from TLE.tle_io import parse_tle, load_tles_from_file

SANITY_CEILING = 1.0e4
MAX_TLES_PER_FILE = 500
R_EMP_CAP_FACTOR = 100.0   # cap empirical R at 100x production R_DIAG


def _wrap_pi(x: float) -> float:
    """Wrap an angle difference to (-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _wrap_2pi(x: float) -> float:
    """Wrap an absolute angle to [0, 2*pi)."""
    return x % (2.0 * math.pi)


def _one_step_residuals(measurements: List) -> Optional[np.ndarray]:
    """One-step secular-prediction residuals of a TLE stream.

    For each consecutive TLE pair, advance the previous element set with the
    Brouwer secular rates over the gap (a, e, i held - drag is tiny over a
    typical 2-day gap; the mean residual absorbs the secular drag decline);
    the residual of the NEXT TLE against that prediction (angles wrapped)
    is exactly the one-step innovation the filter faces. Same-epoch TLE
    pairs (dt <= 0) are included with prediction = previous value: their
    element disagreement is genuine measurement scatter.

    Returns an (n-1) x 6 array, or None if fewer than 3 measurements.
    Shared by empirical_R_from_stream (the honest R) and the residual-mode
    scramble (the eta-calibration null) so both use the SAME definition.
    """
    if len(measurements) < 3:
        return None
    res = []
    for (t_prev, z_prev), (t, z) in zip(measurements[:-1], measurements[1:]):
        dt = t - t_prev
        if dt > 0:
            br = brouwer_rates(z_prev[0], max(1e-9, z_prev[1]), z_prev[2])
            pred = np.array([
                z_prev[0],                                  # a
                z_prev[1],                                  # e
                z_prev[2],                                  # i
                z_prev[3] + br["d_raan"] * dt,
                z_prev[4] + br["d_argp"] * dt,
                z_prev[5] + (br["n"] + br["dn_j2"]) * dt,
            ])
        else:
            pred = np.asarray(z_prev, dtype=float)
        res.append(np.array([
            z[0] - pred[0],
            z[1] - pred[1],
            z[2] - pred[2],
            _wrap_pi(z[3] - pred[3]),
            _wrap_pi(z[4] - pred[4]),
            _wrap_pi(z[5] - pred[5]),
        ]))
    return np.array(res)


def empirical_R_from_stream(measurements: List) -> Optional[np.ndarray]:
    """Honest per-object measurement covariance from the ORIGINAL TLE stream.

    Robustness: a real maneuver is a genuine one-step residual, so the
    scatter is estimated from the MEDIAN of squared residuals. The median of
    a chi^2_1-distributed squared deviation is 0.4549 sigma^2, hence the
    scaling below (unbiased). This keeps a single real event from widening
    R and blinding the detector.

    Returns classical diagonal R [R_a, R_e, R_i, R_Omega, R_argp, R_M],
    or None if too few pairs.
    """
    res = _one_step_residuals(measurements)
    if res is None or len(res) < 6:
        return None
    var = np.median(res ** 2, axis=0) / 0.4549364
    return var


def scramble_stream_residuals(measurements: List,
                              rng: np.random.Generator) -> Optional[List]:
    """RESIDUAL-MODE null: permute one-step prediction residuals.

    Rebuilds the stream as a chain: each step advances the SCRAMBLED state
    with the secular rates, then adds a permuted original residual (whole
    rows, so cross-channel correlation is preserved; temporal coherence is
    destroyed). The injected noise scale per channel equals the empirical
    per-channel scatter by construction, making the null self-consistent
    with the honest per-object R.

    Returns the scrambled measurement list, or None if too few residuals
    (caller falls back to spline mode).
    """
    res = _one_step_residuals(measurements)
    if res is None or len(res) < 6:
        return None
    perm = rng.permutation(len(res))

    out: List = []
    z_prev = np.asarray(measurements[0][1], dtype=float).copy()
    out.append((measurements[0][0], z_prev))
    for i in range(1, len(measurements)):
        t_prev, t = measurements[i - 1][0], measurements[i][0]
        dt = t - t_prev
        z_prev = out[-1][1]
        if dt > 0:
            br = brouwer_rates(z_prev[0], max(1e-9, z_prev[1]), z_prev[2])
            pred = np.array([
                z_prev[0], z_prev[1], z_prev[2],
                z_prev[3] + br["d_raan"] * dt,
                z_prev[4] + br["d_argp"] * dt,
                z_prev[5] + (br["n"] + br["dn_j2"]) * dt,
            ])
        else:
            pred = z_prev.copy()
        z_new = pred + res[perm[i - 1]]
        z_new[1] = min(max(float(z_new[1]), 0.0), 0.99)
        for j in (3, 4, 5):
            z_new[j] = _wrap_2pi(float(z_new[j]))
        out.append((t, z_new))
    return out


def scramble_stream_spline(t_arr: np.ndarray,
                            z_arr: np.ndarray,
                            rng: np.random.Generator) -> List:
    """SPLINE-MODE null (V1 construction, kept for comparability)."""
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
    return [(t_arr[i], z_scrambled[i]) for i in range(len(t_arr))]


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


def scramble_object(
    norad_id: str,
    fpath: str,
    duration_days: float,
    tau: float,
    q: float,
    window_days: float,
    threshold: float,
    seed: int,
    r_from_fit: bool = False,
    r_scale: float = 1.0,
    scramble_mode: str = "spline",
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

        # ------------------------------------------------------------------
        # HONEST-R: per-object covariance from the ORIGINAL stream.
        # ------------------------------------------------------------------
        R_diag = np.asarray(R_DIAG_ELEMENTS, dtype=float) * r_scale
        r_used = R_diag.copy()
        if r_from_fit:
            R_emp = empirical_R_from_stream(measurements)
            if R_emp is not None:
                r_used = np.maximum(R_emp, R_diag)          # floor at production
            r_used = np.minimum(r_used, np.asarray(R_DIAG_ELEMENTS, dtype=float)
                                * R_EMP_CAP_FACTOR * max(1.0, r_scale))  # cap
        r_report = {
            "sigma_used": [float(v) for v in np.sqrt(r_used)],
            "sigma_assumed": [float(v) for v in np.sqrt(np.asarray(R_DIAG_ELEMENTS, dtype=float))],
        }

        # ------------------------------------------------------------------
        # Scramble: residual mode (eta-calibration null) or spline mode
        # (V1-comparable). Residual mode falls back to spline for objects
        # with too few TLE pairs (counted and reported).
        # ------------------------------------------------------------------
        fallback = False
        if scramble_mode == "residual":
            scrambled_meas = scramble_stream_residuals(measurements, rng)
            if scrambled_meas is None:
                fallback = True
                scrambled_meas = scramble_stream_spline(t_arr, z_arr, rng)
        else:
            scrambled_meas = scramble_stream_spline(t_arr, z_arr, rng)

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
            "norad_id": norad_id, "success": True,
            "n_valid_windows": n_valid_windows,
            "peaks": peaks, "detected": detected,
            "mean_peak": float(np.mean(peaks)) if peaks else None,
            "max_peak": float(np.max(peaks)) if peaks else None,
            "r_used": r_report,
            "scramble_fallback": fallback,
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
    parser.add_argument("--r-from-fit", type=int, default=0,
                        help="1: per-object honest measurement covariance from "
                             "the object's own (unscrambled) TLE one-step "
                             "prediction scatter, floored at R_DIAG and "
                             "capped at 100x. 0: production R_DIAG.")
    parser.add_argument("--r-scale", type=float, default=1.0,
                        help="multiplier on the measurement covariance (after "
                             "--r-from-fit flooring/capping)")
    parser.add_argument("--scramble-mode", choices=["spline", "residual"],
                        default="spline",
                        help="spline: V1 construction (conservative floor, "
                             "M-wrap inflated - do NOT calibrate eta on it). "
                             "residual: permute one-step secular-prediction "
                             "residuals - noise scale matches the honest R by "
                             "construction; this is the eta-calibration null.")
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
                        args.threshold, args.seed_offset + idx,
                        bool(args.r_from_fit), args.r_scale,
                        args.scramble_mode))

    print(f"Processing {len(objects)} objects...")
    print(f"  scramble-mode={args.scramble_mode}  r-from-fit={args.r_from_fit}  "
          f"r-scale={args.r_scale}  threshold={args.threshold}  "
          f"seed-offset={args.seed_offset}")

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

    if args.scramble_mode == "residual":
        n_fb = sum(1 for r in successful if r.get("scramble_fallback"))
        print(f"Residual-mode fallbacks to spline (too few TLE pairs): {n_fb}")

    # R misspecification table (median across fleet) - matches run_real_fleet.
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
        exp_beyond = n_total * 3e-4
        if args.scramble_mode == "residual":
            print(f"\nETA RECALIBRATION ({args.scramble_mode} mode): with "
                  f"{n_total} null windows the 99.97th percentile has "
                  f"~{exp_beyond:.1f} expected exceedances (need >=5). "
                  f"If supported, the printed 99.97th IS eta_new; the fleet "
                  f"rate at eta_new is readable from the fleet peaks JSON "
                  f"without re-running.")
        else:
            print(f"\nNOTE ({args.scramble_mode} mode): the spline null's "
                  f"M-channel is inflated by ~pi/sqrt(3) rad wrap artifacts "
                  f"and its a-channel is over-smoothed - use this floor as a "
                  f"CONSERVATIVE upper bound only; do NOT calibrate eta on "
                  f"it (use --scramble-mode residual).")
    else:
        print("WARNING: no peaks computed")

    output_data = {
        "config": {
            "threshold": args.threshold, "tau": args.tau, "q": args.q,
            "window_days": args.window_days, "n_objects": len(objects),
            "n_successful": len(successful), "total_valid_windows": total_valid_windows,
            "r_from_fit": args.r_from_fit, "r_scale": args.r_scale,
            "scramble_mode": args.scramble_mode,
            "seed_offset": args.seed_offset,
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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()