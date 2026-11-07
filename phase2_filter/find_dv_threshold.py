#!/usr/bin/env python3
"""End-to-end TLE-convention consistency test + SYNTHETIC ETA-NULL (V2).

TWO JOBS:
  (a) Convention-consistency experiment (V1 role, unchanged): how much of
      the real-data systematic false-positive floor is caused by the
      mismatch between the SGP4 mean-element convention of published TLEs
      and IMPULSE's own orbit-averaged secular propagation?
  (b) SYNTHETIC ETA-NULL (new): the estimator-limited "ideal-world"
      detection threshold eta_synth. The real nulls are unusable for eta
      (spline mode: a-channel over-smoothed -> anti-conservative; residual
      mode: provably non-exchangeable, biased hot). This script builds the
      one null that CAN carry a threshold: known truth under IMPULSE
      dynamics -> SGP4-consistent positions -> TLE-like fits -> noise
      MATCHED to the per-trial honest R -> the production filter. eta_synth
      = the 99.97th percentile of its peak distribution, with enough
      trials for >=5 expected exceedances. It certifies the ESTIMATOR ONLY;
      it says nothing about real-TLE error structure (that is the paper's
      point, and the fleet rate at eta_synth is reported as a rate-vs-eta
      curve, not as a false-positive rate).

Design per trial:
  Stage A  Generate a KNOWN physical trajectory with the IMPULSE truth
           propagator (sampled every --truth-dt-hours for fitting).
  Stage B  Sample ECI positions from that truth. Truth-position modes:
             twobody         - two-body ECI from the truth mean elements
             sgp4_consistent - SGP4 osculating positions computed from the
                               truth mean elements at each sample; the
                               ETA-NULL mode (full convention pipeline).
             sgp4_static     - MACHINERY CONTROL + PURE-CONVENTION FLOOR
                               (constant element set, positions from ONE
                               pure SGP4 propagation).
  Stage C  Fit SGP4-compatible TLE-like element sets to those positions by
           nonlinear least squares, refitting every --tle-cadence-days
           over a +-fit-span window. Optional B* drag term (--fit-bstar).
  Stage D  Read back fitted mean elements; add noise per --noise-mode:
             off     - no added noise (fitted elements as-is)
             full    - noise at the FULL R_used scale on top of the fitted
                       stream's intrinsic one-step scatter. LEGACY
                       DOUBLE-COUNT (the fitted stream already contains
                       scatter); kept only to reproduce old runs. This is
                       the mode that crashed the backward-EKF guard in
                       ~34% of noise-on trials.
             matched - DEFAULT. The fitted stream's own one-step residual
                       scatter is measured first (the same quantity the
                       honest R is built from); noise is then added at
                       sqrt(max(R_used - R_emp_raw, 0)) per channel, so
                       the stream's ACTUAL one-step scatter equals the
                       covariance the filter assumes. No double count.
  Stage E  Standard IMPULSE window filter with the production 7-day edge
           trim; peak Mahalanobis (a-channel) vs threshold.

FIXES vs the version before this patch (1-8 retained from the earlier fix
round, 9-12 are new):

  1. M/raan/argp initial guesses advanced to the TLE epoch with Brouwer
     secular rates (old code guessed frozen epoch-0 values; fits never
     converged, 2 km RMS).
  2. max_nfev default 2000 + x_scale='jac'; non-converged fits rejected.
  3. Detection statistic evaluated on the window interior only
     (--edge-trim-days, default 7), matching production; argmax location
     recorded per trial.
  4. --r-scale multiplies R_DIAG.
  5. sgp4_static trials draw a random orbit phase per trial.
  6. Element-level diagnostics per trial (fitted a vs truth a, fit
     failures, argmax edge fraction).
  7. A filter exception fails ONLY that trial and is recorded.
  8. --r-from-fit: honest measurement covariance from the fitted stream's
     own one-step prediction scatter, floored per channel at R_DIAG.
  9. HONEST R ALIGNED WITH THE FLEET ESTIMATOR: median of squared one-step
     residuals / 0.4549364 (median of chi^2_1), floored at R_DIAG, capped
     at 100x R_DIAG - byte-consistent with run_real_fleet.py /
     scrambled_tle_null.py, so null and fleet share one R definition.
 10. MATCHED NOISE (default --noise-mode matched): adds only the noise
     deficit, eliminating the legacy double count. Expected effects: the
     backward-EKF guard-crash rate drops from ~34% to ~0, and the null is
     no longer inflated by noise the filter was never told about.
 11. GEOMETRY JITTER (--jitter-geometry 1, default, evolving-truth modes):
     per-trial altitude 700-1000 km, inclination 50-105 deg, e in
     1e-5..1e-3, random phases - the null samples the fleet's regime
     instead of one 800 km / 98 deg orbit. --jitter-geometry 0 restores
     the old fixed geometry.
 12. ETA MACHINERY: percentile ladder, eta_synth with an expected-
     exceedance support check (>=5), a crash census with an explicit
     censoring-bias warning if the failure rate exceeds 2%, and progress/
     ETA reporting (17k-trial runs are overnight).
 13. MEMORY SAFETY (after a paging-file-exhaustion BSOD on Windows):
     (a) default worker count capped at 4 - cpu_count() spawned one full
         scipy process per core and exhausted the system commit limit;
         pass --n-workers explicitly to override, (b) maxtasksperchild=4
         so workers are recycled and any per-trial memory growth is
         bounded, (c) partial results are checkpointed every ~5% to
         <output>.partial.json so a mid-run crash loses nothing.
 14. GEOMETRY RECORDING: every trial records its drawn a0/e0/inclination
     (and phases for the static control), so extreme-peak trials from the
     jittered eta-null can be attributed to a specific orbital regime
     (e.g. the small-e fitting degeneracy) instead of just a seed.
 15. FIT-QUALITY GATE (production QC): individual TLE fits with position
     RMS above --fit-rms-max-m (default 5000 m, ~2x the SGP4
     representational floor) are rejected from the stream, mirroring
     operational TLE production QC. Diagnosis that motivated it: at some
     geometries the fit flips between local minima epoch-to-epoch,
     producing isolated km-scale element JUMPS; the median-based honest R
     is robust to outliers BY DESIGN (so real maneuvers cannot blind the
     detector) and therefore does not widen; a 10 m-confident filter then
     faces 100-sigma one-step innovations -> extreme peaks and a/h guard
     crashes. Real catalogs never contain such fits (QC), so rejecting
     them makes the synthetic stream structurally identical to a real
     catalog; trials that cannot keep >=10 clean fits fail honestly and
     are counted, not silently censored. Per-trial diagnostics record
     n_rejected_fits and max accepted fit RMS.

Recommended sequence:

  1. MACHINERY CONTROL + PURE-CONVENTION FLOOR (unchanged):
       python -m phase2_filter.convention_consistency_test \
          --truth-mode sgp4_static --n-trials 20 --noise-mode off \
          --fit-bstar 1 \
          --output results/convention_machinery_control.json
     EXPECT: fitted a matches truth a to ~meters, edge_peak_frac ~ 0.
     The trigger rate measures the convention mismatch alone.

  2. CONVENTION QUANTIFICATION (unchanged role; honest-R floor):
       python -m phase2_filter.convention_consistency_test \
          --truth-mode sgp4_consistent --n-trials 200 --noise-mode off \
          --fit-bstar 1 --r-from-fit 1 \
          --output results/convention_consistency.json

  3. TIME CHECK before the eta run (extrapolate to full size):
       python -m phase2_filter.convention_consistency_test \
          --mode null --truth-mode sgp4_consistent --n-trials 200 \
          --fit-bstar 1 --r-from-fit 1 --jitter-geometry 1 \
          --seed-offset 400000 \
          --output results/synthetic_null_eta_timing.json
     Full run time ~= (n_target/200) x this wall time. Check: fit
     failures ~0 and filter-crash rate ~0 under matched noise.

  4. ETA-SYNTH CALIBRATION (the paper's ideal-world threshold):
       python -m phase2_filter.convention_consistency_test \
          --mode null --truth-mode sgp4_consistent --n-trials 17000 \
          --fit-bstar 1 --r-from-fit 1 --jitter-geometry 1 \
          --percentile 99.97 --seed-offset 500000 \
          --output results/synthetic_null_eta_honestR.json
     17,000 trials gives ~5 expected exceedances at the 99.97th. If the
     time check shows 17k is infeasible, fall back to --n-trials 5000
     --percentile 99.9 (also ~5 expected exceedances, a more conservative
     0.1% floor) and label the threshold accordingly in the paper.
     The fleet rate at eta_synth is read OFFLINE from the fleet peaks
     JSON (the script prints the exact command); no fleet re-run needed.
     NOTE: eta_synth certifies the estimator only. Applying it to real
     data does NOT give the real-data false-positive rate - the paper
     reports the rate-vs-eta curve and the null bracket instead.

  5. Impulse sweep at eta_synth for the new Gate-4 efficiency curve and
     dv50 (feeds Table 3), after the fleet R-scaling fix has landed.

Requires: numpy, scipy, sgp4.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from sgp4.api import Satrec, WGS72

from constants.constants import MU, F107_BASELINE, QUIET_KP
from phase2_filter.config import R_DIAG_ELEMENTS
from phase2_filter.ekf import EKF
from phase2_filter.smoother import fraser_potter_smoother
from phase2_filter.test_statistics import compute_mahalanobis_distance
from phase2_filter.dynamics import hk_from_e_argp
from phase2_filter.validation import _propagate_truth, _default_el0
from propagator.orbital import MeanElements, mean_to_true_anomaly, brouwer_rates

SANITY_CEILING = 1.0e4
R_EMP_MEDIAN_OF_CHI2_1 = 0.4549364   # median of a chi^2_1 squared deviation
R_EMP_CAP_FACTOR = 100.0             # cap honest R at 100x production R_DIAG
FIT_DEBUG = ""  # last fit-failure reason, for diagnostics


def _wrap_pi(x: float) -> float:
    """Wrap an angle difference to (-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def empirical_R_from_obs(obs: List) -> Optional[np.ndarray]:
    """FIX #8/#9: honest measurement covariance from the fitted-TLE stream.

    For each consecutive pair of fitted TLEs, advance the previous element
    set with the IMPULSE secular (Brouwer) rates over the gap. The residual
    of the NEXT fit against that prediction is exactly the one-step
    innovation the filter faces.

    FIX #9: the scatter is estimated from the MEDIAN of squared residuals
    / 0.4549364 - byte-consistent with the fleet estimator in
    run_real_fleet.py / scrambled_tle_null.py (robust to isolated real
    events; a plain variance would let one outlier widen R). Synthetic
    cadence produces no same-epoch pairs, so every consecutive pair is a
    genuine one-step prediction. No flooring/capping here - the caller
    applies the production floor and cap so the same clip logic serves
    both the filter's R and the matched-noise deficit.

    Returns the RAW per-channel variance [R_a, R_e, R_i, R_Omega,
    R_argp, R_M] (classical elements), or None if too few pairs.
    """
    if len(obs) < 8:
        return None
    res = []
    for (t_prev, z_prev, _), (t, z, _) in zip(obs[:-1], obs[1:]):
        dt = t - t_prev
        if dt <= 0:
            continue
        br = brouwer_rates(z_prev[0], max(1e-9, z_prev[1]), z_prev[2])
        pred = np.array([
            z_prev[0],                          # a: drag neglected (tiny)
            z_prev[1],                          # e
            z_prev[2],                          # i
            _wrap_pi(z_prev[3] + br["d_raan"] * dt),
            _wrap_pi(z_prev[4] + br["d_argp"] * dt),
            _wrap_pi(z_prev[5] + (br["n"] + br["dn_j2"]) * dt),
        ])
        res.append(np.array([
            z[0] - pred[0],
            z[1] - pred[1],
            z[2] - pred[2],
            _wrap_pi(z[3] - pred[3]),
            _wrap_pi(z[4] - pred[4]),
            _wrap_pi(z[5] - pred[5]),
        ]))
    if len(res) < 6:
        return None
    res = np.array(res)
    return np.median(res ** 2, axis=0) / R_EMP_MEDIAN_OF_CHI2_1


# ---------------------------------------------------------------------------
# Stage B: truth mean elements -> ECI position
# ---------------------------------------------------------------------------

def mean_elements_to_rv(el: MeanElements) -> Tuple[np.ndarray, np.ndarray]:
    """Two-body ECI r/v from mean elements (no J2 short-periods)."""
    nu, _ = mean_to_true_anomaly(el.M, el.ecc)
    p = el.a * (1.0 - el.ecc ** 2)
    r = p / (1.0 + el.ecc * math.cos(nu))

    r_pqw = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    v_pqw = math.sqrt(MU / p) * np.array([-math.sin(nu), math.cos(nu) + el.ecc, 0.0])

    cO, sO = math.cos(el.raan), math.sin(el.raan)
    cw, sw = math.cos(el.argp), math.sin(el.argp)
    ci, si = math.cos(el.inc), math.sin(el.inc)

    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return R @ r_pqw, R @ v_pqw


def truth_position(
    el: MeanElements,
    epoch_jd: float,
    truth_mode: str,
    t_s: float = 0.0,
    static_sat: Optional[Satrec] = None,
) -> np.ndarray:
    """Return ECI position [m] for the selected truth mode."""

    params = np.array([
        el.a,
        max(0.0, el.ecc),
        el.inc,
        el.raan,
        el.argp,
        el.M,
        0.0,
    ], dtype=float)

    if truth_mode == "sgp4_static":
        if static_sat is None:
            raise ValueError("sgp4_static requires a reference Satrec")

        e_rc, r, _ = static_sat.sgp4_tsince(float(t_s / 60.0))

    elif truth_mode == "sgp4_consistent":
        # Reconstruct the SGP4 mean elements at this truth epoch.
        sat = _make_satrec(epoch_jd, params)
        e_rc, r, _ = sat.sgp4_tsince(0.0)

    else:
        r, _ = mean_elements_to_rv(el)
        return r

    if e_rc != 0:
        raise RuntimeError(
            f"SGP4 truth propagation failed with error code {e_rc}"
        )

    return np.asarray(r) * 1000.0


# ---------------------------------------------------------------------------
# Stage C: SGP4-compatible fitting (the documented fitting process)
# ---------------------------------------------------------------------------

def _make_satrec(epoch_jd: float, params: np.ndarray) -> Satrec:
    """Build an initialized SGP4 Satrec from [a, e, i, raan, argp, M, bstar]."""

    a, e, inc, raan, argp, M = params[:6]
    bstar = float(params[6]) if len(params) > 6 else 0.0

    # Convert semimajor axis from metres to kilometres.
    a_km = float(a) / 1000.0

    # WGS-72 gravitational parameter, km^3/s^2.
    mu_wgs72 = 398600.8

    # Mean motion in radians per minute.
    n0_rad_min = math.sqrt(mu_wgs72 / a_km**3) * 60.0

    # SGP4 epoch is measured from 1949-12-31.
    epoch_sgp4 = epoch_jd - 2433281.5

    sat = Satrec()
    sat.sgp4init(
        WGS72,
        'i',
        99999,
        epoch_sgp4,
        bstar,
        0.0,  # ndot
        0.0,  # nddot
        float(e),
        float(argp),
        float(inc),
        float(M % (2.0 * math.pi)),
        float(n0_rad_min),
        float(raan),
    )

    return sat


def fit_tle_epoch(epoch_jd: float,
                  tle_time_s: float,
                  fit_times_s: np.ndarray,
                  fit_pos: np.ndarray,
                  guess: np.ndarray,
                  fit_bstar: bool,
                  max_nfev: int = 2000,
                  ) -> Optional[Dict]:
    """Least-squares fit of SGP4 mean elements (+optional B*) to positions.

    sgp4 returns KM; truth positions are metres. Propagation times are
    minutes from the TLE EPOCH (not from the first fit sample).
    """
    n_par = 7 if fit_bstar else 6
    t_minutes = (fit_times_s - tle_time_s) / 60.0

    def residuals(params):
        sat = _make_satrec(epoch_jd, params)
        res = np.empty(3 * len(t_minutes))
        for k, tm in enumerate(t_minutes):
            e_rc, r, v = sat.sgp4_tsince(float(tm))
            if e_rc != 0:
                res[3 * k:3 * k + 3] = 1e6
            else:
                res[3 * k:3 * k + 3] = np.asarray(r) * 1000.0 - fit_pos[k]
        return res

    lo = np.array([6.6e6, 0.0, 0.0, -4 * math.pi, -4 * math.pi, -4 * math.pi,
                   0.0][:n_par])
    hi = np.array([9.0e6, 0.3, math.pi, 4 * math.pi, 4 * math.pi, 4 * math.pi,
                   1.0e-4][:n_par])
    g = np.clip(guess[:n_par], lo + 1e-9, hi - 1e-9)
    if fit_bstar and len(guess) >= 7:
        g[6] = min(guess[6], hi[6] - 1e-9)

    try:
        # FIX #2: higher cap + jac scaling (a and e differ by ~9 orders of
        # magnitude, so unscaled TRF steps are terrible).
        sol = least_squares(residuals, g, bounds=(lo, hi), method="trf",
                            x_scale="jac", max_nfev=max_nfev)
    except Exception as e:
        global FIT_DEBUG
        FIT_DEBUG = f"leastsq: {type(e).__name__}: {str(e)[:120]}"
        return None
    # FIX #2b: reject non-converged fits instead of emitting garbage.
    if sol.status <= 0:
        FIT_DEBUG = f"fit did not converge (status={sol.status})"
        return None
    if np.any(np.abs(sol.fun) > 1e5):
        FIT_DEBUG = f"fit rms too large: {np.sqrt(np.mean(sol.fun**2)):.3e} m"
        return None

    out = {
        "epoch_jd": epoch_jd,
        "a": float(sol.x[0]),
        "ecc": float(max(0.0, sol.x[1])),
        "inc": float(sol.x[2] % (2 * math.pi)),
        "raan": float(sol.x[3] % (2 * math.pi)),
        "argp": float(sol.x[4] % (2 * math.pi)),
        "M": float(sol.x[5] % (2 * math.pi)),
        "fit_rms_m": float(np.sqrt(np.mean(sol.fun ** 2))),
    }
    if fit_bstar:
        out["bstar"] = float(sol.x[6])
    return out


def _secular_advanced_guess(truth_hist: Dict,
                            t_ep: float,
                            i_near: int,
                            truth_mode: str) -> np.ndarray:
    """FIX #1: initial guess with M/raan/argp advanced to the TLE epoch.

    In sgp4_static mode the truth table is frozen at epoch 0 while SGP4
    itself advances M by ~15 rev/day, raan by ~1 deg/day, argp secularly.
    Guessing the frozen values leaves the fit started many radians away in
    phase, which is why the old control 'converged' at 2 km RMS.
    """
    if truth_mode == "sgp4_static":
        a0 = truth_hist["a"][0]
        e0 = truth_hist["e"][0]
        i0 = truth_hist["i"][0]
        br = brouwer_rates(a0, max(1e-9, e0), i0)
        dt = t_ep  # truth table is frozen at epoch 0
        return np.array([
            a0,
            max(1e-9, e0),
            i0,
            truth_hist["Omega"][0] + br["d_raan"] * dt,
            truth_hist["omega"][0] + br["d_argp"] * dt,
            truth_hist["M"][0] + (br["n"] + br["dn_j2"]) * dt,
            0.0,  # bstar guess
        ])

    # Evolving truth: values at i_near are already absolute; only advance
    # the small residual to the exact TLE epoch.
    a_near = truth_hist["a"][i_near]
    e_near = max(1e-9, truth_hist["e"][i_near])
    i_near_v = truth_hist["i"][i_near]
    br = brouwer_rates(a_near, e_near, i_near_v)
    dt = t_ep - truth_hist["t"][i_near]
    return np.array([
        a_near,
        e_near,
        i_near_v,
        truth_hist["Omega"][i_near] + br["d_raan"] * dt,
        truth_hist["omega"][i_near] + br["d_argp"] * dt,
        truth_hist["M"][i_near] + (br["n"] + br["dn_j2"]) * dt,
        0.0,
    ])


def generate_tle_like_observations(truth_hist: Dict,
                                   epoch_jd: float,
                                   tle_cadence_days: float,
                                   fit_span_days: float,
                                   truth_mode: str,
                                   fit_bstar: bool,
                                   max_nfev: int = 2000,
                                   fit_rms_max_m: Optional[float] = 5000.0,
                                   ) -> Tuple[List, Dict]:
    """Stage C+D: TLE-like observation stream + element-level diagnostics.

    FIX #15: fits whose position RMS exceeds fit_rms_max_m are REJECTED
    from the stream (production QC). Without the gate, fits that flip
    between local minima at some geometries inject isolated km-scale
    element jumps; the median-based honest R ignores them (robust by
    design), so a 10 m-confident filter faces 100-sigma innovations.

    Returns (observations, diagnostics). Diagnostics compare each fitted
    element set against the truth elements at the same epoch, so the SOURCE
    of a mismatch (secular drift vs periodic vs noise) is visible.
    """
    t_arr = np.array(truth_hist["t"])
    cad_s = tle_cadence_days * 86400.0
    half_fit = fit_span_days * 86400.0 / 2.0

    obs = []
    da_list, dt_list = [], []
    static_sat = None

    if truth_mode == "sgp4_static":
        el_ref = MeanElements(
            a=truth_hist["a"][0],
            ecc=truth_hist["e"][0],
            inc=truth_hist["i"][0],
            raan=truth_hist["Omega"][0],
            argp=truth_hist["omega"][0],
            M=truth_hist["M"][0],
        )

        static_params = np.array([
            el_ref.a,
            max(0.0, el_ref.ecc),
            el_ref.inc,
            el_ref.raan,
            el_ref.argp,
            el_ref.M,
            0.0,
        ])

        static_sat = _make_satrec(epoch_jd, static_params)

    n_tles = int(t_arr[-1] // cad_s)
    n_fit_attempts = 0
    n_fit_failures = 0
    n_rejected_fits = 0
    for k in range(n_tles + 1):
        t_ep = k * cad_s
        mask = (t_arr >= t_ep - half_fit) & (t_arr <= t_ep + half_fit)
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        fit_t = t_arr[idxs]

        fit_pos = np.zeros((len(idxs), 3))
        for j, idx in enumerate(idxs):
            el = MeanElements(a=truth_hist["a"][idx],
                              ecc=truth_hist["e"][idx],
                              inc=truth_hist["i"][idx],
                              raan=truth_hist["Omega"][idx],
                              argp=truth_hist["omega"][idx],
                              M=truth_hist["M"][idx])
            sample_jd = epoch_jd + truth_hist["t"][idx] / 86400.0
            fit_pos[j] = truth_position(
                el,
                sample_jd,
                truth_mode,
                t_s=truth_hist["t"][idx],
                static_sat=static_sat,
            )

        i_near = int(np.argmin(np.abs(t_arr - t_ep)))
        # FIX #1: secular-advanced initial guess (was frozen epoch-0 values).
        guess = _secular_advanced_guess(truth_hist, t_ep, i_near, truth_mode)

        tle_epoch_jd = epoch_jd + t_ep / 86400.0
        n_fit_attempts += 1
        fitted = fit_tle_epoch(tle_epoch_jd, t_ep, fit_t, fit_pos,
                               guess, fit_bstar, max_nfev=max_nfev)
        if fitted is None:
            n_fit_failures += 1
            continue
        # FIX #15: production-QC gate on the individual TLE fit.
        if fit_rms_max_m is not None and fitted["fit_rms_m"] > fit_rms_max_m:
            n_rejected_fits += 1
            continue

        # element-level diagnostics: fitted a vs truth a at this epoch
        da_list.append(fitted["a"] - truth_hist["a"][i_near])
        dt_list.append(t_ep / 86400.0)

        z = np.array([fitted["a"], fitted["ecc"], fitted["inc"],
                      fitted["raan"], fitted["argp"], fitted["M"]])
        obs.append((t_ep, z, fitted["fit_rms_m"]))

    diag = {"n_fit_attempts": n_fit_attempts,
            "n_fit_failures": n_fit_failures,
            "n_rejected_fits": n_rejected_fits}
    if obs:
        diag["max_fit_rms_m"] = float(max(o[2] for o in obs))
    if len(da_list) >= 3:
        da = np.array(da_list)
        dtt = np.array(dt_list)
        slope = float(np.polyfit(dtt, da, 1)[0])
        diag.update({"mean_da_m": float(np.mean(da)),
                     "std_da_m": float(np.std(da)),
                     "max_abs_da_m": float(np.max(np.abs(da))),
                     "da_slope_m_per_day": slope})
    return obs, diag


# ---------------------------------------------------------------------------
# Stage E/F: standard IMPULSE window filter on TLE-like observations
# ---------------------------------------------------------------------------

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
    R_h = math.sin(argp) ** 2 * R[1] + (e * math.cos(argp)) ** 2 * R[4]
    R_k = math.cos(argp) ** 2 * R[1] + (e * math.sin(argp)) ** 2 * R[4]
    return np.diag([R[0], R_h, R_k,
                    R[2], R[3], R[5],
                    1e-6, 1e-6, 1e-6])


def run_filter_window(measurements: List[Tuple[float, np.ndarray]],
                      epoch_jd: float,
                      tau: float, q: float,
                      edge_trim_s: float = 7.0 * 86400.0,
                      R_diag: Optional[np.ndarray] = None,
                      p0_bwd_inflate: float = 100.0,
                      cd_filter: float = 2.2) -> Optional[Dict]:
    """Fwd EKF + bwd EKF + smoother -> peak Mahalanobis (a-channel only).

    FIX #3: the peak is taken over the window INTERIOR only (edge trim
    matching the production pipeline). Returns the argmax location so edge
    effects remain visible.

    R_diag: classical 6-vector [R_a, R_e, R_i, R_Omega, R_argp, R_M] of
    measurement VARIANCES. Defaults to R_DIAG_ELEMENTS (the production
    assumption). Pass the output of empirical_R_from_obs() for
    honest-covariance runs.
    """
    if R_diag is None:
        R_diag = np.asarray(R_DIAG_ELEMENTS, dtype=float)
    else:
        R_diag = np.asarray(R_diag, dtype=float)

    el0 = MeanElements(a=measurements[0][1][0],
                       ecc=measurements[0][1][1],
                       inc=measurements[0][1][2],
                       raan=measurements[0][1][3],
                       argp=measurements[0][1][4],
                       M=measurements[0][1][5])
    area, mass = 1.0, 200.0

    x0 = _build_initial_state(el0)
    P0 = _build_initial_covariance(el0, R_diag)

    ekf_fwd = EKF(x0, P0, cd_filter, area, mass, epoch_jd,
                  tau=tau, q=q, direction="forward")
    fwd_states, fwd_covs = [], []
    prev_t = 0.0
    for t_rel, z in measurements:
        dt = t_rel - prev_t
        ekf_fwd.predict(dt, F107_BASELINE, QUIET_KP)
        ekf_fwd.update(z, R_diag)
        fwd_states.append(ekf_fwd.x.copy())
        fwd_covs.append(ekf_fwd.P.copy())
        prev_t = t_rel

    if len(fwd_states) < 5:
        return None

    x0_bwd = fwd_states[-1].copy()
    P0_bwd = fwd_covs[-1].copy() * p0_bwd_inflate
    ekf_bwd = EKF(x0_bwd, P0_bwd, cd_filter, area, mass, epoch_jd,
                  tau=tau, q=q, direction="backward")
    bwd_apriori_states, bwd_apriori_covs = [], []
    prev_t = measurements[-1][0]
    for t_rel, z in reversed(measurements):
        dt = t_rel - prev_t
        if abs(dt) > 1e-12:
            ekf_bwd.predict(dt, F107_BASELINE, QUIET_KP)
        bwd_apriori_states.append(ekf_bwd.x.copy())
        bwd_apriori_covs.append(ekf_bwd.P.copy())
        ekf_bwd.update(z, R_diag)
        prev_t = t_rel

    _, sm_covs = fraser_potter_smoother(
        fwd_states, fwd_covs,
        list(reversed(bwd_apriori_states)),
        list(reversed(bwd_apriori_covs)))

    # FIX #3: production-style edge trim on the detection domain.
    times = [t for t, _ in measurements]
    dur = times[-1]
    keep = [i for i, t in enumerate(times)
            if edge_trim_s <= t <= dur - edge_trim_s]
    if len(keep) < 5:
        keep = list(range(len(times)))

    bwd_states_rev = list(reversed(bwd_apriori_states))
    bwd_covs_rev = list(reversed(bwd_apriori_covs))
    fs = [fwd_states[i] for i in keep]
    fc = [fwd_covs[i] for i in keep]
    bs = [bwd_states_rev[i] for i in keep]
    bc = [bwd_covs_rev[i] for i in keep]
    sc = [sm_covs[i] for i in keep]

    mask = np.zeros(9, dtype=bool)
    mask[0] = True
    maha = np.asarray(compute_mahalanobis_distance(
        fs, fc, bs, bc, sc, state_mask=mask))

    i_pk = int(np.argmax(maha))
    peak = float(maha[i_pk])
    if not np.isfinite(peak) or abs(peak) > SANITY_CEILING:
        return None
    return {
        "peak": peak,
        "argmax_t_s": float(times[keep[i_pk]]),
        "argmax_frac": float(keep[i_pk] / max(1, len(times) - 1)),
        "n_epochs_used": len(keep),
    }


# ---------------------------------------------------------------------------
# One trial (module-level for multiprocessing)
# ---------------------------------------------------------------------------

def _run_one(args_tuple) -> Optional[Dict]:
    (seed, window_days, tau, q, threshold, tle_cadence_days,
     fit_span_days, noise_mode, mode, dv_mag,
     truth_mode, fit_bstar, truth_dt_hours,
     max_nfev, edge_trim_days, r_scale, p0_bwd_inflate,
     r_from_fit, jitter_geometry, fit_rms_max_m) = args_tuple

    rng = np.random.default_rng(seed)
    epoch_jd = 2460000.5
    duration_s = window_days * 86400.0
    el0 = _default_el0(alt_km=800.0, inc_deg=98.0)
    area, mass = 1.0, 200.0
    truth_dt_s = truth_dt_hours * 3600.0

    # FIX #11: per-trial geometry jitter for evolving-truth modes, so the
    # eta-null samples the fleet's regime (700-1000 km, incl 50-105 deg,
    # low e) instead of one fixed 800 km / 98 deg orbit. Phases are always
    # randomized; --jitter-geometry 0 restores the old fixed geometry.
    if jitter_geometry and truth_mode in ("sgp4_consistent", "twobody"):
        el0 = MeanElements(
            a=el0.a + (rng.uniform(700.0, 1000.0) - 800.0) * 1000.0,
            ecc=rng.uniform(1.0e-5, 1.0e-3),
            inc=math.radians(rng.uniform(50.0, 105.0)),
            raan=rng.uniform(0.0, 2.0 * math.pi),
            argp=rng.uniform(0.0, 2.0 * math.pi),
            M=rng.uniform(0.0, 2.0 * math.pi),
        )

    # FIX #5: randomize orbit phase per trial for the static control.
    if truth_mode == "sgp4_static":
        el0 = MeanElements(
            a=el0.a, ecc=el0.ecc, inc=el0.inc,
            raan=el0.raan + rng.uniform(0.0, 2.0 * math.pi),
            argp=el0.argp + rng.uniform(0.0, 2.0 * math.pi),
            M=el0.M + rng.uniform(0.0, 2.0 * math.pi),
        )

    # FIX #14: record the drawn geometry so extreme-peak trials can be
    # attributed to an orbital regime (a0/e0/inc), not just a seed.
    trial_geom = {"a0_m": float(el0.a), "e0": float(el0.ecc),
                  "inc0_deg": float(math.degrees(el0.inc))}

    strike_time = duration_s / 2.0
    try:
        if truth_mode == "sgp4_static":
            # MACHINERY CONTROL: constant element set; positions from SGP4.
            if mode != "null":
                return {"seed": seed, "ok": False,
                        "error": "sgp4_static supports null mode only"}
            n_pts = int(duration_s // truth_dt_s) + 1
            truth = {"t": [i * truth_dt_s for i in range(n_pts)],
                     "a": [el0.a] * n_pts, "e": [el0.ecc] * n_pts,
                     "i": [el0.inc] * n_pts,
                     "Omega": [el0.raan] * n_pts,
                     "omega": [el0.argp] * n_pts,
                     "M": [el0.M] * n_pts}
        elif mode == "null":
            truth, _, _ = _propagate_truth(
                el0, epoch_jd, duration_s, truth_dt_s, area, mass, rng=rng)
        else:
            truth, _, _ = _propagate_truth(
                el0, epoch_jd, duration_s, truth_dt_s, area, mass, rng=rng,
                strike_time=strike_time, strike_dv_mag=dv_mag)
    except Exception as e:
        return {"seed": seed, "ok": False, "error": f"truth: {e}"}

    # Stage C+D: pass truth through the SGP4 fitting convention
    try:
        obs, diag = generate_tle_like_observations(
            truth, epoch_jd, tle_cadence_days, fit_span_days,
            truth_mode, fit_bstar, max_nfev=max_nfev,
            fit_rms_max_m=fit_rms_max_m)
    except Exception as e:
        return {"seed": seed, "ok": False, "error": f"tle-fit: {e}"}
    if len(obs) < 10:
        return {"seed": seed, "ok": False,
                "error": f"too few obs ({len(obs)}); fits failed: "
                          f"{diag.get('n_fit_failures', '?')}/"
                          f"{diag.get('n_fit_attempts', '?')}; rejected by "
                          f"QC gate: {diag.get('n_rejected_fits', '?')}; "
                          f"last: {FIT_DEBUG or 'unknown'}",
                "geometry": trial_geom}

    fit_rms = float(np.mean([o[2] for o in obs]))

    # ------------------------------------------------------------------
    # FIX #8/#9: measurement covariance actually used by the filter.
    # Median-based (matches the fleet estimator), floored per channel at
    # production R_DIAG, capped at 100x - the exact policy of
    # run_real_fleet.py, so the synthetic null and the fleet share one R
    # definition.
    # ------------------------------------------------------------------
    R_diag_prod = np.asarray(R_DIAG_ELEMENTS, dtype=float)
    R_used = R_diag_prod * r_scale
    R_emp_raw = None
    # FIX #16: honest R in impulse mode too (previously null-only). The
    # dv sweep must share ONE R policy with the null that calibrated
    # eta_synth, else eps(dv) is on the assumed-R scale and dv50 at
    # eta_synth is meaningless. The median residual absorbs the single
    # strike step, exactly as the fleet's honest R absorbs real maneuvers.
    if r_from_fit:
        R_emp_raw = empirical_R_from_obs(obs)
        if R_emp_raw is not None:
            R_used = np.minimum(
                np.maximum(R_emp_raw, R_diag_prod),
                R_diag_prod * R_EMP_CAP_FACTOR
            ) * r_scale

    # ------------------------------------------------------------------
    # FIX #10: noise construction.
    #   off     - fitted elements as-is (intrinsic fit scatter only)
    #   full    - LEGACY: noise at the full R_used scale on top of the
    #             fitted stream's intrinsic scatter. Double-counted noise
    #             (this is what crashed the backward-EKF guard in ~34% of
    #             old noise-on trials and inflated the old null). Kept
    #             only to reproduce old runs.
    #   matched - DEFAULT: measure the fitted stream's intrinsic one-step
    #             scatter (R_emp_raw, the same quantity the honest R is
    #             built from), then add only the deficit
    #             sqrt(max(R_used - R_emp_raw, 0)) per channel, so the
    #             stream's ACTUAL one-step scatter equals the covariance
    #             the filter assumes. This is the noise-matched
    #             ideal-world construction the eta-null requires.
    # ------------------------------------------------------------------
    if noise_mode == "off":
        measurements = [(t, z) for t, z, _ in obs]
    else:
        if R_emp_raw is None:
            R_emp_raw = empirical_R_from_obs(obs)
        if R_emp_raw is None:
            R_emp_raw = np.zeros(6)
        if noise_mode == "full":
            noise_var = R_used.copy()
        elif noise_mode == "matched":
            noise_var = np.maximum(R_used - R_emp_raw, 0.0)
        else:
            return {"seed": seed, "ok": False,
                    "error": f"unknown noise_mode: {noise_mode}"}
        noise_scale = np.sqrt(noise_var)
        measurements = []
        for t, z, _ in obs:
            zn = z.copy()
            zn += rng.normal(scale=noise_scale)
            measurements.append((t, zn))

    # FIX #7: a filter crash (e.g. the backward-EKF covariance guard in
    # ekf.py) fails only this trial instead of killing the whole pool.
    try:
        res = run_filter_window(measurements, epoch_jd, tau, q,
                                edge_trim_s=edge_trim_days * 86400.0,
                                R_diag=R_used,
                                p0_bwd_inflate=p0_bwd_inflate)
    except Exception as e:
        return {"seed": seed, "ok": False,
                "error": f"filter crashed: {type(e).__name__}: {str(e)[:160]}",
                "geometry": trial_geom}
    if res is None:
        return {"seed": seed, "ok": False, "error": "filter diverged",
                "geometry": trial_geom}

    return {
        "seed": seed, "ok": True,
        "geometry": trial_geom,
        "peak_maha": res["peak"],
        "maha_argmax_t_s": res["argmax_t_s"],
        "maha_argmax_frac": res["argmax_frac"],
        "fit_rms_m": fit_rms,
        "n_obs": len(measurements),
        "r_used": [float(v) for v in R_used],
        "r_emp_raw": [float(v) for v in R_emp_raw] if R_emp_raw is not None
                    else None,
        "noise_mode": noise_mode,
        "detected": bool(res["peak"] > threshold),
        "mode": mode,
        "dv": dv_mag,
        **diag,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["null", "impulse"], default="null")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--window-days", type=float, default=90.0)
    p.add_argument("--tau", type=float, default=3e4)
    p.add_argument("--q", type=float, default=1e-19)
    p.add_argument("--threshold", type=float, default=13.258,
                   help="legacy threshold for the 'detected' flag and the "
                        "trigger-rate line; the eta-null output is the "
                        "percentile ladder, not this flag")
    p.add_argument("--truth-mode",
                   choices=["twobody", "sgp4_consistent", "sgp4_static"],
                   default="twobody",
                   help="twobody: positions from mean elements (rough). "
                        "sgp4_consistent: SGP4 positions from evolving truth "
                        "elements - THE ETA-NULL MODE. "
                        "sgp4_static: MACHINERY CONTROL - pure SGP4 truth.")
    p.add_argument("--fit-bstar", type=int, default=0,
                   help="1: fit the SGP4 B* drag term like real TLE "
                        "production; 0: bstar fixed at 0 (old behavior)")
    p.add_argument("--truth-dt-hours", type=float, default=6.0,
                   help="truth sampling interval used for TLE fitting")
    p.add_argument("--tle-cadence-days", type=float, default=2.0)
    p.add_argument("--fit-span-days", type=float, default=4.0)
    p.add_argument("--noise-mode", choices=["off", "full", "matched"],
                   default=None,
                   help="off: no added noise. full: LEGACY double-counted "
                        "noise (reproduces old runs only). matched (default): "
                        "add only the deficit so the stream's one-step "
                        "scatter equals R_used - the eta-null construction.")
    p.add_argument("--measurement-noise", type=int, default=None,
                   help="(legacy) 0 -> --noise-mode off, 1 -> full. "
                        "Prefer --noise-mode.")
    p.add_argument("--percentile", type=float, default=99.97,
                   help="target percentile for eta_synth. 99.97th needs "
                        "~17,000 trials for >=5 expected exceedances; "
                        "99.9th needs ~5,000 (fallback if runtime binds)")
    p.add_argument("--jitter-geometry", type=int, default=1,
                   help="1 (default): per-trial altitude 700-1000 km, "
                        "inclination 50-105 deg, e in 1e-5..1e-3, random "
                        "phases (evolving-truth modes) - the null samples "
                        "the fleet's regime. 0: old fixed 800 km / 98 deg "
                        "geometry. No effect on sgp4_static (its own "
                        "phase randomization is kept).")
    p.add_argument("--fit-rms-max-m", type=float, default=5000.0,
                   help="FIX #15 production-QC gate: reject individual TLE "
                        "fits with position RMS above this (default 5000 m, "
                        "~2x the SGP4 representational floor). Pass a huge "
                        "value to disable. Mirrors operational TLE "
                        "production, which never publishes garbage fits.")
    p.add_argument("--dv-grid", type=str, default="0.005,0.01,0.02,0.04,0.08")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--seed-offset", type=int, default=900000)
    p.add_argument("--max-nfev", type=int, default=2000,
                   help="cap on least-squares function evaluations per TLE "
                        "fit (was 300, which never converged)")
    p.add_argument("--edge-trim-days", type=float, default=7.0,
                   help="production-style edge trim on the detection "
                        "statistic domain (paper Sec. 2.1)")
    p.add_argument("--r-scale", type=float, default=1.0,
                   help="multiplier on the measurement covariance R (R_DIAG "
                        "by default, or the empirical R if --r-from-fit 1); "
                        "applies to updates, initial covariance, and noise "
                        "generation)")
    p.add_argument("--r-from-fit", type=int, default=0,
                   help="1: build R from the empirical one-step "
                        "prediction-error scatter of the fitted-TLE stream, "
                        "floored per channel at the production R_DIAG and "
                        "capped at 100x (the fleet's exact R policy). "
                        "Null mode only. 0: use R_DIAG.")
    p.add_argument("--p0-bwd-inflate", type=float, default=100.0,
                   help="backward-filter initial covariance inflation "
                        "(100 = production value)")
    p.add_argument("--output", default="results/convention_consistency.json")
    args = p.parse_args()

    # Legacy --measurement-noise compatibility: 0 -> off, 1 -> full.
    if args.noise_mode is None:
        args.noise_mode = {0: "off", 1: "full"}.get(
            args.measurement_noise, "matched")

    # FIX #13a: default worker count capped at 4. With cpu_count() workers
    # (each a full python+numpy+scipy process) a Windows machine with a
    # small or non-growable paging file can exhaust the system commit
    # limit and BSOD. Pass --n-workers explicitly to override.
    n_workers = args.n_workers or max(1, min(4, mp.cpu_count()))

    common_tail = (args.truth_mode, args.fit_bstar, args.truth_dt_hours,
                   args.max_nfev, args.edge_trim_days, args.r_scale,
                   args.p0_bwd_inflate, args.r_from_fit,
                   args.jitter_geometry, args.fit_rms_max_m)

    tasks = []
    if args.mode == "null":
        for i in range(args.n_trials):
            tasks.append((args.seed_offset + i, args.window_days, args.tau,
                           args.q, args.threshold, args.tle_cadence_days,
                           args.fit_span_days, args.noise_mode,
                           "null", 0.0) + common_tail)
    else:
        dvs = [float(d) for d in args.dv_grid.split(",")]
        for dv in dvs:
            for i in range(args.n_trials):
                tasks.append((args.seed_offset + i, args.window_days, args.tau,
                               args.q, args.threshold, args.tle_cadence_days,
                               args.fit_span_days, args.noise_mode,
                               "impulse", dv) + common_tail)

    print(f"TLE-convention consistency test / eta-null: mode={args.mode}, "
          f"{len(tasks)} trials on {n_workers} workers")
    print(f"  truth-mode={args.truth_mode}  fit-bstar={args.fit_bstar}  "
          f"truth-dt={args.truth_dt_hours}h  "
          f"noise-mode={args.noise_mode}")
    print(f"  window={args.window_days:.0f}d  cadence={args.tle_cadence_days}d  "
          f"fit-span={args.fit_span_days}d  max-nfev={args.max_nfev}")
    print(f"  edge-trim={args.edge_trim_days}d  r-scale={args.r_scale}  "
          f"r-from-fit={args.r_from_fit}  "
          f"p0-bwd-inflate={args.p0_bwd_inflate}")
    print(f"  jitter-geometry={args.jitter_geometry}  "
          f"percentile-target={args.percentile}  "
          f"tau={args.tau:.2e}  q={args.q:.2e}  threshold={args.threshold}")

    t0 = time.time()
    results = []
    n_tasks = len(tasks)
    # FIX #13b: recycle workers every 4 tasks so any per-trial memory
    # growth (leaks, pathological jittered geometry) stays bounded.
    with mp.Pool(n_workers, maxtasksperchild=4) as pool:
        chunk = max(1, n_tasks // (n_workers * 8))
        done = 0
        part_path = args.output + ".partial.json"
        for r in pool.imap_unordered(_run_one, tasks, chunksize=chunk):
            results.append(r)
            done += 1
            if done % max(1, n_tasks // 20) == 0 or done == n_tasks:
                elapsed = time.time() - t0
                eta_min = (n_tasks - done) * (elapsed / done) / 60.0
                print(f"  [{done:>6}/{n_tasks}] {100.0*done/n_tasks:5.1f}%  "
                      f"ETA {eta_min:.1f}m")
                # FIX #13c: checkpoint so a mid-run crash loses nothing.
                try:
                    os.makedirs(os.path.dirname(part_path) or ".",
                                exist_ok=True)
                    with open(part_path, "w") as f:
                        json.dump({"config": vars(args), "trials": results},
                                  f)
                except Exception:
                    pass

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    fail_rate = len(failed) / max(1, len(results))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"OK: {len(ok)}  Failed: {len(failed)} ({100*fail_rate:.2f}%)")

    if failed:
        from collections import Counter
        for err, n in Counter(r["error"][:100] for r in failed).most_common(5):
            print(f"  {n}x {err}")

    def _agg(key):
        vals = [r[key] for r in ok if key in r]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    summary = {}
    if args.mode == "null":
        peaks = np.array([r["peak_maha"] for r in ok])
        if len(peaks):
            rate = float(np.mean(peaks > args.threshold))
            fracs = np.array([r.get("maha_argmax_frac", 0.5) for r in ok])
            edge_frac = float(np.mean((fracs <= 0.1) | (fracs >= 0.9)))
            summary = {
                "n_ok": len(ok),
                "n_failed": len(failed),
                "failure_rate": fail_rate,
                "noise_mode": args.noise_mode,
                "jitter_geometry": bool(args.jitter_geometry),
                "percentile_target": args.percentile,
                "mean_peak": float(np.mean(peaks)),
                "std_peak": float(np.std(peaks)),
                "trigger_rate": rate,
                "n_triggers": int(np.sum(peaks > args.threshold)),
                "mean_fit_rms_m": _agg("fit_rms_m")[0],
                "mean_da_m": _agg("mean_da_m")[0],
                "std_da_m_within_trial": _agg("std_da_m")[0],
                "max_abs_da_m": _agg("max_abs_da_m")[0],
                "da_slope_m_per_day": _agg("da_slope_m_per_day")[0],
                "mean_fit_failures": _agg("n_fit_failures")[0],
                "mean_rejected_fits": _agg("n_rejected_fits")[0],
                "max_accepted_fit_rms_m": _agg("max_fit_rms_m")[0],
                "edge_peak_frac": edge_frac,
                "r_used_mean": [float(v) for v in np.mean(
                    [r["r_used"] for r in ok], axis=0)],
            }

            # ------------------------------------------------------------
            # FIX #12: eta machinery - percentile ladder + support check.
            # ------------------------------------------------------------
            pct_ladder = [50, 90, 99, 99.5, 99.9, 99.95, 99.97, 99.99]
            summary["percentiles"] = {
                str(pc): float(np.percentile(peaks, pc)) for pc in pct_ladder
            }
            summary["max_peak"] = float(np.max(peaks))
            eta_synth = float(np.percentile(peaks, args.percentile))
            exp_beyond = len(peaks) * (1.0 - args.percentile / 100.0)
            summary["eta_synth"] = eta_synth
            summary["eta_expected_exceedances"] = float(exp_beyond)
            summary["eta_supported"] = bool(exp_beyond >= 5.0)

            print(f"\nBaseline (no-impulse) trigger rate at legacy "
                  f"eta={args.threshold}: {rate*100:.3f}%  "
                  f"({summary['n_triggers']}/{len(ok)})")
            print(f"  mean peak Mahalanobis: {summary['mean_peak']:.3f} "
                  f"+/- {summary['std_peak']:.3f}")
            print(f"  peaks in first/last 10% of window: "
                  f"{edge_frac*100:.1f}%  (edge-transient indicator)")
            print(f"  mean TLE-fit position RMS: {summary['mean_fit_rms_m']:.1f} m")
            r_used_arr = np.array([r["r_used"] for r in ok])
            sig = np.sqrt(r_used_arr.mean(axis=0))
            sig0 = np.sqrt(np.asarray(R_DIAG_ELEMENTS, dtype=float))
            print(f"  R 1-sigma per channel (used | assumed): "
                  f"a={sig[0]:.4g} | {sig0[0]:.4g} m   "
                  f"e={sig[1]:.4g} | {sig0[1]:.4g}   "
                  f"i={sig[2]:.4g} | {sig0[2]:.4g} rad")
            print(f"    raan={sig[3]:.4g} | {sig0[3]:.4g} rad   "
                  f"argp={sig[4]:.4g} | {sig0[4]:.4g} rad   "
                  f"M={sig[5]:.4g} | {sig0[5]:.4g} rad")
            if summary["max_abs_da_m"] is not None:
                print(f"  element diagnostics: mean(a_fit-a_true) = "
                      f"{summary['mean_da_m']:.2f} m, "
                      f"slope = {summary['da_slope_m_per_day']:.3f} m/day, "
                      f"per-trial std = {summary['std_da_m_within_trial']:.2f} m, "
                      f"max |da| = {summary['max_abs_da_m']:.2f} m")

            print(f"\nETA-SYNTH CALIBRATION (noise-mode={args.noise_mode}, "
                  f"jitter-geometry={args.jitter_geometry}):")
            print(f"  peak-percentile ladder (n={len(peaks)} OK trials):")
            for pc in pct_ladder:
                print(f"    p{pc:<6} {summary['percentiles'][str(pc)]:10.3f}")
            print(f"    max     {summary['max_peak']:10.3f}")
            supported = summary["eta_supported"]
            verdict = ("SUPPORTED (>=5)" if supported else
                       "NOT SUPPORTED - run more trials or lower --percentile")
            print(f"\n  eta_synth = {eta_synth:.4f}  "
                  f"({args.percentile}th percentile of {len(peaks)} trials)")
            print(f"  expected exceedances at this percentile: "
                  f"~{exp_beyond:.1f}  ->  {verdict}")
            print(f"  failed trials: {len(failed)}/{len(results)} "
                  f"({100*fail_rate:.2f}%) are CENSORED from the percentile")
            if fail_rate > 0.02:
                print("  WARNING: failure rate > 2%. If failures correlate "
                      "with extreme peaks (guard crashes fire on the "
                      "noisiest windows), eta_synth is biased LOW and the "
                      "fleet rate at it would be inflated. Fix the failure "
                      "mode before quoting eta_synth.")
            print("\n  Fleet rate at eta_synth (offline, no fleet re-run):")
            readout = (
                "python -c \"import json,numpy as np; "
                "d=json.load(open('results/real_detections_honestR_eta13.json')); "
                "p=np.concatenate([r['peaks'] for r in d['per_object'] "
                "if r.get('success') and r.get('peaks')]); "
                f"print(100*np.mean(p>{eta_synth:.4f}))\""
            )
            print(f"    {readout}")
            print("  (use the corrected fleet JSON once the R-scaling fix "
                  "re-run lands)")

            if args.truth_mode == "sgp4_static":
                print("\n  MACHINERY CHECKS (the actual control gates):")
                if summary.get("max_abs_da_m") is not None \
                        and summary["max_abs_da_m"] < 100.0:
                    print("    fitted a vs truth a: OK "
                          f"(mean {summary['mean_da_m']:.2f} m, "
                          f"max |da| {summary['max_abs_da_m']:.2f} m) "
                          "- fitting machinery is working.")
                else:
                    print("    fitted a vs truth a is off by >100 m - "
                          "fitting machinery is buggy, do not interpret "
                          "the trigger rate.")
                print(f"    peaks at window edges: {summary['edge_peak_frac']*100:.1f}%")
                print("    The trigger rate itself is the PURE-CONVENTION "
                      "floor (SGP4 element convention vs the filter's "
                      "secular convention, zero noise, zero dynamics "
                      "mismatch). It is NOT a fit bug by itself.")
                print("    Sanity-check it against R misspecification with "
                      "--r-scale 10 and --r-scale 100 before quoting it.")
            elif args.truth_mode == "sgp4_consistent":
                print("\n  THIS DISTRIBUTION IS THE CONVENTION+DYNAMICS "
                      "IDEAL-WORLD NULL. eta_synth above is the paper's "
                      "estimator-limited threshold; the real-data rate at "
                      "it is reported as a rate-vs-eta curve, NOT as a "
                      "false-positive rate.")
                if rate > 0.05:
                    print("  (Non-zero triggering by design at the legacy "
                          "threshold: the constant a offset is the "
                          "mean-element convention difference; check the "
                          "percentile ladder rather than this rate.)")
                    if edge_frac > 0.5:
                        print("  NOTE: most peaks are at window edges - "
                              "increase --edge-trim-days and re-run before "
                              "trusting this number.")
                    if summary.get("mean_fit_rms_m", 0) > 1000:
                        print("  NOTE: fit RMS > 1 km - the fits may still "
                              "be poorly converged; check --max-nfev and the "
                              "fit-failure count before trusting this.")
    else:
        dvs = [float(d) for d in args.dv_grid.split(",")]
        print(f"\n{'dv (m/s)':>10} {'N':>6} {'eps':>8} {'mean peak':>10} "
              f"{'fit RMS (m)':>12}")
        print("-" * 52)
        eff = {}
        for dv in dvs:
            sub = [r for r in ok if abs(r["dv"] - dv) < 1e-12]
            if not sub:
                print(f"{dv:10.4f} {0:>6} {'N/A':>8} {'N/A':>10} {'N/A':>12}")
                continue
            pk = np.array([r["peak_maha"] for r in sub])
            eps = float(np.mean(pk > args.threshold))
            eff[dv] = {"n": len(sub), "epsilon": eps,
                       "mean_peak": float(np.mean(pk)),
                       "mean_fit_rms_m": float(np.mean([r["fit_rms_m"] for r in sub]))}
            print(f"{dv:10.4f} {len(sub):>6} {eps:>8.3f} {np.mean(pk):>10.3f} "
                  f"{eff[dv]['mean_fit_rms_m']:>12.1f}")
        summary = {"efficiency": eff,
                   "mean_fit_rms_m": _agg("fit_rms_m")[0],
                   "noise_mode": args.noise_mode,
                   "jitter_geometry": bool(args.jitter_geometry)}
        print("\nCompare eps(dv) against the null baseline above; quote the "
              "new dv50 at eta_synth (not at the legacy threshold) in the "
              "paper. With --jitter-geometry 1 the curve averages over the "
              "fleet's altitude/inclination band.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "trials": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()