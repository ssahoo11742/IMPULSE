#!/usr/bin/env python3
"""
Plot full Mahalanobis-distance time series for one object
(days since epoch style, with threshold line).

Usage example:
    python plot_md_one_object.py \
        --tle 119.tle \
        --threshold 13.258 \
        --tau 3e4 --q 1e-19 \
        --window-days 90 \
        --zoom-start 960 --zoom-end 1040 \
        --output md_118.png
"""

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from constants.constants import MU, R_EARTH, F107_BASELINE, QUIET_KP
from phase2_filter.config import R_DIAG_ELEMENTS
from phase2_filter.ekf import EKF
from phase2_filter.smoother import fraser_potter_smoother
from phase2_filter.test_statistics import compute_mahalanobis_distance
from phase2_filter.dynamics import hk_from_e_argp
from propagator.orbital import MeanElements
from TLE.tle_io import parse_tle, load_tles_from_file

SANITY_CEILING = 1.0e4
MAX_TLES_PER_FILE = 2000   # raise if your history is longer


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
        print(f"Warning: could not read {fpath}: {e}")
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


def process_object_md_series(
    tle_list: List[Dict],
    tau: float,
    q: float,
    window_days: float,
    threshold: float,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Returns three parallel lists:
        days_since_epoch, maha_values, window_id
    Only valid windows that pass the quality cuts are included.
    """
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
        print("Too few measurements")
        return [], [], []

    window_s = window_days * 86400.0
    max_t = measurements[-1][0]
    n_windows = int(max_t // window_s) + 1
    total_data_days = max_t / 86400.0

    all_days = []
    all_maha = []
    all_win  = []

    cd_filter = 2.2
    area, mass = 1.0, 200.0
    mask = np.zeros(9, dtype=bool)
    mask[0] = True          # only semi-major axis component (same as original)

    for w in range(n_windows):
        w_start = w * window_s
        w_end   = (w + 1) * window_s
        window_meas = [(t, z) for t, z in measurements if w_start <= t < w_end]

        if len(window_meas) < 15:
            continue

        times = [t for t, _ in window_meas]
        gaps = [(times[i+1] - times[i]) / 86400.0 for i in range(len(times) - 1)]
        if gaps and max(gaps) > 14.0:
            continue

        a_start = float(window_meas[0][1][0])
        a_end   = float(window_meas[-1][1][0])
        if abs((a_start - a_end) / 1000.0) > 10.0:
            continue

        if w == 0 and total_data_days > 90.0:
            continue

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

            if not np.all(np.isfinite(maha)) or np.max(np.abs(maha)) > SANITY_CEILING:
                continue

            # relative times inside the window → absolute days since epoch0
            times_rel = [t for t, _ in window_meas_rel]
            days = [(t + w_start) / 86400.0 for t in times_rel]

            # maha is usually length = number of measurements (or N-1 depending on implementation)
            n = min(len(days), len(maha))
            all_days.extend(days[:n])
            all_maha.extend(maha[:n].tolist())
            all_win.extend([w] * n)

            peak = float(np.max(maha))
            print(f"  window {w:3d}: {len(window_meas):3d} TLEs, "
                  f"peak MD = {peak:8.3f}  {'*** DETECTED ***' if peak > threshold else ''}")

        except RuntimeError as e:
            if "diverged" in str(e).lower():
                continue
            raise

    return all_days, all_maha, all_win


def main():
    parser = argparse.ArgumentParser(description="Plot MD time series for one object")
    parser.add_argument("--tle", required=True, help="Path to the .tle history file")
    parser.add_argument("--threshold", type=float, default=13.258)
    parser.add_argument("--tau", type=float, default=3e4)
    parser.add_argument("--q", type=float, default=1e-19)
    parser.add_argument("--window-days", type=float, default=90.0)
    parser.add_argument("--zoom-start", type=float, default=None,
                        help="Optional: start of zoom window (days since epoch)")
    parser.add_argument("--zoom-end", type=float, default=None,
                        help="Optional: end of zoom window (days since epoch)")
    parser.add_argument("--output", default="md_timeseries.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    print(f"Loading {args.tle} ...")
    tle_list = load_tles_for_object(args.tle)
    print(f"  {len(tle_list)} TLEs loaded")

    if not tle_list:
        print("No usable TLEs – aborting")
        return

    norad = tle_list[0].get("norad_id") or os.path.basename(args.tle).replace(".tle", "")
    print(f"Processing NORAD {norad} ...")

    days, maha, wins = process_object_md_series(
        tle_list, args.tau, args.q, args.window_days, args.threshold
    )

    if not days:
        print("No valid windows produced any MD values")
        return

    days = np.asarray(days)
    maha = np.asarray(maha)

    # Optional zoom
    if args.zoom_start is not None and args.zoom_end is not None:
        mask = (days >= args.zoom_start) & (days <= args.zoom_end)
        days_plot = days[mask]
        maha_plot = maha[mask]
        title_suffix = f" (Zoomed: Day {args.zoom_start:.1f}-{args.zoom_end:.1f})"
    else:
        days_plot = days
        maha_plot = maha
        title_suffix = ""

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(days_plot, maha_plot, color="darkcyan", linewidth=0.9, label=r"$|w|$ (MD)")
    ax.axhline(args.threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"Threshold = {args.threshold:.3f}")

    ax.set_xlabel("Days since epoch")
    ax.set_ylabel("Mahalanobis distance")
    title = args.title or f"Mahalanobis Distance – NORAD {norad}{title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # nice y-limit if the series is well behaved
    ymax = max(np.percentile(maha_plot, 99.5) * 1.3, args.threshold * 1.5)
    ax.set_ylim(0, ymax)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot → {args.output}")
    plt.show()


if __name__ == "__main__":
    main()