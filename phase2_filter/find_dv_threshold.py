"""Find the minimum detectable single-impact dv (m/s) for a given (tau, q, duration).

Uses the real EKF+smoother pipeline:
  1. Null calibration — peak Mahalanobis under no impulse → detection threshold
  2. dv sweep — find the dv where detection rate crosses 50% (dv50)

Usage:
    python -m phase2_filter.find_dv_threshold \
        --tau 1e5 --q 5e-17 --duration-days 30 \
        --null-trials 40 --sweep-trials 15 --percentile 95
"""
import argparse

import numpy as np

from .validation import run_impulse_test
from .test_statistics import compute_mahalanobis_distance


def _peak_maha(res, mask, sanity_ceiling=1e4):
    # Real peaks top out around ~150-200. Values past the ceiling are
    # numerical garbage from a diverged filter (numpy can produce finite
    # but absurd floats that isfinite still accepts).
    r = res["result"]
    maha = compute_mahalanobis_distance(
        r["fwd_states"], r["fwd_covs"], r["bwd_states"], r["bwd_covs"], r["sm_covs"],
        state_mask=mask,
    )
    peak = maha.max()
    if not np.isfinite(peak) or abs(peak) > sanity_ceiling:
        raise RuntimeError(
            f"peak Mahalanobis={peak:.3e} non-finite or unphysical - filter diverged"
        )
    return peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--duration-days", type=int, default=30)
    p.add_argument("--strike-time-days", type=float, default=None,
                    help="Default: middle of the duration")
    p.add_argument("--null-trials", type=int, default=40)
    p.add_argument("--sweep-trials", type=int, default=15)
    p.add_argument("--percentile", type=float, default=95.0,
                    help="Null percentile used as detection threshold "
                         "(ignored if --threshold-override is set)")
    p.add_argument("--threshold-override", type=float, default=None,
                    help="Skip local null calibration and use this threshold "
                         "(e.g. from calibrate_null_threshold.py)")
    p.add_argument("--dv-min", type=float, default=1.0e-4)
    p.add_argument("--dv-max", type=float, default=0.5)
    p.add_argument("--n-dv-points", type=int, default=14)
    args = p.parse_args()

    strike_days = args.strike_time_days or (args.duration_days / 2.0)
    mask = np.zeros(9, dtype=bool)
    mask[0] = True  # 'a' only

    if args.threshold_override is not None:
        threshold = args.threshold_override
        print(f"Using supplied threshold = {threshold:.3f}")
    else:
        print(f"Null calibration ({args.null_trials} trials, "
              f"tau={args.tau:.2e}, q={args.q:.2e}, duration={args.duration_days}d)...")
        peaks_null = []
        for trial in range(args.null_trials):
            try:
                res = run_impulse_test(
                    duration_days=args.duration_days,
                    strike_time_days=strike_days,
                    strike_dv_mag=0.0,
                    tau=args.tau, q=args.q, seed=9000 + trial,
                )
                peaks_null.append(_peak_maha(res, mask))
            except (RuntimeError, np.linalg.LinAlgError, OverflowError,
                    ValueError, FloatingPointError):
                continue
        if len(peaks_null) < max(5, args.null_trials // 4):
            print(f"ERROR: only {len(peaks_null)}/{args.null_trials} null trials "
                  f"succeeded — (tau, q, duration) too unstable")
            return
        peaks_null = np.array(peaks_null)
        threshold = float(np.percentile(peaks_null, args.percentile))
        print(f"  null peak: mean={peaks_null.mean():.3f} std={peaks_null.std():.3f}")
        print(f"  {args.percentile:.0f}th percentile threshold = {threshold:.3f}")
    print()

    dv_grid = np.logspace(np.log10(args.dv_min), np.log10(args.dv_max), args.n_dv_points)
    print(f"Sweeping dv {args.dv_min:.2e} → {args.dv_max:.2e} m/s "
          f"({args.sweep_trials} trials each)")
    print(f"{'dv_mag(m/s)':>12} {'detect_rate':>12} {'mean_peak_maha':>15}")

    dvs, rates = [], []
    for dv in dv_grid:
        detections = 0
        peaks = []
        n_failed = 0
        for trial in range(args.sweep_trials):
            try:
                res = run_impulse_test(
                    duration_days=args.duration_days,
                    strike_time_days=strike_days,
                    strike_dv_mag=float(dv),
                    tau=args.tau, q=args.q, seed=10000 + trial,
                )
                peak = _peak_maha(res, mask)
                peaks.append(peak)
                if peak > threshold:
                    detections += 1
            except (RuntimeError, np.linalg.LinAlgError, OverflowError,
                    ValueError, FloatingPointError):
                n_failed += 1
                continue
        if len(peaks) < 2:
            print(f"{dv:12.5f}   SKIPPED - all trials diverged")
            continue
        rate = detections / len(peaks)
        fail_note = f"  ({n_failed} skipped)" if n_failed else ""
        print(f"{dv:12.5f} {rate:12.2f} {np.mean(peaks):15.3f}{fail_note}")
        dvs.append(dv)
        rates.append(rate)

    dvs = np.array(dvs)
    rates = np.array(rates)
    print()
    if rates.max() >= 0.5 >= rates.min():
        dv50 = float(np.interp(0.5, rates, dvs))
        print(f"dv50 ≈ {dv50:.5f} m/s")
    else:
        print("Sweep didn't bracket 50% — widen --dv-min/--dv-max")


if __name__ == "__main__":
    main()