"""Parallelized null-case (no-impulse) calibration for a detection
threshold, at whatever percentile you actually need given your fleet size
(see the multiple-testing discussion - 95th percentile is nowhere near
strict enough once you're testing thousands of objects).

Runs many null trials in parallel across CPU cores (each trial is
independent, so this scales close to linearly with core count), then
reports:
  1. Empirical percentiles, flagged as UNRELIABLE if the trial count
     doesn't support that percentile (rule of thumb: want >=5 expected
     samples beyond the target percentile - see the printed warning).
  2. A parametric (normal-distribution) fit to the same samples, used to
     EXTRAPOLATE percentiles beyond what's empirically resolvable. This is
     a cross-check, not a substitute for real samples - if the empirical
     and parametric estimates disagree noticeably in the region where both
     are resolvable, that's a sign the tail isn't well-approximated by a
     normal distribution and the extrapolation should be trusted less.

Usage:
    python -m phase2_filter.calibrate_null_threshold \
        --tau 1e5 --q 1e-19 --duration-days 90 \
        --n-trials 20000 --n-workers 8 \
        --percentiles 95 99 99.9 99.97 99.998
"""
import argparse
import multiprocessing as mp
import statistics

import numpy as np

from .validation import run_impulse_test
from .test_statistics import compute_mahalanobis_distance


def _peak_maha(res, mask, sanity_ceiling=1.0e4):
    """sanity_ceiling: numpy's overflow/invalid-value warnings during matrix
    ops do NOT raise exceptions by default - an overflowed computation can
    silently produce a technically-finite but absurdly large float (seen in
    practice: ~1e103) that np.isfinite() would NOT catch (it's not literally
    inf/nan, just numerical garbage from a diverged filter). Real peak
    Mahalanobis values in this project have topped out around ~150-200 even
    at extreme injected dv, so anything past this ceiling is treated as a
    diverged/failed trial, not a real result - this is what should have made
    the earlier astronomical "best candidate" (~1e103) get correctly
    excluded instead of winning by numerical accident."""
    r = res["result"]
    maha = compute_mahalanobis_distance(
        r["fwd_states"], r["fwd_covs"], r["bwd_states"], r["bwd_covs"], r["sm_covs"],
        state_mask=mask,
    )
    times = np.array(r["times"])
    strike_t = res["strike_time_s"]
    window = (times >= strike_t - 86400) & (times <= strike_t + 86400)
    peak = maha[window].max() if window.any() else maha.max()
    if not np.isfinite(peak) or abs(peak) > sanity_ceiling:
        raise RuntimeError(
            f"peak Mahalanobis={peak:.3e} is non-finite or unphysically large - "
            f"filter has diverged at this (tau, q, duration), not a real result."
        )
    return peak


def _run_one(args_tuple):
    seed, duration_days, strike_days, tau, q = args_tuple
    mask = np.zeros(9, dtype=bool)
    mask[0] = True
    try:
        res = run_impulse_test(duration_days=duration_days, strike_time_days=strike_days,
                                strike_dv_mag=0.0, tau=tau, q=q, seed=seed)
        return _peak_maha(res, mask)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--duration-days", type=int, default=90)
    p.add_argument("--strike-time-days", type=float, default=None)
    p.add_argument("--n-trials", type=int, default=20000)
    p.add_argument("--n-workers", type=int, default=None,
                    help="Default: all available CPU cores")
    p.add_argument("--percentiles", type=float, nargs="+",
                    default=[95, 99, 99.9, 99.97, 99.998])
    p.add_argument("--seed-offset", type=int, default=20000)
    args = p.parse_args()

    strike_days = args.strike_time_days or (args.duration_days / 2.0)
    n_workers = args.n_workers or mp.cpu_count()

    print(f"Running {args.n_trials} null trials across {n_workers} workers "
          f"(tau={args.tau:.3e}, q={args.q:.3e}, duration={args.duration_days}d)...")
    tasks = [(args.seed_offset + i, args.duration_days, strike_days, args.tau, args.q)
             for i in range(args.n_trials)]

    with mp.Pool(n_workers) as pool:
        results = pool.map(_run_one, tasks, chunksize=max(1, args.n_trials // (n_workers * 8)))

    peaks = np.array([r for r in results if r is not None])
    n_failed = len(results) - len(peaks)
    print(f"Completed: {len(peaks)} succeeded, {n_failed} diverged/skipped")
    print()

    mean, std = float(np.mean(peaks)), float(np.std(peaks))
    print(f"Empirical: mean={mean:.3f}  std={std:.3f}")
    print()

    normal_fit = statistics.NormalDist(mean, std)

    print(f"{'percentile':>12} {'empirical':>12} {'reliable?':>10} {'normal-fit extrap':>18}")
    for pct in args.percentiles:
        tail_prob = (100.0 - pct) / 100.0
        expected_beyond = len(peaks) * tail_prob
        reliable = expected_beyond >= 5
        emp_val = float(np.percentile(peaks, pct))
        fit_val = normal_fit.inv_cdf(pct / 100.0)
        flag = "OK" if reliable else f"NO (~{expected_beyond:.1f} samples beyond)"
        print(f"{pct:12.3f} {emp_val:12.3f} {flag:>10} {fit_val:18.3f}")

    print()
    print("If empirical and normal-fit values diverge noticeably where both are "
          "'OK', the tail isn't well-approximated by a normal distribution - "
          "trust the normal-fit extrapolation less for percentiles beyond that point.")


if __name__ == "__main__":
    main()