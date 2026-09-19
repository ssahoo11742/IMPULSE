"""2D grid search over (tau, q) for impulse detection at a fixed duration,
using a cheap separation-score proxy rather than running a full dv50
calibration at every candidate (which would be far too slow to grid search
directly).

Proxy: at a fixed reference dv, compare mean peak Mahalanobis distance WITH
a real impulse against the null (no-impulse) case, normalized by the
null's own spread:
    score = (mean_peak_signal - mean_peak_null) / std_peak_null
Higher score = better separation at that (tau, q). Once you've found the
best candidate here, refine it with find_dv_threshold.py's full null+sweep
calibration for a real dv50 number.

This supersedes the informal one-dimensional searches (fix tau, sweep q;
then fix q, sweep tau) done earlier in this project - those only checked
slices through the space, not the joint optimum.

Usage (this can run a long time - that's expected, not a bug; increase
--coarse-trials for a less noisy result if you have the patience):
    python -m phase2_filter.tune_impulse_filter_2d --duration-days 90 \
        --ref-dv 0.02 --coarse-trials 10 \
        --tau-values 3e4 1e5 3e5 1e6 \
        --q-values 1e-22 1e-21 1e-20 1e-19 1e-18 1e-17
"""
import argparse
import multiprocessing as mp

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


def _run_one(task):
    """Runs one trial and returns its peak Mahalanobis, or None on failure.
    Must be a module-level function (not a closure/lambda) to be picklable
    for multiprocessing."""
    seed, duration_days, strike_days, tau, q, dv_mag = task
    mask = np.zeros(9, dtype=bool)
    mask[0] = True
    try:
        res = run_impulse_test(duration_days=duration_days, strike_time_days=strike_days,
                                strike_dv_mag=dv_mag, tau=tau, q=q, seed=seed)
        return _peak_maha(res, mask)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration-days", type=int, default=90)
    p.add_argument("--ref-dv", type=float, default=0.02)
    p.add_argument("--coarse-trials", type=int, default=10)
    p.add_argument("--strike-time-days", type=float, default=None)
    p.add_argument("--tau-values", type=float, nargs="+",
                    default=[3e4, 1e5, 3e5, 1e6])
    p.add_argument("--q-values", type=float, nargs="+",
                    default=[1e-22, 1e-21, 1e-20, 1e-19, 1e-18, 1e-17])
    p.add_argument("--n-workers", type=int, default=None,
                    help="Default: all available CPU cores")
    args = p.parse_args()

    strike_days = args.strike_time_days or (args.duration_days / 2.0)
    n_workers = args.n_workers or mp.cpu_count()

    # warn about candidates violating the duration >= 10*tau floor
    duration_s = args.duration_days * 86400.0
    for tau in args.tau_values:
        if duration_s < 10.0 * tau:
            print(f"NOTE: tau={tau:.2e} violates duration>=10*tau at "
                  f"duration_days={args.duration_days} - included anyway, "
                  f"but treat its result with suspicion.")

    n_candidates = len(args.tau_values) * len(args.q_values)
    print(f"2D search at duration={args.duration_days}d, ref_dv={args.ref_dv}, "
          f"{args.coarse_trials} trials/candidate ({n_candidates} candidates, "
          f"{n_workers} workers)")

    # Flatten every (tau, q, trial, is_signal) combination into one task list
    # so the whole search parallelizes across cores, not just within a candidate.
    tasks = []
    task_index = []  # (tau, q, is_signal) per task, same order as `tasks`
    for tau in args.tau_values:
        for q in args.q_values:
            for t in range(args.coarse_trials):
                tasks.append((7000 + t, args.duration_days, strike_days, tau, q, 0.0))
                task_index.append((tau, q, False))
                tasks.append((8000 + t, args.duration_days, strike_days, tau, q, args.ref_dv))
                task_index.append((tau, q, True))

    with mp.Pool(n_workers) as pool:
        raw_results = pool.map(_run_one, tasks, chunksize=max(1, len(tasks) // (n_workers * 8)))

    # Reassemble into per-(tau, q) null/signal peak lists
    grouped = {}
    for (tau, q, is_signal), peak in zip(task_index, raw_results):
        key = (tau, q)
        if key not in grouped:
            grouped[key] = {"null": [], "signal": []}
        if peak is not None:
            grouped[key]["signal" if is_signal else "null"].append(peak)

    print(f"{'tau':>10} {'q':>10} {'null_mean':>10} {'null_std':>9} "
          f"{'signal_mean':>12} {'score':>8}")

    results = []
    for tau in args.tau_values:
        for q in args.q_values:
            peaks_null = grouped[(tau, q)]["null"]
            peaks_sig = grouped[(tau, q)]["signal"]
            if len(peaks_null) < max(2, args.coarse_trials // 3) or \
               len(peaks_sig) < max(2, args.coarse_trials // 3):
                print(f"{tau:10.1e} {q:10.1e}   SKIPPED (too many diverged)")
                continue
            nm, ns = np.mean(peaks_null), np.std(peaks_null) + 1e-9
            sm = np.mean(peaks_sig)
            score = (sm - nm) / ns
            print(f"{tau:10.1e} {q:10.1e} {nm:10.3f} {ns:9.3f} {sm:12.3f} {score:8.3f}")
            results.append((tau, q, score))

    print()
    if not results:
        print("ALL candidates diverged.")
        return
    results.sort(key=lambda r: -r[2])
    print("Top 5 candidates:")
    for tau, q, score in results[:5]:
        print(f"  tau={tau:.3e}  q={q:.3e}  score={score:.3f}")
    tau_best, q_best, _ = results[0]
    print()
    print("Refine the winner with the full calibration:")
    print(f"  python -m phase2_filter.find_dv_threshold --tau {tau_best:.4e} "
          f"--q {q_best:.4e} --duration-days {args.duration_days} "
          f"--null-trials 40 --sweep-trials 15")


if __name__ == "__main__":
    main()