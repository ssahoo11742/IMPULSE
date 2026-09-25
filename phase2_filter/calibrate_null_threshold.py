"""
Null-case (no-impulse) calibration for a detection threshold.
Runs independent null trials in parallel and reports empirical percentiles
plus a normal-distribution fit for the far tail.

Usage:
    python -m phase2_filter.calibrate_null_threshold \
        --tau 1e5 --q 1e-19 --duration-days 90 \
        --n-trials 20000 --n-workers 8 \
        --percentiles 95 99 99.9 99.97 99.998
"""

import argparse
# TODO: maybe switch to concurrent.futures if multiprocessing acts up again on linux?
import multiprocessing as mp
import statistics
import numpy as np

from .validation import run_impulse_test
from .test_statistics import compute_mahalanobis_distance

# Some constants we might need later... 
# MAX_ALLOWED_PEAK = 1e4 


def _peak_maha(res, mask, sanity_ceiling=1e4):
    # Real peaks top out around ~150-200 even with large injected dv.
    # Values past the ceiling are numerical garbage from a diverged filter
    # (numpy overflows can produce finite but absurd floats that isfinite
    # still accepts). Treat those as failed trials.
    # Note to self: remember why we added sanity_ceiling in the first place -- 
    # damn covariance matrices blowing up when q is set too high!
    r = res["result"]
    
    maha = compute_mahalanobis_distance(
        r["fwd_states"], 
        r["fwd_covs"], 
        r["bwd_states"], 
        r["bwd_covs"], 
        r["sm_covs"],
        state_mask=mask,
    )
    
    times = np.array(r["times"])
    strike_t = res["strike_time_s"]
    
    # look around a window of 1 day near the strike time
    window = (times >= strike_t - 86400) & (times <= strike_t + 86400)
    
    if window.any():
        peak = maha[window].max()
    else:
        peak = maha.max()
        
    if not np.isfinite(peak) or abs(peak) > sanity_ceiling:
        # Just raise it, we catch it outside anyway
        raise RuntimeError(
            f"peak Mahalanobis={peak:.3e} non-finite or unphysical - filter diverged"
        )
        
    return peak


def _run_one(args_tuple):
    seed, duration_days, strike_days, tau, q = args_tuple
    
    # We only care about the first state variable usually (position/velocity component?)
    mask = np.zeros(9, dtype=bool)
    mask[0] = True
    
    try:
        res = run_impulse_test(
            duration_days=duration_days,
            strike_time_days=strike_days,
            strike_dv_mag=0.0, # zero impulse for null case!!
            tau=tau,
            q=q,
            seed=seed,
        )
        return _peak_maha(res, mask)
    except Exception as e:
        # print(f"Trial failed with seed {seed}: {e}") # debug
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--duration-days", type=int, default=90)
    p.add_argument("--strike-time-days", type=float, default=None)
    p.add_argument("--n-trials", type=int, default=20000)
    p.add_argument("--n-workers", type=int, default=None,
                    help="Default: all available cores")
    p.add_argument("--percentiles", type=float, nargs="+",
                    default=[95, 99, 99.9, 99.97, 99.998])
    p.add_argument("--seed-offset", type=int, default=20000)
    args = p.parse_args()

    # figure out strike days if not given
    if args.strike_time_days is None:
        strike_days = args.duration_days / 2.0
    else:
        strike_days = args.strike_time_days
        
    if args.n_workers is None:
        n_workers = mp.cpu_count()
    else:
        n_workers = args.n_workers

    print(f"Running {args.n_trials} null trials on {n_workers} workers "
          f"(tau={args.tau:.3e}, q={args.q:.3e}, duration={args.duration_days}d)...")
          
    tasks = []
    for i in range(args.n_trials):
        s = args.seed_offset + i
        tasks.append((s, args.duration_days, strike_days, args.tau, args.q))

    # using pool map
    chunk_sz = max(1, args.n_trials // (n_workers * 8))
    
    with mp.Pool(n_workers) as pool:
        results = pool.map(_run_one, tasks, chunksize=chunk_sz)

    # filter out None results from failed runs
    peaks = []
    for r in results:
        if r is not None:
            peaks.append(r)
    peaks = np.array(peaks)
    
    n_failed = len(results) - len(peaks)
    print(f"Completed: {len(peaks)} ok, {n_failed} diverged/skipped\n")

    mean = float(np.mean(peaks))
    std = float(np.std(peaks))
    print(f"Empirical: mean={mean:.3f}  std={std:.3f}\n")

    normal_fit = statistics.NormalDist(mean, std)

    print(f"{'percentile':>12} {'empirical':>12} {'reliable?':>10} {'normal-fit':>12}")
    
    for pct in args.percentiles:
        tail_prob = (100.0 - pct) / 100.0
        expected_beyond = len(peaks) * tail_prob
        
        if expected_beyond >= 5:
            reliable = True
        else:
            reliable = False
            
        emp_val = float(np.percentile(peaks, pct))
        fit_val = normal_fit.inv_cdf(pct / 100.0)
        
        if reliable:
            flag = "OK"
        else:
            flag = f"NO (~{expected_beyond:.1f})"
            
        print(f"{pct:12.3f} {emp_val:12.3f} {flag:>10} {fit_val:12.3f}")


if __name__ == "__main__":
    main()