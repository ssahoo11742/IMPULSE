"""Grid-search driver for tuning (tau, q) at a specific target duration.

Usage:
    python -m phase2_filter.tune_bias_filter --duration-days 365 \
        --sigma-target 1e-6 --seed 43

Picks tau candidates satisfying duration_days >= ~10*tau (the floor needed
for run_bias_test's steady-state trim window to mean anything - see the
tau-aware margin check added to run_bias_test), sizes q from the OU
steady-state rule q = 2*sigma_target^2/tau as a starting point, then sweeps
q around that point (since the sizing rule alone is not sufficient - see
DRIFTS notes) and reports the best (tau, q) found for THIS duration only.
Do not assume the winner here transfers to a different duration_days.
"""
import argparse
import math
import numpy as np

from .validation import run_bias_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration-days", type=int, default=90)
    p.add_argument("--sigma-target", type=float, default=1.0e-6,
                    help="Smallest RSW acceleration [m/s^2] you want resolvable")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--q-mults", type=float, nargs="+",
                    default=[0.1, 0.3, 1.0, 3.0, 10.0],
                    help="Multipliers applied to the sizing-rule q for each tau")
    args = p.parse_args()

    duration_s = args.duration_days * 86400.0
    # tau candidates satisfying duration >= ~10*tau (see run_bias_test's
    # burn-in-margin check - this floor is necessary, not just a suggestion)
    tau_candidates = [t for t in
                       [1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6, 3.0e6]
                       if duration_s >= 10.0 * t]

    if not tau_candidates:
        print(f"No tau candidate satisfies duration_days >= ~10*tau for "
              f"duration_days={args.duration_days}. Use a longer duration "
              f"or manually test smaller tau.")
        return

    print(f"Grid search at duration_days={args.duration_days}, "
          f"sigma_target={args.sigma_target:.1e}")
    print(f"{'tau':>10} {'q_mult':>8} {'q':>10} {'rel_err%':>10}")

    best = None
    for tau in tau_candidates:
        q_size = 2.0 * args.sigma_target ** 2 / tau
        for mult in args.q_mults:
            q = q_size * mult
            try:
                res = run_bias_test(duration_days=args.duration_days,
                                     tau=tau, q=q, seed=args.seed)
                err = res["relative_error"] * 100.0
                print(f"{tau:10.1e} {mult:8.2f} {q:10.2e} {err:10.2f}")
                if best is None or err < best[2]:
                    best = (tau, q, err)
            except ValueError as e:
                print(f"{tau:10.1e} {mult:8.2f} {q:10.2e}   SKIP: {e}")
            except np.linalg.LinAlgError:
                print(f"{tau:10.1e} {mult:8.2f} {q:10.2e}   SKIP: numerical "
                      f"instability (extreme tau/q conditioning)")

    if best:
        tau, q, err = best
        print(f"\nBest for duration_days={args.duration_days}: "
              f"tau={tau:.3e}, q={q:.3e}, rel_err={err:.2f}%")
        print("(Only valid at this duration - rerun for other durations.)")


if __name__ == "__main__":
    main()