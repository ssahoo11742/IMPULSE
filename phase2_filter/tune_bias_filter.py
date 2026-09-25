"""Grid search over (tau, q) for a given duration.

Usage:
    python -m phase2_filter.tune_bias_filter --duration-days 365 \
        --sigma-target 1e-6 --seed 43

Only keeps tau values where duration is at least ~10*tau (needed for the
steady-state trim in run_bias_test to be meaningful). Starts q from the
OU steady-state rule q = 2*sigma^2/tau, then multiplies by a few factors
and reports the best pair for this duration only.
"""
import argparse
import numpy as np

from .validation import run_bias_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration-days", type=int, default=90)
    p.add_argument("--sigma-target", type=float, default=1.0e-6,
                    help="Smallest RSW accel [m/s^2] you want to resolve")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--q-mults", type=float, nargs="+",
                    default=[0.1, 0.3, 1.0, 3.0, 10.0],
                    help="Multipliers on the sizing-rule q for each tau")
    args = p.parse_args()

    duration_s = args.duration_days * 86400.0
    # need duration >= ~10*tau so the burn-in margin in run_bias_test works
    tau_candidates = [t for t in
                       [1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6, 3.0e6]
                       if duration_s >= 10.0 * t]

    if not tau_candidates:
        print(f"No tau candidate with duration >= 10*tau for "
              f"duration_days={args.duration_days}. Try a longer run "
              f"or smaller tau.")
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
                print(f"{tau:10.1e} {mult:8.2f} {q:10.2e}   SKIP: "
                      f"numerical instability")

    if best:
        tau, q, err = best
        print(f"\nBest for duration_days={args.duration_days}: "
              f"tau={tau:.3e}, q={q:.3e}, rel_err={err:.2f}%")
        print("(Only valid at this duration - rerun for other ones.)")


if __name__ == "__main__":
    main()