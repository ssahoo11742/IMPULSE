"""CLI entry point for Phase-2 EKF validation ladder.

Usage:
    python -m phase2_filter.run_validation --step null --duration-days 30
    python -m phase2_filter.run_validation --step impulse --duration-days 30 --plot
    python -m phase2_filter.run_validation --step detect --duration-days 365 --rho-debris 1e-10 --plot
"""

import argparse
import sys
import numpy as np
from .dynamics import e_argp_from_hk
from . import config
from .validation import (
    run_null_test,
    run_bias_test,
    run_impulse_test,
    run_single_detection,
)


def _print_null(res):
    print("\n=== NULL TEST (2a) ===")
    print(f"  Peak |w|       : {res['peak_accel_m_s2']:.3e} m/s^2")
    print(f"  Mean off-peak    : {res['mean_offpeak_m_s2']:.3e} m/s^2")
    print(f"  SNR              : {res['snr']:.3f}")
    print(f"  PASS             : {res['passed']}")


def _print_bias(res):
    print("\n=== BIAS TEST (2b) ===")
    print(f"  Injected w       : {res['injected_w']}")
    print(f"  Estimated w      : {res['estimated_w']}")
    print(f"  Relative error   : {res['relative_error']:.3%}")
    print(f"  PASS             : {res['passed']}")


def _print_impulse(res):
    print("\n=== IMPULSE TEST (2c) ===")
    print(f"  Strike time      : {res['strike_time_s']/86400:.2f} d")
    print(f"  Peak time        : {res['peak_time_s']/86400:.2f} d")
    print(f"  Peak |w_S|       : {res['peak_accel_m_s2']:.3e} m/s^2")
    print(f"  Mean off-peak    : {res['mean_offpeak_m_s2']:.3e} m/s^2")
    print(f"  SNR              : {res['snr']:.3f}")
    print(f"  PASS             : {res['passed']}")


def _print_detect(res):
    print("\n=== DETECTION DEMO ===")
    print(f"  Total impacts    : {res['total_impacts']}")
    print(f"  Peak |w_S|       : {res['accel_stats']['peak_norm']:.3e} m/s^2")
    print(f"  Peak SNR (accel) : {res['accel_stats']['snr']:.3f}")
    print(f"  Peak Mahalanobis : {res['mahalanobis_peak']:.3f}")
    print(f"  Peak McReynolds  : {res['mcreynolds_peak']:.3f}")


def _plot_result(res, step_name):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping plot")
        return

    result = res.get("result")
    if result is None:
        print("[WARN] No filter result to plot")
        return

    times = np.array(result["times"]) / 86400.0  # days
    sm = np.array(result["sm_states"])
    fwd = np.array(result["fwd_states"])
    bwd = np.array(result["bwd_states"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- top: smoothed RSW accelerations ---
    ax = axes[0]
    ax.plot(times, sm[:, 6], label='w_R', alpha=0.8)
    ax.plot(times, sm[:, 7], label='w_S', alpha=0.8)
    ax.plot(times, sm[:, 8], label='w_W', alpha=0.8)
    ax.set_ylabel('Smoothed accel [m/s$^2$]')
    ax.set_title(f'{step_name}: Smoothed unmodeled RSW accelerations')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- middle: forward vs backward state difference (a and M) ---
    ax = axes[1]
    def _to_classical(states):
        out = np.zeros((len(states), 6))
        for i, x in enumerate(states):
            e, argp = e_argp_from_hk(x[1], x[2])
            out[i] = [x[0], e, x[3], x[4], argp, x[5]]
        return out

    fwd_cl = _to_classical(fwd)
    bwd_cl = _to_classical(bwd)
    da = fwd_cl[:, 0] - bwd_cl[:, 0]
    dM = fwd_cl[:, 5] - bwd_cl[:, 5]
    ax.plot(times, da, label='$\Delta a$ [m]', alpha=0.8)
    ax.plot(times, dM * sm[:, 0], label='$\Delta M \cdot a$ [m]', alpha=0.8)
    ax.set_ylabel('Forward–backward difference')
    ax.set_title('Forward vs backward filter discrepancy')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- bottom: innovations (position-like elements) ---
    ax = axes[2]
    innov = np.array(result["innovations"])
    ax.plot(times, innov[:, 0], label='$\nu_a$ [m]', alpha=0.6)
    ax.plot(times, innov[:, 5] * sm[:, 0], label='$\nu_M \cdot a$ [m]', alpha=0.6)
    ax.set_ylabel('Innovation')
    ax.set_xlabel('Time [days]')
    ax.set_title('Measurement innovations')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- draw trim margins (7-day buffer) ---
    margin_days = 7.0
    for ax in axes:
        ax.axvline(margin_days, color='red', linestyle='--', alpha=0.4)
        ax.axvline(times[-1] - margin_days, color='red', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = f'phase2_{step_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {fname}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Phase-2 EKF validation ladder"
    )
    parser.add_argument(
        '--step', choices=['null', 'bias', 'impulse', 'detect', 'all'],
        default='null',
        help='Which validation step to run'
    )
    parser.add_argument(
        '--duration-days', type=int, default=30,
        help='Propagation duration [days]'
    )
    parser.add_argument(
        '--dt-s', type=float, default=86400.0,
        help='Propagation step [s] (default 1 day)'
    )
    parser.add_argument(
        '--tau', type=float, default=None,
        help='FOGM time constant [s] (override config)'
    )
    parser.add_argument(
        '--q', type=float, default=None,
        help='Process noise spectral density (override config)'
    )
    parser.add_argument(
        '--rho-debris', type=float, default=1.0e-10,
        help='Debris density for detection demo [fragments/m^3]'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='RNG seed (default varies by step)'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Save diagnostic plots'
    )
    args = parser.parse_args()

    # Override config if requested
    tau = args.tau if args.tau is not None else config.DEFAULT_TAU_S
    q = args.q if args.q is not None else config.DEFAULT_Q

    print("=" * 60)
    print("Phase-2 EKF Validation")
    print("=" * 60)
    print(f"  Step            : {args.step}")
    print(f"  Duration        : {args.duration_days} days")
    print(f"  dt              : {args.dt_s} s")
    print(f"  tau             : {tau:.3e} s")
    print(f"  q               : {q:.3e}")
    print("=" * 60)

    steps_to_run = []
    if args.step == 'all':
        steps_to_run = ['null', 'bias', 'impulse', 'detect']
    else:
        steps_to_run = [args.step]

    for step in steps_to_run:
        seed = args.seed if args.seed is not None else {
            'null': 42, 'bias': 43, 'impulse': 44, 'detect': 45
        }[step]

        if step == 'null':
            res = run_null_test(
                duration_days=args.duration_days, dt_s=args.dt_s,
                tau=tau, q=q, seed=seed
            )
            _print_null(res)
            if args.plot:
                _plot_result(res, "Null Test")

        elif step == 'bias':
            res = run_bias_test(
                duration_days=args.duration_days, dt_s=args.dt_s,
                tau=tau, q=q, seed=seed
            )
            _print_bias(res)
            if args.plot:
                _plot_result(res, "Bias Test")

        elif step == 'impulse':
            res = run_impulse_test(
                duration_days=args.duration_days, dt_s=args.dt_s,
                tau=tau, q=q, seed=seed
            )
            _print_impulse(res)
            if args.plot:
                _plot_result(res, "Impulse Test")

        elif step == 'detect':
            res = run_single_detection(
                duration_days=args.duration_days, dt_s=args.dt_s,
                rho_debris=args.rho_debris, tau=tau, q=q, seed=seed
            )
            _print_detect(res)
            if args.plot:
                _plot_result(res, "Detection Demo")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())