#!/usr/bin/env python3
"""
fig5.py — Gate 4: Clean-vs-Debris Detection Demonstration
=========================================================
Uses the EXACT validation pipeline code path:
  - _propagate_truth  (validation.py)
  - _run_filter_pair  (validation.py)
  - compute_mahalanobis_distance with state_mask (test_statistics.py)
  - 90-day windows, 7-day edge trim
  - tau=3e4, q=1e-19
  - R_DIAG_ELEMENTS noise (realistic TLE noise levels)

This matches the fleet null test structure (process_fleet.py) so the
Mahalanobis scale is comparable to the real pipeline (~3–20, not ~8000).
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from phase2_filter.validation import (
    _propagate_truth, _default_el0, _run_filter_pair,
    generate_noisy_measurements,
)
from phase2_filter import config
from phase2_filter.test_statistics import (
    compute_smoothed_accel_peak,
    compute_mahalanobis_distance,
)
from propagator.debris_impacts import (
    compute_vmf_kappa, sample_impact_count, apply_impact
)
from constants.constants import CD_MEAN, CD_SIGMA, CD_MIN, CD_MAX, MU


# ===================================================================
# Parameters matching the real IMPULSE operating regime
# ===================================================================
DURATION_DAYS = 90
TRIM_DAYS = 7
DT_S = 86400.0
RHO_DEBRIS = 1.0e-10
SEED_START = 1000
N_TRIALS = 200


def _propagate_debris_truth(
    el0, epoch_jd, duration_s, dt_s, area, mass,
    rho_debris, f107_base=150.0, Cd_base=None, rng=None
):
    """Propagate truth with stochastic debris impacts.
    
    Uses _propagate_truth as the backbone (correct history keys) and
    injects debris impacts each step, matching propagate_debris logic.
    """
    from phase2_filter.validation import _step_truth
    from propagator.atmosphere import (
        sample_f107_phases, f107_at_time, sample_storm_events, kp_at_time
    )
    from constants.constants import CD_TAU_S, CD_DRIFT_FRAC, REENTRY_ALT

    if rng is None:
        rng = np.random.default_rng()
    if Cd_base is None:
        Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))

    el = el0.copy()
    Cd = Cd_base
    t = 0.0
    n_steps = int(duration_s / dt_s)

    storms = sample_storm_events(duration_s, rng)
    f107_phases = sample_f107_phases(rng)
    cd_sigma_step = CD_DRIFT_FRAC * Cd_base * math.sqrt(2 * dt_s / CD_TAU_S)

    hist = {
        "t": [], "a": [], "e": [], "i": [],
        "Omega": [], "omega": [], "M": [], "Cd": []
    }
    total_impacts = 0

    for _ in range(n_steps):
        if el.alt_m() < REENTRY_ALT:
            break

        f107 = f107_at_time(t, f107_phases, f_base=f107_base)
        kp = kp_at_time(t, storms)

        n_sub = 4
        dt_sub = dt_s / n_sub
        for _ in range(n_sub):
            d_a, d_ecc, d_inc, d_raan, d_argp, d_M = _step_truth(
                el, Cd, area, mass, epoch_jd, t, f107, kp
            )
            el.a += d_a * dt_sub
            el.ecc = max(0.0, el.ecc + d_ecc * dt_sub)
            el.inc += d_inc * dt_sub
            el.raan = (el.raan + d_raan * dt_sub) % (2 * math.pi)
            el.argp = (el.argp + d_argp * dt_sub) % (2 * math.pi)
            el.M += d_M * dt_sub

        # --- debris impacts ---
        v_circ = math.sqrt(MU / el.a)
        kappa, _mean_cos, flux_sum_unit = compute_vmf_kappa(el.inc)
        v_rel_mean = flux_sum_unit * v_circ
        n_impacts = sample_impact_count(rho_debris, area, v_rel_mean, dt_s, rng)
        for _ in range(n_impacts):
            apply_impact(el, mass, el.inc, v_circ, kappa, rng)
        total_impacts += n_impacts

        Cd += (1.0 / CD_TAU_S) * (Cd_base - Cd) * dt_s + cd_sigma_step * rng.normal()
        Cd = float(np.clip(Cd, CD_MIN, CD_MAX))

        t += dt_s
        hist["t"].append(t)
        hist["a"].append(el.a)
        hist["e"].append(el.ecc)
        hist["i"].append(el.inc)
        hist["Omega"].append(el.raan)
        hist["omega"].append(el.argp)
        hist["M"].append(el.M)
        hist["Cd"].append(Cd)

    return hist, total_impacts, storms, f107_phases


def run_one_trial(seed, duration_days=DURATION_DAYS, dt_s=DT_S,
                  rho_debris=RHO_DEBRIS, tau=3e4, q=1e-19):
    """Single clean-vs-debris trial using the exact pipeline code path."""
    rng = np.random.default_rng(seed)
    el0 = _default_el0()
    epoch_jd = 2460000.5
    duration_s = duration_days * 86400.0
    area, mass = 1.0, 200.0
    Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))

    # --- Clean trajectory ---
    clean_hist, clean_storms, clean_f107 = _propagate_truth(
        el0, epoch_jd, duration_s, dt_s, area, mass,
        Cd_base=Cd_base, rng=rng
    )
    clean_meas = generate_noisy_measurements(clean_hist, rng=rng)
    clean_result = _run_filter_pair(
        clean_meas, el0, Cd_base, area, mass, epoch_jd, tau, q,
        storms=clean_storms, f107_phases=clean_f107
    )

    # --- Debris trajectory ---
    debris_hist, total_impacts, debris_storms, debris_f107 = _propagate_debris_truth(
        el0, epoch_jd, duration_s, dt_s, area, mass,
        rho_debris=rho_debris, Cd_base=Cd_base, rng=rng
    )
    debris_meas = generate_noisy_measurements(debris_hist, rng=rng)
    debris_result = _run_filter_pair(
        debris_meas, el0, Cd_base, area, mass, epoch_jd, tau, q,
        storms=debris_storms, f107_phases=debris_f107
    )

    # --- Extract stats with 7-day trim ---
    trim_s = TRIM_DAYS * 86400.0

    def _trim(result):
        times = np.array(result["times"])
        mask = (times >= trim_s) & (times <= times[-1] - trim_s)
        sm_trimmed = [s for s, m in zip(result["sm_states"], mask) if m]
        times_trimmed = times[mask].tolist()
        fwd_trimmed = [s for s, m in zip(result["fwd_states"], mask) if m]
        fc_trimmed = [c for c, m in zip(result["fwd_covs"], mask) if m]
        bwd_trimmed = [s for s, m in zip(result["bwd_states"], mask) if m]
        bc_trimmed = [c for c, m in zip(result["bwd_covs"], mask) if m]
        smc_trimmed = [c for c, m in zip(result["sm_covs"], mask) if m]
        return {
            "sm": sm_trimmed, "times": times_trimmed,
            "fwd": fwd_trimmed, "fc": fc_trimmed,
            "bwd": bwd_trimmed, "bc": bc_trimmed,
            "smc": smc_trimmed,
        }

    c = _trim(clean_result)
    d = _trim(debris_result)

    # Peak Mahalanobis — use state_mask on 'a' only, matching process_fleet.py
    state_mask = np.zeros(9, dtype=bool)
    state_mask[0] = True

    d_mh_clean = compute_mahalanobis_distance(
        c["fwd"], c["fc"], c["bwd"], c["bc"], c["smc"],
        state_mask=state_mask
    )
    d_mh_debris = compute_mahalanobis_distance(
        d["fwd"], d["fc"], d["bwd"], d["bc"], d["smc"],
        state_mask=state_mask
    )

    # Smoothed acceleration
    accel_clean = compute_smoothed_accel_peak(c["sm"], c["times"])
    accel_debris = compute_smoothed_accel_peak(d["sm"], d["times"])

    # Time series for plotting (trimmed)
    t_c = np.array(c["times"]) / 86400.0
    a_c = np.array([np.linalg.norm(s[6:9]) for s in c["sm"]])
    t_d = np.array(d["times"]) / 86400.0
    a_d = np.array([np.linalg.norm(s[6:9]) for s in d["sm"]])

    return {
        "peak_maha_clean": float(np.max(d_mh_clean)),
        "peak_maha_debris": float(np.max(d_mh_debris)),
        "peak_accel_clean": accel_clean["peak_norm"],
        "peak_accel_debris": accel_debris["peak_norm"],
        "total_impacts": total_impacts,
        "t_clean": t_c, "a_clean": a_c,
        "t_debris": t_d, "a_debris": a_d,
    }


def main():
    print(f"Running {N_TRIALS} clean-vs-debris trials...")
    print(f"Window: {DURATION_DAYS} days | Trim: {TRIM_DAYS} days each edge")
    print(f"tau={3e4:.0f} s | q={1e-19:.0e} | rho_debris={RHO_DEBRIS:.0e}")
    print("(This may take 5–10 minutes)")

    peak_clean, peak_debris = [], []
    accel_clean, accel_debris = [], []
    representative = None

    for i in range(N_TRIALS):
        trial = run_one_trial(SEED_START + i)
        peak_clean.append(trial["peak_maha_clean"])
        peak_debris.append(trial["peak_maha_debris"])
        accel_clean.append(trial["peak_accel_clean"])
        accel_debris.append(trial["peak_accel_debris"])
        if representative is None:
            representative = trial
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{N_TRIALS}")

    peak_clean = np.array(peak_clean)
    peak_debris = np.array(peak_debris)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: representative acceleration traces (trimmed)
    ax = axes[0]
    ax.plot(representative["t_clean"], representative["a_clean"],
            color='#2E86AB', lw=1.5, label='Clean trajectory')
    ax.plot(representative["t_debris"], representative["a_debris"],
            color='#E63946', lw=1.5, label='Debris trajectory')

    # Mark debris peaks
    thresh = 1e-4
    peaks = representative["t_debris"][representative["a_debris"] > thresh]
    vals = representative["a_debris"][representative["a_debris"] > thresh]
    if len(peaks) > 0:
        ax.scatter(peaks, vals, color='#E63946', s=25, zorder=5)

    ax.axvline(x=TRIM_DAYS, color='#888888', linestyle=':', lw=1.5, alpha=0.5)
    ax.axvline(x=DURATION_DAYS - TRIM_DAYS, color='#888888',
               linestyle=':', lw=1.5, alpha=0.5)

    ax.set_ylabel(r'Smoothed acceleration $|w_S|$ [m/s$^2$]', fontsize=12)
    ax.set_title('Clean vs. Debris Trajectories (90-day window, 7-day trim)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bottom: CDF of peak Mahalanobis
    ax = axes[1]

    def plot_cdf(data, color, label):
        s = np.sort(data)
        ax.plot(s, np.arange(1, len(s) + 1) / len(s),
                color=color, lw=2, label=label)

    plot_cdf(peak_clean, '#2E86AB', f'Clean (n={N_TRIALS})')
    plot_cdf(peak_debris, '#E63946', f'Debris (n={N_TRIALS})')

    ax.axvline(x=13.258, color='#333333', linestyle='--', lw=1.5, alpha=0.5)
    ax.text(13.5, 0.05, r'$\eta = 13.258$', fontsize=10, color='#333333')

    ax.set_xlabel('Peak Mahalanobis distance (semi-major axis only)', fontsize=12)
    ax.set_ylabel('Cumulative probability', fontsize=12)
    ax.set_title('Distribution of peak Mahalanobis distances',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figs/fig5_clean_vs_debris.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    print("\nSaved fig5_clean_vs_debris.png")
    print(f"\n  Clean  — Maha mean: {peak_clean.mean():.2f}, median: {np.median(peak_clean):.2f}")
    print(f"  Debris — Maha mean: {peak_debris.mean():.2f}, median: {np.median(peak_debris):.2f}")
    print(f"  Accel  — Clean peak: {np.mean(accel_clean):.2e}, Debris peak: {np.mean(accel_debris):.2e}")


if __name__ == '__main__':
    main()