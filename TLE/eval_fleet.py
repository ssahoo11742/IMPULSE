"""Score a real fetched fleet's aggregate statistical power for debris
detection, using each object's OWN inclination/altitude/duration, and
folding in a per-object DETECTABILITY threshold - not every Poisson-expected
impact is actually visible above measurement noise, since the fragment-mass
distribution is heavily skewed toward tiny, undetectable fragments.

Rationale for the detectability step: a single impulsive dv_S produces an
immediate jump da = (2/n) * dv_S (same coefficient as the continuous-bias
B[0,1]/n sensitivity used earlier in this project). For that jump to be
visible against one measurement's noise floor at n_sigma confidence, you
need dv_S > n_sigma*sigma_a / (2/n). That dv threshold maps (via
dv = m_frag/mass_sat * v_rel) to a minimum fragment mass, and via the NSBM
power law (fragment_mass is monotonic in Lc, inverted here by bisection -
no scipy dependency, same style as debris_impacts.py's own
_invert_langevin) to a minimum Lc. The fraction of impacts at or above that
Lc follows directly from the NSBM power law CDF already used in
sample_fragment_lc.

CAVEAT: this uses a single-measurement noise threshold. Your actual EKF +
smoother pipeline, run over many measurements, should push the effective
detection floor below a single raw measurement's noise - so this is likely
a pessimistic (lower) bound on the true detectable fraction, not the final
answer.

Usage:
    python -m phase2_filter.evaluate_fleet_power \
        --meta real_tles.tle.meta.json --rho 3.4e-13 \
        --durations real_tles.tle.durations.json \
        --target-impacts 30 --n-sigma 3.0
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

from propagator.debris_impacts import (
    compute_vmf_kappa, sample_fragment_lc, fragment_mass,
)
from constants.constants import MU, R_EARTH, LC_MIN_M, NSBM_POWER_LAW_EXP
from phase2_filter.config import R_DIAG_ELEMENTS

INC_BIN_EDGES_DEG = [0, 10, 20, 28, 40, 51, 65, 74, 82, 90, 97, 98, 100, 110, 120, 150, 180]


def _bin_of(inc_deg):
    for lo, hi in zip(INC_BIN_EDGES_DEG[:-1], INC_BIN_EDGES_DEG[1:]):
        if lo <= inc_deg < hi:
            return lo
    return None


def _invert_fragment_mass(m_target, lo=LC_MIN_M, hi=10.0, iters=60):
    """Find Lc such that fragment_mass(Lc) == m_target, via bisection.
    fragment_mass is monotonically increasing in Lc over the whole NSBM
    piecewise power law domain."""
    if fragment_mass(lo) >= m_target:
        return lo
    if fragment_mass(hi) <= m_target:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if fragment_mass(mid) < m_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _detectable_fraction(v_rel, mass_sat, n_mean, mass_thresh_sigma, n_sigma):
    sens = 2.0 / n_mean  # da per unit dv_S, m per (m/s)
    dv_threshold = (n_sigma * mass_thresh_sigma) / sens
    m_frag_threshold = dv_threshold * mass_sat / v_rel
    if m_frag_threshold <= LC_MIN_M:  # threshold below smallest fragment - everything detectable
        pass
    Lc_threshold = _invert_fragment_mass(m_frag_threshold)
    if Lc_threshold <= LC_MIN_M:
        return 1.0
    return min(1.0, (LC_MIN_M / Lc_threshold) ** NSBM_POWER_LAW_EXP)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True, help="Path to the .meta.json from fetch_tles.py")
    p.add_argument("--rho", type=float, required=True, help="Target debris density [1/m^3]")
    p.add_argument("--area", type=float, default=1.0, help="Satellite cross-section [m^2]")
    p.add_argument("--mass", type=float, default=200.0, help="Satellite mass [kg]")
    p.add_argument("--duration-days", type=float, default=5475.0,
                    help="Fallback duration [days] for objects with no real duration available")
    p.add_argument("--durations", default=None,
                    help="Path to a durations.json from fetch_object_durations.py")
    p.add_argument("--target-impacts", type=float, default=30.0)
    p.add_argument("--n-sigma", type=float, default=3.0,
                    help="Detection confidence threshold (multiples of measurement sigma)")
    p.add_argument("--frag-mc-samples", type=int, default=200000)
    args = p.parse_args()

    with open(args.meta) as f:
        records = json.load(f)

    durations_map = {}
    if args.durations:
        with open(args.durations) as f:
            durations_map = json.load(f)
        print(f"Loaded real durations for {len(durations_map)} objects from {args.durations}")

    rng = np.random.default_rng(0)
    m_frag_mean = np.mean([fragment_mass(l) for l in
                            sample_fragment_lc(rng, n=args.frag_mc_samples)])

    sigma_a = math.sqrt(R_DIAG_ELEMENTS[0])  # measurement noise std on 'a', meters
    duration_s = args.duration_days * 86400.0

    bin_counts = defaultdict(int)
    bin_expected_impacts = defaultdict(float)
    bin_detectable_impacts = defaultdict(float)
    total_expected_impacts = 0.0
    total_detectable_impacts = 0.0
    skipped = 0
    n_real_duration = 0
    n_fallback_duration = 0

    for rec in records:
        try:
            inc_deg = float(rec["INCLINATION"])
            peri_km = float(rec["PERIAPSIS"])
            apo_km = float(rec["APOAPSIS"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        norad_id = str(rec.get("NORAD_CAT_ID", ""))
        if norad_id in durations_map:
            obj_duration_s = durations_map[norad_id]["duration_days"] * 86400.0
            n_real_duration += 1
        else:
            obj_duration_s = duration_s
            n_fallback_duration += 1

        alt_km = 0.5 * (peri_km + apo_km)
        a = alt_km * 1000.0 + R_EARTH
        v_circ = math.sqrt(MU / a)
        n_mean = math.sqrt(MU / a ** 3)
        _, _, flux_sum = compute_vmf_kappa(math.radians(inc_deg))
        v_rel = flux_sum * v_circ

        lam = args.rho * args.area * v_rel
        expected_impacts = lam * obj_duration_s

        frac_detectable = _detectable_fraction(v_rel, args.mass, n_mean, sigma_a, args.n_sigma)
        detectable_impacts = expected_impacts * frac_detectable

        b = _bin_of(inc_deg)
        if b is not None:
            bin_counts[b] += 1
            bin_expected_impacts[b] += expected_impacts
            bin_detectable_impacts[b] += detectable_impacts
        total_expected_impacts += expected_impacts
        total_detectable_impacts += detectable_impacts

    print(f"rho={args.rho:.2e} /m^3   E[m_frag]={m_frag_mean:.2e} kg   "
          f"sigma_a={sigma_a:.1f} m   n_sigma={args.n_sigma:.1f}   "
          f"({skipped} objects skipped, missing fields)")
    if durations_map:
        print(f"durations: {n_real_duration} real (from satcat), "
              f"{n_fallback_duration} fallback (--duration-days={args.duration_days:.0f})")
    else:
        print(f"durations: uniform placeholder ({args.duration_days:.0f} days) for all objects")
    print()
    print(f"{'inc bin':>14} {'N sats':>8} {'expected':>12} {'detectable':>12} {'detect %':>10}")
    for lo, hi in zip(INC_BIN_EDGES_DEG[:-1], INC_BIN_EDGES_DEG[1:]):
        n = bin_counts.get(lo, 0)
        if n == 0:
            continue
        exp = bin_expected_impacts[lo]
        det = bin_detectable_impacts[lo]
        pct = 100.0 * det / exp if exp > 0 else 0.0
        print(f"{lo:>5}-{hi:<5}deg {n:>8} {exp:>12.2f} {det:>12.2f} {pct:>9.2f}%")

    print()
    print(f"TOTAL expected impacts (raw Poisson count):     {total_expected_impacts:.2f}")
    print(f"TOTAL expected DETECTABLE impacts ({args.n_sigma:.0f}-sigma):    {total_detectable_impacts:.2f}")
    print(f"Target for statistical significance:             {args.target_impacts:.0f}")
    if total_detectable_impacts >= args.target_impacts:
        margin = total_detectable_impacts / args.target_impacts
        print(f"-> Fleet meets the target on DETECTABLE impacts ({margin:.1f}x margin).")
    else:
        shortfall = args.target_impacts / total_detectable_impacts
        print(f"-> Short by {shortfall:.1f}x on detectable impacts. Note: this uses a "
              f"single-measurement noise threshold, likely pessimistic vs. the real "
              f"EKF+smoother pipeline - not necessarily infeasible yet.")


if __name__ == "__main__":
    main()