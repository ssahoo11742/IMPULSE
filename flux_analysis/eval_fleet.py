"""Score a real fetched fleet's aggregate statistical power for debris
detection, using REAL per-orbit flux values from ORDEM and MASTER (via
flux_comparison.csv from flux_parser.py) instead of a single flat rho/v_rel
approximation - each object uses whichever (altitude, inclination) grid
point in the flux table is nearest to its own real orbit.

Both ORDEM's and MASTER's flux are reported side by side, since the two
models are now confirmed (against published literature) to disagree by
~2 orders of magnitude in this size range - there is no single "the" rho
to use, so this reports the bracket rather than picking one.

Detectability (the fraction of impacts big enough to clear measurement
noise) still needs v_rel explicitly (not just flux), since that's what
converts a dv threshold into a fragment-mass threshold - see
_detectable_fraction below. Flux itself is used directly for the raw
impact RATE, replacing the rho*v_rel approximation used previously.

Usage:
    python -m phase2_filter.evaluate_fleet_power \
        --meta real_tles.tle.meta.json \
        --flux-table flux_comparison.csv \
        --durations real_tles.tle.durations.json \
        --target-impacts 30 --n-sigma 3.0
"""
import argparse
import csv
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

ALT_GRID = [700, 800, 900, 1000]
INC_GRID = [51.6, 65, 82, 90, 98]


def _bin_of(inc_deg):
    for lo, hi in zip(INC_BIN_EDGES_DEG[:-1], INC_BIN_EDGES_DEG[1:]):
        if lo <= inc_deg < hi:
            return lo
    return None


def _invert_fragment_mass(m_target, lo=LC_MIN_M, hi=10.0, iters=60):
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


def _detectable_fraction(v_rel, mass_sat, n_mean, sigma_a, n_sigma, dv_threshold_override=None):
    if dv_threshold_override is not None:
        dv_threshold = dv_threshold_override
    else:
        sens = 2.0 / n_mean
        dv_threshold = (n_sigma * sigma_a) / sens
    m_frag_threshold = dv_threshold * mass_sat / v_rel
    Lc_threshold = _invert_fragment_mass(m_frag_threshold)
    if Lc_threshold <= LC_MIN_M:
        return 1.0
    return min(1.0, (LC_MIN_M / Lc_threshold) ** NSBM_POWER_LAW_EXP)


def _load_flux_table(path):
    """Returns {(altitude_km, inclination_deg): (ordem_flux, master_flux)}."""
    table = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            alt = float(row["altitude_km"])
            inc = float(row["inclination_deg"])
            ordem = row.get("ordem_flux_1mm_1cm", "")
            master = row.get("master_flux_1mm_1cm", "")
            ordem_val = float(ordem) if ordem not in ("", "None") else None
            master_val = float(master) if master not in ("", "None") else None
            table[(alt, inc)] = (ordem_val, master_val)
    return table


def _nearest_grid_point(alt_km, inc_deg):
    nearest_alt = min(ALT_GRID, key=lambda x: abs(x - alt_km))
    nearest_inc = min(INC_GRID, key=lambda x: abs(x - inc_deg))
    return nearest_alt, nearest_inc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--flux-table", required=True,
                    help="flux_comparison.csv from flux_parser.py")
    p.add_argument("--area", type=float, default=1.0)
    p.add_argument("--mass", type=float, default=200.0)
    p.add_argument("--duration-days", type=float, default=5475.0,
                    help="Fallback duration for objects with no real duration")
    p.add_argument("--durations", default=None)
    p.add_argument("--target-impacts", type=float, default=30.0)
    p.add_argument("--n-sigma", type=float, default=3.0)
    p.add_argument("--dv-threshold", type=float, default=None,
                    help="Empirically-measured dv50 detection threshold [m/s] from an "
                         "impulse-test sweep - overrides the analytic sigma_a-based estimate")
    p.add_argument("--frag-mc-samples", type=int, default=200000)
    args = p.parse_args()

    with open(args.meta) as f:
        records = json.load(f)

    durations_map = {}
    if args.durations:
        with open(args.durations) as f:
            durations_map = json.load(f)
        print(f"Loaded real durations for {len(durations_map)} objects from {args.durations}")

    flux_table = _load_flux_table(args.flux_table)
    print(f"Loaded flux table with {len(flux_table)} grid points from {args.flux_table}")

    rng = np.random.default_rng(0)
    m_frag_mean = np.mean([fragment_mass(l) for l in
                            sample_fragment_lc(rng, n=args.frag_mc_samples)])

    sigma_a = math.sqrt(R_DIAG_ELEMENTS[0])
    duration_s_fallback = args.duration_days * 86400.0

    bin_counts = defaultdict(int)
    bin_ordem_expected = defaultdict(float)
    bin_ordem_detectable = defaultdict(float)
    bin_master_expected = defaultdict(float)
    bin_master_detectable = defaultdict(float)
    total_ordem_expected = 0.0
    total_ordem_detectable = 0.0
    total_master_expected = 0.0
    total_master_detectable = 0.0
    skipped = 0
    n_real_duration = 0
    n_fallback_duration = 0
    n_no_flux_match = 0

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
            obj_duration_s = duration_s_fallback
            n_fallback_duration += 1
        duration_yr = obj_duration_s / 86400.0 / 365.25

        alt_km = 0.5 * (peri_km + apo_km)
        a = alt_km * 1000.0 + R_EARTH
        n_mean = math.sqrt(MU / a ** 3)
        _, _, flux_sum = compute_vmf_kappa(math.radians(inc_deg))
        v_circ = math.sqrt(MU / a)
        v_rel = flux_sum * v_circ

        grid_alt, grid_inc = _nearest_grid_point(alt_km, inc_deg)
        ordem_flux, master_flux = flux_table.get((grid_alt, grid_inc), (None, None))
        if ordem_flux is None and master_flux is None:
            n_no_flux_match += 1
            continue

        frac_detectable = _detectable_fraction(v_rel, args.mass, n_mean, sigma_a, args.n_sigma, dv_threshold_override=args.dv_threshold)

        b = _bin_of(inc_deg)

        if ordem_flux is not None:
            exp_o = ordem_flux * args.area * duration_yr
            det_o = exp_o * frac_detectable
            total_ordem_expected += exp_o
            total_ordem_detectable += det_o
            if b is not None:
                bin_ordem_expected[b] += exp_o
                bin_ordem_detectable[b] += det_o

        if master_flux is not None:
            exp_m = master_flux * args.area * duration_yr
            det_m = exp_m * frac_detectable
            total_master_expected += exp_m
            total_master_detectable += det_m
            if b is not None:
                bin_master_expected[b] += exp_m
                bin_master_detectable[b] += det_m

        if b is not None:
            bin_counts[b] += 1

    print(f"E[m_frag]={m_frag_mean:.2e} kg   sigma_a={sigma_a:.1f} m   n_sigma={args.n_sigma:.1f}")
    print(f"({skipped} objects skipped - missing fields; "
          f"{n_no_flux_match} skipped - no nearby flux grid point)")
    if durations_map:
        print(f"durations: {n_real_duration} real (from satcat), "
              f"{n_fallback_duration} fallback (--duration-days={args.duration_days:.0f})")
    print()
    print(f"{'inc bin':>12} {'N sats':>7} {'ORDEM exp':>10} {'ORDEM det':>10} "
          f"{'MASTER exp':>11} {'MASTER det':>11}")
    for lo, hi in zip(INC_BIN_EDGES_DEG[:-1], INC_BIN_EDGES_DEG[1:]):
        n = bin_counts.get(lo, 0)
        if n == 0:
            continue
        print(f"{lo:>5}-{hi:<5}deg {n:>7} "
              f"{bin_ordem_expected[lo]:>10.2f} {bin_ordem_detectable[lo]:>10.2f} "
              f"{bin_master_expected[lo]:>11.4f} {bin_master_detectable[lo]:>11.4f}")

    print()
    print(f"TOTAL (ORDEM)  expected impacts: {total_ordem_expected:10.2f}   "
          f"detectable ({args.n_sigma:.0f}-sigma): {total_ordem_detectable:8.2f}")
    print(f"TOTAL (MASTER) expected impacts: {total_master_expected:10.4f}   "
          f"detectable ({args.n_sigma:.0f}-sigma): {total_master_detectable:8.4f}")
    print(f"Target for statistical significance: {args.target_impacts:.0f}")
    print()
    for label, det in [("ORDEM", total_ordem_detectable), ("MASTER", total_master_detectable)]:
        if det >= args.target_impacts:
            print(f"-> {label}-based estimate MEETS the target ({det/args.target_impacts:.1f}x margin).")
        else:
            print(f"-> {label}-based estimate falls SHORT by {args.target_impacts/det:.1f}x." if det > 0
                  else f"-> {label}-based estimate: no detectable impacts.")


if __name__ == "__main__":
    main()