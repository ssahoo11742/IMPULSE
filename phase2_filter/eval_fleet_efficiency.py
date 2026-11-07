#!/usr/bin/env python3
"""Compute expected detected impacts for ORDEM and MASTER using empirical ε(Δv).

PATCHED FOR WINDOW-BASED EXPOSURE.
The stock script used `duration_days` from the flux per-object JSON, which is
each object's full TLE span (launch/first-TLE to catalog epoch) — roughly an
order of magnitude more time than the fleet actually searched (the accepted,
edge-trimmed 90-day windows). Predicted counts must share a denominator with
the observed trigger count (6 triggers over 15,760 valid windows ≈ 3,279
object-years), so this version computes exposure as

    T_j = n_valid_windows(object) × --window-days / 365.25

with --window-days defaulting to 76 (the edge-trimmed usable detection time;
use 90 for the full-window convention). Everything else is unchanged.

Usage (1--10 cm band):
    python -m phase2_filter.eval_fleet_efficiency \
        --efficiency results/gate3_efficiency.json \
        --ordem-flux results/ordem_flux_per_object.json \
        --master-flux results/master_flux_per_object.json \
        --windows-file results/window_counts.json \
        --window-days 76 \
        --sat-mass 200.0 --sat-area 1.0 --n-samples 100000 \
        --lc-min 0.01 --lc-max 0.10 \
        --output results/fleet_expected_detected_1to10cm.json

mm--cm band: same, with --lc-min 0.001 --lc-max 0.01 and
    --output results/fleet_expected_detected_eta417.json

Testing options (run on a subset instead of all 3018 objects):
    --max-objects 100      # first 100 objects only
    --stride 30            # every 30th object (~100 objects total)

The --windows-file JSON may be any of:
    {"33xxx": 4, ...}                                  # id -> count
    {"33xxx": {"n_windows": 4}, ...}                   # id -> dict
    [{"norad_id": "33xxx", "n_windows": 4}, ...]       # list of records
Accepted count keys: n_windows, n_valid_windows, valid_windows, windows, count.
"""
import argparse
import json
import math
import os
from typing import Dict, Optional

import numpy as np

from constants.constants import MU, DEBRIS_INC_POP, NSBM_POWER_LAW_EXP
from propagator.debris_impacts import (
    sample_fragment_lc, fragment_mass, sample_impact_velocity,
    compute_vmf_kappa, sample_vmf_direction,
)


def load_efficiency_curve(efficiency_file: str):
    """Load ε(Δv) and build interpolator."""
    with open(efficiency_file) as f:
        data = json.load(f)

    dvs, eps = [], []
    for dv_str, info in sorted(data.get("efficiency", {}).items(), key=lambda x: float(x[0])):
        if info.get("epsilon") is not None and info.get("n", 0) > 0:
            dvs.append(float(dv_str))
            eps.append(float(info["epsilon"]))

    dvs = np.array(dvs, dtype=float)
    eps = np.clip(np.array(eps, dtype=float), 0.0, 1.0)
    sort_idx = np.argsort(dvs)
    dvs, eps = dvs[sort_idx], eps[sort_idx]

    def epsilon(dv):
        if dv <= dvs[0]:
            return float(eps[0])
        if dv >= dvs[-1]:
            return float(eps[-1])
        return float(np.interp(dv, dvs, eps))

    return epsilon, dvs, eps


def find_dv50(dvs, eps):
    """Smallest dv where epsilon crosses 0.5 (eps must be non-decreasing)."""
    eps_sorted = eps[np.argsort(dvs)]
    dvs_sorted = dvs[np.argsort(dvs)]
    idx = np.searchsorted(eps_sorted, 0.5)
    if idx >= len(dvs_sorted):
        return float("nan")
    return float(dvs_sorted[idx])


def load_window_counts(path: str) -> Dict[str, int]:
    """Per-object valid-window counts from the fleet run.

    Tolerates the common output shapes; normalizes NORAD ids to strings.
    """
    with open(path) as f:
        data = json.load(f)

    count_keys = ("n_windows", "n_valid_windows", "valid_windows",
                  "windows", "count")
    counts: Dict[str, int] = {}

    if isinstance(data, dict):
        # Shape A: {"33xxx": 4, ...}  or  {"33xxx": {"n_windows": 4}, ...}
        for k, v in data.items():
            if isinstance(v, dict):
                for key in count_keys:
                    if key in v:
                        counts[str(k)] = int(v[key])
                        break
            elif isinstance(v, (int, float)):
                counts[str(k)] = int(v)
    elif isinstance(data, list):
        # Shape B: [{"norad_id": "33xxx", "n_windows": 4}, ...]
        for v in data:
            if not isinstance(v, dict):
                continue
            nid = v.get("norad_id", v.get("norad", v.get("id")))
            if nid is None:
                continue
            for key in count_keys:
                if key in v:
                    counts[str(nid)] = int(v[key])
                    break
    else:
        raise ValueError(f"Unrecognized windows-file structure in {path}; "
                         f"top-level type: {type(data).__name__}")

    if not counts:
        raise ValueError(f"No per-object window counts parsed from {path}; "
                         f"top-level keys: "
                         f"{list(data.keys())[:10] if isinstance(data, dict) else 'list'}")
    return counts


def sample_band_lc(rng, lc_min, lc_max, n=1):
    """Sample Lc within [lc_min, lc_max] under the NSBM cumulative power
    law N(>Lc) ~ Lc^-NSBM_POWER_LAW_EXP (same exponent eval_fleet.py uses)."""
    lo, hi = float(lc_min), float(lc_max)
    chi = float(NSBM_POWER_LAW_EXP)
    u = rng.random(n)
    return (lo ** (-chi) - u * (lo ** (-chi) - hi ** (-chi))) ** (-1.0 / chi)


def _fragment_masses(Lc):
    """Vectorized fragment_mass: mass (kg) from characteristic length (m)."""
    Lc = np.asarray(Lc, dtype=float)
    boundary = 0.067  # NSBM_AM_REGIME_BOUNDARY_M -- import it instead if exported
    log_am = np.where(
        Lc < boundary,
        -0.6 * np.log10(Lc) - 1.34,   # placeholder small regime
        -0.45 * np.log10(Lc) - 1.50,  # placeholder large regime
    )
    area = np.pi * (Lc / 2.0) ** 2
    return area / (10.0 ** log_am)


def compute_effective_efficiency(
    sat_inc_deg: float,
    sat_mass: float,
    sat_area: float,
    v_circ: float,
    epsilon_func,
    n_samples: int = 100000,
    rng: Optional[np.random.Generator] = None,
    lc_min=None,
    lc_max=None,
) -> Dict:
    """Monte Carlo ε_effective from NSBM fragment population (vectorized)."""
    if rng is None:
        rng = np.random.default_rng()

    sat_inc = math.radians(sat_inc_deg)
    kappa, _, _ = compute_vmf_kappa(sat_inc)

    # --- fragment sizes (vectorized) ---
    if lc_min is not None and lc_max is not None:
        Lc = sample_band_lc(rng, lc_min, lc_max, n_samples)
    else:
        Lc = np.array([sample_fragment_lc(rng, n_samples)[0]])  # fallback, slow path
    m_frag = _fragment_masses(Lc)

    # --- relative velocities (vectorized flux-weighted choice) ---
    incs = [d[0] for d in DEBRIS_INC_POP]
    fracs = np.array([d[1] for d in DEBRIS_INC_POP])
    from propagator.debris_impacts import _rel_velocity_and_angle
    v_rels = np.array([
        _rel_velocity_and_angle(sat_inc, math.radians(i), v_circ)[0] for i in incs
    ])
    weights = fracs * v_rels
    weights = weights / weights.sum()
    idx = rng.choice(len(incs), size=n_samples, p=weights)
    v_rel = v_rels[idx]

    dvs = (m_frag / sat_mass) * v_rel
    effs = np.array([epsilon_func(d) for d in dvs])

    return {
        "epsilon_effective": float(np.mean(effs)),
        "epsilon_std": float(np.std(effs) / math.sqrt(n_samples)),
        "mean_dv": float(np.mean(dvs)),
        "median_dv": float(np.median(dvs)),
        "p95_dv": float(np.percentile(dvs, 95)),
        "p99_dv": float(np.percentile(dvs, 99)),
        "frac_dv_over_1ms": float(np.mean(dvs > 1.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Fleet expected detected impacts")
    parser.add_argument("--efficiency", required=True, help="gate3_efficiency.json")
    parser.add_argument("--ordem-flux", required=True, help="ordem_flux_per_object.json")
    parser.add_argument("--master-flux", required=True, help="master_flux_per_object.json")
    parser.add_argument("--sat-mass", type=float, default=200.0)
    parser.add_argument("--sat-area", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="results/fleet_expected_detected.json")
    parser.add_argument("--lc-min", type=float, default=None)
    parser.add_argument("--lc-max", type=float, default=None)
    # --- exposure patch ---
    parser.add_argument("--windows-file", default=None,
                        help="Per-object valid-window counts from the fleet run. "
                             "If omitted, falls back to the legacy span-based "
                             "duration_days (not comparable with the fleet trigger "
                             "statistics).")
    parser.add_argument("--window-days", type=float, default=76.0,
                        help="Searched days per window: 76 = edge-trimmed usable "
                             "detection time (paper convention), 90 = full window")
    # --- subset testing options ---
    parser.add_argument("--max-objects", type=int, default=None,
                        help="Only process the first N objects (for quick tests)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth object (default 1 = all)")
    args = parser.parse_args()

    epsilon_func, dv_grid, eps_grid = load_efficiency_curve(args.efficiency)
    dv50 = find_dv50(dv_grid, eps_grid)
    print(f"Loaded ε(Δv): {len(dv_grid)} points, DV_50 ≈ {dv50:.5f} m/s")

    # DIAGNOSTIC: what does ε look like across the dv range fragments actually produce?
    print("ε(Δv) diagnostic (this explains zero detections if the low end is 0):")
    for probe in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0]:
        print(f"  ε({probe:>8.4f} m/s) = {epsilon_func(probe):.4f}")

    # --- exposure source ---
    window_counts = None
    if args.windows_file:
        window_counts = load_window_counts(args.windows_file)
        exposure_source = (f"window-based: n_windows × {args.window_days:g} days "
                           f"({len(window_counts)} objects in windows file)")
        print(f"\nExposure convention: {exposure_source}")
    else:
        exposure_source = ("LEGACY span-based duration_days from the flux JSON -- "
                          "NOT comparable with the fleet trigger statistics")
        print(f"\nWARNING: --windows-file not given; falling back to {exposure_source}")

    with open(args.ordem_flux) as f:
        ordem_data = json.load(f)
    with open(args.master_flux) as f:
        master_data = json.load(f)

    rng = np.random.default_rng(args.seed)

    common_ids = sorted(set(ordem_data.keys()) & set(master_data.keys()))
    if args.stride > 1:
        common_ids = common_ids[:: args.stride]
    if args.max_objects is not None:
        common_ids = common_ids[: args.max_objects]

    n_missing_windows = 0
    if window_counts is not None:
        matched = [nid for nid in common_ids if str(nid) in window_counts]
        n_missing_windows = len(common_ids) - len(matched)
        common_ids = matched
        if n_missing_windows:
            print(f"NOTE: {n_missing_windows} flux-matched objects have no window "
                  f"counts and contribute zero exposure.")
    print(f"Processing {len(common_ids)} objects with both ORDEM and MASTER flux...")

    results = []
    totals = {
        "ordem": {"expected": 0.0, "detected": 0.0},
        "master": {"expected": 0.0, "detected": 0.0},
    }
    eff_cache = {}  # (inc, alt) -> eff_info: MC reused for identical geometry

    for idx, nid in enumerate(common_ids):
        o_entry = ordem_data[nid]
        m_entry = master_data[nid]

        # --- exposure: window-based if counts given, else legacy span ---
        if window_counts is not None:
            n_win = window_counts.get(str(nid), 0)
            dur_yr = n_win * args.window_days / 365.25
        else:
            n_win = None
            dur_yr = o_entry.get("duration_days", 0) / 365.25
        if dur_yr <= 0:
            continue

        sat_inc = o_entry.get("inclination_deg", 0)
        alt_km = o_entry.get("altitude_km", 700)
        v_circ = math.sqrt(MU / ((alt_km + 6371.0) * 1000.0)) if alt_km > 0 else 7500.0

        cache_key = (round(sat_inc, 4), round(alt_km, 2))
        if cache_key in eff_cache:
            eff_info = eff_cache[cache_key]
        else:
            eff_info = compute_effective_efficiency(
                sat_inc, args.sat_mass, args.sat_area, v_circ,
                epsilon_func, n_samples=args.n_samples, rng=rng,
                lc_min=args.lc_min, lc_max=args.lc_max
            )
            eff_cache[cache_key] = eff_info

        eps_eff = eff_info["epsilon_effective"]

        o_flux = o_entry.get("flux_impacts_per_m2_per_year", 0.0)
        o_exp = o_flux * args.sat_area * dur_yr
        o_det = o_exp * eps_eff
        totals["ordem"]["expected"] += o_exp
        totals["ordem"]["detected"] += o_det

        m_flux = m_entry.get("flux_impacts_per_m2_per_year", 0.0)
        m_exp = m_flux * args.sat_area * dur_yr
        m_det = m_exp * eps_eff
        totals["master"]["expected"] += m_exp
        totals["master"]["detected"] += m_det

        results.append({
            "norad_id": nid,
            "n_windows": n_win,
            "duration_yr": dur_yr,
            "inclination_deg": sat_inc,
            "altitude_km": alt_km,
            "epsilon_effective": eps_eff,
            "ordem_expected": o_exp,
            "ordem_detected": o_det,
            "master_expected": m_exp,
            "master_detected": m_det,
        })

        if (idx + 1) % max(1, len(common_ids) // 10) == 0:
            print(f"  [{idx+1}/{len(common_ids)}] ORDEM det: {totals['ordem']['detected']:.6e}, "
                  f"MASTER det: {totals['master']['detected']:.6e}")

    print(f"\n{'='*60}")
    print("FLEET EXPECTED DETECTED IMPACTS")
    print(f"{'='*60}")

    total_exposure_yr = sum(r["duration_yr"] for r in results)
    reference_exposure_yr = 15760 * args.window_days / 365.25
    print(f"\nExposure convention: {exposure_source}")
    print(f"Total exposure used: {total_exposure_yr:.1f} object-years "
          f"({len(results)} objects)")
    if window_counts is not None:
        print(f"Full-fleet window-based reference: {reference_exposure_yr:.1f} "
              f"object-years (15,760 valid windows × {args.window_days:g} days); "
              f"the total is smaller because only flux-matched objects count.")

    for model in ["ordem", "master"]:
        t = totals[model]
        print(f"\n{model.upper()}:")
        print(f"  Total expected impacts:  {t['expected']:.6e}")
        print(f"  Total detected:          {t['detected']:.6e}")
        print(f"  Effective efficiency:    {t['detected']/t['expected']:.6f}"
              if t["expected"] > 0 else "  N/A")

    if results:
        effs = np.array([r["epsilon_effective"] for r in results])
        print(f"\nε_effective across objects: min={effs.min():.6f}, "
              f"mean={effs.mean():.6f}, max={effs.max():.6f}")
        if effs.max() < 1e-6:
            print("WARNING: ε_effective is ~0 for every object. Your fragment Δv "
                  "distribution sits entirely below the efficiency curve's "
                  "nonzero range (check DV_50 vs median_dv/p99_dv).")

    output = {
        "config": {
            "efficiency_file": args.efficiency,
            "ordem_flux_file": args.ordem_flux,
            "master_flux_file": args.master_flux,
            "windows_file": args.windows_file,
            "window_days": args.window_days,
            "exposure_convention": exposure_source,
            "total_exposure_object_years": total_exposure_yr,
            "n_objects": len(results),
            "n_flux_matched_without_windows": n_missing_windows,
            "n_samples": args.n_samples,
            "sat_mass": args.sat_mass,
            "sat_area": args.sat_area,
            "lc_min": args.lc_min,
            "lc_max": args.lc_max,
            "max_objects": args.max_objects,
            "stride": args.stride,
        },
        "summary": {
            "ordem": totals["ordem"],
            "master": totals["master"],
            "ratio_ordem_to_master": totals["ordem"]["detected"] / totals["master"]["detected"]
                if totals["master"]["detected"] > 0 else None,
        },
        "per_object": results,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()