#!/usr/bin/env python3
"""Compute expected detected impacts for ORDEM and MASTER using empirical ε(Δv).

Usage:
    python -m phase2_filter.eval_fleet_efficiency \
        --efficiency results/gate3_efficiency.json \
        --ordem-flux results/ordem_flux_per_object.json \
        --master-flux results/master_flux_per_object.json \
        --sat-mass 200.0 --sat-area 1.0 \
        --n-samples 100000 \
        --output results/fleet_expected_detected.json
"""
import argparse
import json
import math
import os
from typing import Dict, Optional

import numpy as np

from constants.constants import MU, DEBRIS_INC_POP
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
            eps.append(info["epsilon"])

    dvs = np.array(dvs)
    eps = np.array(eps)
    sort_idx = np.argsort(dvs)
    dvs, eps = dvs[sort_idx], eps[sort_idx]
    eps = np.clip(eps, 0.0, 1.0)

    def epsilon(dv):
        if dv <= dvs[0]:
            return float(eps[0])
        if dv >= dvs[-1]:
            return float(eps[-1])
        return float(np.interp(dv, dvs, eps))

    return epsilon, dvs, eps


def compute_effective_efficiency(
    sat_inc_deg: float,
    sat_mass: float,
    sat_area: float,
    v_circ: float,
    epsilon_func,
    n_samples: int = 100000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Monte Carlo ε_effective from NSBM fragment population."""
    if rng is None:
        rng = np.random.default_rng()

    sat_inc = math.radians(sat_inc_deg)
    kappa, _, _ = compute_vmf_kappa(sat_inc)

    dvs, effs = [], []
    for _ in range(n_samples):
        Lc = sample_fragment_lc(rng)[0]
        m_frag = fragment_mass(Lc)
        v_rel = sample_impact_velocity(sat_inc, v_circ, rng)
        dv_scalar = (m_frag / sat_mass) * v_rel
        effs.append(epsilon_func(dv_scalar))
        dvs.append(dv_scalar)

    dvs = np.array(dvs)
    effs = np.array(effs)

    return {
        "epsilon_effective": float(np.mean(effs)),
        "epsilon_std": float(np.std(effs) / math.sqrt(n_samples)),
        "mean_dv": float(np.mean(dvs)),
        "median_dv": float(np.median(dvs)),
        "p95_dv": float(np.percentile(dvs, 95)),
        "p99_dv": float(np.percentile(dvs, 99)),
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
    args = parser.parse_args()

    epsilon_func, dv_grid, eps_grid = load_efficiency_curve(args.efficiency)
    print(f"Loaded ε(Δv): {len(dv_grid)} points, DV_50 ≈ {np.interp(0.5, eps_grid, dv_grid):.5f} m/s")

    with open(args.ordem_flux) as f:
        ordem_data = json.load(f)
    with open(args.master_flux) as f:
        master_data = json.load(f)

    rng = np.random.default_rng(args.seed)

    common_ids = sorted(set(ordem_data.keys()) & set(master_data.keys()))
    print(f"Processing {len(common_ids)} objects with both ORDEM and MASTER flux...")

    results = []
    totals = {
        "ordem": {"expected": 0.0, "detected": 0.0},
        "master": {"expected": 0.0, "detected": 0.0},
    }

    for idx, nid in enumerate(common_ids):
        o_entry = ordem_data[nid]
        m_entry = master_data[nid]
        dur_yr = o_entry.get("duration_days", 0) / 365.25
        if dur_yr <= 0:
            continue

        sat_inc = o_entry.get("inclination_deg", 0)
        alt_km = o_entry.get("altitude_km", 700)
        v_circ = math.sqrt(MU / ((alt_km + 6371.0) * 1000.0)) if alt_km > 0 else 7500.0

        eff_info = compute_effective_efficiency(
            sat_inc, args.sat_mass, args.sat_area, v_circ,
            epsilon_func, n_samples=args.n_samples, rng=rng
        )
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
            print(f"  [{idx+1}/{len(common_ids)}] ORDEM det: {totals['ordem']['detected']:.3f}, MASTER det: {totals['master']['detected']:.3f}")

    print(f"\n{'='*60}")
    print("FLEET EXPECTED DETECTED IMPACTS")
    print(f"{'='*60}")

    for model in ["ordem", "master"]:
        t = totals[model]
        print(f"\n{model.upper()}:")
        print(f"  Total expected impacts:  {t['expected']:.4f}")
        print(f"  Total detected:          {t['detected']:.4f}")
        print(f"  Effective efficiency:    {t['detected']/t['expected']:.4f}" if t["expected"] > 0 else "  N/A")

    output = {
        "config": {
            "efficiency_file": args.efficiency,
            "ordem_flux_file": args.ordem_flux,
            "master_flux_file": args.master_flux,
            "n_objects": len(results),
            "n_samples": args.n_samples,
            "sat_mass": args.sat_mass,
            "sat_area": args.sat_area,
        },
        "summary": {
            "ordem": totals["ordem"],
            "master": totals["master"],
            "ratio_ordem_to_master": totals["ordem"]["detected"] / totals["master"]["detected"]
                if totals["master"]["detected"] > 0 else None,
        },
        "per_object": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()