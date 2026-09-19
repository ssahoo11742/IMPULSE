#!/usr/bin/env python3
"""Interpolate ORDEM and MASTER flux from flux_comparison.csv to full fleet.

Reads the 20 discrete (altitude, inclination) points from flux_comparison.csv,
builds 2D interpolators, and assigns per-object flux values based on each
satellite's altitude and inclination from TLE metadata.

Outputs:
  results/ordem_flux_per_object.json
  results/master_flux_per_object.json

Usage:
    python -m phase2_filter.interpolate_flux_to_fleet \
        --flux-csv flux_analysis/flux_comparison.csv \
        --meta TLE/real_tles.tle.meta.json \
        --durations TLE/real_tles.tle.durations.json \
        --output-ordem results/ordem_flux_per_object.json \
        --output-master results/master_flux_per_object.json
"""
import argparse
import json
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from TLE.tle_io import parse_tle, load_tles_from_file


def load_objects_with_altitude(tle_file: str, meta_file: str, durations_file: str):
    """Load objects with computed altitude from TLE."""
    tle_tuples = load_tles_from_file(tle_file)
    with open(meta_file) as f:
        meta_records = json.load(f)
    tle_map = {}
    for name, line1, line2 in tle_tuples:
        tle_dict = parse_tle(name, line1, line2)
        tle_map[tle_dict.get("norad_id", "unknown")] = tle_dict
    with open(durations_file) as f:
        durations_map = json.load(f)

    objects = []
    for rec in meta_records:
        norad_id = str(rec.get("NORAD_CAT_ID", ""))
        if norad_id not in tle_map or norad_id not in durations_map:
            continue
        tle_data = tle_map[norad_id]
        duration_days = durations_map[norad_id].get("duration_days", 0)

        # Compute altitude from semi-major axis
        a = tle_data.get("a", 0)
        # a is in meters from parse_tle
        altitude_km = (a - 6371000.0) / 1000.0 if a > 6371000 else 0.0
        inclination_deg = float(rec.get("INCLINATION", 0))

        objects.append({
            "norad_id": norad_id,
            "altitude_km": altitude_km,
            "inclination_deg": inclination_deg,
            "duration_days": duration_days,
            "tle_data": tle_data,
        })
    return objects


def build_interpolator(flux_csv: str, model: str = "ordem"):
    """Build 2D interpolator from flux_comparison.csv.

    model: 'ordem' or 'master'
    """
    import csv

    alts = []
    incs = []
    fluxes = []

    with open(flux_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            alt = float(row["altitude_km"])
            inc = float(row["inclination_deg"])
            if model == "ordem":
                flux = float(row["ordem_flux_1mm_1cm"])
            else:
                flux_val = row.get("master_flux_1mm_1cm", "").strip()
                if not flux_val:
                    continue
                flux = float(flux_val)

            alts.append(alt)
            incs.append(inc)
            fluxes.append(flux)

    alts = np.array(alts)
    incs = np.array(incs)
    fluxes = np.array(fluxes)

    # Get unique sorted grid points
    alt_grid = np.unique(alts)
    inc_grid = np.unique(incs)

    # Build flux matrix
    flux_grid = np.zeros((len(alt_grid), len(inc_grid)))
    for i, alt in enumerate(alt_grid):
        for j, inc in enumerate(inc_grid):
            mask = (alts == alt) & (incs == inc)
            if mask.any():
                flux_grid[i, j] = fluxes[mask][0]

    interpolator = RegularGridInterpolator(
        (alt_grid, inc_grid),
        flux_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,  # extrapolates with nearest
    )

    return interpolator, alt_grid, inc_grid


def main():
    parser = argparse.ArgumentParser(description="Interpolate flux to fleet")
    parser.add_argument("--flux-csv", default="flux_analysis/flux_comparison.csv")
    parser.add_argument("--tle-file", default="TLE/real_tles.tle")
    parser.add_argument("--meta", default="TLE/real_tles.tle.meta.json")
    parser.add_argument("--durations", default="TLE/real_tles.tle.durations.json")
    parser.add_argument("--output-ordem", default="results/ordem_flux_per_object.json")
    parser.add_argument("--output-master", default="results/master_flux_per_object.json")
    args = parser.parse_args()

    print("Loading fleet...")
    objects = load_objects_with_altitude(args.tle_file, args.meta, args.durations)
    print(f"Loaded {len(objects)} objects")

    # Build interpolators
    print("Building ORDEM interpolator...")
    ordem_interp, alt_g, inc_g = build_interpolator(args.flux_csv, "ordem")
    print(f"  Grid: altitudes {alt_g.tolist()}, inclinations {inc_g.tolist()}")

    print("Building MASTER interpolator...")
    master_interp, _, _ = build_interpolator(args.flux_csv, "master")

    # Interpolate
    ordem_results = {}
    master_results = {}

    alt_min, alt_max = alt_g.min(), alt_g.max()
    inc_min, inc_max = inc_g.min(), inc_g.max()

    n_outside_alt = 0
    n_outside_inc = 0

    for obj in objects:
        nid = obj["norad_id"]
        alt = obj["altitude_km"]
        inc = obj["inclination_deg"]

        # Clamp to interpolation domain for sanity
        if alt < alt_min or alt > alt_max:
            n_outside_alt += 1
        if inc < inc_min or inc > inc_max:
            n_outside_inc += 1

        point = np.array([[alt, inc]])
        ordem_flux = float(ordem_interp(point)[0])
        master_flux = float(master_interp(point)[0])

        ordem_results[nid] = {
            "flux_impacts_per_m2_per_year": ordem_flux,
            "altitude_km": alt,
            "inclination_deg": inc,
            "duration_days": obj["duration_days"],
        }
        master_results[nid] = {
            "flux_impacts_per_m2_per_year": master_flux,
            "altitude_km": alt,
            "inclination_deg": inc,
            "duration_days": obj["duration_days"],
        }

    if n_outside_alt:
        print(f"  Warning: {n_outside_alt} objects outside altitude grid (extrapolated)")
    if n_outside_inc:
        print(f"  Warning: {n_outside_inc} objects outside inclination grid (extrapolated)")

    # Save
    import os
    os.makedirs("results", exist_ok=True)
    with open(args.output_ordem, "w") as f:
        json.dump(ordem_results, f, indent=2)
    with open(args.output_master, "w") as f:
        json.dump(master_results, f, indent=2)

    print(f"\nSaved ORDEM flux for {len(ordem_results)} objects to {args.output_ordem}")
    print(f"Saved MASTER flux for {len(master_results)} objects to {args.output_master}")

    # Summary stats
    ordem_vals = [v["flux_impacts_per_m2_per_year"] for v in ordem_results.values()]
    master_vals = [v["flux_impacts_per_m2_per_year"] for v in master_results.values()]
    print(f"\nORDEM flux:  mean={np.mean(ordem_vals):.4f}, median={np.median(ordem_vals):.4f}")
    print(f"MASTER flux: mean={np.mean(master_vals):.4f}, median={np.median(master_vals):.4f}")
    print(f"Ratio (ORDEM/MASTER): {np.mean(ordem_vals)/np.mean(master_vals):.1f}x")


if __name__ == "__main__":
    main()