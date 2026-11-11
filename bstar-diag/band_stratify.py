#!/usr/bin/env python3
"""FIX C diagnostic + toolkit: altitude-band stratified eta (V2.1).

Answers the Fix C decision questions OFFLINE from saved raw peaks. No
reruns: the null records per-trial geometry (FIX #14), and the fleet and
gate files store norad_id per object, joined to altitude via the repo's
own TLE loader (TLE.tle_io, the same code path gate 3 uses).

Sections (each runs only if its input is given):

  1. NULL (--null, required): per-band percentile ladders of the V2.1
     (Fix A) peak distribution, per-band eta candidates at --percentile
     with the support check, and a fine-binned knee curve (p50/p99/p99.5
     vs altitude) testing whether the 875-905 km knee survives the Fix A
     cleaning. Optional --null-v2 overlays the V2 (Fix B) curve to show
     how much of the band structure was artifact.

  2. FLEET (--fleet + --tle-file): per-band fleet rates at the global eta
     and at the per-band eta candidates (the stratified detection rule:
     a window counts when peak > eta_band(object altitude)); per-band
     excess over the null's 0.1% design rate = the per-band systematic
     floor. This is the stratified rate-vs-eta measurement.

  3. GATE (--gate3 + --tle-file): per-band epsilon(dv) and DV_50 from the
     re-thresholded gate 3 fine file, at the global eta and at the
     per-band eta candidates.

SUPPORT ARITHMETIC (the Fix C constraint): a per-band eta at percentile p
needs n_band * (1 - p/100) >= 5 expected exceedances. The 17k null spread
over 6 default bands gives ~1,700-3,400 per band, so p99.9 per band is
MARGINAL. The script prints the support verdict and the total run size
each band would need; see the canvas for the three adoption options.

Usage:
  python -m bstar-diag.band_stratify --null results/synthetic_null_eta_honestR_v2_warm_start_fixA.json --output results/band_stratify_null.json

  python -m bstar-diag.band_stratify --null results/synthetic_null_eta_honestR_v2_warm_start_fixA.json --null-v2 results/synthetic_null_eta_honestR_v2fixB.json --fleet results/real_detections_honestR_eta13.json --gate3 results/gate3_efficiency_v2fixA.json --tle-file TLE/real_tles.tle --output results/band_stratify_full.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np

from constants.constants import R_EARTH
from TLE.tle_io import load_tles_from_file, parse_tle, tle_to_elements

PCT_LADDER = [50, 90, 99, 99.5, 99.9, 99.95, 99.97, 99.99]
DESIGN_RATE = 0.001   # the 0.1% p99.9 design false-positive rate


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_null(path):
    """Return (list of (alt_km, peak), eta_global) from an eta-null JSON."""
    with open(path) as f:
        d = json.load(f)
    pts = []
    for t in d.get("trials", []):
        if not (isinstance(t, dict) and t.get("ok")):
            continue
        g = t.get("geometry") or {}
        a0 = g.get("a0_m")
        pk = t.get("peak_maha")
        if a0 is None or pk is None:
            continue
        pts.append((a0 / 1000.0 - R_EARTH / 1000.0, float(pk)))
    eta = (d.get("summary") or {}).get("eta_synth")
    return pts, (float(eta) if eta is not None else None)


def parse_tle_altitudes(path):
    """{norad_str: alt_km} via the repo's own TLE loader.

    Uses TLE.tle_io (the exact code path gate 3 uses to build its TLE
    map), so the altitude join can never drift from the pipeline's
    parser. load_tles_from_file handles both 2-line and 3-line files;
    parse_tle derives SMA from mean motion; MeanElements.alt_m() gives
    the mean altitude in metres. Keys are normalized with str(int(...))
    to match how the fleet/gate files store norad_id.
    """
    out = {}
    for name, l1, l2 in load_tles_from_file(path):
        try:
            t = parse_tle(name, l1, l2)
            norad = str(int(t["norad_id"]))
            out[norad] = tle_to_elements(t).alt_m() / 1000.0
        except Exception:
            continue
    return out


def band_index(alt, edges):
    """0-based band index, or None if outside [edges[0], edges[-1])."""
    if alt is None or alt < edges[0] or alt >= edges[-1]:
        return None
    return int(np.searchsorted(edges, alt, side="right") - 1)


def band_label(edges, i):
    return f"{edges[i]:.0f}-{edges[i + 1]:.0f}"


def percentile_ladder(vals):
    a = np.asarray(vals, float)
    return {str(p): float(np.percentile(a, p)) for p in PCT_LADDER}


def dv50_from_eff(eff_by_dv, level=0.5):
    """DV at which epsilon crosses `level` (rethreshold-tool logic)."""
    pts = [(float(dv), e["eps"]) for dv, e in eff_by_dv.items()
           if e.get("eps") is not None]
    if len(pts) < 2:
        return None
    eps_v = np.array([p[1] for p in pts])
    dvs_v = np.array([p[0] for p in pts])
    order = np.argsort(eps_v)
    eps_v, dvs_v = eps_v[order], dvs_v[order]
    if eps_v.min() <= level <= eps_v.max():
        return float(np.interp(level, eps_v, dvs_v))
    return None


# ---------------------------------------------------------------------------
# Section 1: null per-band structure
# ---------------------------------------------------------------------------

def null_section(pts, eta_global, edges, fine_bin, pct, pts_v2=None):
    bands = {}
    for alt, pk in pts:
        i = band_index(alt, edges)
        if i is None:
            continue
        bands.setdefault(i, []).append(pk)

    print("\n=== NULL: per-band structure (V2.1 / Fix A) ===")
    print(f"{'band(km)':>10} {'n':>6} {'p50':>7} {'p90':>7} {'p99':>7} "
          f"{'p99.5':>7} {'p99.9':>8}  {'eta_band':>9}  support")
    band_out = {}
    for i in sorted(bands):
        v = bands[i]
        n = len(v)
        lad = percentile_ladder(v)
        exp_exceed = n * (1.0 - pct / 100.0)
        eta_band = float(np.percentile(v, pct))
        if exp_exceed >= 5.0:
            support = "OK"
        elif exp_exceed >= 2.0:
            support = f"MARGINAL ({exp_exceed:.1f} exp.)"
        else:
            support = f"NOT SUPPORTED ({exp_exceed:.1f} exp.)"
        n_needed = int(math.ceil(5.0 / (1.0 - pct / 100.0)))
        run_needed = int(math.ceil(n_needed * (edges[-1] - edges[0]) /
                                   (edges[i + 1] - edges[i])))
        print(f"{band_label(edges, i):>10} {n:>6} "
              f"{lad['50']:>7.2f} {lad['90']:>7.2f} {lad['99']:>7.2f} "
              f"{lad['99.5']:>7.2f} {lad['99.9']:>8.2f}  "
              f"{eta_band:>9.3f}  {support}  (run ~{run_needed} for support)")
        band_out[band_label(edges, i)] = {
            "n": n, "ladder": lad, "eta_band": eta_band,
            "expected_exceedances": exp_exceed,
            "run_size_for_support": run_needed}

    # fine-binned knee curve
    print(f"\n  knee curve (fine bins of {fine_bin:.0f} km; p99.9 shown "
          f"only where n >= 5000):")
    print(f"  {'alt bin':>12} {'n':>6} {'p50':>7} {'p99':>7} {'p99.5':>7} "
          f"{'p99.9':>8}")
    bins_lo = np.arange(edges[0], edges[-1], fine_bin)
    fine_out = []
    for lo in bins_lo:
        hi = lo + fine_bin
        v = [pk for alt, pk in pts if lo <= alt < hi]
        if not v:
            continue
        lad = percentile_ladder(v)
        p999 = lad["99.9"] if len(v) >= 5000 else None
        print(f"  {lo:>5.0f}-{hi:<5.0f} {len(v):>6} {lad['50']:>7.2f} "
              f"{lad['99']:>7.2f} {lad['99.5']:>7.2f} "
              f"{(f'{p999:>8.2f}' if p999 is not None else '       -')}")
        fine_out.append({"lo": float(lo), "hi": float(hi), "n": len(v),
                         "ladder": lad, "p99_9": p999})

    # optional V2 overlay on the knee curve
    if pts_v2 is not None:
        print("\n  V2 (Fix B) overlay, same bins (how much band structure "
              "was artifact):")
        print(f"  {'alt bin':>12} {'n':>6} {'p50':>7} {'p99':>7} {'p99.9':>8}")
        for lo in bins_lo:
            hi = lo + fine_bin
            v = [pk for alt, pk in pts_v2 if lo <= alt < hi]
            if not v:
                continue
            lad = percentile_ladder(v)
            print(f"  {lo:>5.0f}-{hi:<5.0f} {len(v):>6} {lad['50']:>7.2f} "
                  f"{lad['99']:>7.2f} {lad['99.9']:>8.2f}")

    if eta_global is not None:
        print(f"\n  global eta (from null summary): {eta_global:.4f}")
    return {"bands": band_out, "fine_bins": fine_out,
            "eta_global": eta_global}


# ---------------------------------------------------------------------------
# Section 2: fleet per-band rates (stratified detection rule)
# ---------------------------------------------------------------------------

def fleet_section(path, alt_by_norad, edges, eta_global, eta_by_band):
    with open(path) as f:
        d = json.load(f)
    banded = []
    skipped = 0
    for r in d.get("per_object", []):
        if not (r.get("success") and r.get("peaks")):
            continue
        try:
            nid = str(int(r["norad_id"]))
        except Exception:
            nid = str(r.get("norad_id"))
        alt = alt_by_norad.get(nid)
        if alt is None:
            skipped += 1
            continue
        banded.append((alt, r["peaks"]))

    groups = {}
    for alt, peaks in banded:
        i = band_index(alt, edges)
        if i is None:
            continue
        groups.setdefault(i, []).extend(peaks)

    print("\n=== FLEET: per-band rates (stratified detection rule) ===")
    print(f"{'band(km)':>10} {'n_peak':>7} {'>eta_g':>7} {'rate%':>7} "
          f"{'>eta_band':>9} {'rate%':>7} {'excess':>8}")
    tot_peak = tot_g = tot_b = 0
    out = {}
    for i in sorted(groups):
        pk = np.array(groups[i], float)
        n = len(pk)
        c_g = int((pk > eta_global).sum())
        eta_b = eta_by_band.get(band_label(edges, i), eta_global)
        c_b = int((pk > eta_b).sum())
        excess = c_b - DESIGN_RATE * n
        print(f"{band_label(edges, i):>10} {n:>7} {c_g:>7} "
              f"{100.0 * c_g / n:>7.3f} {c_b:>9} {100.0 * c_b / n:>7.3f} "
              f"{excess:>8.1f}")
        tot_peak += n
        tot_g += c_g
        tot_b += c_b
        out[band_label(edges, i)] = {
            "n_peaks": n, "count_global_eta": c_g,
            "rate_global": c_g / n, "eta_band": eta_b,
            "count_band_eta": c_b, "rate_band": c_b / n,
            "excess_over_design": excess}
    if tot_peak:
        print(f"{'TOTAL':>10} {tot_peak:>7} {tot_g:>7} "
              f"{100.0 * tot_g / tot_peak:>7.3f} {tot_b:>9} "
              f"{100.0 * tot_b / tot_peak:>7.3f} "
              f"{tot_b - DESIGN_RATE * tot_peak:>8.1f}")
    if skipped:
        print(f"  ({skipped} objects skipped: norad_id not found in TLE "
              f"file)")
    out["_totals"] = {"n_peaks": tot_peak, "count_global_eta": tot_g,
                      "count_band_eta": tot_b,
                      "rate_band": tot_b / tot_peak if tot_peak else None}
    return out


# ---------------------------------------------------------------------------
# Section 3: gate 3 per-band epsilon / DV_50
# ---------------------------------------------------------------------------

def gate3_section(path, alt_by_norad, edges, eta_global, eta_by_band):
    with open(path) as f:
        d = json.load(f)
    groups = {}
    skipped = 0
    for r in d.get("per_object_results", []):
        if not r.get("success"):
            continue
        try:
            nid = str(int(r["norad_id"]))
        except Exception:
            nid = str(r.get("norad_id"))
        alt = alt_by_norad.get(nid)
        if alt is None:
            skipped += 1
            continue
        i = band_index(alt, edges)
        if i is None:
            continue
        groups.setdefault(i, []).append(r["results"])

    print("\n=== GATE 3: per-band epsilon / DV_50 ===")
    print(f"{'band(km)':>10} {'n_obj':>6} {'DV50@g':>8} {'DV50@band':>10} "
          f"{'eps@0.1 g/band':>16}")
    out = {}
    for i in sorted(groups):
        objs = groups[i]
        eff_g, eff_b = {}, {}
        dv_seen = set()
        for res in objs:
            for dv_str, e in res.items():
                dv = float(dv_str)
                dv_seen.add(dv)
                pk = e.get("peak_maha")
                if pk is None:
                    continue
                a = eff_g.setdefault(dv, {"eps": 0.0, "n": 0})
                b = eff_b.setdefault(dv, {"eps": 0.0, "n": 0})
                a["eps"] += float(pk) > eta_global
                b["eps"] += float(pk) > eta_by_band.get(
                    band_label(edges, i), eta_global)
                a["n"] += 1
                b["n"] += 1
        for dv in dv_seen:
            if eff_g[dv]["n"]:
                eff_g[dv]["eps"] /= eff_g[dv]["n"]
            if eff_b[dv]["n"]:
                eff_b[dv]["eps"] /= eff_b[dv]["n"]
        dv50_g = dv50_from_eff(eff_g, 0.5)
        dv50_b = dv50_from_eff(eff_b, 0.5)
        e01_g = eff_g.get(0.1, {}).get("eps")
        e01_b = eff_b.get(0.1, {}).get("eps")
        print(f"{band_label(edges, i):>10} {len(objs):>6} "
              f"{dv50_g if dv50_g is not None else float('nan'):>8.4f} "
              f"{dv50_b if dv50_b is not None else float('nan'):>10.4f} "
              f"{e01_g if e01_g is not None else float('nan'):>7.3f}/"
              f"{e01_b if e01_b is not None else float('nan'):.3f}")
        out[band_label(edges, i)] = {
            "n_objects": len(objs), "dv50_global_eta": dv50_g,
            "dv50_band_eta": dv50_b,
            "efficiency_global_eta": {str(k): v["eps"] for k, v in
                                       sorted(eff_g.items())},
            "efficiency_band_eta": {str(k): v["eps"] for k, v in
                                     sorted(eff_b.items())}}
    if skipped:
        print(f"  ({skipped} objects skipped: norad_id not found in TLE "
              f"file)")
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--null", required=True,
                    help="V2.1 (Fix A) eta-null JSON")
    ap.add_argument("--null-v2", default=None,
                    help="optional V2 (Fix B) null JSON for the knee overlay")
    ap.add_argument("--fleet", default=None,
                    help="fleet peaks JSON (per_object with norad_id)")
    ap.add_argument("--gate3", default=None,
                    help="re-thresholded gate 3 fine JSON")
    ap.add_argument("--tle-file", default=None,
                    help="TLE file for the norad->altitude join "
                         "(required with --fleet/--gate3)")
    ap.add_argument("--band-edges",
                    default="700,760,820,875,905,940,1000",
                    help="altitude band edges in km (default isolates the "
                         "875-905 km knee band)")
    ap.add_argument("--fine-bin", type=float, default=25.0,
                    help="fine-bin width for the knee curve (km)")
    ap.add_argument("--percentile", type=float, default=99.9,
                    help="percentile for per-band eta candidates")
    ap.add_argument("--eta-override", type=float, default=None,
                    help="override the global eta read from the null "
                         "summary")
    ap.add_argument("--output", default="results/band_stratify.json")
    args = ap.parse_args()

    edges = [float(x) for x in args.band_edges.split(",") if x.strip()]
    if len(edges) < 3:
        sys.exit("--band-edges needs at least 3 values")

    pts, eta_global = load_null(args.null)
    if args.eta_override is not None:
        eta_global = args.eta_override
    if not pts:
        sys.exit("no usable trials in the null file")
    print(f"Loaded null: {len(pts)} OK trials, global eta = "
          f"{eta_global if eta_global is not None else float('nan')}")

    pts_v2 = None
    if args.null_v2:
        pts_v2, _ = load_null(args.null_v2)
        print(f"Loaded V2 overlay: {len(pts_v2)} OK trials")

    report = {"band_edges": edges, "percentile": args.percentile,
              "eta_global": eta_global}

    ns = null_section(pts, eta_global, edges, args.fine_bin,
                      args.percentile, pts_v2)
    report["null"] = ns
    eta_by_band = {k: v["eta_band"] for k, v in ns["bands"].items()}

    if args.fleet or args.gate3:
        if not args.tle_file:
            sys.exit("--tle-file is required with --fleet/--gate3")
        alt_by_norad = parse_tle_altitudes(args.tle_file)
        print(f"Parsed {len(alt_by_norad)} TLE altitudes from "
              f"{args.tle_file}")
        if args.fleet:
            report["fleet"] = fleet_section(args.fleet, alt_by_norad,
                                            edges, eta_global, eta_by_band)
        if args.gate3:
            report["gate3"] = gate3_section(args.gate3, alt_by_norad,
                                            edges, eta_global, eta_by_band)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()