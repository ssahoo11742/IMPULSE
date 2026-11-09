#!/usr/bin/env python3
"""Re-threshold saved Gate 3 / Gate 4 strike-injection outputs at a new eta.

Both gate scripts store, for every object and every dv point, the raw peak
Mahalanobis statistic (peak_maha). The `detected` flag is just
`peak > threshold`, computed after the fact, and nothing upstream of the
peak depends on the threshold. Re-running the injections is therefore
unnecessary when only eta changes: re-thresholding the saved peaks
reproduces epsilon(dv), DV_50, DV_90, the negligible-dv floor and the
exclusion census EXACTLY, in seconds instead of hours.

Usage:
    python rethreshold_gates.py --input results/gate3_efficiency.json --threshold 152.714 --output results/gate3_efficiency_v2fixB.json
    python rethreshold_gates.py --input results/real_tle_efficiency.json --threshold 152.714 --output results/real_tle_efficiency_v2fixB.json

Gate format is auto-detected: Gate 3 efficiency entries carry "n";
Gate 4 entries carry "n_scored"/"exclusion_reasons". The output keeps the
original file's structure with:
  * config["threshold"] updated, config["rethresholded_from"] added
  * efficiency, dv50, dv90, negligible_dv_floor (gate 4) recomputed
  * per-object "detected" flags flipped where the new threshold moves them
  * peaks, exclusions, and every other config field untouched
"""
import argparse
import json

import numpy as np


def aggregate(per_object_results, threshold):
    """Per-dv census from per-object results. Peaks are threshold-independent."""
    agg = {}
    for r in per_object_results:
        if not r.get("success"):
            continue
        for dv_str, res in r.get("results", {}).items():
            dv_f = float(dv_str)
            a = agg.setdefault(dv_f, {"attempted": 0, "scored": 0, "det": 0,
                                      "excl": {}, "peaks": []})
            a["attempted"] += 1
            peak = res.get("peak_maha")
            if peak is None:
                # diverged / error / censored: census is threshold-independent
                reason = res.get("excluded") or res.get("error") or "no_peak"
                a["excl"][str(reason)] = a["excl"].get(str(reason), 0) + 1
                continue
            a["scored"] += 1
            a["peaks"].append(float(peak))
            if float(peak) > threshold:
                a["det"] += 1
    return agg


def build_efficiency_gate3(agg):
    eff = {}
    for dv_f in sorted(agg):
        a = agg[dv_f]
        n = a["scored"]
        eff[dv_f] = {
            "n": n,
            "epsilon": (a["det"] / n) if n else None,
            "mean_peak": float(np.mean(a["peaks"])) if a["peaks"] else None,
            "std_peak": float(np.std(a["peaks"])) if a["peaks"] else None,
        }
    return eff


def build_efficiency_gate4(agg):
    eff = {}
    for dv_f in sorted(agg):
        a = agg[dv_f]
        eff[dv_f] = {
            "n_attempted": a["attempted"],
            "n_scored": a["scored"],
            "n_excluded": a["attempted"] - a["scored"],
            "exclusion_reasons": dict(a["excl"]),
            "n_detected": a["det"],
            "epsilon": (a["det"] / a["scored"]) if a["scored"] else None,
            "mean_peak": float(np.mean(a["peaks"])) if a["peaks"] else None,
            "std_peak": float(np.std(a["peaks"])) if a["peaks"] else None,
        }
    return eff


def interp_crossing(eff, level):
    """Same DV_x logic as the gate scripts: interp on the eps-sorted curve."""
    pts = [(dv, e["epsilon"]) for dv, e in eff.items()
           if e["epsilon"] is not None]
    if len(pts) < 2:
        return None
    eps_v = np.array([p[1] for p in pts])
    dvs_v = np.array([p[0] for p in pts])
    order = np.argsort(eps_v)
    eps_v, dvs_v = eps_v[order], dvs_v[order]
    if eps_v.min() <= level <= eps_v.max():
        return float(np.interp(level, eps_v, dvs_v))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    por = data["per_object_results"]
    old_thr = data.get("config", {}).get("threshold")
    eff_old = data.get("efficiency", {})
    is_gate4 = bool(eff_old) and all("n_scored" in v for v in eff_old.values())

    # re-flag detections per object / dv (peak_maha is untouched)
    for r in por:
        for res in r.get("results", {}).values():
            if res.get("peak_maha") is not None:
                res["detected"] = bool(float(res["peak_maha"]) > args.threshold)

    agg = aggregate(por, args.threshold)
    eff = build_efficiency_gate4(agg) if is_gate4 else build_efficiency_gate3(agg)

    dv50 = interp_crossing(eff, 0.5)
    dv90 = interp_crossing(eff, 0.9)

    data["efficiency"] = eff
    data["dv50"] = dv50
    data["dv90"] = dv90

    if is_gate4:
        dv_min = min(eff)
        floor = eff[dv_min]
        data["negligible_dv_floor"] = {
            "dv": dv_min,
            "epsilon": floor["epsilon"],
            "n_scored": floor["n_scored"],
            "note": "in-window background trigger rate at negligible dv",
        }

    data.setdefault("config", {})["threshold"] = args.threshold
    data["config"]["rethresholded_from"] = {
        "file": args.input,
        "original_threshold": old_thr,
        "method": ("exact re-threshold of saved peak_maha values; peaks, "
                   "exclusions and honest R are threshold-independent"),
    }

    gate = "gate 4 (real TLE histories)" if is_gate4 else "gate 3 (stratified fleet)"
    print(f"Re-thresholded {gate}: {args.input}  (eta {old_thr} -> {args.threshold})")
    hdr_n = "scored" if is_gate4 else "n"
    print(f"{'dv (m/s)':>12} {hdr_n:>7} {'eps':>8} {'mean peak':>12}")
    for dv_f in sorted(eff):
        e = eff[dv_f]
        n = e["n_scored"] if is_gate4 else e["n"]
        eps = e["epsilon"]
        mp = e["mean_peak"]
        eps_s = f"{eps:.4f}" if eps is not None else "N/A"
        mp_s = f"{mp:.3f}" if mp is not None else "N/A"
        print(f"{dv_f:12.5g} {n:>7} {eps_s:>8} {mp_s:>12}")
    print("\nDV_50 = " + (f"{dv50:.5f} m/s" if dv50 is not None else "not bracketed"))
    print("DV_90 = " + (f"{dv90:.5f} m/s" if dv90 is not None else "not bracketed"))
    if is_gate4 and data["negligible_dv_floor"]["epsilon"] is not None:
        fl = data["negligible_dv_floor"]
        print(f"Negligible-dv floor: {fl['epsilon']:.4%} at dv={fl['dv']:g} m/s "
              f"({fl['n_scored']} scored)")

    with open(args.output, "w") as fo:
        json.dump(data, fo, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()