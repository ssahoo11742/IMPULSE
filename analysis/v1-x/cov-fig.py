#!/usr/bin/env python3
"""
make_cov_ratio_figure.py -- generate fig5_covariance_ratios.png

Figure: per-channel ratio of the used (per-object, empirically estimated)
measurement 1-sigma to the assumed (published-floor) 1-sigma across the
fleet, summarized as median + 75th/90th/99th percentiles per channel.

Usage
-----
    python make_cov_ratio_figure.py --fleet-file results/real_detections_honestR_eta13.json
                                    [--out fig5_covariance_ratios.png]

Data source
-----------
The script reads the fleet run JSON and looks for per-object per-channel
covariance-ratio information in each per_object entry. Auto-detected shapes:

  1. entry has a ready ratio dict:      "cov_ratios" | "r_ratio" | "cov_ratio"
                                        | "noise_ratios" | "r_ratios" | "sigma_ratios"
                                        -> {channel: ratio}
  2. entry has a used/floor pair:       "r_used"+"r_floor" | "R_used"+"R_floor"
                                        | "used_sigma"+"floor_sigma"
                                        | "sigma_used"+"sigma_floor"
                                        -> ratio = used / floor per channel
  3. ratio nested under "covariance":   entry["covariance"]["ratio"],
                                        or entry["covariance"]["used"]/["floor"]

If none of these are present, the script prints the key structure of the
first per_object entry and exits -- paste that output back and the adapter
can be wired to the exact field names in one edit.
"""

import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

RATIO_DICT_KEYS = ("cov_ratios", "r_ratio", "cov_ratio", "noise_ratios",
                   "r_ratios", "sigma_ratios")
USED_FLOOR_PAIRS = (("r_used", "r_floor"), ("R_used", "R_floor"),
                    ("used_sigma", "floor_sigma"),
                    ("sigma_used", "sigma_floor"))

# Canonical display order / labels. Keys not recognized are appended as-is.
CHANNEL_ALIASES = {
    "a": "a", "sma": "a", "semi_major_axis": "a",
    "e": "e", "ecc": "e", "eccentricity": "e",
    "i": "i", "inc": "i", "inclination": "i",
    "Omega": "raan", "raan": "raan", "omega_big": "raan",
    "omega": "argp", "argp": "argp", "omega_small": "argp",
    "M": "M", "mean_anomaly": "M", "ma": "M",
}
CHANNEL_ORDER = ["a", "e", "i", "raan", "argp", "M"]
CHANNEL_LABELS = {"a": "a", "e": "e", "i": "i",
                  "raan": r"$\Omega$", "argp": r"$\omega$", "M": "M"}


def _is_channel_dict(d):
    return (isinstance(d, dict) and len(d) > 0
            and all(isinstance(v, (int, float)) and np.isfinite(v)
                    for v in d.values()))


def _pair_ratio(used, floor):
    out = {}
    for k in set(used) & set(floor):
        f = float(floor[k])
        out[k] = float(used[k]) / f if f > 0 else np.nan
    return out

def _list_ratio(used, floor):
    """used/floor for parallel lists, mapped positionally onto the
    canonical channel order (a, e, i, raan, argp, M)."""
    out = {}
    n = min(len(used), len(floor), len(CHANNEL_ORDER))
    for i in range(n):
        f = float(floor[i])
        out[CHANNEL_ORDER[i]] = float(used[i]) / f if f > 0 else np.nan
    return out


def extract_ratios(entry):
    """Return {raw_key: ratio} from one per_object entry, or None."""
    if not isinstance(entry, dict):
        return None
    for key in RATIO_DICT_KEYS:
        if key in entry and _is_channel_dict(entry[key]):
            return dict(entry[key])
    for ukey, fkey in USED_FLOOR_PAIRS:
        if ukey in entry and fkey in entry:
            u, f = entry[ukey], entry[fkey]
            if _is_channel_dict(u) and _is_channel_dict(f):
                return _pair_ratio(u, f)
            if (isinstance(u, list) and isinstance(f, list)
                    and len(u) == len(f) and len(u) > 0):
                return _list_ratio(u, f)
    # NEW: nested sigma lists, e.g. entry["r_used"] = {"sigma_used": [...],
    #                                                 "sigma_assumed": [...]}
    for container_key in ("r_used", "R_used", "covariance", "noise"):
        cont = entry.get(container_key)
        if not isinstance(cont, dict):
            continue
        for ukey, fkey in (("sigma_used", "sigma_assumed"),
                           ("used", "floor"), ("used_sigma", "floor_sigma")):
            u, f = cont.get(ukey), cont.get(fkey)
            if (isinstance(u, list) and isinstance(f, list)
                    and len(u) == len(f) and len(u) > 0):
                return _list_ratio(u, f)
            if _is_channel_dict(u) and _is_channel_dict(f):
                return _pair_ratio(u, f)
    cov = entry.get("covariance")
    if isinstance(cov, dict):
        if "ratio" in cov and _is_channel_dict(cov["ratio"]):
            return dict(cov["ratio"])
    return None


def dump_key_tree(entry, max_depth=3, indent=0):
    """Print the structure of the first entry so the adapter can be wired."""
    pad = "  " * indent
    if indent >= max_depth or not isinstance(entry, dict):
        print(f"{pad}{entry!r}"[:300])
        return
    for k, v in entry.items():
        if isinstance(v, dict):
            print(f"{pad}{k}: dict with keys {list(v.keys())}")
            dump_key_tree(v, max_depth, indent + 1)
        elif isinstance(v, list):
            print(f"{pad}{k}: list[{len(v)}]" +
                  (f" first={v[0]!r}"[:120] if v else ""))
        else:
            print(f"{pad}{k}: {type(v).__name__} = {v!r}"[:120])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fleet-file", required=True,
                    help="Fleet run JSON, e.g. results/real_detections_honestR_eta13.json")
    ap.add_argument("--out", default="fig5_covariance_ratios.png",
                    help="Output figure path (default: fig5_covariance_ratios.png)")
    args = ap.parse_args()

    with open(args.fleet_file, "r") as fh:
        data = json.load(fh)

    per_object = data.get("per_object", data)
    if isinstance(per_object, dict):
        # {norad_id: entry} form
        entries = [v for v in per_object.values() if isinstance(v, dict)]
    else:
        entries = [e for e in per_object if isinstance(e, dict)]
    if not entries:
        sys.exit("No per_object entries found in the fleet file.")

    # Collect per-object ratios
    per_channel = {}          # normalized channel -> list of per-object ratios
    n_ok, n_fail = 0, 0
    for entry in entries:
        ratios = extract_ratios(entry)
        if ratios is None:
            n_fail += 1
            continue
        n_ok += 1
        for raw_key, r in ratios.items():
            ch = CHANNEL_ALIASES.get(raw_key, raw_key)
            per_channel.setdefault(ch, []).append(float(r))

    if n_ok == 0:
        print("No covariance-ratio fields found in the per_object entries.\n")
        print("Key structure of the first entry:")
        dump_key_tree(entries[0])
        sys.exit(
            "\nPaste the output above back so the adapter can be wired to the "
            "exact field names. If the ratios were never saved by the fleet "
            "run, they must be recomputed with the pipeline's own per-object "
            "covariance estimator (the r_from_fit code path).")

    # Order channels canonically; append unknown ones at the end
    channels = [c for c in CHANNEL_ORDER if c in per_channel]
    channels += [c for c in per_channel if c not in CHANNEL_ORDER]

    print(f"Objects with usable ratio fields: {n_ok}"
          + (f"  (skipped {n_fail} entries without ratio fields)" if n_fail else ""))

    # Console summary (verifies the 'factor 1.0 most channels, 1.5 on M' claim)
    print(f"\n{'channel':>8} {'n':>6} {'median':>8} {'p75':>8} {'p90':>8} "
          f"{'p99':>8} {'% >1.0':>7}")
    stats = {}
    for ch in channels:
        vals = np.asarray(per_channel[ch], dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        p50, p75, p90, p99 = np.percentile(vals, [50, 75, 90, 99])
        frac = 100.0 * np.mean(vals > 1.0)
        stats[ch] = (p50, p75, p90, p99, frac)
        print(f"{ch:>8} {vals.size:>6} {p50:>8.3f} {p75:>8.3f} {p90:>8.3f} "
              f"{p99:>8.3f} {frac:>6.1f}%")

    # ---- Figure ----
    pct_specs = [(50, "o", "tab:blue", "median"),
                 (75, "^", "tab:cyan", "p75"),
                 (90, "s", "tab:orange", "p90"),
                 (99, "D", "tab:red", "p99")]
    x = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    for xi, ch in enumerate(stats):
        vals = np.asarray(per_channel[ch], dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        ys = [np.percentile(vals, p) for p, *_ in pct_specs]
        # thin vertical line spanning median -> p99
        ax.plot([xi, xi], [ys[0], ys[-1]], color="0.6", lw=1.2, zorder=1)
        for (p, m, c, lab), y in zip(pct_specs, ys):
            ms = 9 if p == 50 else 6
            ax.plot(xi, y, marker=m, color=c, markersize=ms, zorder=3,
                    linestyle="none",
                    label=lab if xi == 0 else None)

    ax.axhline(1.0, color="k", lw=1.0, linestyle="--", zorder=2)
    ax.text(len(stats) - 0.45, 1.0, "  published floor", fontsize=9,
            va="bottom", ha="right", color="k")

    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABELS.get(ch, ch) for ch in stats], fontsize=12)
    ax.set_ylabel("Used / assumed measurement 1-sigma", fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=True, fontsize=9)
    ax.set_title("Per-object measurement covariance vs. published floor "
                 "(fleet percentile summary)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {args.out}")
    print("Verify against Section 3.2: median ~1.0 on most channels, "
          "~1.5 on M, widening on a minority of objects.")


if __name__ == "__main__":
    main()