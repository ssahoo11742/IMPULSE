#!/usr/bin/env python3
"""
Figure script v2: confounder ablation (Fig A1), confounders + scrambled null
(Fig A2), and the pooled synthetic null distribution (Fig B).

FIG B FIX, TAKE TWO. The first pass found no candidate pools, which means the
pooled synthetic null (18,582 trials, p99.9 = 417.94, mean 6.31, max 1110.7)
is not stored as a plain numeric array at the top level of results/*.json.
This version searches much harder:

  * recursive scan of results/ (subdirectories included), plus *.json in the
    working directory;
  * also scans .npy / .npz / .csv files for 1-D numeric arrays;
  * pools per-trial peaks from list-of-records layouts, including the
    gate2-style `t["peaks"]["baseline"]` nested-dict form and per-trial
    arrays (pools each record's MAX, i.e. the window-max statistic);
  * if no raw peaks exist, falls back to a saved binned histogram
    (counts + edges) whose total matches 18,582;
  * on failure it prints a full manifest of every JSON it scanned (top-level
    keys, list lengths, record fields) -- paste that back to wire the exact
    field, or pass --null-file PATH once you know the file.

Usage:
    python -m analysis.figs2-5 [--results-dir results] [--null-file PATH]
                               [--out-dir figs]
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Reference values of the pooled synthetic null (from the eta-calibration run)
TARGET_P999 = 417.94
TARGET_MEAN = 6.31
TARGET_N = 18582
EXPECTED_LADDER = {
    "p99.9": 417.94,
    "p99.95": 556.5,
    "p99.97": 670.56,
    "p99.99": 848.79,
    "max": 1110.7,
}
MIN_POOL = 5000          # auto-accept only pools this large
PEAK_FIELD_HINTS = ("peak", "mahalanobis", "score", "s_window", "baseline",
                    "dm", "stat")


def _numeric(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and np.isfinite(v))


def _to_float(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Peak-pool extraction
# ---------------------------------------------------------------------------

def _pool_records(records, path, out):
    """Pool peak-like fields from a list of per-trial/per-object records."""
    rec_keys = set()
    for x in records[:20]:
        if isinstance(x, dict):
            rec_keys.update(x.keys())
    for rk in sorted(rec_keys):
        if not any(h in rk.lower() for h in PEAK_FIELD_HINTS):
            continue
        vals = [x.get(rk) for x in records if isinstance(x, dict)]
        # plain numeric field
        nums = [float(v) for v in vals if _numeric(v)]
        if len(nums) >= 100:
            out.append((f"{path}[].{rk}", nums))
        # field is a dict per record (channel/config breakdown)
        dicts = [v for v in vals if isinstance(v, dict) and v]
        if dicts:
            subkeys = set()
            for d in dicts[:20]:
                subkeys.update(d.keys())
            for sk in sorted(subkeys):
                snums = []
                for d in dicts:
                    v = d.get(sk)
                    if _numeric(v):
                        snums.append(float(v))
                    elif isinstance(v, list) and v:
                        inner = [w for w in v if _numeric(w)]
                        if inner:
                            snums.append(max(inner))
                if len(snums) >= 100:
                    out.append((f"{path}[].{rk}[].{sk}", snums))
        # field is a per-record array (per-epoch scores): pool the record max
        lists = [v for v in vals if isinstance(v, list) and v]
        if lists:
            pooled = []
            for v in lists:
                inner = [w for w in v if _numeric(w)]
                if inner:
                    pooled.append(max(inner))
            if len(pooled) >= 100:
                out.append((f"{path}[].{rk} (per-record max)", pooled))


def collect_peak_pools(obj, path, out, depth=0):
    if depth > 4:
        return out
    if isinstance(obj, dict):
        for key, val in obj.items():
            p = f"{path}.{key}"
            if isinstance(val, list) and val and all(_numeric(x) for x in val):
                out.append((p, [float(x) for x in val]))
            elif (isinstance(val, dict) and val
                  and all(_numeric(x) for x in val.values())):
                out.append((p + " (values)", list(map(float, val.values()))))
            elif (isinstance(val, list) and val
                  and all(isinstance(x, dict) for x in val)):
                _pool_records(val, p, out)
            elif (isinstance(val, list) and val
                  and all(isinstance(x, list) for x in val)):
                pooled = []
                for x in val:
                    inner = [w for w in x if _numeric(w)]
                    if inner:
                        pooled.append(max(inner))
                if pooled:
                    out.append((p + " (per-item max)", pooled))
            else:
                collect_peak_pools(val, p, out, depth + 1)
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        _pool_records(obj, path, out)
    return out


def collect_histograms(obj, path, out, depth=0):
    """Look for saved binned histograms (parallel counts + edges arrays)."""
    if depth > 3 or not isinstance(obj, dict):
        return out
    for key, val in obj.items():
        p = f"{path}.{key}"
        if isinstance(val, dict):
            counts = edges = None
            for k2, v2 in val.items():
                if isinstance(v2, list) and v2 and all(_numeric(x) for x in v2):
                    lk = k2.lower()
                    if "count" in lk or "freq" in lk or "density" in lk:
                        counts = [float(x) for x in v2]
                    elif "edge" in lk or "bin" in lk:
                        edges = [float(x) for x in v2]
            if counts and edges and len(edges) in (len(counts), len(counts) + 1) \
                    and len(counts) >= 10:
                out.append((p, counts, edges))
            else:
                collect_histograms(val, p, out, depth + 1)
        elif isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            for x in val[:5]:
                collect_histograms(x, p + "[]", out, depth + 1)
    return out


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def scan_csv(fp, out):
    try:
        with open(fp, "r", newline="") as fh:
            rows = [r for r in csv.reader(fh)]
    except Exception:
        return
    if not rows:
        return
    header = None
    if any(_to_float(x) is None for x in rows[0]):
        header = rows[0]
        rows = rows[1:]
    if not rows:
        return
    ncols = max(len(r) for r in rows[:200])
    for c in range(ncols):
        col = []
        for r in rows:
            if c < len(r):
                v = _to_float(r[c])
                if v is not None:
                    col.append(v)
        if len(col) >= MIN_POOL:
            label = header[c] if header and c < len(header) else f"col{c}"
            out.append((f"{os.path.basename(fp)}::{label}", col))


def scan_numpy_file(fp, out):
    try:
        if fp.endswith(".npy"):
            arr = np.load(fp, allow_pickle=False)
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                out.append((os.path.basename(fp),
                            arr.astype(float).tolist()))
        elif fp.endswith(".npz"):
            with np.load(fp, allow_pickle=False) as z:
                for k in z.files:
                    a = z[k]
                    if a.ndim == 1 and np.issubdtype(a.dtype, np.number):
                        out.append((f"{os.path.basename(fp)}::{k}",
                                    a.astype(float).tolist()))
    except Exception:
        pass


def discover_files(results_dir, explicit=None):
    if explicit:
        return [explicit]
    files = sorted(set(
        glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
        + glob.glob(os.path.join(results_dir, "**", "*.npy"), recursive=True)
        + glob.glob(os.path.join(results_dir, "**", "*.npz"), recursive=True)
        + glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True)
        + glob.glob("*.json")))
    return files


# ---------------------------------------------------------------------------
# Manifest (printed when auto-discovery fails)
# ---------------------------------------------------------------------------

def describe_entry(v):
    if isinstance(v, dict):
        ks = list(v.keys())
        s = ", ".join(ks[:8])
        return f"dict[{len(ks)}]: {s}" + (" ..." if len(ks) > 8 else "")
    if isinstance(v, list):
        if v and all(_numeric(x) for x in v):
            return f"numeric list[{len(v)}]"
        if v and all(isinstance(x, dict) for x in v):
            ks = sorted({k for x in v[:5] for k in x.keys()})
            return f"record list[{len(v)}]: fields {', '.join(ks[:10])}"
        return f"list[{len(v)}]"
    return f"{type(v).__name__} = {v!r}"[:70]


def print_manifest(files):
    print("\n--- MANIFEST of scanned files (paste this back if discovery "
          "still fails) ---")
    for fp in files:
        if fp.endswith((".npy", ".npz", ".csv")):
            print(f"{fp}: (numeric-format file, scanned for 1-D arrays)")
            continue
        try:
            with open(fp, "r") as fh:
                d = json.load(fh)
        except Exception as e:
            print(f"{fp}: UNREADABLE ({e})")
            continue
        print(f"{fp}")
        if isinstance(d, dict):
            for k, v in list(d.items())[:12]:
                print(f"    .{k}: {describe_entry(v)}")
        else:
            print(f"    top level: {describe_entry(d)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--null-file", default=None,
                    help="Explicit path to the pooled synthetic null file "
                         "(JSON/NPY/NPZ/CSV); skips auto-discovery")
    ap.add_argument("--out-dir", default="figs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )

    with open(os.path.join(args.results_dir, "gate2_confounders.json")) as f:
        conf = json.load(f)
    with open(os.path.join(args.results_dir, "gate2_confounders_v2.json")) as f:
        g2 = json.load(f)
    with open(os.path.join(args.results_dir, "scrambled_null.json")) as f:
        scr = json.load(f)

    # ---- Shared data preparation ----
    names, rates, errs = [], [], []
    for k, s in conf["summary"].items():
        n, p = s["n"], s["fp_rate"]
        names.append(k)
        rates.append(100 * p)  # percent
        se = 100 * np.sqrt(p * (1 - p) / n) if p > 0 else 0
        errs.append(se)

    order = ["baseline", "f107_flat", "kp_quiet", "env_flat", "cd_nominal"]
    display_labels_base = [
        "Baseline",
        "$F_{10.7}$ Flat",
        "$K_p$ Quiet",
        "Env. Flat",
        "$C_d$ Nominal",
    ]
    idx = [names.index(o) for o in order]

    selected_rates = [rates[i] for i in idx]
    selected_errs = [errs[i] for i in idx]

    scr_rate = 100 * scr["summary"]["triggers"] / scr["config"]["total_valid_windows"]
    scr_se = 100 * np.sqrt((scr_rate / 100) * (1 - scr_rate / 100)
                           / scr["config"]["total_valid_windows"])

    # ============================================================
    # Fig A1: Original Confounder Ablation (NO scrambled bar)
    # ============================================================
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    ax.bar(
        range(len(order)),
        selected_rates,
        yerr=selected_errs,
        capsize=5,
        color="#3470a3",
        edgecolor="#1f4260",
        linewidth=0.8,
        alpha=0.85,
        error_kw={"ecolor": "#222222", "linewidth": 1.2},
        width=0.55,
    )

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(display_labels_base)

    ax.axhline(
        0.1,
        color="#d9381e",
        linestyle="--",
        linewidth=1.2,
        label="Nominal 0.1% (99.9th pct)",
    )

    ax.set_ylabel("False-Positive Rate per Window (%)")
    ax.set_title(
        "Confounder Ablation (100-Object Subset, Threshold 13.258)", pad=12
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    sns.despine(top=True, right=True)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_confounder_ablation.png"), dpi=300)
    fig.savefig(os.path.join(args.out_dir, "fig_confounder_ablation.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"Fig A1 written to {args.out_dir}/fig_confounder_ablation.png")

    # ============================================================
    # Fig A2: Confounder Ablation + Scrambled Real-TLE
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=300)

    display_labels_full = display_labels_base + ["Scrambled\nreal-TLE"]
    full_rates = selected_rates + [scr_rate]
    full_errs = selected_errs + [scr_se]

    colors = ["#3470a3"] * 5 + ["#e66101"]
    edge_colors = ["#1f4260"] * 5 + ["#a04000"]

    ax.bar(
        range(len(display_labels_full)),
        full_rates,
        yerr=full_errs,
        capsize=4,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.8,
        alpha=0.85,
        error_kw={"ecolor": "#222222", "linewidth": 1.1},
        width=0.62,
    )

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 50)
    ax.set_xticks(range(len(display_labels_full)))
    ax.set_xticklabels(display_labels_full)

    ax.axhline(
        0.1,
        color="#d9381e",
        linestyle="--",
        linewidth=1.2,
        label="Nominal 0.1% (99.9th pct)",
    )

    for i, rate in enumerate(full_rates):
        if rate < 1:
            label = f"{rate:.2f}%"
            offset = 15
            fontsize = 8
            color_label = "#1f4260"
        else:
            label = f"{rate:.1f}%"
            offset = 8
            fontsize = 9
            color_label = "#a04000"

        ax.annotate(
            label,
            xy=(i, rate),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color=color_label,
        )
    ax.set_ylabel("False-Positive Rate per Window (%)")
    ax.set_title(
        "False-Positive Rate by Source\n"
        "(100-Object Ablations + Scrambled Real-TLE Null)",
        pad=12,
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    sns.despine(top=True, right=True)

    fig.tight_layout()
    fig.savefig(
        os.path.join(args.out_dir, "fig_confounder_ablation_with_scrambled.png"),
        dpi=300)
    fig.savefig(
        os.path.join(args.out_dir, "fig_confounder_ablation_with_scrambled.pdf"),
        bbox_inches="tight")
    plt.close(fig)
    print(f"Fig A2 written to {args.out_dir}/fig_confounder_ablation_with_scrambled.png")

    # ============================================================
    # Fig B: Pooled synthetic null distribution (FIXED)
    # ============================================================
    files = discover_files(args.results_dir, args.null_file)
    pools, hists = [], []
    for fp in files:
        if fp.endswith(".json"):
            try:
                with open(fp, "r") as fh:
                    d = json.load(fh)
            except Exception:
                continue
            base = os.path.basename(fp)
            pools.extend(collect_peak_pools(d, base, []))
            hists.extend(collect_histograms(d, base, []))
        elif fp.endswith((".npy", ".npz")):
            scan_numpy_file(fp, pools)
        elif fp.endswith(".csv"):
            scan_csv(fp, pools)

    min_n = 0 if args.null_file else MIN_POOL
    cands = []
    for label, arr in pools:
        a = np.asarray(arr, float)
        if a.size < max(min_n, 100):
            continue
        p999 = float(np.percentile(a, 99.9))
        mean = float(a.mean())
        score = (abs(p999 - TARGET_P999) / TARGET_P999
                 + 0.5 * abs(mean - TARGET_MEAN) / TARGET_MEAN)
        cands.append({"score": score, "label": label, "file": label.split(".")[0],
                      "n": a.size, "mean": mean, "p99.9": p999, "peaks": a})
    cands.sort(key=lambda c: c["score"])

    print(f"\nPeak pools found (scanned {len(files)} files; showing up to 12, "
          f"any size >= 100):")
    print(f"{'score':>8} {'n':>7} {'mean':>7} {'p99.9':>8}  key")
    for c in cands[:12]:
        print(f"{c['score']:>8.4f} {c['n']:>7} {c['mean']:>7.2f} "
              f"{c['p99.9']:>8.2f}  {c['label']}")

    best = None
    for c in cands:
        if c["n"] >= MIN_POOL or args.null_file:
            best = c
            break

    if best is not None and (best["score"] <= 0.02 or args.null_file):
        a = best["peaks"]
        print(f"\nUsing: {best['label']}")
        found = {
            "mean": float(a.mean()),
            "p99.9": float(np.percentile(a, 99.9)),
            "p99.95": float(np.percentile(a, 99.95)),
            "p99.97": float(np.percentile(a, 99.97)),
            "p99.99": float(np.percentile(a, 99.99)),
            "max": float(a.max()),
        }
        print(f"Statistic   found     expected   (n = {a.size:,})")
        print(f"{'n':>9} {a.size:>9} {TARGET_N:>12,}")
        print(f"{'mean':>9} {found['mean']:>9.2f} {TARGET_MEAN:>12.2f}")
        for k in ("p99.9", "p99.95", "p99.97", "p99.99", "max"):
            print(f"{k:>9} {found[k]:>9.2f} {EXPECTED_LADDER[k]:>12.2f}")
        if abs(found["p99.9"] - TARGET_P999) / TARGET_P999 > 0.02:
            print("WARNING: p99.9 deviates from 417.94 by more than 2% -- "
                  "this may not be the calibration population. Figure saved, "
                  "but verify before using.")
        draw_null_histogram(a, args.out_dir, n_label=f"{a.size:,}")
    else:
        # No acceptable raw pool: try a saved binned histogram.
        drawn = False
        for hlabel, counts, edges in hists:
            total = sum(counts)
            if (abs(total - TARGET_N) / TARGET_N < 0.15
                    and len(edges) >= 10 and edges[-1] >= 800):
                print(f"\nNo raw peaks; drawing saved histogram {hlabel} "
                      f"(total counts = {total:,}).")
                print("NOTE: percentiles cannot be re-verified from binned "
                      "data; caption values remain from the run summary.")
                draw_null_binned(counts, edges, args.out_dir,
                                 n_label=f"{int(total):,}")
                drawn = True
                break
        if not drawn:
            print("\nNo pooled-null peak array or histogram found.")
            print("Likely causes: the null run stored per-trial peaks in a "
                  "layout this scanner missed, or saved only summary "
                  "percentiles.")
            print_manifest(files)
            raise SystemExit(
                "\nPaste the manifest above back, or point --null-file at "
                "the run output that reported p99.9 = 417.94.")

    print(f"\nDone - figures written to {args.out_dir}/")


def draw_null_histogram(a, out_dir, n_label):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    bins = np.linspace(0, 900, 181)          # tighter binning to match new limit
    ax.hist(
        a,
        bins=bins,
        density=True,
        histtype="stepfilled",
        facecolor="#3470a3",
        edgecolor="#1f4260",
        alpha=0.55,
        linewidth=1.2,
        label=f"Pooled synthetic null ($n=18,582$)",
    )
    ax.set_yscale("log")

    # Stronger, more visible threshold lines
    lines = [
        (417.94, r"$\eta$ = 417.94 (p99.9)", "#d9381e", "--", 2.0),   # primary threshold – bright red, thick
        (556.5,  "p99.95 = 556.5",           "#e67e22", ":",  1.6),   # orange
        (670.56, "p99.97 = 670.56",          "#8e44ad", ":",  1.6),   # purple
        (848.79, "p99.99 = 848.79",          "#16a085", ":",  1.6),   # teal
        # max = 1110.7 is now outside the visible range, so omit it
    ]
    for val, lab, color, ls, lw in lines:
        ax.axvline(val, color=color, linestyle=ls, linewidth=lw, label=lab)

    ax.set_xlim(0, 900)
    ax.set_xlabel("Window-Maximum Mahalanobis Distance")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Pooled Synthetic Null Distribution of Peak Mahalanobis "
                 "Distance", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0",
              loc="upper right", fontsize=8.5)
    sns.despine(top=True, right=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_null_distribution.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig_null_distribution.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"Fig B written to {out_dir}/fig_null_distribution.png")


def draw_null_binned(counts, edges, out_dir, n_label):
    counts = np.asarray(counts, float)
    edges = np.asarray(edges, float)
    if len(edges) == len(counts) + 1:
        left, widths = edges[:-1], np.diff(edges)
    else:
        widths = np.gradient(edges)
        left = edges - widths / 2
    density = counts / (counts.sum() * widths)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.bar(left, density, width=widths, align="edge",
           facecolor="#3470a3", edgecolor="#1f4260",
           alpha=0.55, linewidth=1.2,
           label=f"Pooled synthetic null ($n=18,582$, binned)")
    ax.set_yscale("log")

    lines = [
        (417.94, r"$\eta$ = 417.94 (p99.9)", "#d9381e", "--", 2.0),
        (556.5,  "p99.95 = 556.5",           "#e67e22", ":",  1.6),
        (670.56, "p99.97 = 670.56",          "#8e44ad", ":",  1.6),
        (848.79, "p99.99 = 848.79",          "#16a085", ":",  1.6),
    ]
    for val, lab, color, ls, lw in lines:
        ax.axvline(val, color=color, linestyle=ls, linewidth=lw, label=lab)

    ax.set_xlim(0, 900)
    ax.set_xlabel("Window-Maximum Mahalanobis Distance")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Pooled Synthetic Null Distribution of Peak Mahalanobis "
                 "Distance", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0",
              loc="upper right", fontsize=8.5)
    sns.despine(top=True, right=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_null_distribution.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig_null_distribution.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"Fig B written to {out_dir}/fig_null_distribution.png (from saved "
          f"histogram bins)")

if __name__ == "__main__":
    main()