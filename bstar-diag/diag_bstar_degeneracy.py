#!/usr/bin/env python3
"""Phase-0 diagnostic: the B*/a calibration artifact in the synthetic eta-null.

Verifies (or refutes) the V1.1 limitation:

    "The B*/semi-major-axis degeneracy at the band top drives the null's
     extreme tail (all twenty largest null peaks sit at 891-903 km), so eta
     is calibrated partly against a documented construction artifact."

Purely observational: it reads the eta-null trials JSON produced by
convention_consistency_test.py (no pipeline re-run needed) and answers:

  1. Do the extreme peaks cluster by altitude? (peak vs a0 scatter,
     top-N band census with a Monte-Carlo significance test)
  2. Do the extreme peaks carry the fitted-a drift signature? (correlation
     of peak with |da_slope|, std_da, max |da|; drift-to-noise ratio vs
     the sigma_a the filter actually used)
  3. Where is the empirical identifiability knee? (p95 |da_slope| by
     altitude bin: the altitude where the fitted-a drift explodes)
  4. Does the a-channel honest R floor/cap bind in the artifact regime?
  5. Censoring check: crash/failure rate vs altitude (a censored tail
     would bias eta LOW - the script warns above 2%)

Optional deep dive (--refit-top N): re-runs ONLY the TLE-fit stage for the
top-N trials (plus median-peak controls) through the repo's own functions,
capturing the per-refit B* series that the trials JSON does not store:
sign-flip rate, fraction of negative B*, and the fitted-a / fitted-B*
correlation (the degeneracy made visible). Seconds per trial.

Usage (repo root):
    python diag_bstar_degeneracy.py \
        --trials results/synthetic_null_eta_honestR.json
    python diag_bstar_degeneracy.py --trials ... --refit-top 8
    python diag_bstar_degeneracy.py --trials ... --no-plots

Outputs -> results/diag_bstar/:
    census.json                all numbers in machine-readable form
    peak_vs_altitude.png       scatter, log-y, eta line, top-N highlighted
    bins_table.png / .txt      per-altitude-bin stats + identifiability knee
    drift_vs_peak.png          drift-to-noise ratio vs peak
    refit/<seed>.json          per-refit (a_fit, bstar, da) series (deep dive)
"""

import argparse
import json
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# repo conventions (fallbacks if the packages are not importable)
try:
    sys.path.insert(0, REPO_ROOT)
    from constants.constants import R_EARTH          # m
except Exception:
    R_EARTH = 6371000.0
try:
    from phase2_filter.config import R_DIAG_ELEMENTS
    R_DIAG = np.asarray(R_DIAG_ELEMENTS, dtype=float)
except Exception:
    R_DIAG = None


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_trials(path):
    """Tolerant loader for the eta-null output / partial / bare list."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {}, {}, data
    if isinstance(data, dict) and "trials" in data:
        return data.get("config", {}), data.get("summary", {}), data["trials"]
    if isinstance(data, dict) and "partial" in data:      # <output>.partial.json
        return data.get("config", {}), {}, data["partial"]
    raise SystemExit(f"unrecognized trials file structure: {path}")


def alt_km(rec):
    try:
        return (rec["geometry"]["a0_m"] - R_EARTH) / 1000.0
    except (KeyError, TypeError):
        return None


def get(rec, key):
    return rec.get(key)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return None
    x, y = x[m], y[m]
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)
    except ImportError:
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        rx, ry = rx - rx.mean(), ry - ry.mean()
        denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / denom) if denom else None


def binom_mc_p(observed, ntop, band_lo, band_hi, alt_lo, alt_hi, n_mc=200000):
    """Monte-Carlo p-value: P(>= observed of n_top uniform altitudes fall in
    the band) - no scipy needed."""
    rng = np.random.default_rng(0)
    draws = rng.uniform(alt_lo, alt_hi, size=(n_mc, ntop))
    counts = ((draws >= band_lo) & (draws <= band_hi)).sum(axis=1)
    return float((counts >= observed).mean())


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------
def census(config, summary, records, args):
    ok = [r for r in records if r.get("ok")]
    bad = [r for r in records if not r.get("ok")]
    peaks = np.array([r["peak_maha"] for r in ok], float)
    # trials missing geometry are dropped from paired analyses
    paired = [(r, alt_km(r)) for r in ok if alt_km(r) is not None]

    eta = None
    if summary.get("eta_synth") is not None:
        eta = float(summary["eta_synth"])
    elif args.eta is not None:
        eta = args.eta

    print(f"\ntrials: {len(records)} total | ok {len(ok)} | failed/crashed "
          f"{len(bad)} ({100 * len(bad) / max(1, len(records)):.2f}%)")
    if len(bad) / max(1, len(records)) > 0.02:
        print("  WARNING: failure rate > 2% - if failures correlate with "
              "extreme peaks, eta is biased LOW (censoring bias).")

    # ---- 1. top-N altitude cluster ---------------------------------------
    order = np.argsort(peaks)[::-1]
    top = [ok[i] for i in order[:args.top_n]]
    top_alts = [alt_km(r) for r in top if alt_km(r) is not None]
    in_band = [a for a in top_alts if args.band_lo <= a <= args.band_hi]
    band_frac_expected = (args.band_hi - args.band_lo) / (args.alt_max
                                                          - args.alt_min)
    p_cluster = binom_mc_p(len(in_band), len(top_alts), args.band_lo,
                           args.band_hi, args.alt_min, args.alt_max)
    print(f"\n[1] top-{len(top_alts)} peaks by altitude (km):")
    print("    " + ", ".join(f"{a:.1f}" for a in sorted(top_alts)))
    print(f"    in {args.band_lo}-{args.band_hi} km band: {len(in_band)}/"
          f"{len(top_alts)}  (expected under uniform: "
          f"{band_frac_expected * len(top_alts):.1f})")
    print(f"    Monte-Carlo p(>= that count in band): {p_cluster:.2e}")

    # ---- 2. drift signature of extreme trials ----------------------------
    keys = ["da_slope_m_per_day", "std_da_m", "max_abs_da_m", "fit_rms_m",
            "maha_argmax_frac"]
    print(f"\n[2] Spearman corr(peak, x) over all ok trials:")
    corrs = {}
    for k in keys:
        xs = [get(r, k) for r, _ in paired]
        if k == "da_slope_m_per_day":
            xs = [abs(v) if v is not None else np.nan for v in xs]
        else:
            xs = [float(v) if v is not None else np.nan for v in xs]
        c = spearman([r["peak_maha"] for r, _ in paired], xs)
        n_avail = int(np.sum(np.isfinite(np.asarray(xs, float))))
        corrs[k] = c
        print(f"    {k:<22} n={n_avail:<6} rho = "
              f"{'n/a' if c is None else f'{c:+.3f}'}")

    # drift-to-noise ratio: accumulated fitted-a drift vs sigma_a the
    # filter believed
    win_days = float(config.get("window_days", 90.0))
    pk_arr, dnr_arr = [], []
    for r, _ in paired:
        ru = get(r, "r_used")
        slope = get(r, "da_slope_m_per_day")
        if ru is None or slope is None:
            continue
        sig_a = math.sqrt(float(ru[0]))
        pk_arr.append(r["peak_maha"])
        dnr_arr.append(abs(slope) * win_days / sig_a)
    pk_arr, dnr_arr = np.asarray(pk_arr, float), np.asarray(dnr_arr, float)
    if len(dnr_arr) >= 10:
        med_split = np.median(dnr_arr)
        hi = pk_arr[dnr_arr > med_split]
        lo = pk_arr[dnr_arr <= med_split]
        print(f"\n[2b] drift-to-noise ratio DNR = |da_slope|*{win_days:.0f}d"
              f" / sigma_a_used:")
        print(f"    median peak | DNR above median: {np.median(hi):8.2f}")
        print(f"                     DNR below median: {np.median(lo):8.2f}")
        c = spearman(pk_arr, dnr_arr)
        print(f"    Spearman corr(peak, DNR): "
              f"{'n/a' if c is None else f'{c:+.3f}'}")
    else:
        print("[2b] DNR unavailable (missing r_used or da_slope)")

    # ---- 3. altitude bins + identifiability knee -------------------------
    edges = np.arange(args.alt_min, args.alt_max + args.bin_km, args.bin_km)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(r, a) for r, a in paired if lo <= a < hi]
        if not sel:
            continue
        bp = np.array([r["peak_maha"] for r, _ in sel], float)
        slopes = [abs(get(r, "da_slope_m_per_day") or 0.0) for r, _ in sel]
        slopes = np.array(slopes, float)
        sigs = [math.sqrt(float(get(r, "r_used")[0]))
                for r, _ in sel if get(r, "r_used") is not None]
        bins.append({
            "alt_lo": float(lo), "alt_hi": float(hi), "n": len(sel),
            "median_peak": float(np.median(bp)),
            "max_peak": float(np.max(bp)),
            "p95_abs_da_slope": float(np.percentile(slopes, 95)),
            "median_sigma_a": float(np.median(sigs)) if sigs else None,
        })
    print(f"\n[3] per-{args.bin_km:.0f}-km-bin stats "
          f"(p95 |da_slope| = empirical identifiability proxy):")
    print(f"    {'alt (km)':>12} {'n':>6} {'med peak':>9} {'max peak':>9} "
          f"{'p95|da_slope|':>13} {'med sigma_a':>11}")
    for b in bins:
        print(f"    {b['alt_lo']:>5.0f}-{b['alt_hi']:<5.0f} {b['n']:>6} "
              f"{b['median_peak']:>9.2f} {b['max_peak']:>9.2f} "
              f"{b['p95_abs_da_slope']:>13.3f} "
              f"{b['median_sigma_a'] or float('nan'):>11.3f}")
    # knee: lowest bin whose p95 |da_slope| exceeds 10x the lowest bin's
    knee = None
    if len(bins) >= 3 and bins[0]["p95_abs_da_slope"] > 0:
        base = bins[0]["p95_abs_da_slope"]
        for b in bins[1:]:
            if b["p95_abs_da_slope"] > 10 * max(base, 1e-9):
                knee = b
                break
    if knee:
        print(f"    identifiability knee (p95 |da_slope| > 10x lowest bin): "
              f"{knee['alt_lo']:.0f}-{knee['alt_hi']:.0f} km")

    # ---- 4. honest-R floor/cap census in the top tail ---------------------
    floor = float(R_DIAG[0]) if R_DIAG is not None else None
    if floor is None:
        ru_all = [get(r, "r_used") for r, _ in paired if get(r, "r_used")]
        if ru_all:
            floor = min(float(v[0]) for v in ru_all)
    if floor is not None:
        at_floor = sum(1 for r in top
                       if get(r, "r_used") is not None
                       and abs(float(r["r_used"][0]) - floor)
                       < 1e-12 * max(1.0, floor))
        at_cap = sum(1 for r in top
                     if get(r, "r_used") is not None
                     and abs(float(r["r_used"][0]) - 100 * floor)
                     < 1e-9 * max(1.0, floor))
        print(f"\n[4] top-{len(top)} a-channel R: {at_floor} at R_DIAG floor,"
              f" {at_cap} at 100x cap, {len(top) - at_floor - at_cap} between")
        if at_cap > len(top) / 2:
            print("    NOTE: most top peaks ran with CAPPED R - the stream"
                  " scatter was huge and even the cap could not hide it.")

    return {
        "config": config,
        "n_total": len(records), "n_ok": len(ok), "n_bad": len(bad),
        "eta": eta,
        "top_n": len(top),
        "top_alts_km": sorted(top_alts),
        "n_top_in_band": len(in_band),
        "band_km": [args.band_lo, args.band_hi],
        "mc_p_cluster": p_cluster,
        "corr_peak_vs": corrs,
        "bins": bins,
        "identifiability_knee_km": (knee["alt_lo"], knee["alt_hi"])
        if knee else None,
        "r_floor_a": floor,
    }


# ---------------------------------------------------------------------------
# deep dive: re-run the fit stage for selected seeds, capture per-refit B*
# ---------------------------------------------------------------------------
def refit_trial_deep(seed, config):
    """Replicate _run_one's truth+fit stages for one seed and capture the
    per-refit (a_fit, B*, fit RMS) series via a wrapper on fit_tle_epoch."""
    import phase2_filter.convention_consistency_test as cct

    MeanElements = getattr(cct, "MeanElements")
    if MeanElements is None:
        from propagator.orbital import MeanElements

    captured = []
    orig = cct.fit_tle_epoch

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        if out is not None:
            captured.append({
                "t_ep_s": float(a[1]),
                "a_fit_m": float(out["a"]),
                "bstar": float(out["bstar"]) if "bstar" in out else None,
                "fit_rms_m": float(out["fit_rms_m"]),
            })
        return out

    cct.fit_tle_epoch = wrapped
    try:
        rng = np.random.default_rng(seed)
        epoch_jd = 2460000.5
        duration_s = float(config["window_days"]) * 86400.0
        truth_dt_s = float(config.get("truth_dt_hours", 6.0)) * 3600.0
        el0 = cct._default_el0(alt_km=800.0, inc_deg=98.0)
        if config.get("jitter_geometry") and config.get("truth_mode") in (
                "sgp4_consistent", "twobody"):
            el0 = MeanElements(
                a=el0.a + (rng.uniform(700.0, 1000.0) - 800.0) * 1000.0,
                ecc=rng.uniform(1.0e-5, 1.0e-3),
                inc=math.radians(rng.uniform(50.0, 105.0)),
                raan=rng.uniform(0.0, 2.0 * math.pi),
                argp=rng.uniform(0.0, 2.0 * math.pi),
                M=rng.uniform(0.0, 2.0 * math.pi),
            )
        elif config.get("truth_mode") == "sgp4_static":
            el0 = MeanElements(a=el0.a, ecc=el0.ecc, inc=el0.inc,
                               raan=el0.raan + rng.uniform(0.0, 2 * math.pi),
                               argp=el0.argp + rng.uniform(0.0, 2 * math.pi),
                               M=el0.M + rng.uniform(0.0, 2 * math.pi))
        area, mass = 1.0, 200.0
        truth, _, _ = cct._propagate_truth(el0, epoch_jd, duration_s,
                                           truth_dt_s, area, mass, rng=rng)
        obs, diag = cct.generate_tle_like_observations(
            truth, epoch_jd,
            float(config.get("tle_cadence_days", 2.0)),
            float(config.get("fit_span_days", 4.0)),
            config.get("truth_mode", "sgp4_consistent"),
            bool(config.get("fit_bstar", 0)),
            max_nfev=int(config.get("max_nfev", 2000)),
            fit_rms_max_m=float(config.get("fit_rms_max_m", 5000.0)))
    finally:
        cct.fit_tle_epoch = orig

    # truth a at each refit epoch -> da series
    t_truth = np.asarray(truth["t"], float)
    a_truth = np.asarray(truth["a"], float)
    for c in captured:
        i = int(np.argmin(np.abs(t_truth - c["t_ep_s"])))
        c["a_truth_m"] = float(a_truth[i])
        c["da_m"] = c["a_fit_m"] - c["a_truth_m"]

    # B* statistics across refits (apply the same QC gate as the stream)
    gate = float(config.get("fit_rms_max_m", 5000.0))
    kept = [c for c in captured if c["fit_rms_m"] <= gate]
    b = np.array([c["bstar"] for c in kept if c["bstar"] is not None], float)
    a_f = np.array([c["a_fit_m"] for c in kept
                    if c["bstar"] is not None], float)
    stats = {}
    if len(b) >= 3:
        flips = np.mean(np.sign(b[1:]) != np.sign(b[:-1]))
        stats = {
            "n_refits_kept": len(kept),
            "bstar_sign_flip_rate": float(flips),
            "frac_negative_bstar": float(np.mean(b < 0)),
            "bstar_std": float(np.std(b)),
            "corr_a_fit_bstar": (lambda x, y: float(np.corrcoef(x, y)[0, 1])
                                 if np.std(x) > 0 and np.std(y) > 0 else None
                                 )(a_f, b),
        }
    da = np.array([c["da_m"] for c in kept], float)
    dt = np.array([c["t_ep_s"] for c in kept], float) / 86400.0
    if len(da) >= 3:
        stats["da_slope_m_per_day"] = float(np.polyfit(dt, da, 1)[0])
        stats["da_std_m"] = float(np.std(da))
    return {"seed": seed, "geometry": {"a0_m": float(el0.a),
                                       "ecc0": float(el0.ecc),
                                       "inc0_deg": float(math.degrees(
                                           el0.inc))},
            "per_refit": captured, "bstar_stats": stats,
            "diag": diag, "n_obs_kept": len(obs)}


def deep_dive(config, summary, records, args, out_dir):
    ok = [r for r in records if r.get("ok") and alt_km(r) is not None]
    ok.sort(key=lambda r: -r["peak_maha"])
    top = ok[:args.refit_top]
    # controls: median-peak trials (25th-75th percentile band)
    peaks = np.array([r["peak_maha"] for r in ok], float)
    lo_q, hi_q = np.percentile(peaks, [25, 75])
    mids = [r for r in ok if lo_q <= r["peak_maha"] <= hi_q]
    step = max(1, len(mids) // args.refit_controls)
    controls = mids[::step][:args.refit_controls]

    print(f"\n[deep dive] re-running fit stage for {len(top)} top + "
          f"{len(controls)} control trials (captures per-refit B*)...")
    refit_dir = os.path.join(out_dir, "refit")
    os.makedirs(refit_dir, exist_ok=True)
    table = []
    for tag, group in (("TOP", top), ("CONTROL", controls)):
        for r in group:
            seed = r["seed"]
            try:
                res = refit_trial_deep(seed, config)
            except Exception as e:
                print(f"    {tag} seed={seed}: REFIT FAILED: {type(e).__name__}"
                      f": {e}")
                continue
            # sanity: replicated geometry must match the recorded one
            g0 = r.get("geometry", {}).get("a0_m")
            if g0 is not None and abs(res["geometry"]["a0_m"] - g0) > 1.0:
                print(f"    {tag} seed={seed}: geometry mismatch "
                      f"({res['geometry']['a0_m']:.1f} vs {g0:.1f}) - "
                      f"config not reproducible, skipping")
                continue
            fn = os.path.join(refit_dir, f"{seed}.json")
            slim = {k: res[k] for k in ("seed", "geometry", "bstar_stats",
                                        "diag", "n_obs_kept")}
            slim["peak_maha"] = r["peak_maha"]
            slim["alt_km"] = alt_km(r)
            slim["tag"] = tag
            with open(fn, "w") as f:
                json.dump({**slim, "per_refit": res["per_refit"]},
                          f, indent=2)
            s = res["bstar_stats"]
            table.append({**slim, "bstar_stats": s})
            print(f"    {tag:<7} seed={seed:<7} alt={alt_km(r):7.1f} km "
                  f"peak={r['peak_maha']:8.2f}  "
                  f"flip={s.get('bstar_sign_flip_rate', float('nan')):.2f} "
                  f"negB={s.get('frac_negative_bstar', float('nan')):.2f} "
                  f"corr(a,B*)="
                  f"{s.get('corr_a_fit_bstar') if s.get('corr_a_fit_bstar') is not None else float('nan'):+.2f}")

    print(f"\n[deep dive] interpretation guide:")
    print("    degeneracy signature = high B* sign-flip rate, sizeable")
    print("    negative-B* fraction, and a strong NEGATIVE corr(fitted a,")
    print("    fitted B*) - the (a, B*) trade-off made visible. Controls")
    print("    should show pinned B* (low flip rate, ~0 negative fraction).")
    return table


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def make_plots(census_data, records, args, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(no matplotlib -> skipping PNGs; census.json has everything)")
        return

    ok = [r for r in records if r.get("ok") and alt_km(r) is not None]
    alts = np.array([alt_km(r) for r in ok])
    pk = np.array([r["peak_maha"] for r in ok], float)
    top_idx = np.argsort(pk)[::-1][:args.top_n]

    # peak vs altitude
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(alts, np.maximum(pk, 1e-2), s=4, alpha=0.25, c="0.4",
               label="null trials")
    ax.scatter(alts[top_idx], np.maximum(pk[top_idx], 1e-2), s=28, c="crimson",
               label=f"top {args.top_n}")
    if census_data.get("eta"):
        ax.axhline(census_data["eta"], color="navy", ls="--", lw=1,
                   label=f"eta = {census_data['eta']:.2f}")
    ax.axvspan(args.band_lo, args.band_hi, color="orange", alpha=0.15,
               label=f"{args.band_lo}-{args.band_hi} km band")
    ax.set_yscale("log")
    ax.set_xlabel("drawn altitude (km)")
    ax.set_ylabel("window-max Mahalanobis (a-channel)")
    ax.set_title("eta-null extreme tail vs altitude (B*/a artifact check)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "peak_vs_altitude.png"), dpi=150)
    plt.close(fig)

    # bins
    bins = census_data["bins"]
    if bins:
        centers = [(b["alt_lo"] + b["alt_hi"]) / 2 for b in bins]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(centers, [b["p95_abs_da_slope"] for b in bins], "o-",
                label="p95 |da_slope| (m/day)")
        ax2 = ax.twinx()
        ax2.plot(centers, [b["max_peak"] for b in bins], "s--", c="crimson",
                 label="max peak")
        ax.set_xlabel("altitude bin center (km)")
        ax.set_ylabel("p95 |fitted-a drift| (m/day)")
        ax2.set_ylabel("max Mahalanobis peak", color="crimson")
        ax.set_title("empirical identifiability knee vs null tail")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "bins_table.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", default="results/synthetic_null_eta_honestR.json",
                   help="eta-null output JSON (or its .partial.json)")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--band-lo", type=float, default=890.0,
                   help="artifact band lower edge [km] (paper: 891)")
    p.add_argument("--band-hi", type=float, default=905.0,
                   help="artifact band upper edge [km] (paper: 903)")
    p.add_argument("--alt-min", type=float, default=700.0,
                   help="jitter altitude range lower edge [km]")
    p.add_argument("--alt-max", type=float, default=1000.0,
                   help="jitter altitude range upper edge [km]")
    p.add_argument("--bin-km", type=float, default=25.0)
    p.add_argument("--eta", type=float, default=None,
                   help="override eta line (default: summary.eta_synth)")
    p.add_argument("--refit-top", type=int, default=0,
                   help="deep-dive: re-run fit stage for top-N trials")
    p.add_argument("--refit-controls", type=int, default=5,
                   help="deep-dive: number of median-peak control trials")
    p.add_argument("--out-dir", default="results/diag_bstar")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    trials_path = args.trials
    if not os.path.isabs(trials_path):
        trials_path = os.path.join(REPO_ROOT, trials_path)
    config, summary, records = load_trials(trials_path)
    if args.eta is None and not summary.get("eta_synth"):
        print("NOTE: no eta_synth in summary - pass --eta for the line")

    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    census_data = census(config, summary, records, args)

    if args.refit_top > 0:
        census_data["refit_table"] = deep_dive(config, summary, records,
                                                args, out_dir)

    with open(os.path.join(out_dir, "census.json"), "w") as f:
        json.dump(census_data, f, indent=2)

    if not args.no_plots:
        make_plots(census_data, records, args, out_dir)

    # ---- verdict -----------------------------------------------------------
    n_in = census_data["n_top_in_band"]
    n_top = census_data["top_n"]
    p = census_data["mc_p_cluster"]
    knee = census_data["identifiability_knee_km"]
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if p is not None and p < 1e-3:
        print(f"  ALTITUDE CLUSTER CONFIRMED: {n_in}/{n_top} top peaks in "
              f"{args.band_lo:.0f}-{args.band_hi:.0f} km "
              f"(MC p = {p:.1e}). The extreme tail is regime-specific.")
    else:
        print(f"  Altitude cluster NOT significant: {n_in}/{n_top} "
              f"(MC p = {p if p is not None else float('nan'):.2f}).")
    if knee:
        print(f"  Identifiability knee at {knee[0]:.0f}-{knee[1]:.0f} km: "
              f"fitted-a drift explodes above it - consistent with "
              f"B*/a degeneracy (drag below fit-noise floor).")
    c = census_data["corr_peak_vs"].get("da_slope_m_per_day")
    if c is not None and c > 0.3:
        print(f"  Peak correlates with |fitted-a drift| (rho = {c:+.2f}): "
              f"extreme peaks ARE drift events, not estimator noise.")
    if "refit_table" in census_data:
        tops = [t for t in census_data["refit_table"] if t["tag"] == "TOP"]
        ctrl = [t for t in census_data["refit_table"] if t["tag"] == "CONTROL"]
        if tops and ctrl:
            f_top = np.median([t["bstar_stats"].get(
                "bstar_sign_flip_rate", np.nan) for t in tops])
            f_ctl = np.median([t["bstar_stats"].get(
                "bstar_sign_flip_rate", np.nan) for t in ctrl])
            print(f"  B* sign-flip rate, top vs control: "
                  f"{f_top:.2f} vs {f_ctl:.2f}.")
    print("  -> If confirmed, eta is calibrated partly against the artifact;")
    print("     see V2 plan: B* clamp/prior in the fit, fit-covariance into")
    print("     R, step-vs-ramp statistic, stratified eta.")
    print(f"\nSaved: {args.out_dir}/census.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())