#!/usr/bin/env python3
"""Phase-0b: what SHAPE is the band-top artifact?

Follows diag_bstar_degeneracy.py. Phase-0 established:
  - the extreme null tail is 100% altitude-clustered (891-903 km),
  - it is NOT B* sign-flipping (flip=0, negB=0, no corr(a, B*) pattern),
  - 18/20 top trials ran at the R_DIAG floor: honest R saw a quiet
    stream while the window filter saw a huge excursion.

That signature (floor R + monster window peak) fits RARE DISCRETE
STEPS in fitted a: median-based R ignores a few outlier residuals,
but a level shift is exactly what the filter responds to. This script
tests step-vs-ramp directly on the per-refit series saved by the
Phase-0 deep dive (results/diag_bstar/refit/*.json), and runs the
within-regime correlations that pooled (diluted) statistics cannot.

Analyses
--------
  [0] run provenance: n_trials, percentile, eta_synth, window, modes,
      and the FULL percentile ladder - checks which rung the paper's
      eta = 417.94 came from (run flag was 99.97 -> 644.11)
  [1] top-N trial table: altitude, inclination, e0, peak, da_slope,
      std_da, sigma_a_used, drift-to-noise ratio, R-floor flag
  [2] top-N geometry vs pool: is the cluster altitude-only, or also
      inclination/eccentricity-conditioned?
  [3] WITHIN-BAND statistics (the test Phase-0 could not do):
      Spearman corr(peak, |da_slope|) inside the artifact band only,
      plus the within-band tail comparison
  [4] refit-series shape classification for every saved refit trial:
      STEP / RAMP / QUIET, jump size, jump day, slope, residual RMS,
      B* scatter - top group vs controls
  [5] per-trial PNGs of da(t) and bstar(t) (if matplotlib available)

Usage (repo root):
    python diag_refit_series.py \
        --trials results/synthetic_null_eta_honestR.json
    python diag_refit_series.py --trials ... --band-lo 875 --band-hi 905

Outputs -> results/diag_bstar/:
    series_census.json, refit_series_<seed>.png (per refit trial)
"""

import argparse
import json
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_input(path):
    """Resolve an input path against cwd, the script dir, and its parent
    (the script may live in a subpackage while data sits at repo level)."""
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), REPO_ROOT, os.path.dirname(REPO_ROOT)):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return os.path.join(os.getcwd(), path)


def resolve_refit_dir(path):
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), REPO_ROOT, os.path.dirname(REPO_ROOT)):
        cand = os.path.join(base, path)
        if os.path.isdir(cand):
            return cand
    return os.path.join(os.getcwd(), path)
try:
    sys.path.insert(0, REPO_ROOT)
    from constants.constants import R_EARTH
except Exception:
    R_EARTH = 6371000.0


def load_trials(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "trials" in data:
        return data.get("config", {}), data.get("summary", {}), data["trials"]
    if isinstance(data, dict) and "partial" in data:
        return data.get("config", {}), {}, data["partial"]
    if isinstance(data, list):
        return {}, {}, data
    raise SystemExit(f"unrecognized trials file structure: {path}")


def alt_km(rec):
    try:
        return (rec["geometry"]["a0_m"] - R_EARTH) / 1000.0
    except (KeyError, TypeError):
        return None


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


def robust_sigma(v):
    """MAD-based sigma of a 1-D array."""
    v = np.asarray(v, float)
    if len(v) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def classify_series(t_d, da):
    """Shape classification of the fitted-a-minus-truth series.

    STEP:  one (or few) discrete jump(s) dominate the total excursion:
           jump_ratio > 10, judged against the ROBUST scatter of the
           OTHER steps. (Not the OLS residual RMS: a single giant step
           drags the line fit with it, so resid_rms ~ jump/2 and any
           absolute rms threshold is defeated. v1 of this classifier
           mislabeled giant steps as QUIET for exactly this reason.)
           persistence = (post-jump median level - pre-jump median
           level) / jump:
             ~ +1/-1 = level HOLDS  -> a persistent level shift (what
                       a real impulse looks like in fitted a),
             ~ 0     = level REVERTS -> transient one-off fit outlier
                       (killable by a persistence/median filter).
    RAMP:  monotone drift with small robust residuals about a line.
    QUIET: no structure beyond noise.
    """
    n = len(da)
    out = {"n_refits": n}
    if n < 6:
        out["shape"] = "too-short"
        return out
    slope, intercept = np.polyfit(t_d, da, 1)
    resid = da - (slope * t_d + intercept)
    rsig = max(robust_sigma(resid), 1e-9)
    diffs = np.diff(da)
    j = int(np.argmax(np.abs(diffs)))
    jump = float(diffs[j])
    others = np.delete(diffs, j)
    sig_others = max(robust_sigma(others), 1e-9)
    jump_ratio = abs(jump) / sig_others
    mono = float(np.mean(np.sign(diffs) == np.sign(slope))) if slope != 0 \
        else 0.5
    total_drift = abs(slope) * (t_d[-1] - t_d[0])
    excursion = float(np.max(da) - np.min(da))
    persistence = None
    if abs(jump) > 1e-9:
        pre_level = float(np.median(da[:j + 1]))
        post_slice = da[min(j + 3, n):]
        post_level = (float(np.median(post_slice)) if len(post_slice)
                      else float(da[-1]))
        persistence = float((post_level - pre_level) / jump)
    if jump_ratio > 10:
        shape = "STEP"
    elif total_drift > 5 * rsig and mono > 0.7:
        shape = "RAMP"
    else:
        shape = "QUIET"
    out.update({
        "shape": shape,
        "da_slope_m_per_day": float(slope),
        "resid_robust_sigma_m": rsig,
        "resid_rms_m": float(np.sqrt(np.mean(resid ** 2))),
        "jump_m": jump,
        "jump_day": float(t_d[j]),
        "jump_ratio": float(jump_ratio),
        "persistence": persistence,
        "mono_frac": mono,
        "total_drift_m": float(total_drift),
        "excursion_m": excursion,
        "step_scatter_m": robust_sigma(diffs),
    })
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials",
                   default="results/synthetic_null_eta_honestR.json")
    p.add_argument("--refit-dir", default="results/diag_bstar/refit")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--band-lo", type=float, default=875.0)
    p.add_argument("--band-hi", type=float, default=905.0)
    p.add_argument("--paper-n-trials", type=int, default=17000,
                   help="trial count quoted for the eta run (17,000 per "
                        "the author's confirmation); 0 disables the check")
    p.add_argument("--out-dir", default="results/diag_bstar")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    trials_path = resolve_input(args.trials)
    config, summary, records = load_trials(trials_path)
    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(os.getcwd(), args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    ok = [r for r in records if r.get("ok") and alt_km(r) is not None]

    # ---- [0] provenance ----------------------------------------------------
    print("=" * 72)
    print("[0] RUN PROVENANCE")
    print("=" * 72)
    for k in ("n_trials", "percentile", "window_days", "truth_mode",
              "fit_bstar", "noise_mode", "r_from_fit", "jitter_geometry",
              "seed_offset"):
        if k in config:
            print(f"    {k:<16} {config[k]}")
    eta = summary.get("eta_synth")
    print(f"    eta_synth        {eta}")
    # percentile ladder: settles which rung the paper's eta=417.94 came
    # from (the run's target flag was 99.97 -> 644.11, but the ladder
    # stores p99.9 etc. as well).
    pct = summary.get("percentiles")
    if isinstance(pct, dict) and pct:
        print("    percentile ladder (summary['percentiles']):")
        for k in sorted(pct):
            v = pct[k]
            if isinstance(v, (int, float)):
                print(f"        {k:<10} {v:.4f}")
        hits = [str(k) for k, v in pct.items()
                if isinstance(v, (int, float))
                and abs(float(v) - 417.94) < 0.5]
        if hits:
            print(f"    -> paper eta 417.94 matches ladder rung(s): "
                  f"{', '.join(hits)}")
            print("       PROVENANCE RECONCILED: the paper quotes this "
                  "percentile of the")
            print("       same 17,000-trial run (run target flag was "
                  "99.97 -> 644.11).")
        else:
            print("    -> NO ladder rung matches 417.94: the paper's eta "
                  "came from a")
            print("       different run; locate that trials file before "
                  "using its numbers")
            print("       in V2.")
    n_in_file = len(records)
    if args.paper_n_trials and n_in_file != args.paper_n_trials:
        print(f"\n    *** PROVENANCE WARNING: this file holds {n_in_file} "
              f"trials but the paper quotes {args.paper_n_trials}. Confirm")
        print(f"        this is the run that produced the quoted eta before")
        print(f"        using its numbers in V2.")

    # ---- [1] top-N table ----------------------------------------------------
    ok.sort(key=lambda r: -r["peak_maha"])
    top = ok[:args.top_n]
    win_days = float(config.get("window_days", 90.0))
    floor = None
    try:
        sys.path.insert(0, REPO_ROOT)
        from phase2_filter.config import R_DIAG_ELEMENTS
        floor = float(R_DIAG_ELEMENTS[0])
    except Exception:
        ru = [r.get("r_used") for r in ok if r.get("r_used")]
        if ru:
            floor = min(float(v[0]) for v in ru)
    print("\n" + "=" * 72)
    print(f"[1] TOP-{len(top)} TRIALS (full geometry + drift forensics)")
    print("=" * 72)
    hdr = (f"    {'seed':>7} {'alt km':>7} {'inc deg':>8} {'e0':>8} "
           f"{'peak':>8} {'da_slope':>9} {'std_da':>8} {'sig_a':>6} "
           f"{'DNR':>6} {'R':>6}")
    print(hdr)
    top_rows = []
    for r in top:
        g = r.get("geometry", {})
        slope = r.get("da_slope_m_per_day")
        std_da = r.get("std_da_m")
        ru = r.get("r_used")
        sig_a = math.sqrt(float(ru[0])) if ru else float("nan")
        dnr = abs(slope) * win_days / sig_a if (slope and ru) else float("nan")
        rflag = "floor" if (floor is not None and ru
                            and abs(float(ru[0]) - floor)
                            < 1e-12 * max(1.0, floor)) else "-"
        row = {"seed": r["seed"], "alt_km": alt_km(r),
               "inc0_deg": g.get("inc0_deg"), "e0": g.get("e0"),
               "peak_maha": r["peak_maha"], "da_slope": slope,
               "std_da_m": std_da, "sigma_a_m": sig_a, "dnr": dnr,
               "r_flag": rflag}
        top_rows.append(row)
        print(f"    {r['seed']:>7} {alt_km(r):>7.1f} "
              f"{g.get('inc0_deg', float('nan')):>8.2f} "
              f"{g.get('e0', float('nan')):>8.1e} "
              f"{r['peak_maha']:>8.2f} "
              f"{slope if slope is not None else float('nan'):>9.3f} "
              f"{std_da if std_da is not None else float('nan'):>8.2f} "
              f"{sig_a:>6.2f} {dnr:>6.1f} {rflag:>6}")

    # ---- [2] geometry conditioning -----------------------------------------
    print("\n" + "=" * 72)
    print("[2] IS THE CLUSTER ALTITUDE-ONLY? (top-N vs pool)")
    print("=" * 72)
    pool_inc = np.array([r["geometry"].get("inc0_deg", float("nan"))
                          for r in ok], float)
    pool_e = np.array([r["geometry"].get("e0", float("nan"))
                       for r in ok], float)
    top_inc = np.array([t["inc0_deg"] for t in top_rows
                        if t["inc0_deg"] is not None], float)
    top_e = np.array([t["e0"] for t in top_rows
                      if t["e0"] is not None], float)
    print(f"    inclination: top-N  range {np.nanmin(top_inc):.1f}-"
          f"{np.nanmax(top_inc):.1f}, median {np.nanmedian(top_inc):.1f}"
          f" | pool median {np.nanmedian(pool_inc):.1f} "
          f"(pool range 50-105 by construction)")
    print(f"    eccentricity: top-N range {np.nanmin(top_e):.1e}-"
          f"{np.nanmax(top_e):.1e}, median {np.nanmedian(top_e):.1e}"
          f" | pool median {np.nanmedian(pool_e):.1e}")
    if np.nanstd(top_inc) < 0.25 * np.nanstd(pool_inc):
        print("    -> inclination is ALSO clustered: the artifact is "
              "geometry-conditioned, not altitude-only.")
    if np.nanstd(top_e) < 0.25 * np.nanstd(pool_e):
        print("    -> eccentricity is ALSO clustered: check the small-e "
              "fitting degeneracy as a co-condition.")

    # ---- [3] within-band statistics -----------------------------------------
    print("\n" + "=" * 72)
    print(f"[3] WITHIN-BAND STATISTICS "
          f"({args.band_lo:.0f}-{args.band_hi:.0f} km only)")
    print("=" * 72)
    band = [r for r in ok if args.band_lo <= alt_km(r) <= args.band_hi]
    hi, lo = [], []
    bp = np.array([r["peak_maha"] for r in band], float)
    bs = np.array([abs(r["da_slope_m_per_day"]) if
                   r.get("da_slope_m_per_day") is not None else np.nan
                   for r in band], float)
    n_drift = int(np.sum(np.isfinite(bs)))
    rho = spearman(bp, bs)
    print(f"    n in band: {len(band)} (with da_slope: {n_drift})")
    print(f"    Spearman corr(peak, |da_slope|) WITHIN band: "
          f"{'n/a' if rho is None else format(rho, '+.3f')}")
    # within-band tail comparison: top decile vs rest
    q90 = np.percentile(bp, 90)
    hi = [abs(r["da_slope_m_per_day"]) for r in band
          if r["peak_maha"] >= q90
          and r.get("da_slope_m_per_day") is not None]
    lo = [abs(r["da_slope_m_per_day"]) for r in band
          if r["peak_maha"] < q90
          and r.get("da_slope_m_per_day") is not None]
    if hi and lo:
        print(f"    |da_slope| median, top-decile peaks vs rest: "
              f"{np.median(hi):.3f} vs {np.median(lo):.3f} m/day "
              f"(x{np.median(hi) / max(np.median(lo), 1e-9):.1f})")
        if np.median(hi) > 5 * max(np.median(lo), 1e-9):
            print("    -> WITHIN the regime, drift tracks the tail: the")
            print("       fitted-a drift IS the within-band driver.")
        else:
            print("    -> drift does NOT separate the within-band tail;")
            print("       the extreme peaks need a different signature")
            print("       (check [4]: discrete steps?)")

    # ---- [4] refit-series shapes --------------------------------------------
    print("\n" + "=" * 72)
    print("[4] REFIT-SERIES SHAPE CLASSIFICATION (saved deep-dive trials)")
    print("=" * 72)
    refit_dir = resolve_refit_dir(args.refit_dir)
    if not os.path.isdir(refit_dir):
        print(f"    no refit dir ({args.refit_dir}); run Phase-0 with "
              f"--refit-top N first. Skipping.")
        refit_table = []
    else:
        refit_table = []
        files = sorted(os.listdir(refit_dir))
        gate = float(config.get("fit_rms_max_m", 5000.0)) \
            if isinstance(config, dict) else 5000.0
        for fn in files:
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(refit_dir, fn)) as f:
                rr = json.load(f)
            pr_all = rr.get("per_refit", [])
            has_rms = bool(pr_all) and all(
                c.get("fit_rms_m") is not None for c in pr_all)
            # QC gate: the Phase-0 capture saved PRE-gate fits too.
            # Classify the ACCEPTED stream (what the filter actually
            # saw); report how many captured fits the gate dropped and
            # whether the jumping fit itself passed QC.
            if has_rms:
                pr = [c for c in pr_all if float(c["fit_rms_m"]) <= gate]
            else:
                pr = pr_all
            n_qc_rej = len(pr_all) - len(pr)
            if len(pr) < 6:
                continue
            t_d = np.array([c["t_ep_s"] for c in pr], float) / 86400.0
            da = np.array([c["da_m"] for c in pr], float)
            rms_series = (np.array([c["fit_rms_m"] for c in pr], float)
                          if has_rms else None)
            bs = np.array([c["bstar"] for c in pr if c["bstar"] is not None],
                          float)
            cls = classify_series(t_d, da)
            cls["tag"] = rr.get("tag")
            cls["seed"] = rr.get("seed")
            cls["alt_km"] = rr.get("alt_km")
            cls["peak_maha"] = rr.get("peak_maha")
            cls["bstar_std"] = float(np.std(bs)) if len(bs) else None
            cls["n_captured_fits"] = len(pr_all)
            cls["n_qc_rejected_fits"] = n_qc_rej if has_rms else None
            if has_rms and cls.get("jump_m") is not None:
                k = int(np.argmin(np.abs(t_d - cls["jump_day"])))
                kk = min(k + 1, len(pr) - 1)
                cls["jump_fit_rms_m"] = float(rms_series[kk])
                cls["jump_fit_passed_qc"] = bool(rms_series[kk] <= gate)
            refit_table.append(cls)
            pers = cls.get("persistence")
            pers_s = f"{pers:+.2f}" if pers is not None else "  n/a"
            qc_s = "-"
            if cls.get("jump_fit_rms_m") is not None:
                qc_s = ("PASS" if cls["jump_fit_passed_qc"] else "REJ") \
                    + f" rms={cls['jump_fit_rms_m']:.0f}m"
            print(f"    {str(rr.get('tag')):<7} seed={rr.get('seed'):<7} "
                  f"alt={rr.get('alt_km'):7.1f} "
                  f"peak={rr.get('peak_maha'):8.2f}  "
                  f"{cls.get('shape'):<6} "
                  f"slope={cls.get('da_slope_m_per_day', float('nan')):8.3f} "
                  f"jump={cls.get('jump_m', float('nan')):8.1f} m "
                  f"(x{cls.get('jump_ratio', float('nan')):6.0f} scatter, "
                  f"day {cls.get('jump_day', float('nan')):5.0f})  "
                  f"persist={pers_s}  jumpQC={qc_s}  "
                  f"qcRej={n_qc_rej if has_rms else 'n/a'}")
        # group summary
        for tag in ("TOP", "CONTROL"):
            grp = [c for c in refit_table if c.get("tag") == tag]
            if not grp:
                continue
            shapes = [c.get("shape") for c in grp]
            jumps = [abs(c.get("jump_m") or 0.0) for c in grp]
            pers = [c.get("persistence") for c in grp
                    if c.get("persistence") is not None]
            line = (f"\n    {tag}: shapes {shapes}; median |jump| "
                    f"{np.median(jumps):.1f} m")
            if pers:
                line += f"; persistence median {np.median(pers):+.2f}"
            print(line)
        print("\n    Interpretation: STEP = discrete fitted-a jumps. The")
        print("    discriminator vs real impulses is PERSISTENCE, not")
        print("    size: a real along-track strike at dv = 1.19 m/s makes")
        print("    da ~ 2.3 km, the same scale as these artifact jumps.")
        print("    jumpQC=PASS on a km-scale jump means the fit found a")
        print("    degenerate element combination that still reproduces")
        print("    the positions (convention aliasing / small-e null")
        print("    space); jumpQC=REJ means the position gate caught it")
        print("    and the accepted stream is already cleaner than the")
        print("    raw capture suggested.")

    # ---- [5] plots -----------------------------------------------------------
    if not args.no_plots and refit_table:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for cls in refit_table:
                fn = f"refit_series_{cls['seed']}.png"
                path = os.path.join(refit_dir, fn)
                # re-read per_refit from the refit json
                with open(os.path.join(
                        refit_dir,
                        f"{cls['seed']}.json")) as f:
                    rr = json.load(f)
                pr_all = rr["per_refit"]
                if all(c.get("fit_rms_m") is not None for c in pr_all):
                    pr = [c for c in pr_all
                          if float(c["fit_rms_m"]) <= gate]
                else:
                    pr = pr_all
                t_d = np.array([c["t_ep_s"] for c in pr], float) / 86400.0
                da = np.array([c["da_m"] for c in pr], float)
                bt = np.array([c["bstar"] for c in pr
                               if c["bstar"] is not None], float)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5),
                                               sharex=True)
                ax1.plot(t_d, da, "o-", ms=3)
                ax1.set_ylabel("fitted a - truth a (m)")
                ax1.set_title(f"seed {cls['seed']} | {cls['tag']} | "
                              f"peak {cls['peak_maha']:.1f} | "
                              f"{cls['shape']}")
                if len(bt) == len(da):
                    ax2.plot(t_d, bt, "s-", ms=3, color="darkorange")
                    ax2.set_ylabel("fitted B*")
                ax2.set_xlabel("days since window start")
                fig.tight_layout()
                fig.savefig(path, dpi=150)
                plt.close(fig)
            print(f"\n[5] saved {len(refit_table)} series PNGs -> "
                  f"{args.refit_dir}/refit_series_<seed>.png")
        except ImportError:
            print("\n[5] matplotlib unavailable - skipping series PNGs")

    # ---- save + verdict ------------------------------------------------------
    census = {
        "provenance": {"config": config, "eta_synth": eta,
                       "n_trials_in_file": n_in_file,
                       "paper_n_trials": args.paper_n_trials},
        "top_rows": top_rows,
        "within_band": {
            "band_km": [args.band_lo, args.band_hi], "n": len(band),
            "spearman_peak_vs_abs_da_slope": rho,
            "abs_da_slope_median_top_decile": float(np.median(hi))
            if hi else None,
            "abs_da_slope_median_rest": float(np.median(lo)) if lo else None,
        },
        "refit_table": refit_table,
    }
    with open(os.path.join(out_dir, "series_census.json"), "w") as f:
        json.dump(census, f, indent=2)

    print("\n" + "=" * 72)
    print("VERDICT (revised-mechanism check)")
    print("=" * 72)
    tops = [c for c in refit_table if c.get("tag") == "TOP"]
    if tops:
        n_step = sum(1 for c in tops if c.get("shape") == "STEP")
        n_ramp = sum(1 for c in tops if c.get("shape") == "RAMP")
        print(f"  TOP series: {n_step} STEP, {n_ramp} RAMP, "
              f"{len(tops) - n_step - n_ramp} other")
        step_rows = [c for c in tops if c.get("shape") == "STEP"]
        pers = [c.get("persistence") for c in step_rows
                if c.get("persistence") is not None]
        qc_pass = [c for c in step_rows
                   if c.get("jump_fit_passed_qc") is True]
        qc_rej = [c for c in step_rows
                  if c.get("jump_fit_passed_qc") is False]
        if step_rows:
            if qc_pass or qc_rej:
                print(f"  STEP jumps vs position-QC gate: "
                      f"{len(qc_pass)} PASSED, {len(qc_rej)} REJECTED "
                      f"(accepted stream is what the filter saw)")
            if pers:
                med_p = float(np.median(pers))
                print(f"  STEP persistence: median {med_p:+.2f} "
                      f"(|p|~0 transient, |p|~1 persistent)")
                if abs(med_p) < 0.5:
                    print("  -> TRANSIENT excursions: levels revert within")
                    print("     a few refits. A persistence/median (Hampel)")
                    print("     pre-filter on the fitted-a stream kills the")
                    print("     artifact and leaves real impulses (which")
                    print("     persist) intact: safe estimator-side fix,")
                    print("     apply to null + gates + fleet identically.")
                else:
                    print("  -> PERSISTENT level shifts: fits stay on the")
                    print("     new branch. Only source-side fixes work:")
                    print("     warm-start/sequential refits + element-space")
                    print("     prior QC, or fit-covariance R. A median")
                    print("     filter would ALSO absorb real impulses.")
        elif n_step > n_ramp:
            print("  -> Mechanism: DISCRETE fitted-a steps, not smooth")
            print("     B*-driven ramps.")
        elif n_ramp > n_step:
            print("  -> RAMP-dominated: the smooth-degeneracy story "
                  "stands; V2 fix targets: B* prior/clamp, slow")
            print("     nuisance state, fit-covariance into R.")
        else:
            print("  -> Mixed or inconclusive: inspect the PNGs and "
                  "consider refitting more trials (--refit-top 20).")
    if eta is not None:
        print(f"  eta_synth in this file: {eta:.4f} (verify vs the "
              f"paper's 417.94)")
    print(f"\nSaved: {args.out_dir}/series_census.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())