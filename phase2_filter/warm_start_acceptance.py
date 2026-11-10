#!/usr/bin/env python3
"""FIX A acceptance harness (V2.1).

Replays exact eta-null trials from the saved V2 null run, with Fix A off
(baseline reproduction check) and on, and reports peak movement for the
leftover seeds, the persistent-excursion class, and impulse-mode safety.

Legs:
  1. REPRO: baseline (IMPULSE_FIXA off) must reproduce the peak recorded
     in the null JSON for the same seed. If not, the local module has
     drifted from the one that produced the null: the paired A/B verdict
     below is still internally valid (both legs run on the current module),
     but investigate before quoting.
  2. QUIET: with Fix A on, each named seed's peak must drop below the
     quiet threshold (default: the null run's p99.9 rung).
  3. SCAN (--auto-select [cutoff], default 146): rerun every null trial
     with recorded peak >= cutoff both ways; report the class-wide
     before/after distribution.
  4. SAFETY (--impulse-dv, default 0.5): one seed in impulse mode, both
     ways; the Fix A peak must not collapse vs baseline (no strike
     erasure by the QC).

Examples:
  python -m phase2_filter.fixa_acceptance
  python -m phase2_filter.fixa_acceptance --auto-select 146 --max-scan 40
  python -m phase2_filter.fixa_acceptance --sigma 8 --impulse-dv 0.5
"""

import argparse
import json
import os
import sys

import numpy as np

from phase2_filter.find_dv_threshold import _run_one

DEFAULT_NULL_CONFIG = "results/synthetic_null_eta_honestR_v2fixB.json"
DEFAULT_LEFTOVER_SEEDS = "503523,504718,513933"

_REQUIRED_CFG_KEYS = [
    "window_days", "tau", "q", "threshold", "tle_cadence_days",
    "fit_span_days", "noise_mode", "truth_mode", "fit_bstar",
    "truth_dt_hours", "max_nfev", "edge_trim_days", "r_scale",
    "p0_bwd_inflate", "r_from_fit", "jitter_geometry", "fit_rms_max_m",
]


def build_task(cfg, seed, mode="null", dv=0.0):
    """Rebuild the exact 21-field trial tuple _run_one expects."""
    return (int(seed), float(cfg["window_days"]), float(cfg["tau"]),
            float(cfg["q"]), float(cfg["threshold"]),
            float(cfg["tle_cadence_days"]), float(cfg["fit_span_days"]),
            cfg["noise_mode"], mode, float(dv),
            cfg["truth_mode"], int(cfg["fit_bstar"]),
            float(cfg["truth_dt_hours"]), int(cfg["max_nfev"]),
            float(cfg["edge_trim_days"]), float(cfg["r_scale"]),
            float(cfg["p0_bwd_inflate"]), int(cfg["r_from_fit"]),
            int(cfg["jitter_geometry"]), float(cfg["fit_rms_max_m"]),
            int(cfg["persistence_filter"]))


def run_trial(cfg, seed, fixa, sigma, mode="null", dv=0.0):
    """Run one trial with Fix A toggled via env (read at call time)."""
    os.environ["IMPULSE_FIXA"] = "1" if fixa else "0"
    if fixa and sigma is not None:
        os.environ["IMPULSE_FIXA_QC_SIGMA"] = str(sigma)
    else:
        os.environ.pop("IMPULSE_FIXA_QC_SIGMA", None)
    return _run_one(build_task(cfg, seed, mode, dv))


def peak_of(res):
    if res is None or not res.get("ok"):
        return None
    return float(res["peak_maha"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=
                                 argparse.RawDescriptionHelpFormatter)
    p.add_argument("--null-config", default=DEFAULT_NULL_CONFIG)
    p.add_argument("--seeds", default=DEFAULT_LEFTOVER_SEEDS,
                   help="comma-separated seeds to replay (default: the "
                        "three V2 leftover trials)")
    p.add_argument("--auto-select", type=float, nargs="?", const=146.0,
                   default=None,
                   help="also rerun every null trial with recorded peak "
                        ">= cutoff (default cutoff 146, the bottom of the "
                        "leftover band)")
    p.add_argument("--max-scan", type=int, default=60,
                   help="cap on auto-selected scan trials (peak-sorted)")
    p.add_argument("--sigma", type=float, default=None,
                   help="override IMPULSE_FIXA_QC_SIGMA for the Fix A leg")
    p.add_argument("--quiet", type=float, default=None,
                   help="quiet threshold; default = the null run's p99.9 "
                        "rung")
    p.add_argument("--impulse-dv", type=float, default=0.5,
                   help="impulse-mode safety check dv (0 disables the leg)")
    p.add_argument("--impulse-seed", type=int, default=None)
    p.add_argument("--output", default="results/fixa_acceptance.json")
    args = p.parse_args()

    with open(args.null_config) as f:
        d = json.load(f)
    cfg = dict(d["config"])

    missing = [k for k in _REQUIRED_CFG_KEYS if k not in cfg]
    if missing:
        sys.exit(f"null config missing keys: {missing} - wrong file?")
    if "persistence_filter" not in cfg:
        print("WARNING: config lacks 'persistence_filter'; assuming 1 "
              "(v2fixB run). Verify against your V2 invocation.")
        cfg["persistence_filter"] = 1
    if "mode" in cfg and cfg["mode"] != "null":
        sys.exit("this harness replays null-mode trials only")

    quiet = args.quiet
    if quiet is None:
        pct = d.get("summary", {}).get("percentiles", {})
        quiet = float(pct.get("99.9", 152.714))

    print("FIX A ACCEPTANCE HARNESS")
    print(f"  null config : {args.null_config}")
    print(f"  window={cfg['window_days']}d cadence={cfg['tle_cadence_days']}d "
          f"fit-span={cfg['fit_span_days']}d truth-dt={cfg['truth_dt_hours']}h")
    print(f"  tau={cfg['tau']:.2e} q={cfg['q']:.2e} r-from-fit="
          f"{cfg['r_from_fit']} persistence={cfg['persistence_filter']}")
    print(f"  quiet threshold (target rung): {quiet}")
    print(f"  QC sigma: {args.sigma if args.sigma is not None else 'module default (10)'}")

    recorded = {}
    for t in d.get("trials", []):
        if isinstance(t, dict) and "seed" in t:
            recorded[int(t["seed"])] = t

    report = {"config_source": args.null_config, "quiet": quiet,
              "sigma": args.sigma, "seeds": [], "scan": [], "impulse": None}

    def replay(seed, mode="null", dv=0.0):
        base = run_trial(cfg, seed, False, args.sigma, mode, dv)
        fix = run_trial(cfg, seed, True, args.sigma, mode, dv)
        return base, fix

    # ---- Leg 1+2: the named leftover seeds -----------------------------
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"\n--- Named seeds (leftovers): {seeds} ---")
    print(f"{'seed':>8} {'recorded':>10} {'base(Off)':>10} {'FixA(On)':>10} "
          f"{'warmR':>6} {'suspect':>8} {'verdict':>8}")
    all_quiet = True
    for seed in seeds:
        rec = recorded.get(seed, {})
        rec_peak = rec.get("peak_maha") if rec.get("ok") else None
        base, fix = replay(seed)
        pb, pf = peak_of(base), peak_of(fix)
        if rec_peak is not None and pb is not None:
            repro_ok = abs(pb - rec_peak) <= 1e-6 * (1.0 + abs(rec_peak))
        else:
            repro_ok = None
        diag = fix or {}
        quiet_ok = pf is not None and pf < quiet
        all_quiet = all_quiet and quiet_ok
        verdict = "QUIET" if quiet_ok else "LOUD"
        if repro_ok is False:
            verdict += "/REPRO?"
        print(f"{seed:>8} "
              f"{rec_peak if rec_peak is not None else float('nan'):>10.3f} "
              f"{pb if pb is not None else float('nan'):>10.3f} "
              f"{pf if pf is not None else float('nan'):>10.3f} "
              f"{diag.get('n_warm_recovered', 0):>6} "
              f"{diag.get('n_suspect_epochs', 0):>8} {verdict:>8}")
        if pb is None:
            print(f"    baseline trial FAILED: {base.get('error')}")
        if pf is None:
            print(f"    fixA trial FAILED: {fix.get('error')}")
        else:
            print(f"    fixA diag: suspects={diag.get('n_suspect_epochs')} "
                  f"confirmed={diag.get('n_suspect_confirmed')} "
                  f"continuity={diag.get('n_suspect_continuity')} "
                  f"max_dev={diag.get('fixa_max_norm_dev', float('nan')):.1f} sigma")
        report["seeds"].append({
            "seed": seed, "recorded_peak": rec_peak,
            "base_peak": pb, "fixa_peak": pf, "repro_ok": repro_ok,
            "quiet": quiet_ok,
            "fixa_diag": {k: diag.get(k) for k in
                          ("n_warm_fits", "n_warm_recovered",
                           "n_suspect_epochs", "n_suspect_confirmed",
                           "n_suspect_continuity", "fixa_max_norm_dev")}})

    # ---- Leg 3: class-wide scan ---------------------------------------
    n_class_quiet = 0
    n_class = 0
    if args.auto_select is not None:
        cutoff = float(args.auto_select)
        pool = [t for t in recorded.values()
                if t.get("ok") and t.get("peak_maha") is not None
                and t["peak_maha"] >= cutoff]
        pool.sort(key=lambda t: -t["peak_maha"])
        pool = pool[:args.max_scan]
        print(f"\n--- Class scan: {len(pool)} null trials with recorded "
              f"peak >= {cutoff} ---")
        before, after = [], []
        for i, t in enumerate(pool, 1):
            seed = int(t["seed"])
            base, fix = replay(seed)
            pb, pf = peak_of(base), peak_of(fix)
            if pb is not None:
                before.append(pb)
            if pf is not None:
                after.append(pf)
                if pf < quiet:
                    n_class_quiet += 1
            n_class += 1
            print(f"  [{i:>3}/{len(pool)}] seed={seed} "
                  f"recorded={t['peak_maha']:.1f} base="
                  f"{pb if pb is not None else float('nan'):.1f} fixa="
                  f"{pf if pf is not None else float('nan'):.1f} "
                  f"{'QUIET' if (pf is not None and pf < quiet) else 'LOUD'}")
        report["scan"] = {"cutoff": cutoff, "n": n_class,
                           "n_quiet_after": n_class_quiet,
                           "before": before, "after": after}
        if after:
            print(f"  class: {n_class_quiet}/{n_class} quiet after Fix A "
                  f"({100.0 * n_class_quiet / n_class:.0f}%)")
            print(f"  class peak before: median={np.median(before):.1f} "
                  f"max={np.max(before):.1f}")
            print(f"  class peak after : median={np.median(after):.1f} "
                  f"max={np.max(after):.1f}")

    # ---- Leg 4: impulse-mode safety ------------------------------------
    safety_ok = None
    if args.impulse_dv and args.impulse_dv > 0:
        seed = args.impulse_seed or seeds[0]
        print(f"\n--- Impulse-mode safety: seed={seed}, "
              f"dv={args.impulse_dv} m/s ---")
        base, fix = replay(seed, mode="impulse", dv=args.impulse_dv)
        pb, pf = peak_of(base), peak_of(fix)
        print(f"  baseline peak: {pb}")
        print(f"  fixA     peak: {pf}")
        if pb is not None and pf is not None:
            safety_ok = pf >= 0.5 * pb
            print(f"  detection survives: "
                  f"{'YES' if safety_ok else 'NO - QC IS ERASING STRIKES'}")
        else:
            print("  WARNING: a leg failed; inspect errors below")
            if base is not None and not base.get("ok"):
                print(f"    base error: {base.get('error')}")
            if fix is not None and not fix.get("ok"):
                print(f"    fixA error: {fix.get('error')}")
        report["impulse"] = {"seed": seed, "dv": args.impulse_dv,
                             "base_peak": pb, "fixa_peak": pf,
                             "safety_ok": safety_ok}

    # ---- Verdict -------------------------------------------------------
    print("\n=== VERDICT ===")
    print(f"  named seeds quiet : {'PASS' if all_quiet else 'FAIL'}")
    if args.auto_select is not None and n_class:
        frac = n_class_quiet / n_class
        print(f"  class scan        : {n_class_quiet}/{n_class} quiet "
              f"({'PASS' if frac >= 0.8 else 'REVIEW'} at 80%)")
    if safety_ok is not None:
        print(f"  impulse safety    : {'PASS' if safety_ok else 'FAIL'}")
    repro_fail = [r for r in report["seeds"] if r["repro_ok"] is False]
    if repro_fail:
        print(f"  REPRO WARNING: {len(repro_fail)} baseline peaks differ "
              f"from the null record - module drift; paired A/B still "
              f"valid, investigate before quoting")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {args.output}")

    bad = (not all_quiet) or (safety_ok is False)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()