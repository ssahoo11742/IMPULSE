#!/usr/bin/env python3
"""IMPULSE V1.1 one-shot reproduction driver.

Runs every V1.1 pipeline stage end-to-end, in dependency order, with the
paper's configuration as defaults. Place this file at the REPO ROOT (next
to phase2_filter/, TLE/, constants/, propagator/, flux_analysis/) and run:

    python pipeline.py                      # full reproduction
    python pipeline.py --list-stages        # show the stage map
    python pipeline.py --stages prepare,validate,eta
    python pipeline.py --eta 417.94         # skip the overnight eta run
    python pipeline.py --dry-run            # print commands, run nothing
    python pipeline.py --stages report      # just summarize existing JSONs

Stage map (run order)
---------------------
  prepare   verify deps/data; derive meta + durations JSONs if absent
  validate  quick filter sanity ladder (run_validation --step all)
  eta       pooled synthetic null -> eta_synth            [overnight]
  gate2     confounder nulls at eta (gate2_confounders)
  gate3     real-TLE strike injection -> dv50_real        [hours]
  gate4     synthetic impulse sweep -> eps(dv), dv50_synth [hours]
  fleet     full fleet run, honest R (run_real_fleet)     [overnight]
  null      permuted reference null (scrambled_tle_null, residual mode)
  flux      flux_comparison.csv (if absent) + per-object ORDEM/MASTER flux
  predict   expected detectable events (eval_fleet_efficiency, 76 searched
            days per window - the paper convention)
  fig5      Figure 5 covariance ratios (analysis/v1-x/cov-fig.py;
            needs matplotlib)
  report    side-by-side vs the paper's numbers + results/pipeline_manifest.json

Behavior
--------
* Stages call the module CLIs (python -m phase2_filter....), so each stage
  keeps its own multiprocessing, checkpointing and resume behavior.
* A stage whose output file(s) already exist is SKIPPED; use --force to
  redo. Nothing already computed is thrown away.
* eta: if --eta is given, the eta stage is skipped and the value is used
  for every downstream stage. Otherwise eta is read from the eta-null JSON
  (summary.eta_synth).
* If TLE/real_tles.tle.meta.json or a durations JSON is absent, prepare
  derives stand-ins from the repo's own data (real_tles.tle and
  TLE/histories). The derived durations use each history's epoch span
  capped to 2009-01-01..2026-09-01, which can differ slightly from the
  Space-Track satcat LAUNCH/DECAY durations used for the paper - pass
  --meta/--durations to use the official files if you have them.
* Runtime: eta (18,582 trials) and fleet (3,325 objects) are overnight on a
  4-core laptop; gate3/gate4 are hours; the rest are minutes.

Terminology note: windows whose peak exceeds eta are CANDIDATES, never
detections. The pipeline preserves that wording in its report.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Paper reference values (V1.1 abstract / tables) for the report comparison
# ---------------------------------------------------------------------------
PAPER = {
    "eta": 417.94,
    "dv50_synth": 1.19,          # m/s (Gate 4 synthetic sweep)
    "dv50_real": 1.21,           # m/s (Gate 3 real-TLE injection)
    "n_objects": 3325,
    "n_valid_windows": 15760,
    "candidates": 6,
    "expected_at_design": 15.76,  # = 0.1% design rate x 15,760 windows
}

# Canonical stage outputs (relative to repo root)
ETA_JSON = "results/synthetic_null_eta_honestR.json"
GATE2_JSON = "results/gate2_confounders_v2.json"
GATE3_JSON = "results/real_tle_efficiency.json"
GATE4_JSON = "results/gate4_efficiency_sweep.json"
FLEET_JSON = "results/real_detections_honestR.json"
NULL_JSON = "results/scrambled_null_residual.json"
ORDEM_FLUX_JSON = "results/ordem_flux_per_object.json"
MASTER_FLUX_JSON = "results/master_flux_per_object.json"
PREDICT_JSON = "results/fleet_expected_detected.json"
MANIFEST_JSON = "results/pipeline_manifest.json"
FLUX_CSV = "flux_analysis/flux_comparison.csv"
FIG5_PNG = "fig5_covariance_ratios.png"

STAGE_ORDER = [
    "prepare", "validate", "eta", "gate2", "gate3", "gate4",
    "fleet", "null", "flux", "predict", "fig5", "report",
]

CAP_START = datetime(2009, 1, 1, tzinfo=timezone.utc)
CAP_END = datetime(2026, 9, 1, tzinfo=timezone.utc)

STAGE_LOG = []   # [{name, cmd, wall_s, outputs, status}]


class StageError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def py():
    return sys.executable


def mod(name):
    return [py(), "-m", name]


def read_json(path):
    with open(os.path.join(REPO_ROOT, path)) as f:
        return json.load(f)


def exists(path):
    return os.path.exists(os.path.join(REPO_ROOT, path))


def run_stage(args, name, cmd, outputs):
    """Run one stage CLI; skip if all outputs exist unless --force."""
    if (not args.force) and outputs and all(exists(o) for o in outputs):
        print(f"[{name}] outputs already exist; skipping (--force to redo)")
        STAGE_LOG.append({"name": name, "cmd": cmd, "wall_s": 0.0,
                          "outputs": outputs, "status": "skipped"})
        return True
    if args.dry_run:
        print(f"[{name}] DRY-RUN: {' '.join(cmd)}")
        STAGE_LOG.append({"name": name, "cmd": cmd, "wall_s": 0.0,
                          "outputs": outputs, "status": "dry-run"})
        return True
    print("\n" + "=" * 72)
    print(f"[{name}] $ {' '.join(cmd)}")
    print("=" * 72)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    wall = time.time() - t0
    ok = proc.returncode == 0
    STAGE_LOG.append({"name": name, "cmd": cmd, "wall_s": round(wall, 1),
                      "outputs": outputs,
                      "status": "ok" if ok else f"exit {proc.returncode}"})
    if not ok:
        raise StageError(
            f"stage '{name}' failed with exit code {proc.returncode}; "
            f"stopping. Re-run with --stages {name} after fixing, or pass "
            f"--keep-going to continue anyway.")
    print(f"[{name}] done in {wall / 60.0:.1f} min")
    return True


def _tle_epoch(line1):
    """Epoch datetime from a TLE line 1 (same convention as tle_io)."""
    ep = line1[18:32].strip()
    y2 = int(ep[:2])
    yr = 2000 + y2 if y2 < 57 else 1900 + y2
    doy = float(ep[2:])
    return datetime(yr, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)


def derive_meta(tle_file, out_path):
    """Synthesize a meta JSON (NORAD_CAT_ID, INCLINATION, ...) from the
    fleet TLE snapshot - the two keys downstream consumers actually read."""
    from TLE.tle_io import load_tles_from_file, parse_tle
    seen = {}
    for name, l1, l2 in load_tles_from_file(tle_file):
        d = parse_tle(name, l1, l2)
        seen[d["norad_id"]] = {
            "NORAD_CAT_ID": d["norad_id"],
            "OBJECT_NAME": d["name"],
            "INCLINATION": math.degrees(d["inc"]),
            "ECCENTRICITY": d["ecc"],
            "ALTITUDE_KM": (d["a"] - 6371000.0) / 1000.0,
            "EPOCH": d["epoch_dt"].isoformat(),
            "MEAN_MOTION_REV_DAY": d["n_revday"],
        }
    records = list(seen.values())
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    return len(records)


def derive_durations(hist_dir, nids, out_path):
    """Synthesize a durations JSON {nid: {launch, decay, duration_days}} from
    each history file's epoch span, capped to 2009-01-01..2026-09-01
    (the same cap window as TLE/cap_durs.py)."""
    durations = {}
    for nid in sorted(nids):
        fpath = os.path.join(hist_dir, nid + ".tle")
        if not os.path.isfile(fpath):
            continue
        epochs = []
        with open(fpath, errors="replace") as f:
            for line in f:
                if line.startswith("1 "):
                    try:
                        epochs.append(_tle_epoch(line))
                    except (ValueError, IndexError):
                        pass
        if not epochs:
            continue
        start = min(max(min(epochs), CAP_START), CAP_END)
        end = max(min(max(epochs), CAP_END), CAP_START)
        d = max(0, (end - start).days)
        durations[nid] = {
            "launch": start.date().isoformat(),
            "decay": end.date().isoformat(),
            "duration_days": d,
            "source": "derived_from_tle_history_span",
        }
    with open(out_path, "w") as f:
        json.dump(durations, f, indent=2)
    return len(durations)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_prepare(args, ctx):
    problems = []

    # -- importable packages ---------------------------------------------
    probe = subprocess.run(
        [py(), "-c", "import numpy, scipy, sgp4, phase2_filter, TLE, "
                     "constants, propagator"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if probe.returncode != 0:
        problems.append("missing python packages or repo layout:\n"
                        + (probe.stderr.strip() or probe.stdout.strip()))
    mpl = subprocess.run([py(), "-c", "import matplotlib"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    has_mpl = mpl.returncode == 0
    if not has_mpl:
        print("[prepare] NOTE: matplotlib not installed - stage 'fig5' "
              "will fail (pip install matplotlib if you want Figure 5)")

    # -- data --------------------------------------------------------------
    if not exists(args.tle_file):
        problems.append(f"missing fleet snapshot: {args.tle_file}")
    if not os.path.isdir(os.path.join(REPO_ROOT, args.tle_history_dir)):
        problems.append(f"missing TLE history dir: {args.tle_history_dir}")
    else:
        n_hist = len([f for f in os.listdir(os.path.join(REPO_ROOT,
                     args.tle_history_dir)) if f.endswith(".tle")])
        print(f"[prepare] TLE histories: {n_hist} files "
              f"(paper fleet: {PAPER['n_objects']})")

    # -- meta: use given file, else derive ---------------------------------
    meta_path = args.meta
    if not exists(meta_path):
        derived = os.path.join(REPO_ROOT, args.derived_meta)
        if args.dry_run:
            print(f"[prepare] DRY-RUN: would derive meta -> {args.derived_meta}")
        else:
            os.makedirs(os.path.dirname(derived), exist_ok=True)
            n = derive_meta(os.path.join(REPO_ROOT, args.tle_file), derived)
            print(f"[prepare] meta absent; derived {n} records from "
                  f"{args.tle_file} -> {args.derived_meta}")
            if abs(n - PAPER["n_objects"]) > max(50, 0.02 * PAPER["n_objects"]):
                print(f"[prepare] WARNING: derived meta has {n} objects, "
                      f"paper used {PAPER['n_objects']}")
        meta_path = args.derived_meta
    ctx["meta"] = meta_path

    # -- durations: use given file, else derive ----------------------------
    dur_path = args.durations
    if not exists(dur_path):
        derived = os.path.join(REPO_ROOT, args.derived_durations)
        if args.dry_run:
            print(f"[prepare] DRY-RUN: would derive durations -> "
                  f"{args.derived_durations}")
        else:
            with open(os.path.join(REPO_ROOT, meta_path)) as f:
                nids = [str(r.get("NORAD_CAT_ID", ""))
                        for r in json.load(f)]
            os.makedirs(os.path.dirname(derived), exist_ok=True)
            n = derive_durations(
                os.path.join(REPO_ROOT, args.tle_history_dir), nids, derived)
            print(f"[prepare] durations absent; derived {n} spans from "
                  f"{args.tle_history_dir} -> {args.derived_durations}")
            print("[prepare] NOTE: derived durations = TLE-history epoch "
                  "spans capped to 2009-2026, which can differ slightly "
                  "from the satcat LAUNCH/DECAY durations used in the "
                  "paper (TLE/fetch_dur.py + cap_durs.py).")
        dur_path = args.derived_durations
    ctx["durations"] = dur_path

    if problems:
        raise StageError("prepare failed:\n  " + "\n  ".join(problems))
    print(f"[prepare] meta -> {ctx['meta']}")
    print(f"[prepare] durations -> {ctx['durations']}")
    return True


def stage_validate(args, ctx):
    cmd = mod("phase2_filter.run_validation") + ["--step", "all"]
    return run_stage(args, "validate", cmd, outputs=[])


def stage_eta(args, ctx):
    cmd = mod("phase2_filter.convention_consistency_test") + [
        "--mode", "null",
        "--truth-mode", "sgp4_consistent",
        "--n-trials", str(args.n_trials_eta),
        "--fit-bstar", "1",
        "--r-from-fit", "1",
        "--jitter-geometry", "1",
        "--percentile", str(args.percentile),
        "--seed-offset", str(args.seed_offset_eta),
        "--n-workers", str(args.n_workers),
        "--output", ETA_JSON,
    ]
    run_stage(args, "eta", cmd, outputs=[ETA_JSON])
    ctx["eta"] = float(read_json(ETA_JSON)["summary"]["eta_synth"])
    print(f"[eta] eta_synth = {ctx['eta']:.4f} (paper: {PAPER['eta']})")
    return True


def resolve_eta(args, ctx):
    if ctx.get("eta") is None:
        if args.eta is not None:
            ctx["eta"] = float(args.eta)
        elif exists(ETA_JSON):
            ctx["eta"] = float(read_json(ETA_JSON)["summary"]["eta_synth"])
        else:
            raise StageError(
                "no eta available: run the 'eta' stage, or pass --eta "
                "(paper value: 417.94)")
    return ctx["eta"]


def stage_gate2(args, ctx):
    eta = resolve_eta(args, ctx)
    cmd = mod("phase2_filter.gate2_confounders") + [
        "--threshold", f"{eta:.4f}",
        "--n-trials", str(args.n_trials_gate2),
        "--n-workers", str(args.n_workers),
        "--output", GATE2_JSON,
    ]
    return run_stage(args, "gate2", cmd, outputs=[GATE2_JSON])


def stage_gate3(args, ctx):
    eta = resolve_eta(args, ctx)
    cmd = mod("phase2_filter.real_tle_strike_injection") + [
        "--tle-history-dir", args.tle_history_dir,
        "--meta", ctx["meta"],
        "--durations", ctx["durations"],
        "--threshold", f"{eta:.4f}",
        "--r-from-fit", "1",
        "--n-workers", str(args.n_workers),
        "--output", GATE3_JSON,
    ]
    return run_stage(args, "gate3", cmd, outputs=[GATE3_JSON])


def stage_gate4(args, ctx):
    eta = resolve_eta(args, ctx)
    cmd = mod("phase2_filter.find_dv_threshold") + [
        "--mode", "impulse",
        "--truth-mode", "sgp4_consistent",
        "--threshold", f"{eta:.4f}",
        "--n-trials", str(args.n_trials_gate4),
        "--dv-grid", args.dv_grid,
        "--fit-bstar", "1",
        "--r-from-fit", "1",
        "--jitter-geometry", "1",
        "--seed-offset", str(args.seed_offset_gate4),
        "--n-workers", str(args.n_workers),
        "--output", GATE4_JSON,
    ]
    return run_stage(args, "gate4", cmd, outputs=[GATE4_JSON])


def stage_fleet(args, ctx):
    eta = resolve_eta(args, ctx)
    cmd = mod("phase2_filter.run_real_fleet") + [
        "--tle-history-dir", args.tle_history_dir,
        "--meta", ctx["meta"],
        "--durations", ctx["durations"],
        "--threshold", f"{eta:.4f}",
        "--r-from-fit", "1",
        "--n-workers", str(args.n_workers),
        "--output", FLEET_JSON,
    ]
    return run_stage(args, "fleet", cmd, outputs=[FLEET_JSON])


def stage_null(args, ctx):
    eta = resolve_eta(args, ctx)
    cmd = mod("phase2_filter.scrambled_tle_null") + [
        "--tle-history-dir", args.tle_history_dir,
        "--meta", ctx["meta"],
        "--durations", ctx["durations"],
        "--threshold", f"{eta:.4f}",
        "--r-from-fit", "1",
        "--scramble-mode", "residual",
        "--n-workers", str(args.n_workers),
        "--output", NULL_JSON,
    ]
    return run_stage(args, "null", cmd, outputs=[NULL_JSON])


def stage_flux(args, ctx):
    if not exists(args.flux_csv):
        cmd = [py(), "flux_analysis/flux_compare.py",
               "--ordem-dir", "flux_analysis/results-ORDEM",
               "--ordem-manifest", "flux_analysis/manifest.txt",
               "--master-dir", "flux_analysis/results-MASTER",
               "--output", args.flux_csv]
        run_stage(args, "flux (parse ORDEM/MASTER)", cmd, outputs=[args.flux_csv])
    cmd = mod("phase2_filter.interpolate_flux_to_fleet") + [
        "--flux-csv", args.flux_csv,
        "--tle-file", args.tle_file,
        "--meta", ctx["meta"],
        "--durations", ctx["durations"],
        "--output-ordem", ORDEM_FLUX_JSON,
        "--output-master", MASTER_FLUX_JSON,
    ]
    return run_stage(args, "flux", cmd,
                     outputs=[ORDEM_FLUX_JSON, MASTER_FLUX_JSON])


def stage_predict(args, ctx):
    eff_json = GATE4_JSON if args.efficiency_source == "gate4" else GATE3_JSON
    if not exists(eff_json):
        raise StageError(f"efficiency source {eff_json} missing - run "
                         f"'{args.efficiency_source}' first")
    cmd = mod("phase2_filter.eval_fleet_efficiency") + [
        "--efficiency", eff_json,
        "--ordem-flux", ORDEM_FLUX_JSON,
        "--master-flux", MASTER_FLUX_JSON,
        "--windows-file", FLEET_JSON,
        "--window-days", str(args.window_days),
        "--output", PREDICT_JSON,
    ]
    if args.lc_min is not None:
        cmd += ["--lc-min", str(args.lc_min)]
    if args.lc_max is not None:
        cmd += ["--lc-max", str(args.lc_max)]
    return run_stage(args, "predict", cmd, outputs=[PREDICT_JSON])


def stage_fig5(args, ctx):
    cmd = [py(), "analysis/v1-x/cov-fig.py",
           "--fleet-file", FLEET_JSON,
           "--out", args.fig_out]
    return run_stage(args, "fig5", cmd, outputs=[args.fig_out])


def _dv50_from_efficiency(efficiency):
    """Interpolate dv at eps = 0.5 from {dv: {epsilon}} (string or float keys)."""
    pts = []
    for k, v in efficiency.items():
        try:
            eps = v.get("epsilon")
        except AttributeError:
            continue
        if eps is None:
            continue
        pts.append((float(k), float(eps)))
    if len(pts) < 2:
        return None
    pts.sort()
    return _interp_crossing(pts, 0.5)


def _interp_crossing(pts, level):
    """Linear interpolation of dv where eps crosses `level`."""
    for (dv0, e0), (dv1, e1) in zip(pts[:-1], pts[1:]):
        if (e0 - level) * (e1 - level) <= 0 and e0 != e1:
            return dv0 + (level - e0) * (dv1 - dv0) / (e1 - e0)
    return None


def stage_report(args, ctx):
    eta = None
    if args.eta is not None:
        eta = float(args.eta)
    elif exists(ETA_JSON):
        eta = float(read_json(ETA_JSON)["summary"]["eta_synth"])

    rows = []   # (quantity, pipeline_value, paper_value)

    rows.append(("eta_synth", f"{eta:.4f}" if eta else "n/a",
                 f"{PAPER['eta']}"))

    if exists(GATE4_JSON):
        eff = read_json(GATE4_JSON)["summary"]["efficiency"]
        dv50 = _dv50_from_efficiency(eff)
        rows.append(("dv50 synthetic (Gate 4)",
                     f"{dv50:.3f} m/s" if dv50 else "not bracketed",
                     f"{PAPER['dv50_synth']} m/s"))
    if exists(GATE3_JSON):
        g3 = read_json(GATE3_JSON)
        dv50r = g3.get("dv50")
        rows.append(("dv50 real-TLE (Gate 3)",
                     f"{dv50r:.3f} m/s" if dv50r else "not bracketed",
                     f"{PAPER['dv50_real']} m/s"))

    n_cand = n_win = expected = p_le = None
    if exists(FLEET_JSON):
        fleet = read_json(FLEET_JSON)
        n_win = fleet["config"]["total_valid_windows"]
        n_cand = fleet["summary"]["raw_detections"]
        rate = fleet["summary"]["raw_trigger_rate"]
        rows.append(("valid windows", f"{n_win}", f"{PAPER['n_valid_windows']}"))
        rows.append(("candidates above eta",
                     f"{n_cand} ({100 * rate:.3f}% of windows)",
                     f"{PAPER['candidates']} (0.038%)"))
        if eta is not None:
            # design expectation: the eta percentile defines the per-window
            # design trigger probability
            design_p = 1.0 - args.percentile / 100.0
            expected = design_p * n_win
            rows.append(("expected at design rate",
                         f"{expected:.2f}", f"{PAPER['expected_at_design']}"))
            try:
                from scipy.stats import poisson
                p_le = float(poisson.cdf(n_cand, expected))
                rows.append((f"P(<= {n_cand} candidates | design)",
                             f"{p_le:.4f}", "0.0047"))
            except ImportError:
                pass

    if exists(NULL_JSON):
        nn = read_json(NULL_JSON)
        fp = nn.get("scrambled_fp_rate")
        if fp is None and "summary" in nn:
            fp = nn["summary"].get("scrambled_fp_rate")
        if fp is not None:
            rows.append(("permuted-null trigger rate", f"{fp * 100:.4f}%",
                         "(reference; see paper Sec. 4.3)"))

    if exists(PREDICT_JSON):
        rows.append(("expected detectable events",
                     f"see {PREDICT_JSON}", "see paper Sec. 5.4/6.2"))

    print("\n" + "=" * 72)
    print("IMPULSE V1.1 REPRODUCTION REPORT")
    print("=" * 72)
    w = max(len(r[0]) for r in rows) if rows else 1
    print(f"{'quantity':<{w}}  {'this run':>28}  {'paper':>28}")
    print("-" * (w + 60))
    for q, got, ref in rows:
        print(f"{q:<{w}}  {got:>28}  {ref:>28}")
    print("-" * (w + 60))
    print("Candidates are candidate impulses, NOT detections. No positive")
    print("excess is certifiable against either background; absence of a")
    print("certifiable excess is not evidence of absence of impacts.")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in vars(args).items()},
        "eta": eta,
        "stages": STAGE_LOG,
        "summary": {
            "valid_windows": n_win, "candidates": n_cand,
            "expected_at_design": expected, "p_le_candidates": p_le,
        },
    }
    os.makedirs(os.path.join(REPO_ROOT, "results"), exist_ok=True)
    with open(os.path.join(REPO_ROOT, MANIFEST_JSON), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {MANIFEST_JSON}")
    return True


STAGES = {
    "prepare": stage_prepare,
    "validate": stage_validate,
    "eta": stage_eta,
    "gate2": stage_gate2,
    "gate3": stage_gate3,
    "gate4": stage_gate4,
    "fleet": stage_fleet,
    "null": stage_null,
    "flux": stage_flux,
    "predict": stage_predict,
    "fig5": stage_fig5,
    "report": stage_report,
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default="all",
                   help=f"comma-separated stages, or 'all'. "
                        f"Order enforced: {','.join(STAGE_ORDER)}")
    p.add_argument("--list-stages", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-run stages whose outputs already exist")
    p.add_argument("--keep-going", action="store_true",
                   help="continue after a stage failure")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands, run nothing")

    # ---- thresholds / trial counts (V1.1 paper defaults) --------------
    p.add_argument("--eta", type=float, default=None,
                   help="use this eta_synth instead of running the "
                        "overnight eta calibration (paper: 417.94)")
    p.add_argument("--n-trials-eta", type=int, default=18582,
                   help="eta-null trials (paper: 18,582)")
    p.add_argument("--percentile", type=float, default=99.9,
                   help="eta percentile (paper: 99.9th)")
    p.add_argument("--seed-offset-eta", type=int, default=500000)
    p.add_argument("--n-trials-gate2", type=int, default=2000)
    p.add_argument("--n-trials-gate4", type=int, default=200,
                   help="trials PER dv point in the Gate-4 sweep")
    p.add_argument("--dv-grid", default="0.05,0.1,0.2,0.4,0.7,1.0,1.2,1.5,"
                                        "2.0,3.0,5.0,10.0,20.0",
                   help="Gate-4 dv grid [m/s]; must bracket dv50 ~ 1.19")
    p.add_argument("--seed-offset-gate4", type=int, default=560000)
    p.add_argument("--window-days", type=float, default=76.0,
                   help="searched days per window for the prediction stage "
                        "(76 = edge-trimmed paper convention)")
    p.add_argument("--efficiency-source", choices=["gate4", "gate3"],
                   default="gate4",
                   help="which efficiency curve feeds the prediction stage")
    p.add_argument("--lc-min", type=float, default=None)
    p.add_argument("--lc-max", type=float, default=None)

    # ---- paths ------------------------------------------------------------
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--tle-history-dir", default="TLE/histories")
    p.add_argument("--tle-file", default="TLE/real_tles.tle")
    p.add_argument("--meta", default="TLE/real_tles.tle.meta.json",
                   help="Space-Track meta (fetch_tles.py output). If "
                        "absent, prepare derives a stand-in.")
    p.add_argument("--durations",
                   default="TLE/real_tles.tle.durations_2009_2026.json",
                   help="satcat durations (fetch_dur.py + cap_durs.py "
                        "output). If absent, prepare derives a stand-in.")
    p.add_argument("--derived-meta", default="results/derived_meta.json")
    p.add_argument("--derived-durations",
                   default="results/derived_durations_2009_2026.json")
    p.add_argument("--flux-csv", default=FLUX_CSV)
    p.add_argument("--fig-out", default=FIG5_PNG)
    args = p.parse_args()

    if args.list_stages:
        print("Stages (run order):")
        for s in STAGE_ORDER:
            print(f"  {s:<10} {STAGES[s].__doc__ or ''}")
        return 0

    # parse + validate stage list, enforce dependency order
    if args.stages == "all":
        selected = list(STAGE_ORDER)
    else:
        wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
        bad = [s for s in wanted if s not in STAGES]
        if bad:
            p.error(f"unknown stage(s): {','.join(bad)}; "
                    f"valid: {','.join(STAGE_ORDER)}")
        selected = [s for s in STAGE_ORDER if s in wanted]
    if args.eta is not None and "eta" in selected and not args.force:
        selected = [s for s in selected if s != "eta"]
        print(f"[eta] --eta {args.eta} given; skipping the eta stage")

    ctx = {}
    print(f"IMPULSE V1.1 pipeline | stages: {','.join(selected)}")
    print(f"repo root: {REPO_ROOT}")
    t_all = time.time()

    for s in selected:
        try:
            STAGES[s](args, ctx)
        except StageError as e:
            print(f"\n[STOP] {e}")
            if not args.keep_going:
                return 1

    print(f"\nAll requested stages finished in "
          f"{(time.time() - t_all) / 60.0:.1f} min.")
    if "report" not in selected:
        print(f"Run 'python pipeline.py --stages report' for the "
              f"side-by-side comparison vs the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())