#!/usr/bin/env python3
"""V2 Fix B: persistence-based transient-excursion filter for fitted a.

Phase-0b (series_census.json, 17,000-trial honestR null run) established:
  - the extreme null tail is discrete km-scale STEPs in fitted a
    (8/8 TOP trials; jumps 4.2-7.6 km at 541-42,367x the scatter of
    every other step; controls 2-17 m);
  - the steps are TRANSIENT: persistence median +0.00, i.e. the fitted
    level reverts to the pre-jump value within a few refits;
  - they PASS the position-RMS QC gate (8/8, rms 843-4863 m vs the
    5000 m gate): the position gate is structurally blind to the
    aliased small-e / mean-element-convention fit branches, so the
    fix must live at the ESTIMATOR, not the position gate;
  - real impulses are the SAME SIZE (along-track dv = 1.19 m/s gives
    da ~ 2.3 km) but PERSIST: the level holds for every later refit.

Therefore jump SIZE cannot discriminate artifact from signal;
PERSISTENCE can. This module implements the discriminator as a
reusable function plus an offline validation harness.

Filter contract (per per-refit channel; designed for fitted a):
    A refit point is DROPPED only if BOTH hold:
      (1) SIZE:      it belongs to an excursion whose entry jump
                     exceeds n_jump x the robust scatter of the
                     other steps (the same statistic Phase-0b used);
      (2) REVERSION: the level after the excursion returns to the
                     pre-jump level (|post - pre| <= revert_frac x
                     |jump|), confirmed by >= confirm_pts later
                     refits.
    Persistent level shifts (real impulses) fail (2) and survive.
    Small steps (controls, 2-17 m) fail (1) and survive.
    SAFEGUARD: an excursion with fewer than confirm_pts refits left
    after it (window-end tail) is NEVER dropped - a real strike near
    the window end has exactly that shape. It is flagged
    "tail_unconfirmed" instead. Same for excursions whose level never
    returns: they are flagged "persistent" and kept. Both leftovers
    are Fix A's job (warm-start source fits), not this filter's.

Pipeline integration (V2, merged convention_consistency_test /
find_dv_threshold module): apply after the position-RMS QC gate,
before R estimation and the Mahalanobis window filter, identically
for null calibration, gates 2-4, fleet scoring, and candidate
rescoring. Dropped refits leave the stream exactly like QC-rejected
fits (the stream machinery already tolerates gaps).

Offline validation (repo root, ~1 min, no pipeline needed):
    python -m bstar-diag.persistence_filter --self-test
    python -m bstar-diag.persistence_filter --refit-dir results/diag_bstar/refit

Expected (established by the two prior offline runs on the same
series): seeds 506954, 507728, 508369, 509336, 511638 clean to
near-control std_da (the staircase excursions of 509336/511638
need the TIGHT step tolerance - that is self-test 9); seed 503523
(down ~day 84, back up ~day 86, fewer than 2 refits left to confirm)
and seed 504718 (level still displaced at the window end) flag
tail/persistent and are KEPT - correctly, because a real strike
near the window end has the same shape; seed 513933 settles at a
new level and is KEPT. Those three leftovers are Fix A's job.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_dir(path):
    """Resolve a directory against cwd, the script dir, and its parent
    (the script may live in a subpackage while data sits at repo
    level)."""
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), REPO_ROOT, os.path.dirname(REPO_ROOT)):
        cand = os.path.join(base, path)
        if os.path.isdir(cand):
            return cand
    return os.path.join(os.getcwd(), path)


from phase2_filter.persistence import (robust_sigma,
                                       detect_transient_excursions)

def self_test():
    """Synthetic unit checks for the detector (no data needed)."""
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 90.0, 1.5)          # 60 refits over 90 days
    base = rng.normal(0.0, 8.0, len(t)) + 0.02 * t
    checks = []

    # 1) transient excursion (the artifact): out at day 30, back day 36
    x = base.copy()
    m = (t >= 30) & (t < 36)
    x[m] += 5000.0
    r = detect_transient_excursions(x, t)
    ok = (r["n_excursions"] == 1
          and r["details"][0]["decision"] == "transient"
          and bool(r["drop_mask"][m].all())
          and not bool(r["drop_mask"][~m].any()))
    checks.append(("transient excursion dropped, rest kept", ok))

    # 2) persistent level shift (a real impulse at day 30): kept
    x = base.copy()
    x[t >= 30] += 5000.0
    r = detect_transient_excursions(x, t)
    ok = (r["n_excursions"] == 1
          and r["details"][0]["decision"] == "persistent"
          and r["n_dropped"] == 0)
    checks.append(("persistent level shift survives", ok))

    # 3) tail excursion (day 88, no post refits): flagged, kept
    x = base.copy()
    x[t >= 88] += 5000.0
    r = detect_transient_excursions(x, t)
    ok = (r["n_excursions"] == 1
          and r["details"][0]["decision"] == "tail_unconfirmed"
          and r["n_dropped"] == 0)
    checks.append(("tail excursion flagged, never dropped", ok))

    # 4) quiet control: nothing detected
    r = detect_transient_excursions(base, t)
    checks.append(("quiet stream untouched",
                   r["n_excursions"] == 0 and r["n_dropped"] == 0))

    # 5) small persistent step (control-type, 17 m): below size gate
    x = base.copy()
    x[t >= 45] -= 17.0
    r = detect_transient_excursions(x, t)
    checks.append(("small persistent step survives",
                   r["n_excursions"] == 0 and r["n_dropped"] == 0))

    # 6) transient excursion with a one-point in-level blip inside:
    #    blip is a good refit and is KEPT; the rest of the event drops
    x = base.copy()
    x[(t >= 30) & (t < 40)] += 5000.0
    blip = int(np.argmin(np.abs(t - 35)))
    x[blip] -= 5000.0                      # one blip back at level
    r = detect_transient_excursions(x, t)
    m = (t >= 30) & (t < 40)
    m[blip] = False
    ok = (r["n_excursions"] == 1
          and r["details"][0]["decision"] == "transient"
          and bool(r["drop_mask"][m].all())
          and not bool(r["drop_mask"][~m | (np.arange(len(t)) == blip)]
                     .any())
          and not bool(r["drop_mask"][blip]))
    checks.append(("blip inside excursion kept, event dropped", ok))

    # 7) compound event (Phase-0b seed 503523 shape): small entry
    #    wiggle, then a km-scale excursion, then revert. Must be judged
    #    by the EXCURSION magnitude, not the entry wiggle, and the
    #    km-scale segment must drop while the wiggle level survives.
    x = base.copy()
    x[(t >= 30) & (t < 36)] += 60.0        # entry wiggle level
    x[(t >= 36) & (t < 44)] += 5060.0     # km-scale excursion
    r = detect_transient_excursions(x, t)
    m = (t >= 36) & (t < 44)
    ok = (r["n_excursions"] == 1
          and r["details"][0]["decision"] == "transient"
          and bool(r["drop_mask"][m].all())
          and not bool(r["drop_mask"][~m].any()))
    checks.append(("compound wiggle+excursion: km part drops, "
                   "wiggle kept", ok))

    # 8) SAFETY: artifact excursion early, REAL impulse later:
    #    the artifact must drop and the impulse must survive
    x = base.copy()
    x[(t >= 20) & (t < 26)] += 5000.0        # transient artifact
    x[t >= 50] += 2300.0                      # persistent real impulse
    r = detect_transient_excursions(x, t)
    art = (t >= 20) & (t < 26)
    imp = t >= 50
    ok = (bool(r["drop_mask"][art].all())
          and not bool(r["drop_mask"][imp].any()))
    checks.append(("artifact before real impulse: artifact dropped, "
                   "impulse kept", ok))

    # 9) REGRESSION GUARD (v3 bug shape, Phase-0b seed 509336):
    #    small entry step, large staircase excursion, revert. The
    #    walk must track the WHOLE event with the tight step
    #    tolerance and drop it; an excursion-scaled step tolerance
    #    breaks the walk at the entry step and keeps fragments.
    x = base.copy()
    x[(t >= 30) & (t < 33)] += 130.0        # small entry step
    x[(t >= 33) & (t < 40)] += 2230.0      # staircase excursion
    r = detect_transient_excursions(x, t)
    m = (t >= 30) & (t < 40)
    ok = (r["n_excursions"] == 1
          and r["details"][0]["decision"] == "transient"
          and bool(r["drop_mask"][m].all())
          and not bool(r["drop_mask"][~m].any()))
    checks.append(("small entry + staircase excursion: whole event "
                   "drops", ok))

    print("SELF-TEST " + "=" * 55)
    all_ok = True
    for name, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= bool(ok)
    print(f"    -> {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit-dir", default="results/diag_bstar/refit")
    ap.add_argument("--gate", type=float, default=5000.0,
                    help="position-RMS QC gate (config fit_rms_max_m)")
    ap.add_argument("--n-jump", type=float, default=10.0)
    ap.add_argument("--revert-frac", type=float, default=0.5)
    ap.add_argument("--confirm-pts", type=int, default=2)
    ap.add_argument("--level-floor", type=float, default=5.0)
    ap.add_argument("--out-dir", default="results/diag_bstar")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic unit checks only, no data")
    args = ap.parse_args()

    if args.self_test:
        return 0 if self_test() else 1

    refit_dir = resolve_dir(args.refit_dir)
    if not os.path.isdir(refit_dir):
        print(f"no refit dir at {args.refit_dir}; run Phase-0 with "
              f"--refit-top N first, or pass --refit-dir explicitly")
        return 1
    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(os.getcwd(), args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("FIX B OFFLINE VALIDATION: persistence filter vs saved "
          "Phase-0b refit series")
    print("=" * 72)
    print(f"    params: n_jump={args.n_jump} revert_frac="
          f"{args.revert_frac} confirm_pts={args.confirm_pts} "
          f"gate={args.gate:.0f} m")

    census = []
    files = sorted(fn for fn in os.listdir(refit_dir)
                   if fn.endswith(".json"))
    for fn in files:
        with open(os.path.join(refit_dir, fn)) as f:
            rr = json.load(f)
        pr_all = rr.get("per_refit", [])
        if len(pr_all) < 6:
            continue
        has_rms = all(c.get("fit_rms_m") is not None for c in pr_all)
        pr = ([c for c in pr_all
               if float(c["fit_rms_m"]) <= args.gate]
              if has_rms else pr_all)
        if len(pr) < 6:
            continue
        t_d = np.array([c["t_ep_s"] for c in pr], float) / 86400.0
        da = np.array([c["da_m"] for c in pr], float)
        det = detect_transient_excursions(
            da, t_d, n_jump=args.n_jump, revert_frac=args.revert_frac,
            confirm_pts=args.confirm_pts, level_floor=args.level_floor)
        keep = ~det["drop_mask"]
        std_before = float(np.std(da))
        std_after = float(np.std(da[keep]))
        maxdev_before = float(np.max(np.abs(da - np.median(da))))
        maxdev_after = float(np.max(np.abs(da[keep] - np.median(da[keep]))))
        row = {"tag": rr.get("tag"), "seed": rr.get("seed"),
               "alt_km": rr.get("alt_km"), "peak_maha": rr.get("peak_maha"),
               "n_captured": len(pr_all),
               "n_qc_rejected": (len(pr_all) - len(pr)) if has_rms
               else None,
               "n_dropped": det["n_dropped"],
               "std_da_before": std_before, "std_da_after": std_after,
               "max_dev_before": maxdev_before,
               "max_dev_after": maxdev_after,
               "excursions": det["details"]}
        census.append(row)

        print(f"\n    {str(rr.get('tag')):<7} seed={rr.get('seed'):<7} "
              f"alt={rr.get('alt_km'):7.1f} "
              f"peak={rr.get('peak_maha'):8.2f}")
        if not det["details"]:
            print("        no excursions detected (stream kept as-is)")
        for d in det["details"]:
            day_s = (f"day {d['day']:5.0f}" if d["day"] is not None
                     else "idx {0}".format(d["enter_idx"]))
            print(f"        {day_s}  jump={d['jump']:+9.1f} m  "
                  f"exc={d.get('exc_dev', 0.0):+9.1f} m  "
                  f"-> {d['decision']:<16} "
                  f"{'dropped' if d['dropped'] else 'KEPT'}")
        if det["n_dropped"]:
            print(f"        std_da: {std_before:8.1f} -> "
                  f"{std_after:8.1f} m | max|da|: "
                  f"{maxdev_before:8.1f} -> {maxdev_after:8.1f} m")

    # ---- group summary + verdict -------------------------------------------
    print("\n" + "=" * 72)
    print("GROUP SUMMARY")
    print("=" * 72)
    for tag in ("TOP", "CONTROL"):
        grp = [c for c in census if c.get("tag") == tag]
        if not grp:
            continue
        dec = {}
        for c in grp:
            for d in c["excursions"]:
                dec[d["decision"]] = dec.get(d["decision"], 0) + 1
        stds_b = [c["std_da_before"] for c in grp]
        stds_a = [c["std_da_after"] for c in grp]
        print(f"    {tag}: {len(grp)} trials | excursions {dec or 'none'}")
        print(f"        median std_da {np.median(stds_b):8.1f} -> "
              f"{np.median(stds_a):8.1f} m")

    tops = [c for c in census if c.get("tag") == "TOP"]
    if tops:
        n_tr = sum(1 for c in tops for d in c["excursions"]
                   if d["decision"] == "transient")
        n_pen = sum(1 for c in tops for d in c["excursions"]
                    if d["decision"] == "persistent")
        n_tail = sum(1 for c in tops for d in c["excursions"]
                      if d["decision"] == "tail_unconfirmed")
        print("\nVERDICT")
        print(f"    TOP excursions: {n_tr} transient (dropped), "
              f"{n_tail} tail_unconfirmed (kept), "
              f"{n_pen} persistent (kept)")
        if n_tr and not n_pen and not n_tail:
            print("    -> Fix B alone removes the artifact tail; wire it")
            print("       into the merged module and re-run the pipeline.")
        elif n_tr:
            print("    -> Fix B removes the bulk of the tail; the kept")
            print("       leftovers (persistent + tail-unconfirmed) are")
            print("       Fix A's job: warm-start/sequential source fits")
            print("       + element-space prior QC in the generator.")
        else:
            print("    -> nothing dropped: check params vs the 0b census")
            print("       before wiring into the pipeline.")

    out_path = os.path.join(out_dir, "persistence_filter_census.json")
    with open(out_path, "w") as f:
        json.dump(census, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())