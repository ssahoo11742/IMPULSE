"""V2 Fix B: persistence-based transient-excursion filter.

Phase-0/0b established (17,000-trial honestR null run): the extreme
null tail is discrete km-scale STEPS in fitted a (8/8 top trials,
jumps 4.2-7.6 km at 541-42,367x step scatter), they are TRANSIENT
(persistence median +0.00, level reverts within a few refits), and
they PASS the position-RMS QC gate (8/8, rms 843-4863 m vs 5000 m) -
the aliased small-e / mean-element-convention fit branches reproduce
the positions, so the fix lives at the estimator, not the gate. Real
impulses are the SAME SIZE (dv=1.19 m/s gives da ~ 2.3 km) but
PERSIST, so jump size cannot discriminate; persistence can.

Contract: a refit is dropped only if BOTH hold:
  (1) SIZE: its excursion's entry jump exceeds n_jump x the robust
      scatter of the other steps;
  (2) REVERSION: the level after the excursion returns to the
      pre-jump level, confirmed by >= confirm_pts later refits.
Persistent shifts fail (2); small steps (2-17 m) fail (1); window-end
excursions with too few confirming refits are flagged and KEPT (a
real end-of-window strike has the same shape).

Two tolerances (validated on the Phase-0b saved series):
  tol_step (TIGHT) = max(revert_frac*|jump|, level_floor*step scatter)
    -> drives the excursion walk and the drop test; excursion-scaled
       versions here let km-scale plateaus hide (v3 regression).
  tol_revert (GENEROUS) = max(tol_step, revert_frac*excursion deviation)
    -> drives only the transient/persistent label, so compound events
       (small entry wiggle, km excursion, revert) are judged by the
       excursion. Can mislabel a report, can never misDROP: a point
       drops only when the walk demonstrably returned to the pre
       level, which a persistent impulse makes impossible.

Config frozen from the offline validation; do not retune:
  n_jump=10.0, revert_frac=0.5, confirm_pts=2, level_floor=5.0.
"""

import numpy as np


def robust_sigma(v):
    """MAD-based sigma of a 1-D array (robust to a few giant steps)."""
    v = np.asarray(v, float)
    if len(v) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def detect_transient_excursions(x, t=None, n_jump=10.0, revert_frac=0.5,
                                confirm_pts=2, blip_tol=1, min_points=6,
                                level_floor=5.0):
    """Returns {"drop_mask", "n_excursions", "n_dropped", "details"}.

    x : 1-D channel values in refit order (fitted a minus truth).
    t : optional 1-D times in days; used only for reporting.
    drop_mask[i] is True = remove refit i from the stream.
    """
    x = np.asarray(x, float)
    n = len(x)
    drop = np.zeros(n, dtype=bool)
    covered = np.zeros(n, dtype=bool)
    details = []
    if n < max(int(min_points), 2 * int(confirm_pts) + 3):
        return {"drop_mask": drop, "n_excursions": 0, "n_dropped": 0,
                "details": details}
    diffs = np.diff(x)
    sig_steps = max(robust_sigma(diffs), 1e-9)
    t_d = None if t is None else np.asarray(t, float)

    i = 0
    while i < n - 1:
        if covered[i + 1]:
            i += 1
            continue
        jump = float(diffs[i])
        others = np.delete(diffs, i)
        sig_others = max(robust_sigma(others), 1e-9)
        if abs(jump) <= n_jump * sig_others:
            i += 1
            continue

        # candidate excursion entering at point i+1
        pre_pts = x[: i + 1][~covered[: i + 1]]
        pre_level = (float(np.median(pre_pts)) if len(pre_pts)
                     else float(x[i]))
        # excursion magnitude relative to the pre level (over the
        # rest of the stream): compound events (small entry wiggle,
        # km-scale excursion, revert) exist - see Phase-0b seed 503523
        exc_dev = (float(np.max(np.abs(x[i + 1:] - pre_level)))
                   if i + 1 < n else 0.0)
        tol_step = max(revert_frac * abs(jump),
                       float(level_floor) * sig_steps)
        tol_revert = max(tol_step, revert_frac * exc_dev)

        # walk the excursion span: points away from the pre level,
        # tolerating up to blip_tol brief in-level blips
        end = i + 1
        blips = 0
        while end < n:
            if abs(x[end] - pre_level) <= tol_step:
                blips += 1
                if blips > int(blip_tol):
                    break
            else:
                blips = 0
            end += 1

        day = (float(t_d[min(i + 1, n - 1)]) if t_d is not None
               else None)
        rec = {"enter_idx": i + 1, "exit_idx": min(end, n - 1),
               "day": day, "jump": jump, "exc_dev": exc_dev,
               "pre_level": pre_level}

        if end == n:
            # level never returns within the series: judge by the
            # level the stream settles at (>= confirm_pts after entry)
            post_lo = min(i + 1 + int(confirm_pts), n)
            post = x[post_lo:]
            if len(post) < 1:
                rec["decision"] = "tail_unconfirmed"
            else:
                post_level = float(np.median(post))
                rec["post_level"] = post_level
                if abs(post_level - pre_level) <= tol_revert:
                    rec["decision"] = "tail_unconfirmed"
                else:
                    rec["decision"] = "persistent"
                # a real impulse near the window end is also "level
                # never returns": indistinguishable, so NEVER drop.
            rec["dropped"] = False
        else:
            post = x[end:]
            if len(post) < int(confirm_pts):
                rec["decision"] = "tail_unconfirmed"
                rec["dropped"] = False
            else:
                post_level = float(np.median(post))
                rec["post_level"] = post_level
                if abs(post_level - pre_level) <= tol_revert:
                    # drop ONLY the points that are actually away
                    # from the pre level; in-level blips inside the
                    # event window are good refits and are kept
                    n_span_dropped = 0
                    for k in range(i + 1, min(end, n)):
                        if abs(x[k] - pre_level) > tol_step:
                            drop[k] = True
                            n_span_dropped += 1
                    rec["decision"] = "transient"
                    rec["dropped"] = n_span_dropped > 0
                    rec["n_dropped_pts"] = n_span_dropped
                else:
                    rec["decision"] = "persistent"
                    rec["dropped"] = False

        details.append(rec)
        # cover the span AND the return landing point so the exit
        # step of the same event is never scanned as a new excursion
        covered[i + 1: min(end + 1, n)] = True
        i = max(end, i + 1)

    return {"drop_mask": drop, "n_excursions": len(details),
            "n_dropped": int(drop.sum()), "details": details}