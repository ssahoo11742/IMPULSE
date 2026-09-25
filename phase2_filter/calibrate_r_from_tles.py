#!/usr/bin/env python3
"""Calibrate measurement noise covariance R from TLE residuals."""
import argparse
import json
import math
import os
import warnings
from typing import Dict, List, Optional

import numpy as np

from TLE.tle_io import parse_tle
from constants.constants import MU

ELEMENT_NAMES = ["a", "e", "i", "Omega", "omega", "M"]
ELEMENT_UNITS = ["m", "", "rad", "rad", "rad", "rad"]
MAX_TLES_PER_FILE = 500


def tle_to_element_dict(tle: Dict) -> Optional[Dict[str, float]]:
    """Convert parsed TLE dict to orbital elements (a, e, i, Omega, omega, M)."""
    a = tle.get("a")
    if a is None or a <= 0:
        n_rev_per_day = tle.get("n_revday", 0.0)
        if n_rev_per_day <= 0:
            return None
        n_rad_s = n_rev_per_day * 2 * math.pi / 86400.0
        a = (MU / (n_rad_s ** 2)) ** (1.0 / 3.0)

    e = tle.get("ecc", 0.0)
    i = tle.get("inc", 0.0)       # already rad from parse_tle
    raan = tle.get("raan", 0.0)
    argp = tle.get("argp", 0.0)
    M = tle.get("M", 0.0)

    return {"a": a, "e": e, "i": i, "Omega": raan, "omega": argp, "M": M}


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    if len(angles) == 0:
        return angles
    unwrapped = np.zeros_like(angles, dtype=float)
    unwrapped[0] = angles[0]
    for j in range(1, len(angles)):
        diff = angles[j] - angles[j - 1]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        unwrapped[j] = unwrapped[j - 1] + diff
    return unwrapped


def fit_polynomial(t: np.ndarray, y: np.ndarray, degree: int):
    """Fit polynomial with centered time for numerical stability."""
    if len(t) < degree + 1:
        return None

    t_mean = np.mean(t)
    t_centered = t - t_mean

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            coeffs = np.polyfit(t_centered, y, degree)
        except (np.linalg.LinAlgError, ValueError):
            return None

    return coeffs, t_mean


def eval_polynomial(t: float, coeffs: np.ndarray, t_mean: float) -> float:
    t_shifted = t - t_mean
    result = 0.0
    for c in coeffs:
        result = result * t_shifted + c
    return result


def compute_object_residuals(
    tle_list: List[Dict],
    fit_span_days: float = 30.0,
    poly_degrees: Optional[Dict[str, int]] = None,
    maneuver_sigma: float = 5.0,
) -> Optional[Dict[str, List[float]]]:
    if poly_degrees is None:
        poly_degrees = {"a": 2, "M": 2, "e": 1, "i": 1, "Omega": 1, "omega": 1}

    tle_list = sorted(tle_list, key=lambda t: t.get("epoch_jd", 0))
    epochs = []
    elements = {k: [] for k in ELEMENT_NAMES}

    for tle in tle_list:
        el = tle_to_element_dict(tle)
        if el is None:
            continue
        epoch_jd = tle.get("epoch_jd", 0)
        if epoch_jd == 0:
            continue
        epochs.append(epoch_jd)
        for k in ELEMENT_NAMES:
            elements[k].append(el[k])

    if len(epochs) < 10:
        return None

    epochs = np.array(epochs)
    t_days = epochs - epochs[0]

    for elem in ["M", "Omega", "omega"]:
        elements[elem] = list(unwrap_angles(np.array(elements[elem])))

    residuals = {k: [] for k in ELEMENT_NAMES}
    half_span = fit_span_days / 2.0

    for i in range(len(epochs)):
        t_center = t_days[i]
        mask = (t_days >= t_center - half_span) & (t_days <= t_center + half_span)
        idxs = np.where(mask)[0]

        if len(idxs) < 5:
            continue

        t_win = t_days[idxs]

        for elem in ELEMENT_NAMES:
            y_arr = np.array(elements[elem])
            y_win = y_arr[idxs]
            deg = poly_degrees.get(elem, 1)

            if len(t_win) < deg + 1:
                continue

            fit_result = fit_polynomial(t_win, y_win, deg)
            if fit_result is None:
                continue
            coeffs, t_mean = fit_result

            pred = eval_polynomial(t_center, coeffs, t_mean)
            resid = elements[elem][i] - pred
            residuals[elem].append(resid)

    # MAD-based sigma clip for maneuver outliers
    if maneuver_sigma > 0:
        for elem in ELEMENT_NAMES:
            vals = np.array(residuals[elem])
            if len(vals) == 0:
                continue
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            if mad > 0:
                robust_std = 1.4826 * mad
                mask = np.abs(vals - med) <= maneuver_sigma * robust_std
                residuals[elem] = vals[mask].tolist()

    return residuals


def load_tle_histories_streaming(tle_dir: str, meta_file: str):
    with open(meta_file) as f:
        meta_records = json.load(f)
    norad_ids = {str(r.get("NORAD_CAT_ID", "")) for r in meta_records}

    histories = {}
    for fname in sorted(os.listdir(tle_dir)):
        if not fname.endswith(".tle"):
            continue
        nid_from_fname = fname.replace(".tle", "")
        if nid_from_fname not in norad_ids:
            continue

        fpath = os.path.join(tle_dir, fname)
        tles = []
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines.append(line)
                    if len(lines) >= 3:
                        name = lines[0] if not lines[0].startswith("1 ") and not lines[0].startswith("2 ") else ""
                        l1 = lines[1] if name else lines[0]
                        l2 = lines[2] if name else lines[1]
                        if l1.startswith("1 ") and l2.startswith("2 "):
                            try:
                                tle = parse_tle(name, l1, l2)
                                if tle.get("epoch_jd", 0) > 0:
                                    tles.append(tle)
                            except Exception:
                                pass
                        lines = []
                    if len(tles) >= MAX_TLES_PER_FILE:
                        break
        except Exception as e:
            print(f"  Warning: could not read {fname}: {e}")
            continue

        if tles:
            histories[nid_from_fname] = tles

    return histories


def main():
    parser = argparse.ArgumentParser(description="Calibrate R from TLE residuals")
    parser.add_argument("--tle-history-dir", default=None)
    parser.add_argument("--tle-history-file", default=None)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--fit-span-days", type=float, default=30.0)
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--min-tles", type=int, default=20)
    parser.add_argument("--maneuver-sigma", type=float, default=5.0,
                        help="Sigma threshold for outlier rejection (0=off)")
    parser.add_argument("--output", default="results/empirical_r.json")
    args = parser.parse_args()

    if not args.tle_history_dir and not args.tle_history_file:
        print("ERROR: provide --tle-history-dir or --tle-history-file")
        return

    print(f"Loading histories (streaming, max {MAX_TLES_PER_FILE} per file)...")
    histories = load_tle_histories_streaming(args.tle_history_dir, args.meta)
    print(f"Loaded histories for {len(histories)} objects")

    all_residuals = {k: [] for k in ELEMENT_NAMES}
    object_counts = 0
    skipped = 0

    for norad_id, tle_list in histories.items():
        if len(tle_list) < args.min_tles:
            skipped += 1
            continue

        resids = compute_object_residuals(
            tle_list,
            fit_span_days=args.fit_span_days,
            maneuver_sigma=args.maneuver_sigma,
        )
        if resids is None:
            skipped += 1
            continue

        for elem in ELEMENT_NAMES:
            all_residuals[elem].extend(resids[elem])
        object_counts += 1

    total_samples = sum(len(v) for v in all_residuals.values())
    print(f"Processed {object_counts} objects ({skipped} skipped)")
    print(f"Total residual samples: {total_samples}")
    print()

    if total_samples == 0:
        print("ERROR: No residuals computed. Check that TLE files parse correctly.")
        return

    results = {}
    print(f"{'Element':>8} {'N':>8} {'Mean':>12} {'Std':>12} {'R (var)':>14} {'Unit':>8}")
    print("-" * 65)

    for elem, unit in zip(ELEMENT_NAMES, ELEMENT_UNITS):
        vals = np.array(all_residuals[elem])
        n = len(vals)
        if n == 0:
            print(f"{elem:>8} {0:>8} {'N/A':>12} {'N/A':>12} {'N/A':>14} {unit:>8}")
            results[elem] = {"n": 0, "mean": None, "std": None, "variance": None}
            continue

        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        var = float(np.var(vals, ddof=1))

        print(f"{elem:>8} {n:>8} {mean:>12.6e} {std:>12.6e} {var:>14.6e} {unit:>8}")
        results[elem] = {"n": n, "mean": mean, "std": std, "variance": var, "unit": unit}

    r_diag = [results[k]["variance"] for k in ELEMENT_NAMES]
    print()
    print("Suggested R_DIAG_ELEMENTS (variances):")
    current_r = [1.0e2, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-6]
    for elem, emp, cur in zip(ELEMENT_NAMES, r_diag, current_r):
        print(f"  {elem:>8}: {emp:>14.6e}  (current: {cur:.0e})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": {
                "fit_span_days": args.fit_span_days,
                "poly_degree": args.poly_degree,
                "min_tles": args.min_tles,
                "maneuver_sigma": args.maneuver_sigma,
                "n_objects": object_counts,
            },
            "results": results,
            "suggested_r_diag": r_diag,
        }, f, indent=2)
    print(f"\nSaved to {args.output}")

    print()
    print("Ratio (empirical / current):")
    for elem, emp, cur in zip(ELEMENT_NAMES, r_diag, current_r):
        if emp is not None and cur > 0:
            ratio = emp / cur
            print(f"  {elem:>8}: {ratio:>8.3f}x")


if __name__ == "__main__":
    main()