#!/usr/bin/env python3
"""Parse ORDEM SIZEFLUX_SC.OUT and MASTER _d.dia outputs into a single
1cm-10cm flux comparison table, indexed by (altitude_km, inclination_deg).

Identical logic to flux_compare.py (the 1mm-1cm version) with two changes:

  1. ORDEM: SIZEFLUX_SC.OUT is reverse-cumulative, so the differential
     flux IN the 1cm-10cm bin is
         flux(>=1cm) - flux(>=10cm),
     read from the SAME results-ORDEM files as before. No new ORDEM runs.

  2. MASTER: totals come from results-MASTER-1to10, whose runs must have
     been configured with diameter thresholds 1cm/10cm - then the _d.dia
     footer line ("Total flux in this spectrum is X 1/m^2/yr") IS the bin
     flux. VERIFY THAT FIRST:
       Get-Content (Get-ChildItem <master-dir>/task-1/*_d.dia).FullName -TotalCount 40
     If the thresholds were NOT 1cm/10cm, the footer is the flux of
     whatever range the run was configured for, and the table will be
     wrong.

Usage:
    python flux_compare_1to10.py \
        --ordem-dir flux_analysis/results-ORDEM \
        --ordem-manifest flux_analysis/manifest.txt \
        --master-dir flux_analysis/results-MASTER-1to10 \
        --output flux_analysis/flux_comparison_1to10.csv \
        --show-endpoints

Task grid (same order both tools, identical to the 1mm-1cm run):
    for alt in [700, 800, 900, 1000]:
        for inc in [51.6, 65, 82, 90, 98]:
            task_number += 1
"""
import argparse
import csv
import glob
import os
import re

TASK_GRID = []
for _alt in [700, 800, 900, 1000]:
    for _inc in [51.6, 65, 82, 90, 98]:
        TASK_GRID.append((_alt, _inc))


def parse_ordem_sizeflux(filepath, size_lo=1.0e-2, size_hi=1.0e-1,
                         tol=0.02, show=False):
    """Return differential flux in [size_lo, size_hi) from an ORDEM
    SIZEFLUX_SC.OUT file, using the reverse-cumulative subtraction.
    1cm-10cm bin: size_lo=1e-2 m, size_hi=1e-1 m.
    tol is the allowed relative mismatch when matching size_lo/size_hi
    against the file's fixed size grid."""
    flux_lo = flux_hi = None
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                size = float(parts[0])
                flux = float(parts[1])
            except ValueError:
                continue
            if abs(size - size_lo) / size_lo < tol:
                flux_lo = flux
            if abs(size - size_hi) / size_hi < tol:
                flux_hi = flux

    if flux_lo is None or flux_hi is None:
        raise ValueError(f"{filepath}: could not find rows near "
                          f"{size_lo} and/or {size_hi} m in the size grid")
    if show:
        print(f"    {os.path.basename(os.path.dirname(filepath))}: "
              f"flux(>={size_lo:g} m)={flux_lo:.6e}  "
              f"flux(>={size_hi:g} m)={flux_hi:.6e}  "
              f"diff={flux_lo - flux_hi:.6e}")
    return flux_lo - flux_hi


def parse_master_dia(filepath):
    """Return the total flux [1/m^2/yr] from a MASTER _d.dia footer line.
    Only correct as 'the 1cm-10cm flux' if the run's lower/upper diameter
    thresholds were actually set to 1cm/10cm - otherwise this is the total
    flux across whatever range the run was configured for."""
    pattern = re.compile(r"Total flux in this spectrum is\s+([\d.EeD+-]+)\s+1/m\^2/yr")
    with open(filepath) as f:
        text = f.read()
    m = pattern.search(text)
    if not m:
        raise ValueError(f"{filepath}: could not find the 'Total flux in this "
                          f"spectrum is ...' footer line")
    val_str = m.group(1).replace("D", "E").replace("d", "e")
    return float(val_str)


def parse_ordem_manifest(filepath):
    """Parse the raw job-manifest text into {hash: (altitude_km, inclination_deg)}."""
    mapping = {}
    line_re = re.compile(
        r"^([0-9a-f]{20,})\s.*?apogeeKm=([\d.]+).*?perigeeKm=([\d.]+).*?"
        r"inclinationDeg=([\d.]+)",
        re.IGNORECASE,
    )
    with open(filepath) as f:
        for line in f:
            m = line_re.search(line.strip())
            if m:
                hash_id, apogee, perigee, inc = m.groups()
                alt = round((float(apogee) + float(perigee)) / 2.0)
                mapping[hash_id] = (alt, float(inc))
    return mapping


def collect_ordem(ordem_dir, manifest_path, show=False):
    manifest = parse_ordem_manifest(manifest_path)
    results = {}
    for folder in sorted(os.listdir(ordem_dir)):
        full = os.path.join(ordem_dir, folder)
        if not os.path.isdir(full):
            continue
        sizeflux_path = os.path.join(full, "SIZEFLUX_SC.OUT")
        if not os.path.exists(sizeflux_path):
            continue
        if folder not in manifest:
            print(f"  WARNING: folder {folder} not found in manifest, skipping")
            continue
        alt, inc = manifest[folder]
        try:
            flux = parse_ordem_sizeflux(sizeflux_path, show=show)
            results[(alt, inc)] = flux
        except ValueError as e:
            print(f"  WARNING: {e}")
    return results


def collect_master(master_dir):
    results = {}
    task_folders = sorted(glob.glob(os.path.join(master_dir, "task-*")))
    for folder in task_folders:
        m = re.search(r"task-(\d+)", os.path.basename(folder))
        if not m:
            continue
        task_num = int(m.group(1))
        if task_num < 1 or task_num > len(TASK_GRID):
            print(f"  WARNING: task number {task_num} out of expected range 1-{len(TASK_GRID)}")
            continue
        alt, inc = TASK_GRID[task_num - 1]
        dia_files = glob.glob(os.path.join(folder, "*_d.dia"))
        if not dia_files:
            print(f"  WARNING: no *_d.dia file found in {folder}")
            continue
        try:
            flux = parse_master_dia(dia_files[0])
            results[(alt, inc)] = flux
        except ValueError as e:
            print(f"  WARNING: {e}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ordem-dir", required=True)
    p.add_argument("--ordem-manifest", required=True)
    p.add_argument("--master-dir", required=True)
    p.add_argument("--output", default="flux_comparison_1to10.csv")
    p.add_argument("--show-endpoints", action="store_true",
                   help="Print the matched cumulative flux rows per ORDEM task")
    args = p.parse_args()

    print("Parsing ORDEM results (1cm-10cm bin: flux(>=1cm) - flux(>=10cm))...")
    ordem = collect_ordem(args.ordem_dir, args.ordem_manifest, show=args.show_endpoints)
    print(f"  parsed {len(ordem)} ORDEM tasks")

    print("Parsing MASTER results (footer total of the 1cm/10cm-threshold runs)...")
    master = collect_master(args.master_dir)
    print(f"  parsed {len(master)} MASTER tasks")

    all_keys = sorted(set(ordem) | set(master))
    rows = []
    print()
    print(f"{'alt_km':>7} {'inc_deg':>8} {'ORDEM_1cm_10cm':>15} {'MASTER_1cm_10cm':>16} {'ratio_O/M':>10}")
    for alt, inc in all_keys:
        o = ordem.get((alt, inc))
        m = master.get((alt, inc))
        ratio = (o / m) if (o is not None and m not in (None, 0)) else None
        rows.append({
            "altitude_km": alt, "inclination_deg": inc,
            "ordem_flux_1cm_10cm": o, "master_flux_1cm_10cm": m,
            "ratio_ordem_to_master": ratio,
        })
        o_s = f"{o:.6f}" if o is not None else "MISSING"
        m_s = f"{m:.6f}" if m is not None else "MISSING"
        r_s = f"{ratio:.3f}" if ratio is not None else "N/A"
        print(f"{alt:7} {inc:8.1f} {o_s:>15} {m_s:>16} {r_s:>10}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "altitude_km", "inclination_deg",
            "ordem_flux_1cm_10cm", "master_flux_1cm_10cm", "ratio_ordem_to_master",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()