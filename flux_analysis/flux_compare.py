"""Parse ORDEM SIZEFLUX_SC.OUT and MASTER _d.dia outputs into a single
1mm-1cm flux comparison table, indexed by (altitude_km, inclination_deg).

ORDEM's SIZEFLUX_SC.OUT gives REVERSE-CUMULATIVE flux (flux of objects with
diameter >= given size). The differential flux actually IN the 1mm-1cm bin
is flux(>=1mm) - flux(>=1cm), NOT the raw value at either endpoint alone -
using the raw 1mm-row value directly overcounts everything above 1cm too.

MASTER's _d.dia file, when configured with lower/upper thresholds already
set to 1mm/1cm (as this run was), reports the differential total directly
in its footer line ("Total flux in this spectrum is X 1/m^2/yr") - no
cumulative subtraction needed there, since the whole spectrum IS the
1mm-1cm bin already.

Usage:
    python flux_parser.py \
        --ordem-dir results-ORDEM \
        --ordem-manifest ordem_manifest.txt \
        --master-dir results-MASTER \
        --output flux_comparison.csv

ordem_manifest.txt should contain the raw job-manifest text Space-Track/
ORDEM gave you (hash <tab> "taskNumber=N, ..., apogeeKm=X, perigeeKm=X,
inclinationDeg=Y, ..." <tab> ...) - paste it as-is into a text file.

MASTER task folders are expected named task-1 .. task-20 (any 'task-N'
pattern), each containing one *_d.dia file, in the SAME altitude x
inclination grid order used for both tools:
    for alt in [700, 800, 900, 1000]:
        for inc in [51.6, 65, 82, 90, 98]:
            task_number += 1
If your MASTER task order differs from this, edit TASK_GRID below to match.
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


def parse_ordem_sizeflux(filepath, size_lo=1.0e-3, size_hi=1.0e-2, tol=0.02):
    """Return differential flux in [size_lo, size_hi) from an ORDEM
    SIZEFLUX_SC.OUT file, using the reverse-cumulative subtraction.
    tol is the allowed relative mismatch when matching size_lo/size_hi
    against the file's fixed size grid (should be near-exact; ORDEM's grid
    includes round decade points, but allow a little slack)."""
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
    return flux_lo - flux_hi


def parse_master_dia(filepath):
    """Return the total flux [1/m^2/yr] from a MASTER _d.dia footer line.
    Only correct as 'the 1mm-1cm flux' if the run's lower/upper diameter
    thresholds were actually set to 1mm/1cm - otherwise this is the total
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
    """Parse the raw job-manifest text into {hash: (altitude_km, inclination_deg)}.
    Handles lines like:
      <hash>\ttaskNumber=17, year=2026, ..., apogeeKm=1000, perigeeKm=1000,
      inclinationDeg=65, ...\t..."""
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


def collect_ordem(ordem_dir, manifest_path):
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
            flux = parse_ordem_sizeflux(sizeflux_path)
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
    p.add_argument("--output", default="flux_comparison.csv")
    args = p.parse_args()

    print("Parsing ORDEM results...")
    ordem = collect_ordem(args.ordem_dir, args.ordem_manifest)
    print(f"  parsed {len(ordem)} ORDEM tasks")

    print("Parsing MASTER results...")
    master = collect_master(args.master_dir)
    print(f"  parsed {len(master)} MASTER tasks")

    all_keys = sorted(set(ordem) | set(master))
    rows = []
    print()
    print(f"{'alt_km':>7} {'inc_deg':>8} {'ORDEM_1mm_1cm':>15} {'MASTER_1mm_1cm':>16} {'ratio_O/M':>10}")
    for alt, inc in all_keys:
        o = ordem.get((alt, inc))
        m = master.get((alt, inc))
        ratio = (o / m) if (o is not None and m not in (None, 0)) else None
        rows.append({
            "altitude_km": alt, "inclination_deg": inc,
            "ordem_flux_1mm_1cm": o, "master_flux_1mm_1cm": m,
            "ratio_ordem_to_master": ratio,
        })
        o_s = f"{o:.6f}" if o is not None else "MISSING"
        m_s = f"{m:.6f}" if m is not None else "MISSING"
        r_s = f"{ratio:.3f}" if ratio is not None else "N/A"
        print(f"{alt:7} {inc:8.1f} {o_s:>15} {m_s:>16} {r_s:>10}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "altitude_km", "inclination_deg",
            "ordem_flux_1mm_1cm", "master_flux_1mm_1cm", "ratio_ordem_to_master",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()