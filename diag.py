#!/usr/bin/env python3
"""Diagnostic: check what's in the TLE history files."""
import os
import json

from TLE.tle_io import parse_tle

with open("TLE/real_tles.tle.meta.json") as f:
    meta = json.load(f)
valid_ids = {str(r.get("NORAD_CAT_ID", "")) for r in meta}

files = sorted(os.listdir("TLE/histories/"))
tle_files = [f for f in files if f.endswith(".tle")]
print(f"Total .tle files: {len(tle_files)}")
print()

# Check first 5 files
for fname in tle_files[:5]:
    nid = fname.replace(".tle", "")
    if nid not in valid_ids:
        print(f"{fname}: not in meta, skipping")
        continue

    fpath = os.path.join("TLE/histories/", fname)
    size = os.path.getsize(fpath)
    print(f"\n{fname} ({size} bytes):")

    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        lines = []
        parsed = 0
        epoch_jds = []
        mean_motions = []
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
                        parsed += 1
                        epoch_jds.append(tle.get("epoch_jd", 0))
                        mean_motions.append(tle.get("mean_motion", 0))
                    except Exception as e:
                        print(f"  Parse error: {e}")
                lines = []
            if parsed >= 5:
                break

        print(f"  Parsed {parsed} TLEs in first 5 attempts")
        print(f"  epoch_jds: {epoch_jds}")
        print(f"  mean_motions: {mean_motions}")