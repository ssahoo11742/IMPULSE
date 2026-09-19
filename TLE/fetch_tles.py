#!/usr/bin/env python3
"""
Fetch passive LEO TLEs from Space-Track.org, constrained to a specific
altitude band, and bin results by inclination.

Requires: pip install requests
Usage:    python fetch_tles.py <email> <password> [--alt-min 700] [--alt-max 1000]
                                [--output real_tles.tle] [--limit 2000]

Fetches debris and rocket bodies (passive, non-maneuvering - the candidate
"sensor" population for DRIFTS) with:
  - PERIAPSIS and APOAPSIS both within [alt-min, alt-max] km altitude
    (both constrained, not just one, so the whole orbit sits inside the
    band rather than an eccentric orbit just clipping through it)
  - Eccentricity < 0.25 (redundant with the periapsis/apoapsis constraint
    in most cases, kept as a sanity filter)
  - Updated within the last 3 days (currently tracked, not decayed)
  - Object type: DEBRIS or ROCKET BODY

Outputs:
  - <output>            : raw TLE lines (2 lines per object), for the
                           propagator/EKF pipeline
  - <output>.meta.json   : one record per object (NORAD ID, name, object
                            type, inclination, periapsis/apoapsis, epoch),
                            for filtering/joining against the TLE file later
  - stdout               : population counts binned by inclination, using
                            the SAME bin edges as constants.py's
                            DEBRIS_INC_POP, for direct comparison against
                            the assumed distribution already in the codebase
"""
import argparse
import json
import sys

import requests

# Same bin edges as constants.py's DEBRIS_INC_POP (deg), so counts here can
# be compared directly against the assumed population distribution already
# baked into the simulation.
INC_BIN_EDGES_DEG = [0, 10, 20, 28, 40, 51, 65, 74, 82, 90, 97, 98, 100, 110, 120, 150, 180]


def fetch(email, password, alt_min_km, alt_max_km, output, limit):
    session = requests.Session()
    r = session.post(
        "https://www.space-track.org/ajaxauth/login",
        data={"identity": email, "password": password},
    )
    assert r.status_code == 200, f"Login failed: {r.status_code}"
    print("Login OK")

    fields = ",".join([
        "NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "EPOCH",
        "INCLINATION", "ECCENTRICITY", "PERIAPSIS", "APOAPSIS",
        "TLE_LINE1", "TLE_LINE2",
    ])

    # Space-Track range predicate syntax: FIELD/low--high/
    url = (
        "https://www.space-track.org/basicspacedata/query/class/gp/"
        f"PERIAPSIS/{alt_min_km}--{alt_max_km}/"
        f"APOAPSIS/{alt_min_km}--{alt_max_km}/"
        "ECCENTRICITY/%3C0.25/"
        "OBJECT_TYPE/DEBRIS,ROCKET%20BODY/"
        "EPOCH/%3Enow-3/"
        f"orderby/INCLINATION/limit/{limit}/"
        f"format/json/predicates/{fields}"
    )
    r = session.get(url)
    assert r.status_code == 200, f"Fetch failed: {r.status_code}"

    records = json.loads(r.text)
    print(f"Fetched {len(records)} objects in [{alt_min_km}, {alt_max_km}] km "
          f"(periapsis and apoapsis both constrained to the band)")

    # --- write TLE lines for the propagator/EKF pipeline ---
    tle_lines = []
    for rec in records:
        l1, l2 = rec.get("TLE_LINE1"), rec.get("TLE_LINE2")
        if l1 and l2:
            tle_lines.append(l1)
            tle_lines.append(l2)
    with open(output, "w") as f:
        f.write("\n".join(tle_lines) + "\n")
    print(f"Saved {len(tle_lines)//2} TLEs to {output}")

    # --- write metadata sidecar for filtering/joining later ---
    meta_path = output + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved metadata for {len(records)} objects to {meta_path}")

    # --- bin by inclination, using the same edges as DEBRIS_INC_POP ---
    bins = {edge: 0 for edge in INC_BIN_EDGES_DEG[:-1]}
    unbinned = 0
    for rec in records:
        try:
            inc = float(rec["INCLINATION"])
        except (KeyError, TypeError, ValueError):
            unbinned += 1
            continue
        placed = False
        for lo, hi in zip(INC_BIN_EDGES_DEG[:-1], INC_BIN_EDGES_DEG[1:]):
            if lo <= inc < hi:
                bins[lo] += 1
                placed = True
                break
        if not placed:
            unbinned += 1

    print(f"\n{'inclination bin':>20} {'count':>8}")
    for lo, hi in zip(INC_BIN_EDGES_DEG[:-1], INC_BIN_EDGES_DEG[1:]):
        print(f"{lo:>8}-{hi:<8}deg {bins[lo]:>8}")
    if unbinned:
        print(f"{'unbinned/missing':>20} {unbinned:>8}")

    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("email")
    p.add_argument("password")
    p.add_argument("--alt-min", type=float, default=700.0, help="Min altitude [km]")
    p.add_argument("--alt-max", type=float, default=1000.0, help="Max altitude [km]")
    p.add_argument("--output", default="real_tles.tle")
    p.add_argument("--limit", type=int, default=2000)
    args = p.parse_args()

    fetch(args.email, args.password, args.alt_min, args.alt_max,
          args.output, args.limit)


if __name__ == "__main__":
    main()