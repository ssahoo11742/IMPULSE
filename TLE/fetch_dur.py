#!/usr/bin/env python3
"""
Fetch real per-object observation durations for a fleet already pulled by
fetch_tles.py, using Space-Track's `satcat` class (LAUNCH/DECAY dates -
one row per object) rather than `gp_history` (thousands of historical TLE
rows per object). duration = (DECAY or today) - LAUNCH.

Requires: pip install requests
Usage:    python fetch_object_durations.py <email> <password> \
              --meta real_tles.tle.meta.json \
              --output real_tles.tle.durations.json
"""
import argparse
import json
import time
from datetime import date, datetime

import requests

CHUNK_SIZE = 500          # keep query URLs a reasonable length
SLEEP_BETWEEN_CALLS_S = 2  # stay well under Space-Track's rate limit


def _parse_date(s):
    if not s:
        return None
    # satcat LAUNCH/DECAY are typically "YYYY-MM-DD"
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def fetch_durations(email, password, norad_ids):
    session = requests.Session()
    r = session.post(
        "https://www.space-track.org/ajaxauth/login",
        data={"identity": email, "password": password},
    )
    assert r.status_code == 200, f"Login failed: {r.status_code}"
    print("Login OK")

    today = date.today()
    results = {}
    ids = list(norad_ids)

    for i in range(0, len(ids), CHUNK_SIZE):
        chunk = ids[i:i + CHUNK_SIZE]
        id_list = ",".join(str(x) for x in chunk)
        url = (
            "https://www.space-track.org/basicspacedata/query/class/satcat/"
            f"NORAD_CAT_ID/{id_list}/"
            "format/json/predicates/NORAD_CAT_ID,LAUNCH,DECAY"
        )
        r = session.get(url)
        assert r.status_code == 200, f"Fetch failed on chunk {i}: {r.status_code}"
        recs = json.loads(r.text)

        for rec in recs:
            nid = rec.get("NORAD_CAT_ID")
            launch = _parse_date(rec.get("LAUNCH"))
            decay = _parse_date(rec.get("DECAY"))
            if nid is None or launch is None:
                continue
            end = decay if decay is not None else today
            duration_days = max(0, (end - launch).days)
            results[str(nid)] = {
                "launch": str(launch),
                "decay": str(decay) if decay else None,
                "duration_days": duration_days,
            }

        print(f"  chunk {i//CHUNK_SIZE + 1}/{-(-len(ids)//CHUNK_SIZE)}: "
              f"{len(recs)} records, {len(results)} durations resolved so far")
        time.sleep(SLEEP_BETWEEN_CALLS_S)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("email")
    p.add_argument("password")
    p.add_argument("--meta", required=True, help="meta.json from fetch_tles.py")
    p.add_argument("--output", default=None,
                    help="Output path (default: <meta>.durations.json)")
    args = p.parse_args()

    with open(args.meta) as f:
        records = json.load(f)
    norad_ids = sorted({rec["NORAD_CAT_ID"] for rec in records if "NORAD_CAT_ID" in rec})
    print(f"Resolving durations for {len(norad_ids)} objects from {args.meta}")

    durations = fetch_durations(args.email, args.password, norad_ids)

    missing = len(norad_ids) - len(durations)
    if missing:
        print(f"WARNING: {missing} objects had no LAUNCH date in satcat - "
              f"they'll need a fallback duration when merged")

    out_path = args.output or (args.meta.replace(".meta.json", "") + ".durations.json")
    with open(out_path, "w") as f:
        json.dump(durations, f, indent=2)
    print(f"Saved durations for {len(durations)} objects to {out_path}")


if __name__ == "__main__":
    main()