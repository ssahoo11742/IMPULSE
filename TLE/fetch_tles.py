#!/usr/bin/env python3
"""
Fetch passive LEO TLEs from Space-Track.org.
Requires: pip install requests
Usage:    python fetch_tles.py <email> <password> [output_file] [limit]

Fetches debris and rocket bodies with:
  - Mean motion > 11.25 rev/day (LEO)
  - Eccentricity < 0.25 (not highly elliptical)
  - Updated within last 3 days
  - Object type: DEBRIS or ROCKET BODY (passive, non-maneuvering)
"""
import requests, sys

def fetch_tles(email, password, output="real_tles.tle", n=500):
    session = requests.Session()
    r = session.post("https://www.space-track.org/ajaxauth/login",
                     data={"identity": email, "password": password})
    assert r.status_code == 200, f"Login failed: {r.status_code}"
    print("Login OK")

    url = ("https://www.space-track.org/basicspacedata/query/class/gp/"
           "MEAN_MOTION/%3E11.25/ECCENTRICITY/%3C0.25/"
           "OBJECT_TYPE/DEBRIS,ROCKET%20BODY/"
           "EPOCH/%3Enow-3/"
           f"orderby/NORAD_CAT_ID/limit/{n}/format/tle")
    r = session.get(url)
    assert r.status_code == 200, f"Fetch failed: {r.status_code}"

    with open(output, "w") as f:
        f.write(r.text)
    lines = [l for l in r.text.strip().splitlines() if l.strip()]
    print(f"Saved {len(lines)//3} TLEs to {output}")

if __name__ == "__main__":
    fetch_tles(sys.argv[1], sys.argv[2],
               sys.argv[3] if len(sys.argv) > 3 else "real_tles.tle",
               int(sys.argv[4]) if len(sys.argv) > 4 else 500)
