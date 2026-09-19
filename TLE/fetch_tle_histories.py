#!/usr/bin/env python3
"""Fetch historical TLEs from Space-Track.org — BATCHED + DEBUG version.

Usage:
    python fetch_tle_histories_batched_debug.py user@email.com password \\
        --meta TLE/real_tles.tle.meta.json \\
        --output-dir TLE/histories/ \\
        --start-date 2019-01-01 --end-date 2024-12-31 \\
        --resume --verbose
"""
import argparse
import json
import os
import sys
import time
from typing import List, Optional

import requests

BATCH_SIZE = 50           # Reduced from 100 — safer for URL length limits
SLEEP_BETWEEN_BATCHES = 4


def login(session: requests.Session, email: str, password: str) -> bool:
    r = session.post(
        "https://www.space-track.org/ajaxauth/login",
        data={"identity": email, "password": password},
    )
    if r.status_code != 200:
        print(f"Login failed: {r.status_code}", file=sys.stderr)
        return False
    print("Login OK")
    return True


def fetch_batch(
    session: requests.Session,
    norad_ids: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
    verbose: bool = False,
) -> Optional[List[dict]]:
    """Fetch historical TLEs for a batch of NORAD IDs."""
    id_list = ",".join(norad_ids)
    fields = ",".join([
        "NORAD_CAT_ID", "OBJECT_NAME", "EPOCH",
        "TLE_LINE1", "TLE_LINE2",
    ])

    date_preds = ""
    if start_date:
        date_preds += f"EPOCH/%3E{start_date}/"
    if end_date:
        date_preds += f"EPOCH/%3C{end_date}/"

    url = (
        "https://www.space-track.org/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{id_list}/"
        f"{date_preds}"
        "orderby/EPOCH%20asc/"
        f"format/json/predicates/{fields}"
    )

    if verbose:
        print(f"    URL length: {len(url)} chars")
        print(f"    First 200 chars: {url[:200]}")

    r = session.get(url)
    
    if verbose:
        print(f"    HTTP status: {r.status_code}")
        print(f"    Response length: {len(r.text)} chars")
        if len(r.text) < 500:
            print(f"    Response preview: {r.text[:300]}")

    if r.status_code != 200:
        print(f"  Batch fetch failed: {r.status_code}", file=sys.stderr)
        if len(r.text) < 1000:
            print(f"  Response: {r.text[:500]}", file=sys.stderr)
        return None

    try:
        records = json.loads(r.text)
    except json.JSONDecodeError as e:
        print(f"  JSON decode failed: {e}", file=sys.stderr)
        print(f"  Raw text (first 500 chars): {r.text[:500]}", file=sys.stderr)
        return None

    if isinstance(records, dict):
        if "error" in records:
            print(f"  API error: {records['error']}", file=sys.stderr)
            return []
        # Sometimes Space-Track returns a single object as a dict, not a list
        if "NORAD_CAT_ID" in records:
            records = [records]
        else:
            print(f"  Unexpected dict response: {list(records.keys())}", file=sys.stderr)
            return []

    if not isinstance(records, list):
        print(f"  Unexpected response type: {type(records)}", file=sys.stderr)
        return []

    if verbose and records:
        sample_ids = [str(r.get("NORAD_CAT_ID", "?")) for r in records[:3]]
        print(f"    Parsed {len(records)} records. Sample NORAD IDs: {sample_ids}")

    return records


def write_object_tles(records: List[dict], output_dir: str, verbose: bool = False) -> dict:
    """Write records grouped by NORAD ID. Returns counts."""
    counts = {}
    by_id = {}
    for rec in records:
        nid = str(rec.get("NORAD_CAT_ID", ""))
        if not nid:
            continue
        by_id.setdefault(nid, []).append(rec)

    if verbose and by_id:
        print(f"    Unique NORAD IDs in response: {list(by_id.keys())[:5]}...")

    for nid, recs in by_id.items():
        lines = []
        for rec in recs:
            l1 = rec.get("TLE_LINE1", "")
            l2 = rec.get("TLE_LINE2", "")
            name = rec.get("OBJECT_NAME", "")
            if l1 and l2:
                if name:
                    lines.append(name)
                lines.append(l1)
                lines.append(l2)
        if lines:
            path = os.path.join(output_dir, f"{nid}.tle")
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
            counts[nid] = len(lines) // 3
    return counts


def main():
    p = argparse.ArgumentParser(description="Fetch TLE histories (batched + debug)")
    p.add_argument("email")
    p.add_argument("password")
    p.add_argument("--meta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_BATCHES)
    p.add_argument("--verbose", action="store_true", help="Print debug info for every batch")
    p.add_argument("--max-batches", type=int, default=None, help="Stop after N batches (for testing)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.meta) as f:
        meta = json.load(f)
    all_ids = [str(r.get("NORAD_CAT_ID", "")) for r in meta if r.get("NORAD_CAT_ID")]

    print(f"Total IDs from meta: {len(all_ids)}")
    print(f"First 5 IDs: {all_ids[:5]}")

    if args.resume:
        existing = set()
        for fname in os.listdir(args.output_dir):
            if fname.endswith(".tle"):
                nid = fname.replace(".tle", "")
                if os.path.getsize(os.path.join(args.output_dir, fname)) > 0:
                    existing.add(nid)
        all_ids = [nid for nid in all_ids if nid not in existing]
        print(f"Resume: {len(existing)} already fetched, {len(all_ids)} remaining")
    else:
        print(f"Total objects to fetch: {len(all_ids)}")

    if not all_ids:
        print("Nothing to fetch.")
        return

    print(f"Date range: {args.start_date or 'beginning'} to {args.end_date or 'now'}")
    print(f"Batch size: {args.batch_size}, Sleep: {args.sleep}s")
    print(f"Estimated batches: {len(all_ids) // args.batch_size + 1}")
    print()

    session = requests.Session()
    if not login(session, args.email, args.password):
        sys.exit(1)

    total_tles = 0
    total_objects_fetched = 0
    t0 = time.time()

    n_batches = 0
    for batch_num in range(0, len(all_ids), args.batch_size):
        batch_ids = all_ids[batch_num:batch_num + args.batch_size]
        n_batches += 1
        
        print(f"\nBatch {n_batches}: IDs {batch_ids[0]} to {batch_ids[-1]} ({len(batch_ids)} objects)")
        
        records = fetch_batch(session, batch_ids, args.start_date, args.end_date, verbose=args.verbose)

        if records is None:
            print(f"  -> FAILED (network or HTTP error)")
        elif len(records) == 0:
            print(f"  -> EMPTY (0 records returned)")
            # Try without date filter on first empty batch to diagnose
            if n_batches == 1 and (args.start_date or args.end_date):
                print(f"  -> DIAGNOSTIC: retrying without date filter...")
                records_nodate = fetch_batch(session, batch_ids[:5], None, None, verbose=True)
                if records_nodate:
                    print(f"  -> Without dates: {len(records_nodate)} records. Date filter is the problem.")
                else:
                    print(f"  -> Without dates: also empty. IDs may have no history at all.")
        else:
            counts = write_object_tles(records, args.output_dir, verbose=args.verbose)
            total_tles += sum(counts.values())
            total_objects_fetched += len(counts)
            print(f"  -> SUCCESS: {len(counts)} objects, {sum(counts.values())} TLEs written")
            if args.verbose:
                for nid, n in sorted(counts.items())[:5]:
                    print(f"      {nid}: {n} TLEs")

        if args.max_batches and n_batches >= args.max_batches:
            print(f"\nStopped after {args.max_batches} batches (testing mode)")
            break

        if batch_num + args.batch_size < len(all_ids):
            time.sleep(args.sleep)

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  Batches run: {n_batches}")
    print(f"  Objects with data: {total_objects_fetched}")
    print(f"  Total TLEs: {total_tles}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()