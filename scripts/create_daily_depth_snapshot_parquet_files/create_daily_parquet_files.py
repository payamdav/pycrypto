#!/usr/bin/env python3
"""
Create one Parquet file per calendar day from depth snapshot JSON files.

Input files (current working directory):
  {asset}_depth_snapshot_{ts_ms}.json
  {asset}_depth_snapshot_{ts_ms}.json.gz   (decompressed in memory, file untouched)

Output:
  {asset}_depth_snapshot_{year}_{month}_{day}.parquet

Processed source files are renamed to *.del after their Parquet is written.
Only "complete" days are written — the last group of files is left untouched
because the day may still be receiving new snapshots.

Usage:
  python create_daily_parquet_files.py            # all complete days
  python create_daily_parquet_files.py --days 3   # stop after 3 daily files
"""

import gzip
import glob
import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ── PyArrow schema ────────────────────────────────────────────────────────────

_LEVEL = pa.struct([
    pa.field("price",  pa.float64()),
    pa.field("volume", pa.float64()),
])
_LIST_LEVEL = pa.list_(_LEVEL)

SCHEMA = pa.schema([
    pa.field("ts",     pa.timestamp("ms")),
    pa.field("bids",   _LIST_LEVEL),
    pa.field("asks",   _LIST_LEVEL),
    pa.field("bcount", pa.int32()),
    pa.field("acount", pa.int32()),
    pa.field("bmin",   pa.float64()),
    pa.field("bmax",   pa.float64()),
    pa.field("amin",   pa.float64()),
    pa.field("amax",   pa.float64()),
    pa.field("arange", pa.float64()),
    pa.field("brange", pa.float64()),
    pa.field("spread", pa.float64()),
    pa.field("mid",    pa.float64()),
    pa.field("av",     pa.float64()),
    pa.field("bv",     pa.float64()),
])

# ── File discovery ────────────────────────────────────────────────────────────

def find_snapshot_files() -> list[str]:
    """Return sorted list of unprocessed snapshot files in CWD."""
    files = glob.glob("*.json") + glob.glob("*.json.gz")
    files = [f for f in files if not f.endswith(".del")]
    return sorted(files)


def extract_asset(filename: str) -> str:
    """'btcusdt_depth_snapshot_1234567890.json' → 'btcusdt'"""
    return Path(filename).name.split("_")[0]

# ── JSON loading ──────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    """Load JSON, decompressing gzip in memory if needed."""
    if path.endswith(".json.gz"):
        with gzip.open(path, "rb") as fh:
            return json.loads(fh.read())
    with open(path, "r") as fh:
        return json.load(fh)

# ── Record building ───────────────────────────────────────────────────────────

def build_record(data: dict) -> dict:
    """
    Parse one depth snapshot dict into a flat record ready for Parquet.
    bids are sorted descending (bids[0] = best bid, highest price).
    asks are sorted ascending  (asks[0] = best ask, lowest price).
    """
    b = [(float(row[0]), float(row[1])) for row in data["bids"]]
    a = [(float(row[0]), float(row[1])) for row in data["asks"]]

    bmax = b[0][0]  if b else 0.0
    bmin = b[-1][0] if b else 0.0
    amin = a[0][0]  if a else 0.0
    amax = a[-1][0] if a else 0.0
    av   = sum(v for _, v in a)
    bv   = sum(v for _, v in b)

    return {
        "ts":     data["T"],
        "bids":   [{"price": p, "volume": v} for p, v in b],
        "asks":   [{"price": p, "volume": v} for p, v in a],
        "bcount": len(b),
        "acount": len(a),
        "bmin":   bmin,
        "bmax":   bmax,
        "amin":   amin,
        "amax":   amax,
        "arange": amax - amin,
        "brange": bmax - bmin,
        "spread": amin - bmax,
        "mid":    (amin + bmax) / 2.0,
        "av":     av,
        "bv":     bv,
    }

# ── Parquet writing ───────────────────────────────────────────────────────────

def write_parquet(asset: str, day, records: list[dict]) -> str:
    """Build and write a Parquet file for one day. Returns the output filename."""
    filename = (
        f"{asset}_depth_snapshot_"
        f"{day.year}_{day.month:02d}_{day.day:02d}.parquet"
    )

    table = pa.table(
        {
            "ts":     pa.array([r["ts"]     for r in records], type=pa.timestamp("ms")),
            "bids":   pa.array([r["bids"]   for r in records], type=_LIST_LEVEL),
            "asks":   pa.array([r["asks"]   for r in records], type=_LIST_LEVEL),
            "bcount": pa.array([r["bcount"] for r in records], type=pa.int32()),
            "acount": pa.array([r["acount"] for r in records], type=pa.int32()),
            "bmin":   pa.array([r["bmin"]   for r in records], type=pa.float64()),
            "bmax":   pa.array([r["bmax"]   for r in records], type=pa.float64()),
            "amin":   pa.array([r["amin"]   for r in records], type=pa.float64()),
            "amax":   pa.array([r["amax"]   for r in records], type=pa.float64()),
            "arange": pa.array([r["arange"] for r in records], type=pa.float64()),
            "brange": pa.array([r["brange"] for r in records], type=pa.float64()),
            "spread": pa.array([r["spread"] for r in records], type=pa.float64()),
            "mid":    pa.array([r["mid"]    for r in records], type=pa.float64()),
            "av":     pa.array([r["av"]     for r in records], type=pa.float64()),
            "bv":     pa.array([r["bv"]     for r in records], type=pa.float64()),
        },
        schema=SCHEMA,
    )

    pq.write_table(table, filename)
    return filename


def mark_del(file_paths: list[str]) -> None:
    """Rename each file to <filename>.del to mark it as processed."""
    for fp in file_paths:
        os.rename(fp, fp + ".del")

# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(
    filename: str,
    n_files: int,
    first_ts: int,
    last_ts: int,
    last_rec: dict,
    elapsed_s: float,
) -> None:
    coverage_min = (last_ts - first_ts) / 60_000
    print(
        f"{filename} | {n_files} files | {coverage_min:.1f}m coverage | "
        f"spread={last_rec['spread']:.4f} mid={last_rec['mid']:.2f} "
        f"av={last_rec['av']:.2f} bv={last_rec['bv']:.2f} | "
        f"{elapsed_s:.2f}s"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create daily Parquet files from depth snapshot JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--days", type=int, default=0, metavar="N",
        help="Stop after writing N daily files (default: 0 = unlimited).",
    )
    args = parser.parse_args()
    max_days = args.days

    files = find_snapshot_files()
    if not files:
        print("No snapshot files (*.json / *.json.gz) found in current directory.")
        sys.exit(0)

    asset = extract_asset(files[0])
    print(f"Asset       : {asset}")
    print(f"Files found : {len(files):,}")
    if max_days:
        print(f"Day limit   : {max_days}")
    print()

    current_day  = None
    day_files:   list[str]  = []
    day_records: list[dict] = []
    days_written = 0

    for fp in files:
        # ── Load file ────────────────────────────────────────────────────────
        try:
            data = load_json(fp)
        except Exception as exc:
            print(f"[WARN] Cannot read {fp}: {exc} — skipping.")
            continue

        ts   = data["T"]
        date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()

        # ── Day boundary detected ────────────────────────────────────────────
        if current_day is not None and date != current_day:
            t0 = time.perf_counter()
            out_file = write_parquet(asset, current_day, day_records)
            elapsed  = time.perf_counter() - t0

            print_report(
                out_file,
                len(day_files),
                day_records[0]["ts"],
                day_records[-1]["ts"],
                day_records[-1],
                elapsed,
            )
            mark_del(day_files)
            days_written += 1

            if max_days and days_written >= max_days:
                print(f"\nReached --days {max_days} limit. Exiting.")
                sys.exit(0)

            # Reset for the new day
            current_day  = date
            day_files    = []
            day_records  = []

        # ── Accumulate record ────────────────────────────────────────────────
        if current_day is None:
            current_day = date

        day_records.append(build_record(data))
        day_files.append(fp)

    # ── End of file list ─────────────────────────────────────────────────────
    if day_files:
        print(
            f"\n{len(day_files):,} file(s) from {current_day} were NOT written — "
            f"day may still be receiving snapshots. "
            f"Run again once the day is complete (a newer day's files appear)."
        )

    print(f"\nDone. {days_written} daily Parquet file(s) written.")


if __name__ == "__main__":
    main()
