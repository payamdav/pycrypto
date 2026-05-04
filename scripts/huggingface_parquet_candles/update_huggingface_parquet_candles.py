#!/usr/bin/env python3
"""
Build and maintain a HuggingFace Parquet candle dataset.

For each asset in config.toml:
  1. Ensures the asset folder exists in payamdavaee/candles.
  2. Finds the last month already uploaded as Parquet.
  3. Downloads missing monthly CSVs from Binance, converts via
     load_numpy_candles_from_binance_file(), writes Parquet, uploads.

Parquet path in dataset : {asset}/{asset}-1m-{year}-{month:02d}.parquet
                example : btcusdt/btcusdt-1m-2026-03.parquet

Binance publishes a monthly file 2 days after the month ends:
  last_day(M) + 2 days <= today  →  month M is available.

Modes
-----
  TEST (default)  — upload at most 1 missing month per asset.
  FULL (--full)   — upload every missing month for every asset.

Env
---
  HUGGINGFACE_FULL_TOKEN   HuggingFace write access token.
"""

import calendar
import io
import os
import sys
import tempfile
import tomllib
import zipfile
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, CommitOperationAdd, login

# Make packages/numpy_candles importable when running from repo root or via
# the GitHub Action (PYTHONPATH=…/packages is also set in the workflow).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))
from numpy_candles import load_numpy_candles_from_binance_file, NC  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH    = Path(__file__).parent / "config.toml"
REPO_ID        = "payamdavaee/candles"
START_YEAR     = 2024
START_MONTH    = 1

# ── PyArrow schema (column order matches NC indices 0-10) ─────────────────────

_COLS: list[tuple[str, int, pa.DataType]] = [
    ("ts",   NC.ts,   pa.timestamp("ms")),
    ("o",    NC.o,    pa.float64()),
    ("h",    NC.h,    pa.float64()),
    ("l",    NC.l,    pa.float64()),
    ("c",    NC.c,    pa.float64()),
    ("v",    NC.v,    pa.float64()),
    ("q",    NC.q,    pa.float64()),
    ("n",    NC.n,    pa.int32()),
    ("vwap", NC.vwap, pa.float64()),
    ("vb",   NC.vb,   pa.float64()),
    ("vs",   NC.vs,   pa.float64()),
]

SCHEMA = pa.schema([pa.field(name, pa_type) for name, _, pa_type in _COLS])

# ── Date helpers ──────────────────────────────────────────────────────────────

def get_latest_binance_month(today: date | None = None) -> tuple[int, int]:
    """Return (year, month) of the latest month available on Binance.

    Binance publishes the file for month M once last_day(M) + 2 days have passed.
    """
    if today is None:
        today = date.today()
    check = today - timedelta(days=2)
    _, last_day = calendar.monthrange(check.year, check.month)
    if check.day >= last_day:
        return check.year, check.month
    if check.month == 1:
        return check.year - 1, 12
    return check.year, check.month - 1


def months_range(sy: int, sm: int, ey: int, em: int):
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1

# ── HuggingFace helpers ───────────────────────────────────────────────────────

def ensure_repo_exists(api: HfApi) -> None:
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="dataset")
    except Exception:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False)
        print(f"Created dataset repo: {REPO_ID}")


def get_existing_parquet_months(api: HfApi, asset: str) -> set[tuple[int, int]]:
    """Return set of (year, month) already present as Parquet for this asset."""
    existing: set[tuple[int, int]] = set()
    try:
        prefix = f"{asset}/{asset}-1m-"
        for path in api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"):
            if path.startswith(prefix) and path.endswith(".parquet"):
                # e.g. btcusdt/btcusdt-1m-2026-03.parquet → ['btcusdt', '1m', '2026', '03']
                parts = Path(path).stem.split("-")
                if len(parts) == 4:
                    try:
                        existing.add((int(parts[2]), int(parts[3])))
                    except ValueError:
                        pass
    except Exception as exc:
        print(f"  Warning: could not list existing files for {asset}: {exc}")
    return existing

# ── Binance + conversion ──────────────────────────────────────────────────────

def download_and_convert(asset: str, year: int, month: int) -> np.ndarray | None:
    """
    Download monthly kline zip from Binance, extract CSV, return numpy array.
    Returns None on 404 (asset not yet listed for that month).
    asset should be lowercase; Binance URL uses uppercase.
    """
    asset_up = asset.upper()
    url = (
        f"https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{asset_up}/1m/{asset_up}-1m-{year}-{month:02d}.zip"
    )
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found in zip for {asset} {year}-{month:02d}")
        csv_bytes = zf.read(csv_names[0])

    # load_numpy_candles_from_binance_file expects a file path or URL;
    # write to a temp file so we can hand it a path.
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(csv_bytes)
        tmp_path = tmp.name
    try:
        arr = load_numpy_candles_from_binance_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    return arr


def numpy_to_parquet_bytes(arr: np.ndarray) -> bytes:
    """Convert a numpy candle array (shape N×11) to Parquet bytes."""
    columns: dict[str, pa.Array] = {}
    for name, col_idx, pa_type in _COLS:
        col = arr[:, col_idx]
        if pa_type == pa.int32():
            col = col.astype(np.int32)
        columns[name] = pa.array(col, type=pa_type)

    table = pa.table(columns, schema=SCHEMA)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload monthly candle Parquet files to HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Upload ALL missing months (default: 1 missing month per asset).",
    )
    args = parser.parse_args()
    test_mode = not args.full

    if test_mode:
        print("TEST MODE — at most 1 missing month per asset.")
        print("Run with --full to upload all missing months.\n")

    hf_token = os.environ.get("HUGGINGFACE_FULL_TOKEN")
    if not hf_token:
        print("ERROR: HUGGINGFACE_FULL_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH, "rb") as fh:
        config = tomllib.load(fh)
    assets = [a.lower() for a in config["settings"]["assets"]]

    login(token=hf_token, add_to_git_credential=False)
    api = HfApi()
    ensure_repo_exists(api)

    today = date.today()
    latest_year, latest_month = get_latest_binance_month(today)
    print(f"Date        : {today}")
    print(f"Latest Binance month available: {latest_year}-{latest_month:02d}")
    print(f"Assets      : {', '.join(assets)}\n")

    grand_uploaded   = 0
    grand_skipped    = 0
    grand_not_found  = 0

    for asset in assets:
        print(f"{'=' * 55}")
        print(f"Asset: {asset}")

        existing  = get_existing_parquet_months(api, asset)
        all_months = list(months_range(START_YEAR, START_MONTH, latest_year, latest_month))
        missing   = [(y, m) for y, m in all_months if (y, m) not in existing]

        print(f"  Parquet in dataset : {len(existing)} / {len(all_months)}")
        print(f"  Missing            : {len(missing)}")

        if not missing:
            print("  Status: UP TO DATE")
            grand_skipped += len(existing)
            print()
            continue

        targets = missing[:1] if test_mode else missing
        if test_mode:
            print(f"  [TEST] Only uploading: {targets[0][0]}-{targets[0][1]:02d}")

        operations: list[CommitOperationAdd] = []
        not_found = 0

        for year, month in targets:
            label = f"{year}-{month:02d}"
            try:
                arr = download_and_convert(asset, year, month)
                if arr is None:
                    print(f"  [{label}] 404 — not on Binance yet, skipping.")
                    not_found += 1
                    continue

                parquet_bytes = numpy_to_parquet_bytes(arr)
                path_in_repo  = f"{asset}/{asset}-1m-{label}.parquet"

                operations.append(
                    CommitOperationAdd(
                        path_in_repo=path_in_repo,
                        path_or_fileobj=parquet_bytes,
                    )
                )
                print(
                    f"  [{label}] {arr.shape[0]:,} candles → "
                    f"{len(parquet_bytes):,} bytes"
                )
            except Exception as exc:
                print(f"  [{label}] ERROR: {exc}")

        if operations:
            print(f"  Committing {len(operations)} file(s) to HuggingFace …")
            api.create_commit(
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=(
                    f"{asset}: add {len(operations)} Parquet candle file(s)"
                ),
                operations=operations,
            )
            print(f"  Committed.")

        grand_uploaded  += len(operations)
        grand_skipped   += len(existing)
        grand_not_found += not_found

        print(
            f"  Result: +{len(operations)} uploaded | "
            f"{len(existing)} already present | "
            f"{not_found} not on Binance"
        )
        print()

    print("=" * 55)
    print("OVERALL SUMMARY")
    print(f"  Uploaded        : {grand_uploaded}")
    print(f"  Already present : {grand_skipped}")
    print(f"  Not on Binance  : {grand_not_found}")


if __name__ == "__main__":
    main()
