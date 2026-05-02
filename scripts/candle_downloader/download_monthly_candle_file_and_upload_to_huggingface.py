#!/usr/bin/env python3
"""
Download monthly 1-minute candle files from Binance and upload to HuggingFace dataset.

For each asset in iconfig.toml the script:
  1. Lists what is already present in payamdavaee/candles on HuggingFace.
  2. Determines which months (from 2024-01 up to the latest Binance-available month) are missing.
  3. Downloads the missing files from Binance, decompresses them, and uploads the CSV.

Binance publishes a monthly file 2 days after the month ends, so the latest available
month M satisfies: last_day(M) + 2 days <= today.
"""

import os
import sys
import calendar
import requests
import zipfile
import io
import tomllib
from datetime import date, timedelta
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, login

CONFIG_PATH = Path(__file__).parent / "iconfig.toml"
REPO_ID = "payamdavaee/candles"
START_YEAR, START_MONTH = 2024, 1


def load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def get_latest_binance_available_month(today=None):
    """
    Returns (year, month) of the latest month whose full data is available on Binance.
    Binance publishes data 2 days after the month ends, so last_day(M) + 2 <= today.
    """
    if today is None:
        today = date.today()
    check_date = today - timedelta(days=2)
    _, last_day = calendar.monthrange(check_date.year, check_date.month)
    if check_date.day >= last_day:
        return check_date.year, check_date.month
    if check_date.month == 1:
        return check_date.year - 1, 12
    return check_date.year, check_date.month - 1


def months_range(start_year, start_month, end_year, end_month):
    """Yield (year, month) tuples from start to end inclusive."""
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def get_existing_months(api, asset):
    """Return a set of (year, month) tuples already uploaded for this asset."""
    existing = set()
    try:
        all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        prefix = f"{asset}/{asset}-1m-"
        for f in all_files:
            if f.startswith(prefix) and f.endswith(".csv"):
                name = Path(f).name  # e.g. BTCUSDT-1m-2026-03.csv
                parts = name.replace(".csv", "").split("-")
                # Expected parts: [ASSET, 1m, YEAR, MONTH]
                if len(parts) == 4:
                    try:
                        existing.add((int(parts[2]), int(parts[3])))
                    except ValueError:
                        pass
    except Exception as e:
        print(f"  Warning: could not list existing files for {asset}: {e}")
    return existing


def download_binance_csv(asset, year, month):
    """
    Download and decompress a monthly kline zip from Binance.
    Returns CSV bytes, or None if the file does not exist (404).
    """
    url = (
        f"https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{asset}/1m/{asset}-1m-{year}-{month:02d}.zip"
    )
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV found in zip for {asset} {year}-{month:02d}")
        return zf.read(csv_files[0])


def ensure_repo_exists(api):
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="dataset")
    except Exception:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False)
        print(f"Created HuggingFace dataset repo: {REPO_ID}")


def main():
    hf_token = os.environ.get("HUGGINGFACE_FULL_TOKEN")
    if not hf_token:
        print("ERROR: HUGGINGFACE_FULL_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    config = load_config()
    assets = config["settings"]["assets"]

    login(token=hf_token, add_to_git_credential=False)
    api = HfApi()
    ensure_repo_exists(api)

    today = date.today()
    latest_year, latest_month = get_latest_binance_available_month(today)
    print(f"Date        : {today}")
    print(f"Latest month available on Binance: {latest_year}-{latest_month:02d}")
    print(f"Assets      : {', '.join(assets)}")
    print(f"Coverage    : {START_YEAR}-{START_MONTH:02d} → {latest_year}-{latest_month:02d}\n")

    grand_uploaded = 0
    grand_skipped = 0
    grand_not_found = 0

    for asset in assets:
        print(f"{'=' * 50}")
        print(f"Asset: {asset}")

        existing = get_existing_months(api, asset)
        all_months = list(months_range(START_YEAR, START_MONTH, latest_year, latest_month))
        missing = [(y, m) for y, m in all_months if (y, m) not in existing]

        print(f"  Files in dataset : {len(existing)} / {len(all_months)}")
        print(f"  Files to fetch   : {len(missing)}")

        if not missing:
            print("  Status: UP TO DATE — nothing to do.")
            grand_skipped += len(existing)
            print()
            continue

        operations = []
        not_found_count = 0

        for year, month in missing:
            label = f"{year}-{month:02d}"
            try:
                csv_bytes = download_binance_csv(asset, year, month)
                if csv_bytes is None:
                    print(f"  [{label}] 404 — not yet available on Binance, skipping.")
                    not_found_count += 1
                    continue
                path_in_repo = f"{asset}/{asset}-1m-{label}.csv"
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=path_in_repo,
                        path_or_fileobj=csv_bytes,
                    )
                )
                print(f"  [{label}] Downloaded {len(csv_bytes):,} bytes — queued for upload.")
            except Exception as e:
                print(f"  [{label}] ERROR: {e}")

        if operations:
            print(f"  Committing {len(operations)} file(s) to HuggingFace ...")
            api.create_commit(
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Add {asset} 1m candles: {len(operations)} file(s)",
                operations=operations,
            )
            print(f"  Committed {len(operations)} file(s) successfully.")

        grand_uploaded += len(operations)
        grand_skipped += len(existing)
        grand_not_found += not_found_count

        print(
            f"  Result: +{len(operations)} uploaded | "
            f"{len(existing)} already present | "
            f"{not_found_count} not on Binance"
        )
        print()

    print("=" * 50)
    print("OVERALL SUMMARY")
    print(f"  Uploaded        : {grand_uploaded}")
    print(f"  Already present : {grand_skipped}")
    print(f"  Not on Binance  : {grand_not_found}")


if __name__ == "__main__":
    main()
