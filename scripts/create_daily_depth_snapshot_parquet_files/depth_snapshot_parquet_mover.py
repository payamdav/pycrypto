#!/usr/bin/env python3
"""
Upload depth snapshot Parquet files to HuggingFace and remove them locally.

Scans the base directory and one level of sub-folders for files matching:
  {asset}_depth_snapshot_{YYYY}_{MM}_{DD}.parquet

Each matched file is uploaded to:
  payamdavaee/depth_snapshot / {asset} / {filename}

After a successful upload the local file is deleted.

Usage:
  python depth_snapshot_parquet_mover.py            # uses current directory
  python depth_snapshot_parquet_mover.py /some/path # uses /some/path
"""

import os
import re
import sys
import time
import argparse
from pathlib import Path

from huggingface_hub import HfApi

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ID   = "payamdavaee/depth_snapshot"
REPO_TYPE = "dataset"
TOKEN_ENV  = "HUGGINGFACE_FULL_TOKEN"

# Pattern:  {asset}_depth_snapshot_{YYYY}_{MM}_{DD}.parquet
_PARQUET_RE = re.compile(
    r"^([a-z0-9]+)_depth_snapshot_(\d{4})_(\d{2})_(\d{2})\.parquet$"
)

# Small delay between uploads to stay well within HuggingFace rate limits.
UPLOAD_DELAY_S = 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────


def collect_parquet_files(base_dir: Path) -> list[Path]:
    """
    Return all depth-snapshot Parquet files found in *base_dir* and in each of
    its immediate sub-directories (one level deep only).
    """
    candidates: list[Path] = []

    dirs_to_scan: list[Path] = [base_dir]
    for entry in base_dir.iterdir():
        if entry.is_dir():
            dirs_to_scan.append(entry)

    for directory in dirs_to_scan:
        for fp in sorted(directory.iterdir()):
            if fp.is_file() and _PARQUET_RE.match(fp.name):
                candidates.append(fp)

    return candidates


def extract_asset(filename: str) -> str:
    """Return the asset portion from a depth-snapshot Parquet filename."""
    m = _PARQUET_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename!r}")
    return m.group(1)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload depth snapshot Parquet files to HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=None,
        metavar="PATH",
        help="Base directory to scan (default: current working directory).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    if not base_dir.is_dir():
        print(f"[ERROR] Not a directory: {base_dir}", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"[ERROR] Environment variable {TOKEN_ENV!r} is not set.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    files = collect_parquet_files(base_dir)
    if not files:
        print("No depth snapshot Parquet files found.")
        sys.exit(0)

    print(f"Base directory : {base_dir}")
    print(f"Files found    : {len(files)}")
    print()

    uploaded = 0
    failed   = 0

    for fp in files:
        try:
            asset      = extract_asset(fp.name)
            repo_path  = f"{asset}/{fp.name}"

            print(f"Uploading {fp}  →  {REPO_ID}/{repo_path} … ", end="", flush=True)

            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )

            fp.unlink()
            uploaded += 1
            print("OK")

        except Exception as exc:
            failed += 1
            print(f"FAILED — {exc}")

        time.sleep(UPLOAD_DELAY_S)

    print()
    print(f"Done. {uploaded} uploaded, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
