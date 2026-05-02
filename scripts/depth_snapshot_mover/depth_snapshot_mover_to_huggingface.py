#!/usr/bin/env python3
"""
Move depth snapshot files from /var/data/{asset}/ to HuggingFace dataset
payamdavaee/depth_snapshot.

File formats expected in /var/data/{asset}/:
  {asset}_depth_snapshot_{timestamp_ms}.json
  {asset}_depth_snapshot_{timestamp_ms}.json.gz   (decompressed before upload)

After a successful upload each file is renamed to *.del instead of being
deleted immediately.  This protects against data loss from bugs — you can
safely remove *.del files once you are confident the upload is working.

Modes
-----
TEST (default)  — 1 asset × 1 batch of 100 files, then exit.
FULL (--full)   — all assets, all batches.

Usage
-----
  # Test mode (safe first run):
  python depth_snapshot_mover_to_huggingface.py

  # Full mode (production):
  python depth_snapshot_mover_to_huggingface.py --full

Environment
-----------
  HUGGINGFACE_FULL_TOKEN   HuggingFace access token with write permissions.
"""

import gzip
import os
import sys
import time
import tomllib
import argparse
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, login

# ── Constants ────────────────────────────────────────────────────────────────

CONFIG_PATH   = Path(__file__).parent / "iconfig.toml"
REPO_ID       = "payamdavaee/depth_snapshot"
DATA_BASE_DIR = Path("/var/data")
BATCH_SIZE    = 100


# ── Config ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


# ── File helpers ─────────────────────────────────────────────────────────────

def scan_asset_files(asset_dir: Path) -> list[str]:
    """
    Return a sorted list of absolute file paths inside asset_dir that are
    eligible for upload (.json or .json.gz, not already marked .del).
    Sorting by name gives natural timestamp order.
    """
    files: list[str] = []
    with os.scandir(asset_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if name.endswith(".del"):
                continue
            if name.endswith(".json") or name.endswith(".json.gz"):
                files.append(entry.path)
    files.sort()
    return files


def read_file(file_path: str) -> tuple[str, bytes]:
    """
    Read a snapshot file, decompressing .gz on the fly.
    Returns (filename_for_huggingface, raw_bytes).
    The .gz extension is stripped so the stored file is always plain JSON.
    """
    if file_path.endswith(".json.gz"):
        with gzip.open(file_path, "rb") as fh:
            data = fh.read()
        hf_name = Path(file_path).name[:-3]   # drop .gz → .json
    else:
        with open(file_path, "rb") as fh:
            data = fh.read()
        hf_name = Path(file_path).name
    return hf_name, data


# ── HuggingFace helpers ───────────────────────────────────────────────────────

def ensure_repo_exists(api: HfApi) -> None:
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="dataset")
    except Exception:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False)
        print(f"Created HuggingFace dataset repo: {REPO_ID}")


# ── Core logic ────────────────────────────────────────────────────────────────

def process_batch(
    api: HfApi,
    asset: str,
    batch_files: list[str],
    batch_num: int,
) -> int:
    """
    Read, upload, and rename one batch of files.
    Returns the number of files successfully uploaded.
    """
    operations: list[CommitOperationAdd] = []
    ready_paths: list[str] = []

    for file_path in batch_files:
        try:
            hf_name, data = read_file(file_path)
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{asset}/{hf_name}",
                    path_or_fileobj=data,
                )
            )
            ready_paths.append(file_path)
        except Exception as exc:
            print(f"  [ERROR] Reading {Path(file_path).name}: {exc}")

    if not operations:
        return 0

    try:
        api.create_commit(
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=(
                f"{asset}: add {len(operations)} depth snapshots (batch {batch_num})"
            ),
            operations=operations,
        )
    except Exception as exc:
        print(f"  [ERROR] Commit failed for batch {batch_num}: {exc}")
        # Files are NOT renamed — they stay for the next run.
        return 0

    # Rename only after a confirmed successful commit.
    for fp in ready_paths:
        os.rename(fp, fp + ".del")

    return len(operations)


def process_asset(api: HfApi, asset: str, test_mode: bool) -> int:
    asset_dir = DATA_BASE_DIR / asset
    if not asset_dir.exists():
        print(f"  [SKIP] {asset_dir} does not exist.")
        return 0

    print(f"  Scanning {asset_dir} …")
    all_files = scan_asset_files(asset_dir)
    total_files = len(all_files)
    print(f"  Found {total_files:,} file(s) pending upload.")

    if not all_files:
        print(f"  Nothing to do.")
        return 0

    total_uploaded = 0

    for batch_idx in range(0, total_files, BATCH_SIZE):
        batch_num   = (batch_idx // BATCH_SIZE) + 1
        batch_files = all_files[batch_idx : batch_idx + BATCH_SIZE]
        t_start     = time.perf_counter()

        uploaded = process_batch(api, asset, batch_files, batch_num)
        total_uploaded += uploaded

        duration        = time.perf_counter() - t_start
        files_remaining = total_files - batch_idx - len(batch_files)

        print(
            f"Asset: {asset} - batch: {batch_num} - "
            f"time duration: {duration:.1f}s - batch remains: {files_remaining}"
        )

        if test_mode:
            print("  [TEST MODE] Stopping after first batch.")
            break

    return total_uploaded


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move depth snapshots from VPS to HuggingFace dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full mode: all assets, all batches (default: test mode).",
    )
    args = parser.parse_args()
    test_mode = not args.full

    if test_mode:
        print("=" * 60)
        print("TEST MODE  —  1 asset × 1 batch (≤100 files)")
        print("Processed files will be renamed to *.del, not deleted.")
        print("Run with --full to process everything.")
        print("=" * 60)

    hf_token = os.environ.get("HUGGINGFACE_FULL_TOKEN")
    if not hf_token:
        print(
            "ERROR: HUGGINGFACE_FULL_TOKEN environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_config()
    assets = [a.lower() for a in config["settings"]["assets"]]

    login(token=hf_token, add_to_git_credential=False)
    api = HfApi()
    ensure_repo_exists(api)

    grand_total = 0

    for asset in assets:
        print(f"\n{'=' * 60}")
        print(f"Asset: {asset}")
        uploaded = process_asset(api, asset, test_mode=test_mode)
        grand_total += uploaded
        print(f"  Uploaded: {uploaded:,} file(s)")

        if test_mode:
            print("\n[TEST MODE] Stopping after first asset.")
            break

    print(f"\n{'=' * 60}")
    print(f"GRAND TOTAL UPLOADED: {grand_total:,} file(s)")


if __name__ == "__main__":
    main()
