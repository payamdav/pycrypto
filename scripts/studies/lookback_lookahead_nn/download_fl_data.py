#!/usr/bin/env python3
"""
One-time copy of every asset's fl_data blob from GCS to the local ~/data cache.

All lookback/lookahead studies share the SAME fl_data. Downloading it from GCS
for every observation/run is slow, so this script pulls each asset's blob from
gs://payamdprojectbucket/fl_data_{asset} ONCE and writes it to
~/data/fl_data_{asset}.npy. The study trainers then load from ~/data (see each
study's load_fl_data), eliminating repeated GCS downloads.

Data spec: agents/datasets/lookback_lookahead_fl.md
"""
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_GCS_TOOLS_DIR = os.path.join(
    _REPO_ROOT, "packages", "tools", "google_cloud_storage_tools")
if _GCS_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _GCS_TOOLS_DIR)

from gcs_tools import gcs_json_key_file, read_file  # noqa: E402

DATA_BUCKET = "payamdprojectbucket"
LOCAL_DATA_DIR = os.path.expanduser("~/data")
ASSETS = ["btcusdt", "ethusdt", "trumpusdt", "vineusdt",
          "adausdt", "xrpusdt", "dogeusdt"]


def main():
    parser = argparse.ArgumentParser(
        description="Copy fl_data blobs from GCS to the local ~/data cache.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the local file exists.")
    args = parser.parse_args()

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    gcs_json_key_file()  # resolve credentials once before any GCS call

    for asset in ASSETS:
        local_path = os.path.join(LOCAL_DATA_DIR, f"fl_data_{asset}.npy")
        if os.path.exists(local_path) and not args.force:
            print(f"  skip {asset}: already at {local_path}", flush=True)
            continue
        try:
            print(f"  downloading fl_data_{asset} ...", flush=True)
            data_bytes = read_file(DATA_BUCKET, f"fl_data_{asset}")
            with open(local_path, "wb") as fh:
                fh.write(data_bytes)
            print(f"  saved -> {local_path} "
                  f"({len(data_bytes) / (1024 ** 2):.1f} MiB)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {asset}: {exc}", flush=True)

    print("\nDone. Local cache dir: " + LOCAL_DATA_DIR, flush=True)


if __name__ == "__main__":
    main()
