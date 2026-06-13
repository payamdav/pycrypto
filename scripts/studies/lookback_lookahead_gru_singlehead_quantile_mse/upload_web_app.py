#!/usr/bin/env python3
"""
Upload the Observation Report Viewer SPA to the public GCS report bucket.

Walks the local ``web/`` directory (next to this script) recursively and
uploads every file, preserving the relative folder structure under a fixed
prefix in the bucket. Each object is uploaded with the correct ``Content-Type``
so the SPA serves cleanly when fetched over HTTP(S).

The viewer is deployed per-study under ``{OBSERVATION_SET_NAME}/app/`` so each
study folder is self-contained (its own reports/ AND its own viewer).

Resulting layout (with the default prefix
``lookback_lookahead_gru_singlehead_quantile_mse/app``):

    gs://payamdpycryptoreports/lookback_lookahead_gru_singlehead_quantile_mse/app/index.html
    gs://payamdpycryptoreports/lookback_lookahead_gru_singlehead_quantile_mse/app/css/styles.css
    gs://payamdpycryptoreports/lookback_lookahead_gru_singlehead_quantile_mse/app/js/config.js
    ...

Uploads use the shared ``gcs_tools`` package
(``packages/tools/google_cloud_storage_tools/``):
``gcs_json_key_file`` resolves credentials, ``write_file`` uploads each object.

Usage:
    python upload_web_app.py                          # -> {study}/app/ prefix
    python upload_web_app.py --dry-run                # list planned uploads
    python upload_web_app.py --prefix OtherStudy/app  # deploy another study's viewer
"""

import os
import sys
import argparse
import mimetypes

# --------------------------------------------------------------------------- #
#  Repository / package import bootstrap
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_GCS_TOOLS_DIR = os.path.join(
    _REPO_ROOT, "packages", "tools", "google_cloud_storage_tools"
)
if _GCS_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _GCS_TOOLS_DIR)

from gcs_tools import gcs_json_key_file, write_file  # noqa: E402

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
REPORT_BUCKET = "payamdpycryptoreports"
# Keep identical to the python study's OBSERVATION_SET_NAME so the viewer lands
# inside the same study folder as that study's reports/.
OBSERVATION_SET_NAME = "lookback_lookahead_gru_singlehead_quantile_mse"
# The SPA is served per-study under "{OBSERVATION_SET_NAME}/app/".
WEB_PREFIX = f"{OBSERVATION_SET_NAME}/app"
WEB_DIR = os.path.join(_THIS_DIR, "web")  # local source directory (via __file__)

# Explicit content types per extension. Falls back to mimetypes / octet-stream.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}

# Files / directories that should never be uploaded.
SKIP_NAMES = {".DS_Store", ".gitkeep"}
SKIP_DIRS = {"__pycache__", ".git"}


def guess_content_type(filename: str) -> str:
    """Resolve a Content-Type for a filename, preferring the explicit table."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in CONTENT_TYPES:
        return CONTENT_TYPES[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def iter_web_files(root: str):
    """Yield (absolute_path, relative_posix_path) for every uploadable file."""
    for dirpath, dirnames, filenames in os.walk(root):
        # prune unwanted directories in place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name in SKIP_NAMES:
                continue
            abs_path = os.path.join(dirpath, name)
            rel_path = os.path.relpath(abs_path, root)
            # normalize to forward slashes for the object key
            rel_posix = rel_path.replace(os.sep, "/")
            yield abs_path, rel_posix


def main():
    parser = argparse.ArgumentParser(
        description="Upload the web/ SPA hierarchy to the GCS report bucket.")
    parser.add_argument(
        "--bucket", default=REPORT_BUCKET,
        help=f"Target GCS bucket (default: {REPORT_BUCKET}).")
    parser.add_argument(
        "--prefix", default=WEB_PREFIX,
        help=f"Object key prefix in the bucket (default: {WEB_PREFIX}).")
    parser.add_argument(
        "--web-dir", default=WEB_DIR,
        help="Local web/ directory to upload (default: ./web next to script).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the planned uploads without performing them.")
    args = parser.parse_args()

    web_dir = os.path.abspath(args.web_dir)
    if not os.path.isdir(web_dir):
        print(f"ERROR: web directory not found: {web_dir}", flush=True)
        sys.exit(1)

    prefix = args.prefix.strip("/")
    files = sorted(iter_web_files(web_dir), key=lambda t: t[1])
    if not files:
        print(f"No files found under {web_dir}.", flush=True)
        return

    print(f"Uploading {len(files)} file(s) from {web_dir}", flush=True)
    print(f"  -> gs://{args.bucket}/{prefix}/", flush=True)

    if not args.dry_run:
        gcs_json_key_file()  # resolve credentials once before any upload

    uploaded = 0
    failures = []
    for abs_path, rel_posix in files:
        key = f"{prefix}/{rel_posix}" if prefix else rel_posix
        ctype = guess_content_type(rel_posix)
        if args.dry_run:
            print(f"  [dry-run] {rel_posix:40s} ({ctype})", flush=True)
            continue
        try:
            with open(abs_path, "rb") as fh:
                write_file(args.bucket, key, fh, content_type=ctype)
            uploaded += 1
            print(f"  uploaded {rel_posix:40s} -> "
                  f"gs://{args.bucket}/{key}  ({ctype})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {rel_posix}: {exc}", flush=True)
            failures.append((rel_posix, str(exc)))

    print("\n========== UPLOAD COMPLETE ==========", flush=True)
    if args.dry_run:
        print(f"  dry-run: {len(files)} file(s) would be uploaded.", flush=True)
    else:
        print(f"  uploaded: {uploaded}/{len(files)}", flush=True)
        print(f"  failures: {len(failures)}", flush=True)
        for f in failures:
            print(f"    - {f}", flush=True)
        if uploaded:
            entry = f"https://storage.googleapis.com/{args.bucket}/{prefix}/index.html"
            print(f"\n  App entry point: {entry}", flush=True)


if __name__ == "__main__":
    main()
