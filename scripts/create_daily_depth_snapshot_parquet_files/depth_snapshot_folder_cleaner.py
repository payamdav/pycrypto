#!/usr/bin/env python3
"""
Move .del files from the base directory (and its immediate sub-folders) to
an archive directory, compressing uncompressed ones along the way.

Rules
-----
* Files already compressed:  extension ends with ".gz.del"  → move as-is.
* Files not yet compressed:  extension ends with ".del" only → compress with
  ``nice -n 19 gzip``, rename the result to "<stem>.gz.del", then move.
* Only the base directory and its *immediate* sub-folders are processed
  (one level of depth).  Sub-folders of sub-folders are ignored.
* The mirror sub-folder structure is created inside the archive directory
  as needed.

Usage
-----
  python depth_snapshot_folder_cleaner.py <base_dir> <archive_dir>
"""

import shutil
import subprocess
import sys
import argparse
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────


def is_gz_del(path: Path) -> bool:
    """Return True when the file already has the '.gz.del' extension."""
    return path.name.endswith(".gz.del")


def is_del(path: Path) -> bool:
    """Return True when the file has the '.del' extension (including '.gz.del')."""
    return path.name.endswith(".del")


def compress_to_gz_del(src: Path) -> Path:
    """
    Compress *src* (a plain '.del' file) using ``nice -n 19 gzip``.

    ``gzip`` produces ``<name>.gz`` from ``<name>``, so the compressed file
    will be named ``<stem>.del.gz``.  We rename it to ``<stem>.gz.del`` to
    match the required convention, then return the new path.
    """
    subprocess.run(
        ["nice", "-n", "19", "gzip", str(src)],
        check=True,
    )
    # gzip removes the original and creates <src>.gz
    gz_path = src.with_suffix(src.suffix + ".gz")   # e.g. foo.del.gz
    # Rename to the required convention:  foo.gz.del
    stem = src.name[: -len(".del")]                  # strip trailing ".del"
    target = src.parent / f"{stem}.gz.del"
    gz_path.rename(target)
    return target


def move_del_files(src_dir: Path, dst_dir: Path) -> None:
    """
    Find all '.del' files in *src_dir*, compress the uncompressed ones, and
    move every resulting '.gz.del' file to *dst_dir*.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    for fp in sorted(src_dir.iterdir()):
        if not fp.is_file() or not is_del(fp):
            continue

        if is_gz_del(fp):
            # Already compressed — move straight to archive.
            shutil.move(str(fp), str(dst_dir / fp.name))
            print(f"  moved   {fp.name}  →  {dst_dir}")
        else:
            # Plain .del file — compress first, then move.
            gz_del = compress_to_gz_del(fp)
            shutil.move(str(gz_del), str(dst_dir / gz_del.name))
            print(f"  compressed+moved  {fp.name}  →  {dst_dir / gz_del.name}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move .del files to an archive directory, compressing as needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("base_dir", metavar="BASE_DIR", help="Source base directory.")
    parser.add_argument("archive_dir", metavar="ARCHIVE_DIR", help="Archive destination directory.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()

    if not base_dir.is_dir():
        print(f"[ERROR] Not a directory: {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Base directory   : {base_dir}")
    print(f"Archive directory: {archive_dir}")
    print()

    # ── 1. Process files directly inside base_dir ─────────────────────────────
    print(f"[base] {base_dir}")
    move_del_files(base_dir, archive_dir)

    # ── 2. Process each immediate sub-folder ──────────────────────────────────
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        sub_archive = archive_dir / entry.name
        print(f"[sub]  {entry}")
        move_del_files(entry, sub_archive)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
