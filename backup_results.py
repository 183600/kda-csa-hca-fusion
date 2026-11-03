#!/usr/bin/env python3
"""Data backup utility for results_lm/.

Creates timestamped tar.gz archives of results_lm/ under backups/,
with simple rotation to keep the latest N backups.
Designed to be runnable inside Docker containers and locally.

Usage:
    python3 backup_results.py [--keep 5]

Constraints:
    - Does not modify results_lm/ contents.
    - Creates backups/ if missing.
    - Safe for repeated runs.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import tarfile
import sys

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_lm"
BACKUP_DIR = ROOT / "backups"
DEFAULT_KEEP = 5


def make_backup(keep: int = DEFAULT_KEEP) -> pathlib.Path:
    if not RESULTS_DIR.is_dir():
        msg = f"results_lm/ not found at {RESULTS_DIR}"
        print(msg, file=sys.stderr)
        sys.exit(1)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"results_lm_backup_{timestamp}.tar.gz"
    archive_path = BACKUP_DIR / archive_name

    print(f"[backup] Archiving {RESULTS_DIR} -> {archive_path}")
    with tarfile.open(archive_path, "w:gz") as tar:
        # Add contents preserving relative path
        tar.add(RESULTS_DIR, arcname=RESULTS_DIR.name)

    # Rotation
    backups = sorted(BACKUP_DIR.glob("results_lm_backup_*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(backups) > keep:
        for old in backups[keep:]:
            print(f"[backup] Removing old backup {old.name}")
            old.unlink()

    print(f"[backup] Done. {len(backups[:keep])} backup(s) retained.")
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup results_lm directory")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="Number of backups to retain")
    args = parser.parse_args()
    make_backup(keep=args.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
