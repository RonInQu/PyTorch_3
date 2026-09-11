"""
Fix a mistake made by build_train_test_split_183.py.

The source dataset has TWO folders with the same 183 filenames but different content:
  - processedResults/training/  → blanked version (unlabeled regions flattened to
                                   baseline). Use for TRAINING.
  - processedResults/testing/   → raw version (real signal preserved). Use for
                                   INFERENCE / TESTING.

The original split script pulled both train and test files from the training/
folder. This script overwrites workspace test_data/*.parquet with the raw
versions from processedResults/testing/, keeping training_data/ untouched.

Behavior:
  - Backs up current workspace test_data/ to test_data.backup_testfix_<ts>/
  - For each parquet currently in test_data/, copies the same filename from
    processedResults/testing/ into test_data/, replacing the old file.
  - Prints per-file size delta and a summary.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_TEST = ROOT / "test_data"
SOURCE_TESTING = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents"
    r"\Working\Ronald Kurnik\19August2026\processedResults\testing"
)


def main() -> None:
    if not WORKSPACE_TEST.exists():
        raise SystemExit(f"Workspace test_data missing: {WORKSPACE_TEST}")
    if not SOURCE_TESTING.exists():
        raise SystemExit(f"Source testing folder missing: {SOURCE_TESTING}")

    current_files = sorted(WORKSPACE_TEST.glob("*_labeled_segment.parquet"))
    if not current_files:
        raise SystemExit(f"No parquet files in {WORKSPACE_TEST}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT / f"test_data.backup_testfix_{ts}"
    print(f"Backing up {len(current_files)} files from test_data/ → {backup_dir.name}/")
    backup_dir.mkdir(exist_ok=False)
    for f in current_files:
        shutil.copy2(f, backup_dir / f.name)
    print("Backup done.\n")

    missing: list[str] = []
    replaced: list[tuple[str, int, int]] = []
    for f in current_files:
        src = SOURCE_TESTING / f.name
        if not src.exists():
            missing.append(f.name)
            continue
        old_size = f.stat().st_size
        shutil.copy2(src, f)
        new_size = f.stat().st_size
        replaced.append((f.name, old_size, new_size))

    print(f"Replaced {len(replaced)} test files from {SOURCE_TESTING.name}/")
    print(f"Missing at source: {len(missing)}")
    if missing:
        for name in missing:
            print(f"  MISSING: {name}")

    same_size = sum(1 for _, o, n in replaced if o == n)
    print(f"\nFiles where size did NOT change (blanked==raw): {same_size}")

    print("\nTop 10 largest size changes:")
    replaced.sort(key=lambda t: abs(t[2] - t[1]), reverse=True)
    for name, old, new in replaced[:10]:
        delta = new - old
        pct = 100.0 * delta / old if old else 0.0
        print(f"  {name:>40s}  {old:>10,d} → {new:>10,d}  ({delta:+,d}, {pct:+.1f}%)")

    print(f"\nDone. Rollback: copy files from {backup_dir.name}/ back into test_data/")


if __name__ == "__main__":
    main()
