"""
setup_split.py — Copy the baseline 62/14 denoised parquets into train_data/ and test_data/.

Reads study IDs from the baseline split file, then copies the matching
*_labeled_segment_denoised.parquet files from denoised_data/parquets/.
"""

import shutil
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent

DENOISED_SOURCE = PROJECT_ROOT / "denoised_data" / "parquets"
SPLIT_FILE = PROJECT_ROOT / "src" / "data" / "baseline_split_2026-04-07.txt"

TRAIN_OUT = EXPERIMENT_DIR / "train_data"
TEST_OUT  = EXPERIMENT_DIR / "test_data"


def parse_split(split_path: Path):
    """Parse the baseline split file → (train_ids, test_ids)."""
    train_ids, test_ids = [], []
    section = None
    for line in split_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("=== TRAINING"):
            section = "train"
            continue
        if line.startswith("=== TEST"):
            section = "test"
            continue
        if section == "train":
            train_ids.append(line)
        elif section == "test":
            test_ids.append(line)
    return train_ids, test_ids


def copy_studies(study_ids, source_dir, dest_dir, label):
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing
    for old in dest_dir.glob("*.parquet"):
        old.unlink()

    copied, missing = 0, []
    for sid in study_ids:
        src = source_dir / f"{sid}_labeled_segment_denoised.parquet"
        if src.exists():
            shutil.copy2(src, dest_dir / src.name)
            copied += 1
        else:
            missing.append(sid)

    print(f"  {label}: copied {copied}/{len(study_ids)}")
    if missing:
        print(f"  ⚠️  Missing: {missing}")
    return missing


def main():
    if not DENOISED_SOURCE.is_dir():
        print(f"ERROR: denoised source not found: {DENOISED_SOURCE}")
        sys.exit(1)
    if not SPLIT_FILE.exists():
        print(f"ERROR: split file not found: {SPLIT_FILE}")
        sys.exit(1)

    train_ids, test_ids = parse_split(SPLIT_FILE)
    print(f"Baseline split: {len(train_ids)} train, {len(test_ids)} test")
    print(f"Source: {DENOISED_SOURCE}")

    m1 = copy_studies(train_ids, DENOISED_SOURCE, TRAIN_OUT, "Train")
    m2 = copy_studies(test_ids, DENOISED_SOURCE, TEST_OUT, "Test")

    if m1 or m2:
        print("\n⚠️  Some studies missing — check denoised source!")
        sys.exit(1)
    else:
        print("\n✅ All files copied successfully.")


if __name__ == "__main__":
    main()
