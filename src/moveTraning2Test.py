# split_test_files.py
"""
Select specific labeled_segment files and move them from training_data to test_data.
"""

from pathlib import Path
import shutil

# ================= CONFIG =================
# BASE_DIR = Path(r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\merged_expts_with_events_April6\event_files\processedResults")
BASE_DIR = Path(r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\merged_expts_with_events_April10\merged_expts_with_events\parquet\processedResults")

TRAINING_DIR = BASE_DIR / "training"
TEST_DIR = BASE_DIR / "testing"

# List of filenames you want to move to test (add or remove as needed)
TEST_FILENAMES = [
    "00F628C9_labeled_segment.parquet",
    "33CFB812_labeled_segment.parquet",
    "43140EA7_labeled_segment.parquet",
    "4E3747A0_labeled_segment.parquet",
    "530618CC_labeled_segment.parquet",
    "819421BC_labeled_segment.parquet",
    "A225B105_labeled_segment.parquet",
    "AFF18ECE_labeled_segment.parquet",
    "CENT0008_labeled_segment.parquet",
    "HACK0140_labeled_segment.parquet",
    "HUNT0120_labeled_segment.parquet",
    "STCLD001_labeled_segment.parquet",
    "SUMM0127_labeled_segment.parquet",
    "UHMAX001_labeled_segment.parquet",
]

# Create test directory if it doesn't exist
TEST_DIR.mkdir(parents=True, exist_ok=True)

print(f"Moving {len(TEST_FILENAMES)} files from training_data to test_data...\n")

moved_count = 0
for filename in TEST_FILENAMES:
    src_path = TRAINING_DIR / filename
    dst_path = TEST_DIR / filename

    if src_path.exists():
        try:
            shutil.move(str(src_path), str(dst_path))
            print(f"✓ Moved: {filename}")
            moved_count += 1
        except Exception as e:
            print(f"✗ Error moving {filename}: {e}")
    else:
        print(f"⚠ File not found in training_data: {filename}")

print("\n" + "="*60)
print(f"Done! {moved_count}/{len(TEST_FILENAMES)} files moved to test_data folder.")
print(f"Test folder path: {TEST_DIR}")