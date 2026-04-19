# clean_test_folder.py
"""
Ensure only the 14 baseline test files exist in the testing folder.
Removes any files that are NOT in the approved test list.
"""

from pathlib import Path

# ================= CONFIG =================
BASE_DIR = Path(r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\merged_expts_with_events_April6\event_files\processedResults")
TEST_DIR = BASE_DIR / "testing"

# The 14 approved test studies (baseline split 2026-04-07)
KEEP_STUDIES = [
    "00F628C9",
    "33CFB812",
    "43140EA7",
    "4E3747A0",
    "530618CC",
    "819421BC",
    "A225B105",
    "AFF18ECE",
    "CENT0008",
    "HACK0140",
    "HUNT0120",
    "STCLD001",
    "SUMM0127",
    "UHMAX001",
]

KEEP_FILES = {f"{s}_labeled_segment.parquet" for s in KEEP_STUDIES}

# ================= MAIN =================
if not TEST_DIR.exists():
    print(f"Testing folder not found: {TEST_DIR}")
    exit(1)

existing = list(TEST_DIR.glob("*.parquet"))
print(f"Found {len(existing)} files in testing folder.")
print(f"Expected: {len(KEEP_FILES)} files.\n")

removed = 0
for f in existing:
    if f.name not in KEEP_FILES:
        f.unlink()
        print(f"  Removed: {f.name}")
        removed += 1

# Check for missing files
missing = KEEP_FILES - {f.name for f in TEST_DIR.glob("*.parquet")}
if missing:
    print(f"\nWARNING: {len(missing)} expected files are missing:")
    for m in sorted(missing):
        print(f"  {m}")

print(f"\nDone. Removed {removed} files. {len(KEEP_FILES) - len(missing)} of {len(KEEP_FILES)} expected files present.")
