from __future__ import annotations

from pathlib import Path
import shutil


PROJECT_ROOT = Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3")
SOURCE_BASE = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\19August2026\processedResults"
)

SOURCE_TESTING = SOURCE_BASE / "testing"
SOURCE_TRAINING = SOURCE_BASE / "training"

DEST_TEST = PROJECT_ROOT / "test_data"
DEST_TRAIN = PROJECT_ROOT / "training_data"
NEXT_TESTSET_FILE = PROJECT_ROOT / "versions" / "NextTestSet.txt"


def read_test_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"NextTestSet file not found: {path}")

    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(set(ids))


def extract_study_id(parquet_path: Path) -> str:
    return parquet_path.stem.replace("_labeled_segment", "")


def clear_parquets(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("*.parquet"):
        f.unlink()


def copy_split() -> None:
    test_ids = read_test_ids(NEXT_TESTSET_FILE)
    test_id_set = set(test_ids)

    print(f"Loaded {len(test_ids)} test IDs from: {NEXT_TESTSET_FILE}")

    clear_parquets(DEST_TEST)
    clear_parquets(DEST_TRAIN)

    testing_files = list(SOURCE_TESTING.glob("*.parquet"))
    training_files = list(SOURCE_TRAINING.glob("*.parquet"))

    testing_map = {extract_study_id(f): f for f in testing_files}

    copied_test = 0
    missing_test_ids: list[str] = []
    for study_id in test_ids:
        src = testing_map.get(study_id)
        if src is None:
            missing_test_ids.append(study_id)
            continue
        shutil.copy2(src, DEST_TEST / src.name)
        copied_test += 1

    copied_train = 0
    for src in training_files:
        study_id = extract_study_id(src)
        if study_id in test_id_set:
            continue
        shutil.copy2(src, DEST_TRAIN / src.name)
        copied_train += 1

    final_test_files = list(DEST_TEST.glob("*.parquet"))
    final_train_files = list(DEST_TRAIN.glob("*.parquet"))

    final_test_ids = {extract_study_id(f) for f in final_test_files}
    final_train_ids = {extract_study_id(f) for f in final_train_files}
    overlap = sorted(final_test_ids.intersection(final_train_ids))

    print("\nSummary")
    print("=" * 60)
    print(f"Copied test files     : {copied_test}")
    print(f"Copied training files : {copied_train}")
    print(f"Final test_data count : {len(final_test_files)}")
    print(f"Final training count  : {len(final_train_files)}")

    if missing_test_ids:
        print("\nWARNING: Missing test IDs in source testing folder:")
        for study_id in missing_test_ids:
            print(f"  {study_id}")
    else:
        print("\nAll NextTestSet IDs were found in source testing folder.")

    if overlap:
        print("\nWARNING: Overlap between test_data and training_data:")
        for study_id in overlap:
            print(f"  {study_id}")
    else:
        print("\nNo overlap between test_data and training_data IDs.")


if __name__ == "__main__":
    copy_split()
