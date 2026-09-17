from __future__ import annotations

from pathlib import Path

from scripts.move_postmay_split import transfer_remaining_files, transfer_selected_files, clear_parquet_files, read_ids


PROCESSED_ROOT = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\19August2026\processedResults"
)
TEST_ID_FILE = Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Desktop\TestSet.txt")
REPO_ROOT = Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3")

# Safe default for Spyder use: copy files and preserve source folders.
MOVE_FILES = False
CLEAR_TARGET = True


def apply_split(
    processed_root: Path = PROCESSED_ROOT,
    test_id_file: Path = TEST_ID_FILE,
    repo_root: Path = REPO_ROOT,
    move_files: bool = MOVE_FILES,
    clear_target: bool = CLEAR_TARGET,
) -> None:
    testing_src = processed_root / "testing"
    training_src = processed_root / "training"
    test_dest = repo_root / "test_data"
    train_dest = repo_root / "training_data"

    if not testing_src.exists() or not training_src.exists():
        raise FileNotFoundError(f"Expected source folders not found under: {processed_root}")
    if not test_id_file.exists():
        raise FileNotFoundError(f"Test ID file not found: {test_id_file}")

    test_dest.mkdir(parents=True, exist_ok=True)
    train_dest.mkdir(parents=True, exist_ok=True)

    test_ids = read_ids(test_id_file)
    test_id_set = set(test_ids)
    if len(test_ids) != len(test_id_set):
        raise ValueError("Duplicate IDs found in test ID file.")

    if clear_target:
        removed_test = clear_parquet_files(test_dest)
        removed_train = clear_parquet_files(train_dest)
        print(f"Cleared destination files: test_data={removed_test}, training_data={removed_train}")

    transferred_test, missing_test = transfer_selected_files(
        testing_src,
        test_id_set,
        test_dest,
        move_files=move_files,
    )
    transferred_train, skipped_from_training = transfer_remaining_files(
        training_src,
        test_id_set,
        train_dest,
        move_files=move_files,
    )

    action_word = "Moved" if move_files else "Copied"

    print(f"Requested test IDs: {len(test_ids)}")
    print(f"{action_word} to test_data: {transferred_test}")
    print(f"{action_word} to training_data: {transferred_train}")
    print(f"Training files excluded because in test list: {len(skipped_from_training)}")

    if missing_test:
        print("Missing expected test files in source testing folder:")
        for study_id in missing_test:
            print(f"  - {study_id}")

    final_test = len(list(test_dest.glob("*.parquet")))
    final_train = len(list(train_dest.glob("*.parquet")))
    print(f"Final destination counts: test_data={final_test}, training_data={final_train}")


if __name__ == "__main__":
    apply_split()