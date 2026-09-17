#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def read_ids(file_path: Path) -> list[str]:
    ids: list[str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        study_id = line.strip()
        if not study_id or study_id.startswith("#"):
            continue
        ids.append(study_id)
    return ids


def clear_parquet_files(folder: Path) -> int:
    removed = 0
    for parquet in folder.glob("*.parquet"):
        parquet.unlink()
        removed += 1
    return removed


def extract_study_id(file_name: str) -> str | None:
    match = re.match(r"^([A-Z0-9]{8})(?:_|\.)", file_name)
    return match.group(1) if match else None


def find_file_for_study_id(source_dir: Path, study_id: str) -> Path | None:
    candidates = sorted(source_dir.glob(f"{study_id}*.parquet"))
    if not candidates:
        return None
    return candidates[0]


def transfer_selected_files(
    source_dir: Path,
    selected_ids: set[str],
    dest_dir: Path,
    move_files: bool,
) -> tuple[int, list[str]]:
    transferred = 0
    missing: list[str] = []
    for study_id in sorted(selected_ids):
        src = find_file_for_study_id(source_dir, study_id)
        if src is None:
            missing.append(study_id)
            continue
        dst = dest_dir / src.name
        if move_files:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)
        transferred += 1
    return transferred, missing


def transfer_remaining_files(
    source_dir: Path,
    excluded_ids: set[str],
    dest_dir: Path,
    move_files: bool,
) -> tuple[int, list[str]]:
    transferred = 0
    skipped: list[str] = []
    for src in sorted(source_dir.glob("*.parquet")):
        study_id = extract_study_id(src.name)
        if study_id is not None and study_id in excluded_ids:
            skipped.append(study_id)
            continue
        dst = dest_dir / src.name
        if move_files:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)
        transferred += 1
    return transferred, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy test IDs from processedResults/testing and remaining training files from processedResults/training."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\PostMay_strict\processedResults"),
        help="Path to processedResults folder containing testing/training and UseAsTest.",
    )
    parser.add_argument(
        "--test-id-file",
        type=Path,
        default=Path(r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\PostMay_strict\processedResults\UseAsTest\UseAsTestData.txt"),
        help="Text file with one test study ID per line.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3"),
        help="Path to repo root containing test_data and training_data.",
    )
    parser.add_argument(
        "--no-clear-target",
        action="store_true",
        help="Do not clear existing parquet files in destination folders before transfer.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them. Default behavior is copy.",
    )
    args = parser.parse_args()

    testing_src = args.processed_root / "testing"
    training_src = args.processed_root / "training"
    test_dest = args.repo_root / "test_data"
    train_dest = args.repo_root / "training_data"

    if not testing_src.exists() or not training_src.exists():
        raise FileNotFoundError(f"Expected source folders not found under: {args.processed_root}")
    if not args.test_id_file.exists():
        raise FileNotFoundError(f"Test ID file not found: {args.test_id_file}")

    test_dest.mkdir(parents=True, exist_ok=True)
    train_dest.mkdir(parents=True, exist_ok=True)

    test_ids = read_ids(args.test_id_file)
    test_id_set = set(test_ids)

    if len(test_ids) != len(test_id_set):
        raise ValueError("Duplicate IDs found in test ID file.")

    if not args.no_clear_target:
        removed_test = clear_parquet_files(test_dest)
        removed_train = clear_parquet_files(train_dest)
        print(f"Cleared destination files: test_data={removed_test}, training_data={removed_train}")

    transferred_test, missing_test = transfer_selected_files(
        testing_src, test_id_set, test_dest, move_files=args.move
    )
    transferred_train, skipped_from_training = transfer_remaining_files(
        training_src, test_id_set, train_dest, move_files=args.move
    )

    action_word = "Moved" if args.move else "Copied"

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
    main()
