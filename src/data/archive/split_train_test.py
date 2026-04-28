# split_train_test.py
"""
Splits LabelingMax output into training (80%) and testing (20%) by study.
No overlap — each study goes to exactly one split.

Reads from: processedResults/training/ and processedResults/testing/
Writes to:  training_data/ and test_data/

Usage:
    python split_train_test.py                      # from the folder containing processedResults/
    python split_train_test.py --seed 42            # reproducible split
    python split_train_test.py --test-fraction 0.3  # 70/30 split
    python split_train_test.py --list                # show current split without copying
"""

import argparse
import shutil
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split labeled data into train/test sets")
    parser.add_argument("--input-dir", type=str, default=".",
                        help="Root folder containing processedResults/ (default: current dir)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Root folder for training_data/ and test_data/ (default: PyTorch_3 project root)")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of studies for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--list", action="store_true",
                        help="Show the split without copying files")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    training_src = input_root / "processedResults" / "training"
    testing_src = input_root / "processedResults" / "testing"

    if not training_src.exists():
        print(f"Error: {training_src} not found. Run LabelingMax first.")
        return
    if not testing_src.exists():
        print(f"Error: {testing_src} not found. Run LabelingMax first.")
        return

    # Discover studies from training source (both folders should have the same studies)
    train_files = sorted(training_src.glob("*_labeled_segment.parquet"))
    test_files = sorted(testing_src.glob("*_labeled_segment.parquet"))

    train_studies = {f.stem.replace("_labeled_segment", ""): f for f in train_files}
    test_studies = {f.stem.replace("_labeled_segment", ""): f for f in test_files}

    # Only use studies present in both folders
    common = sorted(set(train_studies.keys()) & set(test_studies.keys()))
    train_only = set(train_studies.keys()) - set(test_studies.keys())
    test_only = set(test_studies.keys()) - set(train_studies.keys())

    if train_only:
        print(f"Warning: {len(train_only)} studies only in training/: {train_only}")
    if test_only:
        print(f"Warning: {len(test_only)} studies only in testing/: {test_only}")

    if not common:
        print("Error: No common studies found in both folders.")
        return

    # Shuffle and split
    random.seed(args.seed)
    shuffled = list(common)
    random.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * args.test_fraction))
    n_train = len(shuffled) - n_test

    test_set = sorted(shuffled[:n_test])
    train_set = sorted(shuffled[n_test:])

    print(f"\nTotal studies: {len(common)}")
    print(f"Train: {n_train} ({100*n_train/len(common):.0f}%)  |  Test: {n_test} ({100*n_test/len(common):.0f}%)")
    print(f"Seed: {args.seed}")

    print(f"\n{'─'*50}")
    print(f"TRAINING ({n_train} studies):")
    for s in train_set:
        print(f"  {s}")

    print(f"\nTESTING ({n_test} studies):")
    for s in test_set:
        print(f"  {s}")
    print(f"{'─'*50}")

    if args.list:
        return

    # Output directories
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        # Default: PyTorch_3 project root
        output_root = Path(__file__).resolve().parents[2]

    train_dst = output_root / "training_data"
    test_dst = output_root / "test_data"

    # Clear existing files
    for folder in [train_dst, test_dst]:
        folder.mkdir(parents=True, exist_ok=True)
        existing = list(folder.glob("*_labeled_segment.parquet"))
        if existing:
            print(f"\nClearing {len(existing)} existing files in {folder.name}/")
            for f in existing:
                f.unlink()

    # Copy training files (from processedResults/training/ → training_data/)
    print(f"\nCopying training files → {train_dst}")
    for study in train_set:
        src = train_studies[study]
        dst = train_dst / src.name
        shutil.copy2(src, dst)
        print(f"  {src.name}")

    # Copy test files (from processedResults/testing/ → test_data/)
    print(f"\nCopying test files → {test_dst}")
    for study in test_set:
        src = test_studies[study]
        dst = test_dst / src.name
        shutil.copy2(src, dst)
        print(f"  {src.name}")

    print(f"\nDone. {n_train} training + {n_test} testing files copied.")


if __name__ == "__main__":
    main()
