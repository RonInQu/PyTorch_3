from __future__ import annotations

from pathlib import Path
import argparse
import shutil


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Set file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transfer_train_set(
    source_training_dir: Path,
    destination_training_dir: Path,
    set_file: Path,
    move_files: bool = False,
    clear_destination: bool = True,
) -> tuple[int, list[str], int]:
    if not source_training_dir.exists():
        raise FileNotFoundError(f"Source training folder not found: {source_training_dir}")

    destination_training_dir.mkdir(parents=True, exist_ok=True)

    if clear_destination:
        for p in destination_training_dir.glob("*.parquet"):
            p.unlink()

    ids = read_ids(set_file)
    copied = 0
    missing: list[str] = []

    for sid in ids:
        src = source_training_dir / f"{sid}_labeled_segment.parquet"
        if not src.exists():
            missing.append(sid)
            continue

        dst = destination_training_dir / src.name
        if move_files:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)
        copied += 1

    final_count = sum(1 for _ in destination_training_dir.glob("*.parquet"))
    return copied, missing, final_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a candidate training set into training_data.")
    parser.add_argument(
        "--source-training-dir",
        type=Path,
        default=Path(r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\19August2026\processedResults\training"),
        help="Source processedResults/training directory",
    )
    parser.add_argument(
        "--destination-training-dir",
        type=Path,
        default=Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\training_data"),
        help="Destination training_data directory",
    )
    parser.add_argument(
        "--set-file",
        type=Path,
        default=Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\versions\2026-09-08_candidate_train_sets\candidate_train85_from_train162.txt"),
        help="Path to candidate training set file",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying files",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear destination *.parquet files before transfer",
    )

    args = parser.parse_args()

    copied, missing, final_count = transfer_train_set(
        source_training_dir=args.source_training_dir,
        destination_training_dir=args.destination_training_dir,
        set_file=args.set_file,
        move_files=args.move,
        clear_destination=not args.no_clear,
    )

    ids_count = len(read_ids(args.set_file))
    print(f"Requested IDs: {ids_count}")
    print(f"Transferred : {copied}")
    print(f"Final count : {final_count}")

    if missing:
        print("Missing IDs:")
        for sid in missing:
            print(f"  {sid}")
    else:
        print("Missing IDs: none")


if __name__ == "__main__":
    main()
