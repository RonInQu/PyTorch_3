from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.move_postmay_split import (
    clear_parquet_files,
    transfer_remaining_files,
    transfer_selected_files,
)

DEFAULT_PROCESSED_ROOT = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\19August2026\processedResults"
)
DEFAULT_INVENTORY_CSV = PROJECT_ROOT / "analysis_data_drift" / "data_inventory_183.csv"
RUNS_ROOT = PROJECT_ROOT / "analysis_data_drift" / "kfold_runs"


@dataclass
class FoldSplit:
    fold_index: int
    train_ids: list[str]
    test_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a study-level V8 k-fold benchmark with isolated split and artifact folders."
    )
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--inventory-csv", type=Path, default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--limit-folds", type=int, default=None)
    return parser.parse_args()


def ensure_source_dirs(processed_root: Path) -> tuple[Path, Path]:
    training_src = processed_root / "training"
    testing_src = processed_root / "testing"
    if not training_src.exists() or not testing_src.exists():
        raise FileNotFoundError(f"Expected training/ and testing/ under {processed_root}")
    return training_src, testing_src


def build_inventory_from_source(testing_src: Path) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for parquet_path in sorted(testing_src.glob("*.parquet")):
        match = re.match(r"^([A-Z0-9]{8})(?:_|\.)", parquet_path.name)
        if not match:
            continue
        study_id = match.group(1)
        df = pd.read_parquet(parquet_path, columns=["label"])
        labels = df["label"]
        valid = labels.isin([0, 1, 2])
        labels = labels[valid]
        n_samples = int(len(labels))
        if n_samples == 0:
            frac_blood = frac_clot = frac_wall = 0.0
        else:
            counts = labels.value_counts(normalize=True).to_dict()
            frac_blood = float(counts.get(0, 0.0))
            frac_clot = float(counts.get(1, 0.0))
            frac_wall = float(counts.get(2, 0.0))
        rows.append(
            {
                "study_id": study_id,
                "n_samples": n_samples,
                "frac_blood": frac_blood,
                "frac_clot": frac_clot,
                "frac_wall": frac_wall,
            }
        )
    if not rows:
        raise RuntimeError(f"No studies found in {testing_src}")
    return pd.DataFrame(rows).sort_values("study_id").reset_index(drop=True)


def load_inventory(inventory_csv: Path, testing_src: Path) -> pd.DataFrame:
    if inventory_csv.exists():
        df = pd.read_csv(inventory_csv)
        if "study_id" not in df.columns and "stem" in df.columns:
            df["study_id"] = df["stem"].astype(str).str.extract(r"^([A-Z0-9]{8})", expand=False)
        required = ["study_id", "n_samples", "frac_blood", "frac_clot", "frac_wall"]
        if all(col in df.columns for col in required):
            return (
                df[required]
                .dropna(subset=["study_id"])
                .drop_duplicates("study_id")
                .sort_values("study_id")
                .reset_index(drop=True)
            )
    return build_inventory_from_source(testing_src)


def _bucket(value: float, edges: list[float]) -> int:
    for idx, edge in enumerate(edges):
        if value < edge:
            return idx
    return len(edges)


def choose_split_labels(inventory: pd.DataFrame, n_splits: int) -> tuple[str, list[str] | None]:
    candidates = [
        (
            "stratified_clot_wall_3x3",
            inventory.apply(
                lambda row: f"c{_bucket(float(row['frac_clot']), [0.05, 0.15])}_w{_bucket(float(row['frac_wall']), [0.03, 0.10])}",
                axis=1,
            ).tolist(),
        ),
        (
            "stratified_clot_only_3bin",
            inventory["frac_clot"].apply(lambda value: f"c{_bucket(float(value), [0.05, 0.15])}").tolist(),
        ),
        (
            "stratified_wall_only_3bin",
            inventory["frac_wall"].apply(lambda value: f"w{_bucket(float(value), [0.03, 0.10])}").tolist(),
        ),
        (
            "stratified_any_tissue",
            inventory.apply(
                lambda row: f"cw{int(float(row['frac_clot']) >= 0.05)}{int(float(row['frac_wall']) >= 0.05)}",
                axis=1,
            ).tolist(),
        ),
    ]

    for name, labels in candidates:
        counts = pd.Series(labels).value_counts()
        if not counts.empty and int(counts.min()) >= n_splits:
            return name, labels

    return "plain_kfold_shuffle", None


def build_folds(inventory: pd.DataFrame, n_splits: int, seed: int) -> tuple[str, list[FoldSplit]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    inventory = inventory.sort_values("study_id").reset_index(drop=True)
    split_strategy, labels = choose_split_labels(inventory, n_splits)
    ids = inventory["study_id"].to_numpy()

    if labels is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(ids, labels)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(ids)

    folds: list[FoldSplit] = []
    for fold_index, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_ids = sorted(ids[train_idx].tolist())
        test_ids = sorted(ids[test_idx].tolist())
        folds.append(FoldSplit(fold_index=fold_index, train_ids=train_ids, test_ids=test_ids))

    return split_strategy, folds


def write_fold_manifest(run_dir: Path, folds: list[FoldSplit]) -> Path:
    rows: list[dict[str, int | str]] = []
    for split in folds:
        for study_id in split.train_ids:
            rows.append({"fold": split.fold_index, "study_id": study_id, "split": "train"})
        for study_id in split.test_ids:
            rows.append({"fold": split.fold_index, "study_id": study_id, "split": "test"})
    manifest_path = run_dir / "fold_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def prepare_fold_split(split: FoldSplit, training_src: Path, testing_src: Path, fold_dir: Path) -> tuple[Path, Path]:
    train_dir = fold_dir / "training_data"
    test_dir = fold_dir / "test_data"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    clear_parquet_files(train_dir)
    clear_parquet_files(test_dir)

    test_id_set = set(split.test_ids)
    copied_test, missing_test = transfer_selected_files(testing_src, test_id_set, test_dir, move_files=False)
    copied_train, skipped_train = transfer_remaining_files(training_src, test_id_set, train_dir, move_files=False)

    if missing_test:
        raise FileNotFoundError(f"Missing test studies for fold {split.fold_index}: {', '.join(missing_test)}")
    if copied_test != len(split.test_ids):
        raise RuntimeError(f"Fold {split.fold_index}: copied {copied_test} test files but expected {len(split.test_ids)}")
    if copied_train != len(split.train_ids):
        raise RuntimeError(
            f"Fold {split.fold_index}: copied {copied_train} train files but expected {len(split.train_ids)}; skipped {len(skipped_train)}"
        )

    return train_dir, test_dir


def run_step(command: list[str], env: dict[str, str], cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\nSee log: {log_path}")


SUMMARY_PATTERNS = {
    "da_accuracy": r"DA  Accuracy: ([0-9.]+)",
    "da_f1_macro": r"DA  Accuracy: [0-9.]+\s+F1-macro: ([0-9.]+)",
    "ml_accuracy": r"ML  Accuracy: ([0-9.]+)",
    "ml_f1_macro": r"ML  Accuracy: [0-9.]+\s+F1-macro: ([0-9.]+)",
    "da_precision": r"DA  Precision: ([0-9.]+)",
    "da_recall": r"DA  Precision: [0-9.]+\s+Recall: ([0-9.]+)",
    "ml_precision": r"ML  Precision: ([0-9.]+)",
    "ml_recall": r"ML  Precision: [0-9.]+\s+Recall: ([0-9.]+)",
    "total_overrides": r"Total overrides across all studies: ([0-9]+)",
    "correct_overrides": r"Correct overrides \(ML right, DA wrong\): ([0-9]+)",
    "harmful_overrides": r"Harmful overrides \(DA right, ML wrong\): ([0-9]+)",
    "neither_overrides": r"Neither correct \(both wrong differently\): ([0-9]+)",
    "override_precision": r"Override Precision:\s+([0-9.]+)",
    "override_recall": r"Override Recall:\s+([0-9.]+)",
    "net_benefit": r"Net benefit:\s+([+-]?[0-9]+) samples",
}


def parse_summary(summary_path: Path) -> dict[str, float | int]:
    text = summary_path.read_text(encoding="utf-8")
    metrics: dict[str, float | int] = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not parse {key} from {summary_path}")
        value = match.group(1)
        metrics[key] = int(value) if re.fullmatch(r"[+-]?[0-9]+", value) else float(value)
    return metrics


def aggregate_summary(results_df: pd.DataFrame, split_strategy: str) -> list[str]:
    lines = [
        "=" * 78,
        "V8 K-FOLD SUMMARY",
        "=" * 78,
        f"Split strategy: {split_strategy}",
        f"Folds completed: {len(results_df)}",
        "",
    ]

    metric_cols = [
        "da_accuracy",
        "ml_accuracy",
        "da_f1_macro",
        "ml_f1_macro",
        "da_precision",
        "ml_precision",
        "da_recall",
        "ml_recall",
        "override_precision",
        "override_recall",
        "net_benefit",
    ]
    for col in metric_cols:
        series = results_df[col]
        lines.append(
            f"{col:>20}: mean={series.mean():.4f}  std={series.std(ddof=0):.4f}"
            f"  min={series.min():.4f}  max={series.max():.4f}"
        )

    lines.append("")
    lines.append(f"Total overrides summed   : {int(results_df['total_overrides'].sum())}")
    lines.append(f"Correct overrides summed : {int(results_df['correct_overrides'].sum())}")
    lines.append(f"Harmful overrides summed : {int(results_df['harmful_overrides'].sum())}")
    lines.append(f"Neither overrides summed : {int(results_df['neither_overrides'].sum())}")
    return lines


def main() -> None:
    args = parse_args()
    training_src, testing_src = ensure_source_dirs(args.processed_root)
    inventory = load_inventory(args.inventory_csv, testing_src)
    split_strategy, folds = build_folds(inventory, args.n_splits, args.seed)
    if args.limit_folds is not None:
        folds = folds[: args.limit_folds]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    run_dir = RUNS_ROOT / f"{timestamp}_V8_kfold{tag_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest_path = write_fold_manifest(run_dir, folds)
    print(f"Run directory : {run_dir}")
    print(f"Split strategy: {split_strategy}")
    print(f"Manifest      : {manifest_path}")
    print(f"Total studies : {inventory['study_id'].nunique()}")
    print(f"Folds planned : {len(folds)}")

    if args.plan_only:
        print("Plan-only mode: no scale/train/test commands executed.")
        return

    results: list[dict[str, float | int]] = []

    for split in folds:
        fold_dir = run_dir / f"fold_{split.fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Fold {split.fold_index}/{len(folds)} ===")
        print(f"Train studies: {len(split.train_ids)}")
        print(f"Test studies : {len(split.test_ids)}")

        train_dir, test_dir = prepare_fold_split(split, training_src, testing_src, fold_dir)
        artifacts_dir = fold_dir / "artifacts"
        results_dir = fold_dir / "results"
        cache_dir = fold_dir / "cache"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "PYTORCH3_TRAINING_DATA_DIR": str(train_dir),
                "PYTORCH3_TEST_DATA_DIR": str(test_dir),
                "PYTORCH3_SCALER_PATH": str(artifacts_dir / "clot_feature_scaler_5s_seq8_clot_wall_focused.pkl"),
                "PYTORCH3_MODEL_OUTPUT_DIR": str(artifacts_dir),
                "PYTORCH3_MODEL_PATH": str(artifacts_dir / "clot_gru_trained.pt"),
                "PYTORCH3_OUTPUT_FOLDER": str(results_dir),
                "PYTORCH3_CACHE_DIR": str(cache_dir),
                "PYTORCH3_SKIP_SAVE_VERSION": "1",
            }
        )

        run_step([args.python, "src/data/fit_scaler_V8.py"], env, PROJECT_ROOT, fold_dir / "01_fit_scaler.log")
        run_step([args.python, "src/training/train_gru_V8.py", "--force-extract"], env, PROJECT_ROOT, fold_dir / "02_train.log")
        run_step([args.python, "src/models/gru_torch_V8.py"], env, PROJECT_ROOT, fold_dir / "03_infer.log")

        summary_path = results_dir / "global_summary.txt"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file after fold {split.fold_index}: {summary_path}")

        fold_metrics = parse_summary(summary_path)
        fold_metrics["fold"] = split.fold_index
        fold_metrics["n_train_files"] = len(split.train_ids)
        fold_metrics["n_test_files"] = len(split.test_ids)
        results.append(fold_metrics)

    results_df = pd.DataFrame(results).sort_values("fold")
    results_path = run_dir / "kfold_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_lines = aggregate_summary(results_df, split_strategy)
    summary_path = run_dir / "kfold_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(summary_lines))
    print(f"\nPer-fold results: {results_path}")
    print(f"Aggregate summary: {summary_path}")


if __name__ == "__main__":
    main()