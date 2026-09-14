"""
Train and evaluate a DA-override model for clot/wall using grouped file splits.

Goal:
- Learn a better override policy than fixed hand thresholds.
- Predict GT clot/wall from impedance features + DA label context.
- Apply policy: keep DA unless model disagrees with enough confidence.

Key constraints:
- Clot/wall only (ignore blood labels in training/eval).
- File-grouped splits to avoid leakage across the same study.
- Metrics focused on override utility:
  correct_overrides, harmful_overrides, override_precision, net_benefit.

Outputs:
- analysis_data_drift/override_model_v1/threshold_sweep_summary.csv
- analysis_data_drift/override_model_v1/best_policy_summary.txt
- analysis_data_drift/override_model_v1/per_fold_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_torch_V9 import (  # noqa: E402
    ClotFeatureExtractor,
    REPORT_INTERVAL_MS,
    active_dim,
    active_idx,
)

DEFAULT_SOURCE_DIR = PROJECT_ROOT / "training_data"
OUT_DIR = PROJECT_ROOT / "analysis_data_drift" / "override_model_v1"


def iter_files(source_dir: Path, max_files: int | None = None) -> list[Path]:
    files = sorted(source_dir.glob("*_labeled_segment.parquet"))
    if max_files is not None:
        files = files[:max_files]
    return files


def build_emit_dataset(files: list[Path]) -> pd.DataFrame:
    rows = []
    for i, fp in enumerate(files, start=1):
        study = fp.stem.replace("_labeled_segment", "")
        print(f"[{i:3d}/{len(files)}] {study}")

        df = pd.read_parquet(fp, columns=["timeInMS", "magRLoadAdjusted", "label", "da_label"])

        t_ms = df["timeInMS"].to_numpy(dtype=np.int64)
        r = df["magRLoadAdjusted"].to_numpy(dtype=np.float32)
        gt = df["label"].to_numpy(dtype=np.int16)
        da = df["da_label"].to_numpy(dtype=np.int16)

        extractor = ClotFeatureExtractor(active_features=active_idx)
        last_report = -REPORT_INTERVAL_MS

        for j, (tm, rr) in enumerate(zip(t_ms, r)):
            extractor.update(float(rr))

            if tm - last_report < REPORT_INTERVAL_MS:
                continue
            last_report = tm

            gt_now = int(gt[j])
            da_now = int(da[j])

            # Clot/wall only
            if gt_now not in (1, 2) or da_now not in (1, 2):
                continue

            feats = extractor.compute_features()
            if feats is None or len(feats) != active_dim:
                continue
            if not np.all(np.isfinite(feats)):
                continue

            # 2-class mapping: clot=0, wall=1
            y = gt_now - 1
            da2 = da_now - 1

            row = {
                "file": study,
                "time_s": tm / 1000.0,
                "y": y,
                "da2": da2,
            }
            for k, v in enumerate(feats):
                row[f"f{k:02d}"] = float(v)

            rows.append(row)

    if not rows:
        raise RuntimeError("No clot/wall emits extracted.")

    dfe = pd.DataFrame(rows)
    print(f"\nEmit dataset: {len(dfe):,} rows, {dfe['file'].nunique()} files")
    return dfe


def evaluate_policy(y_true: np.ndarray, da2: np.ndarray, ml_pred: np.ndarray, ml_conf: np.ndarray, threshold: float) -> dict:
    do_override = (ml_pred != da2) & (ml_conf >= threshold)
    final = np.where(do_override, ml_pred, da2)

    correct_overrides = int(np.sum(do_override & (final == y_true)))
    harmful_overrides = int(np.sum(do_override & (da2 == y_true) & (final != y_true)))
    n_overrides = int(np.sum(do_override))

    override_precision = (correct_overrides / n_overrides) if n_overrides > 0 else np.nan
    net_benefit = correct_overrides - harmful_overrides

    return {
        "threshold": threshold,
        "n_overrides": n_overrides,
        "correct_overrides": correct_overrides,
        "harmful_overrides": harmful_overrides,
        "override_precision": override_precision,
        "net_benefit": net_benefit,
        "acc": float(accuracy_score(y_true, final)),
        "f1": float(f1_score(y_true, final, average="macro")),
    }


def run_grouped_cv(dfe: pd.DataFrame, n_splits: int, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    feat_cols = [c for c in dfe.columns if c.startswith("f") and c[1:].isdigit()]
    X = dfe[feat_cols].to_numpy(dtype=np.float32)
    y = dfe["y"].to_numpy(dtype=np.int16)
    da2 = dfe["da2"].to_numpy(dtype=np.int16)
    groups = dfe["file"].to_numpy()

    # Add DA as one-hot context features.
    da_onehot = np.zeros((len(da2), 2), dtype=np.float32)
    da_onehot[np.arange(len(da2)), da2] = 1.0
    X_all = np.hstack([X, da_onehot])

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    thresholds = np.round(np.arange(0.50, 0.96, 0.02), 2)

    fold_rows = []
    thresh_rows = []

    for fold_idx, (tr, te) in enumerate(splitter.split(X_all, y, groups=groups), start=1):
        Xtr, Xte = X_all[tr], X_all[te]
        ytr, yte = y[tr], y[te]
        da_te = da2[te]

        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=6,
            max_iter=320,
            min_samples_leaf=30,
            random_state=random_state + fold_idx,
        )
        model.fit(Xtr, ytr)

        proba = model.predict_proba(Xte)
        ml_pred = np.argmax(proba, axis=1)
        ml_conf = np.max(proba, axis=1)

        # Baselines
        da_acc = float(accuracy_score(yte, da_te))
        da_f1 = float(f1_score(yte, da_te, average="macro"))
        ml_acc = float(accuracy_score(yte, ml_pred))
        ml_f1 = float(f1_score(yte, ml_pred, average="macro"))

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_test": len(te),
                "n_test_files": int(pd.Series(groups[te]).nunique()),
                "da_acc": da_acc,
                "da_f1": da_f1,
                "ml_acc": ml_acc,
                "ml_f1": ml_f1,
            }
        )

        for t in thresholds:
            m = evaluate_policy(yte, da_te, ml_pred, ml_conf, float(t))
            m["fold"] = fold_idx
            thresh_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    thr_df = pd.DataFrame(thresh_rows)

    # Aggregate threshold performance across folds
    agg = (
        thr_df.groupby("threshold", as_index=False)
        .agg(
            n_overrides_mean=("n_overrides", "mean"),
            correct_overrides_mean=("correct_overrides", "mean"),
            harmful_overrides_mean=("harmful_overrides", "mean"),
            override_precision_mean=("override_precision", "mean"),
            net_benefit_mean=("net_benefit", "mean"),
            acc_mean=("acc", "mean"),
            f1_mean=("f1", "mean"),
            acc_std=("acc", "std"),
            f1_std=("f1", "std"),
        )
        .sort_values("threshold")
    )

    # Choose best threshold by net benefit, break ties by F1 then precision.
    best = agg.sort_values(["net_benefit_mean", "f1_mean", "override_precision_mean"], ascending=False).iloc[0]

    summary = {
        "best_threshold": float(best["threshold"]),
        "best_net_benefit_mean": float(best["net_benefit_mean"]),
        "best_override_precision_mean": float(best["override_precision_mean"]),
        "best_acc_mean": float(best["acc_mean"]),
        "best_f1_mean": float(best["f1_mean"]),
        "da_acc_mean": float(fold_df["da_acc"].mean()),
        "da_f1_mean": float(fold_df["da_f1"].mean()),
        "ml_acc_mean": float(fold_df["ml_acc"].mean()),
        "ml_f1_mean": float(fold_df["ml_f1"].mean()),
        "n_total_emits": int(len(dfe)),
        "n_files": int(dfe["file"].nunique()),
    }

    return fold_df, agg, summary, feat_cols


def train_final_model(dfe: pd.DataFrame, feat_cols: list[str], seed: int):
    """Train one final model on all available rows for deployment.

    This is for real-time inference runtime packaging only; performance claims
    should still come from grouped CV above.
    """
    X = dfe[feat_cols].to_numpy(dtype=np.float32)
    y = dfe["y"].to_numpy(dtype=np.int16)
    da2 = dfe["da2"].to_numpy(dtype=np.int16)

    da_onehot = np.zeros((len(da2), 2), dtype=np.float32)
    da_onehot[np.arange(len(da2)), da2] = 1.0
    X_all = np.hstack([X, da_onehot])

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=6,
        max_iter=320,
        min_samples_leaf=30,
        random_state=seed,
    )
    model.fit(X_all, y)
    return model


def save_runtime_bundle(model, feat_cols: list[str], summary: dict, out_path: Path) -> None:
    bundle = {
        "model_type": "HistGradientBoostingClassifier",
        "version": "override_model_v1",
        "feature_cols": feat_cols,
        "da_encoding": {"clot": 0, "wall": 1},
        "label_encoding": {"clot": 0, "wall": 1},
        "best_threshold": float(summary["best_threshold"]),
        "notes": "Use at each 200ms emit. Causal-only: current feature window + current DA label.",
        "model": model,
    }
    joblib.dump(bundle, out_path)


def write_summary(summary: dict, out_path: Path) -> None:
    lines = []
    lines.append("DA Override Model V1 - Grouped CV Summary")
    lines.append("=" * 44)
    lines.append(f"Files: {summary['n_files']}")
    lines.append(f"Emits (clot/wall only): {summary['n_total_emits']:,}")
    lines.append("")
    lines.append("Baselines (mean over folds)")
    lines.append(f"  DA only   : acc={summary['da_acc_mean']:.4f}, f1={summary['da_f1_mean']:.4f}")
    lines.append(f"  ML only   : acc={summary['ml_acc_mean']:.4f}, f1={summary['ml_f1_mean']:.4f}")
    lines.append("")
    lines.append("Best confidence-gated override policy")
    lines.append(f"  threshold : {summary['best_threshold']:.2f}")
    lines.append(f"  acc       : {summary['best_acc_mean']:.4f}")
    lines.append(f"  f1        : {summary['best_f1_mean']:.4f}")
    lines.append(f"  precision : {summary['best_override_precision_mean']:.4f}")
    lines.append(f"  net ben.  : {summary['best_net_benefit_mean']:+.2f} (correct - harmful, mean/fold)")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DA override model and sweep confidence threshold.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick runs")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-test-source",
        action="store_true",
        help="Allow fitting from a path that appears to be test/testing data.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = iter_files(args.source_dir, args.max_files)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {args.source_dir}")

    src_lower = str(args.source_dir).lower()
    looks_like_test = ("testing" in src_lower) or ("test_data" in src_lower)
    if looks_like_test and not args.allow_test_source:
        raise ValueError(
            "Refusing to fit override model from test/testing source. "
            "Use training_data (default) or pass --allow-test-source intentionally."
        )

    dfe = build_emit_dataset(files)
    fold_df, thresh_df, summary, feat_cols = run_grouped_cv(
        dfe,
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.seed,
    )

    final_model = train_final_model(dfe, feat_cols=feat_cols, seed=args.seed)

    fold_df.to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)
    thresh_df.to_csv(OUT_DIR / "threshold_sweep_summary.csv", index=False)
    write_summary(summary, OUT_DIR / "best_policy_summary.txt")
    save_runtime_bundle(
        final_model,
        feat_cols=feat_cols,
        summary=summary,
        out_path=OUT_DIR / "override_policy_v1.joblib",
    )

    print("\nSaved:")
    print(f"  {OUT_DIR / 'per_fold_metrics.csv'}")
    print(f"  {OUT_DIR / 'threshold_sweep_summary.csv'}")
    print(f"  {OUT_DIR / 'best_policy_summary.txt'}")
    print(f"  {OUT_DIR / 'override_policy_v1.joblib'}")

    print("\nBest policy:")
    print(f"  threshold={summary['best_threshold']:.2f}")
    print(f"  DA only   acc={summary['da_acc_mean']:.4f} f1={summary['da_f1_mean']:.4f}")
    print(f"  ML only   acc={summary['ml_acc_mean']:.4f} f1={summary['ml_f1_mean']:.4f}")
    print(f"  Override  acc={summary['best_acc_mean']:.4f} f1={summary['best_f1_mean']:.4f}")
    print(f"  Override precision={summary['best_override_precision_mean']:.4f}")
    print(f"  Net benefit={summary['best_net_benefit_mean']:+.2f} mean/fold")


if __name__ == "__main__":
    main()
