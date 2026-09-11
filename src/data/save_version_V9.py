"""
save_version_V9.py — Snapshot a trained V9 (2-class) pipeline version.

Copies model, scaler, and train/test split into a timestamped folder
under  versions/  at the project root. V9-specific — matches the imports,
paths, and config knobs used by the V9 pipeline.

Usage (standalone):
    python src/data/save_version_V9.py
    python src/data/save_version_V9.py --tag "150train_33test"
    python src/data/save_version_V9.py --note "Post source-fix rerun" --f1 0.85

Usage (from train_gru_V9.py):
    from src.data.save_version_V9 import save_version_v9
    save_version_v9(f1=0.85, tag="after_retrain", note="...")
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

VERSIONS_DIR = PROJECT_ROOT / "versions"


def _collect_studies(data_dir: Path) -> list[str]:
    return sorted(
        p.stem.replace("_labeled_segment", "")
        for p in data_dir.glob("*.parquet")
    )


def save_version_v9(
    f1: float | None = None,
    tag: str | None = None,
    note: str | None = None,
    global_summary: str | None = None,
) -> Path:
    """Snapshot the current V9 model + scaler + split into versions/<timestamp>_v9[_tag]/."""

    # ── lazy import so the module can be used without torch installed ──
    from src.models.gru_torch_V9 import (
        FEATURE_SET, SEQ_LEN, active_dim, dim_str,
        WINDOW_SEC, REPORT_INTERVAL_MS, TEMPERATURE,
        EMA_HISTORY,
        DA_LABEL_CONFIDENCE,
        ML_STABILITY_STREAK, ML_STABILITY_MEAN_CONF,
        ML_STABILITY_CONF_RANGE,
        SCALER_PATH, MODEL_PATH,
    )
    from src.training.train_gru_V9 import (
        SEEDS_TO_TRY, STRIDE_SAMPLES, BATCH_SIZE, N_EPOCHS,
        PATIENCE, LR, WEIGHT_DECAY, CLINICAL_WEIGHTS,
        CLASS_NAMES, NUM_CLASSES,
    )

    # ── resolve source files ──
    train_dir = PROJECT_ROOT / "training_data"
    test_dir = PROJECT_ROOT / "test_data"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"V9 model not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"V9 scaler not found: {SCALER_PATH}")

    # ── build version folder name: <timestamp>_v9[_tag] ──
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_name = f"{timestamp}_v9"
    if tag:
        safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
        folder_name = f"{folder_name}_{safe_tag}"

    version_dir = VERSIONS_DIR / folder_name
    version_dir.mkdir(parents=True, exist_ok=False)

    # ── copy artefacts ──
    shutil.copy2(MODEL_PATH, version_dir / MODEL_PATH.name)
    shutil.copy2(SCALER_PATH, version_dir / SCALER_PATH.name)

    # ── copy pipeline source scripts for reproducibility ──
    script_copies = [
        PROJECT_ROOT / "src" / "data" / "fit_scaler_V9.py",
        PROJECT_ROOT / "src" / "training" / "train_gru_V9.py",
        PROJECT_ROOT / "src" / "models" / "gru_torch_V9.py",
        PROJECT_ROOT / "src" / "data" / "save_version_V9.py",
        PROJECT_ROOT / "scripts" / "build_train_test_split_183.py",
        PROJECT_ROOT / "scripts" / "data_inventory_183.py",
    ]
    for src_script in script_copies:
        if src_script.exists():
            shutil.copy2(src_script, version_dir / src_script.name)

    # ── copy only the latest per-seed model (by modification time) ──
    seed_pattern = f"clot_gru_trained_V9_2class_seq{SEQ_LEN}_{FEATURE_SET}_seed*.pt"
    seed_files = sorted(
        (PROJECT_ROOT / "src" / "training").glob(seed_pattern),
        key=lambda p: p.stat().st_mtime,
    )
    if seed_files:
        latest_seed = seed_files[-1]
        shutil.copy2(latest_seed, version_dir / latest_seed.name)

    # ── gather study lists ──
    train_studies = _collect_studies(train_dir)
    test_studies = _collect_studies(test_dir)

    # ── write manifest ──
    lines: list[str] = []
    lines.append(f"# V9 Pipeline version — {timestamp}")
    lines.append(f"# Architecture: 2-class (clot vs wall) with DA guardrail")
    lines.append(f"# Feature set: {FEATURE_SET} ({active_dim} features, dim_str={dim_str})")
    lines.append(f"# SEQ_LEN: {SEQ_LEN}")
    lines.append(f"# NUM_CLASSES: {NUM_CLASSES}  ({CLASS_NAMES})")
    if f1 is not None:
        lines.append(f"# Best F1-macro: {f1:.4f}")
    if note:
        lines.append(f"# Note: {note}")
    lines.append(f"# Model:  {MODEL_PATH.name}")
    lines.append(f"# Scaler: {SCALER_PATH.name}")
    lines.append("")

    lines.append("=== LABELING CONFIG (per Labeling_5Names_V6 conventions) ===")
    lines.append("Blood events:    [6, 12]")
    lines.append("Clot events:     [7, 11]")
    lines.append("Wall events:     [23]")
    lines.append("Artifact events: [8 (contrast), 15 (saline)] — blanked to baseline")
    lines.append("V9 note: model trains on {clot=1, wall=2} only; blood windows skipped.")
    lines.append("")

    lines.append("=== TRAINING CONFIG (V9) ===")
    lines.append(f"Seeds:          {SEEDS_TO_TRY}")
    lines.append(f"Window sec:     {WINDOW_SEC}")
    lines.append(f"Stride samples: {STRIDE_SAMPLES}")
    lines.append(f"Batch size:     {BATCH_SIZE}")
    lines.append(f"Epochs:         {N_EPOCHS}")
    lines.append(f"Patience:       {PATIENCE}")
    lines.append(f"Learning rate:  {LR}")
    lines.append(f"Weight decay:   {WEIGHT_DECAY}")
    lines.append(f"Class weights:  {CLINICAL_WEIGHTS}  (order: {CLASS_NAMES})")
    lines.append("")

    lines.append("=== INFERENCE CONFIG (V9) ===")
    lines.append(f"Temperature:           {TEMPERATURE}")
    lines.append(f"DA label confidence:   {DA_LABEL_CONFIDENCE}")
    lines.append(f"EMA history:           {EMA_HISTORY}")
    lines.append(f"Init posterior (2c):   [0.65, 0.35]  (clot, wall)")
    lines.append(f"Report interval ms:    {REPORT_INTERVAL_MS}")
    lines.append("")
    lines.append("--- V9 ML-override stability gate ---")
    lines.append(f"streak:      {ML_STABILITY_STREAK}   (raw GRU same class for N samples)")
    lines.append(f"mean_conf:   {ML_STABILITY_MEAN_CONF}   (mean raw conf over run)")
    lines.append(f"conf_range:  {ML_STABILITY_CONF_RANGE}   (max - min conf over run)")
    lines.append("")

    lines.append(f"=== TRAINING SPLIT ({len(train_studies)} studies) ===")
    lines.extend(train_studies)
    lines.append("")
    lines.append(f"=== TEST SPLIT ({len(test_studies)} studies) ===")
    lines.extend(test_studies)
    lines.append("")

    if global_summary:
        lines.append("=== GLOBAL SUMMARY (from inference run) ===")
        lines.append(global_summary.strip())
        lines.append("")

    manifest = version_dir / "manifest.txt"
    manifest.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  V9 VERSION SAVED  →  {version_dir.relative_to(PROJECT_ROOT)}")
    print(f"{'=' * 60}")
    print(f"  Model:   {MODEL_PATH.name}")
    print(f"  Scaler:  {SCALER_PATH.name}")
    print(f"  Train:   {len(train_studies)} studies")
    print(f"  Test:    {len(test_studies)} studies")
    if f1 is not None:
        print(f"  F1:      {f1:.4f}")
    if note:
        print(f"  Note:    {note}")
    print(f"{'=' * 60}\n")

    return version_dir


# ─────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a versioned snapshot of the current V9 pipeline."
    )
    parser.add_argument("--tag", type=str, default=None,
                        help="Short label appended to folder name")
    parser.add_argument("--note", type=str, default=None,
                        help="Free-text note for the manifest")
    parser.add_argument("--f1", type=float, default=None,
                        help="Best F1-macro to record")
    parser.add_argument("--summary-file", type=str, default=None,
                        help="Path to a text file containing the global inference summary")
    args = parser.parse_args()

    summary_text: str | None = None
    if args.summary_file:
        summary_text = Path(args.summary_file).read_text(encoding="utf-8")

    save_version_v9(
        f1=args.f1,
        tag=args.tag,
        note=args.note,
        global_summary=summary_text,
    )
