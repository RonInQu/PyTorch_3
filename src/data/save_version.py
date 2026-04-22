"""
save_version.py — Snapshot a trained pipeline version.

Copies model, scaler, and train/test split into a timestamped folder
under  versions/  at the project root.

Usage (standalone):
    python src/data/save_version.py                       # auto-detect latest
    python src/data/save_version.py --tag "baseline_v1"   # custom tag
    python src/data/save_version.py --note "First 62/14 split, +24k net benefit"

Usage (from train_gru_V6.py):
    from src.data.save_version import save_version
    save_version(f1=0.8091, tag="after_retrain", note="Added 5 new studies")
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

VERSIONS_DIR = PROJECT_ROOT / "versions"


def _collect_training_studies(data_dir: Path) -> list[str]:
    return sorted(p.stem.replace("_labeled_segment", "")
                  for p in data_dir.glob("*.parquet"))


def _collect_test_studies(data_dir: Path) -> list[str]:
    return sorted(p.stem.replace("_labeled_segment", "")
                  for p in data_dir.glob("*.parquet"))


def save_version(
    f1: float | None = None,
    tag: str | None = None,
    note: str | None = None,
    global_summary: str | None = None,
) -> Path:
    """Snapshot the current model + scaler + split into versions/<timestamp>/."""

    # ── lazy import so the module can be used without torch installed ──
    from src.models.gru_torch_V6 import (
        FEATURE_SET, SEQ_LEN, active_dim, dim_str,
        WINDOW_SEC, REPORT_INTERVAL_MS, TEMPERATURE,
        GRU_OVERRIDE_THRD_CLOT, GRU_OVERRIDE_THRD_WALL,
        EMA_BLOOD_PRIOR_HISTORY, EMA_EXIT_TO_BLOOD_HISTORY,
        EMA_SAME_CLASS_HISTORY, EMA_CROSS_CLASS_HISTORY,
        DA_LABEL_CONFIDENCE,
        INIT_BLOOD_PROB, INIT_CLOT_PROB, INIT_WALL_PROB,
    )
    from src.training.train_gru_V6 import (
        SEEDS_TO_TRY, STRIDE_SAMPLES, BATCH_SIZE, N_EPOCHS,
        PATIENCE, LR, WEIGHT_DECAY, CLINICAL_WEIGHTS,
    )

    # ── resolve source files ──
    model_path = PROJECT_ROOT / "src" / "training" / "clot_gru_trained.pt"
    scaler_path = (PROJECT_ROOT / "src" / "data" /
                   f"clot_feature_scaler_5s_seq{SEQ_LEN}_{dim_str}.pkl")
    train_dir = PROJECT_ROOT / "training_data"
    test_dir  = PROJECT_ROOT / "test_data"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    # ── build version folder name ──
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_name = timestamp
    if tag:
        # sanitize tag for filesystem
        safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
        folder_name = f"{timestamp}_{safe_tag}"

    version_dir = VERSIONS_DIR / folder_name
    version_dir.mkdir(parents=True, exist_ok=False)

    # ── copy artefacts ──
    shutil.copy2(model_path,  version_dir / model_path.name)
    shutil.copy2(scaler_path, version_dir / scaler_path.name)

    # ── copy pipeline source scripts for reproducibility ──
    script_copies = [
        PROJECT_ROOT / "src" / "data" / "Labeling_5Names_V6.py",
        PROJECT_ROOT / "src" / "data" / "fit_scaler_V6.py",
        PROJECT_ROOT / "src" / "training" / "train_gru_V6.py",
        PROJECT_ROOT / "src" / "models" / "gru_torch_V6.py",
    ]
    for src_script in script_copies:
        if src_script.exists():
            shutil.copy2(src_script, version_dir / src_script.name)

    # ── copy global summary ONLY if explicitly provided ──
    # Do NOT auto-copy from inference_deploy/Results/ — it may be stale
    # from a different run.  Pass --summary-file or global_summary param.
    if global_summary is not None:
        (version_dir / "global_summary.txt").write_text(
            global_summary, encoding="utf-8")

    # ── copy only the latest per-seed model (by modification time) ──
    seed_pattern = f"clot_gru_trained_seq{SEQ_LEN}_{FEATURE_SET}_seed*.pt"
    seed_files = sorted(
        (PROJECT_ROOT / "src" / "training").glob(seed_pattern),
        key=lambda p: p.stat().st_mtime,
    )
    if seed_files:
        latest_seed = seed_files[-1]
        shutil.copy2(latest_seed, version_dir / latest_seed.name)

    # ── gather study lists ──
    train_studies = _collect_training_studies(train_dir)
    test_studies  = _collect_test_studies(test_dir)

    # ── write manifest ──
    lines = [
        f"# Pipeline version — {timestamp}",
        f"# Feature set: {FEATURE_SET} ({active_dim} features)",
        f"# SEQ_LEN: {SEQ_LEN}",
    ]
    if f1 is not None:
        lines.append(f"# Best F1-macro: {f1:.4f}")
    if note:
        lines.append(f"# Note: {note}")
    lines.append(f"# Model: {model_path.name}")
    lines.append(f"# Scaler: {scaler_path.name}")
    lines.append("")

    # ── Configuration settings ──
    lines.append("=== LABELING CONFIG ===")
    lines.append(f"Blood events:    [6, 12]")
    lines.append(f"Clot events:     [7, 11]")
    lines.append(f"Wall events:     [23]")
    lines.append(f"Artifact events: [8 (contrast), 15 (saline)] — always blanked")
    lines.append(f"Noise value:     5")
    lines.append(f"Blank mask:      (da_label==0 & not tissue) | (R>5000) | artifact")
    lines.append("")

    lines.append("=== TRAINING CONFIG ===")
    lines.append(f"Seeds:          {SEEDS_TO_TRY}")
    lines.append(f"Window sec:     {WINDOW_SEC}")
    lines.append(f"Stride samples: {STRIDE_SAMPLES}")
    lines.append(f"Batch size:     {BATCH_SIZE}")
    lines.append(f"Epochs:         {N_EPOCHS}")
    lines.append(f"Patience:       {PATIENCE}")
    lines.append(f"Learning rate:  {LR}")
    lines.append(f"Weight decay:   {WEIGHT_DECAY}")
    lines.append(f"Class weights:  {CLINICAL_WEIGHTS}")
    lines.append("")

    lines.append("=== INFERENCE CONFIG ===")
    lines.append(f"Temperature:           {TEMPERATURE}")
    lines.append(f"GRU override clot:     {GRU_OVERRIDE_THRD_CLOT}")
    lines.append(f"GRU override wall:     {GRU_OVERRIDE_THRD_WALL}")
    lines.append(f"DA label confidence:   {DA_LABEL_CONFIDENCE}")
    lines.append(f"EMA blood prior:       {EMA_BLOOD_PRIOR_HISTORY}")
    lines.append(f"EMA exit to blood:     {EMA_EXIT_TO_BLOOD_HISTORY}")
    lines.append(f"EMA same class:        {EMA_SAME_CLASS_HISTORY}")
    lines.append(f"EMA cross class:       {EMA_CROSS_CLASS_HISTORY}")
    lines.append(f"Init posterior:         [{INIT_BLOOD_PROB}, {INIT_CLOT_PROB}, {INIT_WALL_PROB}]")
    lines.append(f"Report interval ms:    {REPORT_INTERVAL_MS}")
    lines.append("")

    lines.append(f"=== TRAINING ({len(train_studies)}) ===")
    lines.extend(train_studies)
    lines.append("")
    lines.append(f"=== TEST ({len(test_studies)}) ===")
    lines.extend(test_studies)
    lines.append("")

    # ── Global summary from inference (if provided) ──
    if global_summary:
        lines.append("=== GLOBAL SUMMARY ===")
        lines.append(global_summary.strip())
        lines.append("")

    manifest = version_dir / "manifest.txt"
    manifest.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  VERSION SAVED  →  {version_dir.relative_to(PROJECT_ROOT)}")
    print(f"{'='*60}")
    print(f"  Model:   {model_path.name}")
    print(f"  Scaler:  {scaler_path.name}")
    print(f"  Train:   {len(train_studies)} studies")
    print(f"  Test:    {len(test_studies)} studies")
    if f1 is not None:
        print(f"  F1:      {f1:.4f}")
    if note:
        print(f"  Note:    {note}")
    print(f"{'='*60}\n")

    return version_dir


# ─────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a versioned snapshot of the current pipeline.")
    parser.add_argument("--tag",  type=str, default=None, help="Short label appended to folder name")
    parser.add_argument("--note", type=str, default=None, help="Free-text note for the manifest")
    parser.add_argument("--f1",   type=float, default=None, help="Best F1-macro to record")
    parser.add_argument("--summary-file", type=str, default=None,
                        help="Path to a text file containing the global inference summary")
    args = parser.parse_args()

    global_summary = None
    if args.summary_file:
        global_summary = Path(args.summary_file).read_text(encoding="utf-8")

    save_version(f1=args.f1, tag=args.tag, note=args.note, global_summary=global_summary)
