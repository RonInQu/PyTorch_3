"""
infer_dn.py — Run inference on denoised test data using the denoised model+scaler.

Reuses process_file() and main() logic from gru_torch_V6.py but overrides
MODEL_PATH, SCALER_PATH, TEST_DATA_DIR, and OUTPUT_FOLDER to point to the
denoised experiment paths.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Denoised experiment paths ──
DN_MODEL_PATH  = EXPERIMENT_DIR / "models" / "clot_gru_denoised.pt"
DN_SCALER_PATH = EXPERIMENT_DIR / "models" / "scaler_denoised.pkl"
DN_TEST_DIR    = EXPERIMENT_DIR / "test_data"
DN_RESULTS_DIR = EXPERIMENT_DIR / "results"

# ── Patch module globals BEFORE any function uses them ──
import src.models.gru_torch_V6 as _gru_mod

_gru_mod.MODEL_PATH   = DN_MODEL_PATH
_gru_mod.SCALER_PATH  = DN_SCALER_PATH
_gru_mod.OUTPUT_FOLDER = DN_RESULTS_DIR

# Now import the functions (they will read the patched globals)
from src.models.gru_torch_V6 import process_file, SAVE_CSV_PARQUET

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():
    DN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    glob_pattern = "*_labeled_segment_denoised.parquet"
    files = sorted(DN_TEST_DIR.glob(glob_pattern))

    print("=" * 60)
    print("INFERENCE — DENOISED EXPERIMENT")
    print("=" * 60)
    print(f"  Model:    {DN_MODEL_PATH}")
    print(f"  Scaler:   {DN_SCALER_PATH}")
    print(f"  Test dir: {DN_TEST_DIR}")
    print(f"  Output:   {DN_RESULTS_DIR}")
    print(f"  Files:    {len(files)}")
    print("-" * 60)

    if not files:
        print(f"No files matching {glob_pattern} in {DN_TEST_DIR}")
        sys.exit(1)

    all_gt_labels      = []
    all_da_labels      = []
    all_ml_preds       = []
    all_override_times = []

    for f in files:
        process_file(f,
                     all_gt_labels=all_gt_labels,
                     all_da_labels=all_da_labels,
                     all_ml_preds=all_ml_preds,
                     all_override_times=all_override_times,
                     save_csv_parquet=True)

    # ── Global summary ──
    if all_gt_labels:
        print("\n" + "=" * 60)
        print("GLOBAL SUMMARY — DENOISED EXPERIMENT")
        print("=" * 60)

        acc_da  = accuracy_score(all_gt_labels, all_da_labels)
        f1_da   = f1_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        prec_da = precision_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        rec_da  = recall_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)

        acc_ml  = accuracy_score(all_gt_labels, all_ml_preds)
        f1_ml   = f1_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        prec_ml = precision_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        rec_ml  = recall_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)

        print(f"DA  Accuracy: {acc_da:.4f}    F1-macro: {f1_da:.4f}")
        print(f"ML  Accuracy: {acc_ml:.4f}    F1-macro: {f1_ml:.4f}")
        print(f"Improvement: Acc {acc_ml - acc_da:+.4f}   F1 {f1_ml - f1_da:+.4f}")
        print(f"DA  Precision: {prec_da:.4f}    Recall: {rec_da:.4f}")
        print(f"ML  Precision: {prec_ml:.4f}    Recall: {rec_ml:.4f}")

        gt_arr = np.array(all_gt_labels)
        da_arr = np.array(all_da_labels)
        ml_arr = np.array(all_ml_preds)

        g_override_mask = (ml_arr != da_arr)
        g_n_overrides = g_override_mask.sum()

        print(f"\nGLOBAL OVERRIDE ANALYSIS")
        print(f"Total overrides: {g_n_overrides}")

        if g_n_overrides > 0:
            g_correct = (ml_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_harmful = (da_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_neither = g_n_overrides - g_correct - g_harmful
            g_override_prec = g_correct / g_n_overrides

            g_da_cw_errors = ((da_arr != gt_arr) & ((gt_arr == 1) | (gt_arr == 2))).sum()
            g_override_rec = g_correct / g_da_cw_errors if g_da_cw_errors > 0 else 0.0

            print(f"  Correct:    {g_correct}")
            print(f"  Harmful:    {g_harmful}")
            print(f"  Neither:    {g_neither}")
            print(f"  Precision:  {g_override_prec:.4f}")
            print(f"  Recall:     {g_override_rec:.4f}  (of {g_da_cw_errors} DA errors)")
            print(f"  Net benefit: {g_correct - g_harmful:+d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
