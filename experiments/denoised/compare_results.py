"""
compare_results.py — Compare denoised vs baseline inference results.

Loads detection_results CSVs from both pipelines, joins with ground-truth labels
from the test parquets, and prints a side-by-side comparison table.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ──
# Baseline: original model on original test data
BASELINE_RESULTS = PROJECT_ROOT / "inference_deploy" / "Results" / "Results_CWF_04.10.2025_ActualTestData" / "files"
BASELINE_TEST_DATA = PROJECT_ROOT / "test_data"

# Denoised: denoised model on denoised test data
DENOISED_RESULTS = EXPERIMENT_DIR / "results"
DENOISED_TEST_DATA = EXPERIMENT_DIR / "test_data"

REPORT_INTERVAL_MS = 200  # same as inference


def load_study_metrics(result_csv: Path, test_parquet: Path):
    """Load a detection result CSV and its ground-truth parquet, return per-study metrics."""
    df_res = pd.read_csv(result_csv)
    df_gt = pd.read_parquet(test_parquet)

    time_ms = df_gt['timeInMS'].values
    gt = df_gt['label'].values.astype(int)
    da = df_gt['da_label'].values.astype(int) if 'da_label' in df_gt.columns else gt.copy()
    resistance = df_gt['magRLoadAdjusted'].values

    # Interpolate ML predictions to full 150 Hz
    full_times = time_ms / 1000.0
    interp_ml = np.interp(full_times, df_res['time'].values, df_res['prediction'].values)
    interp_ml = np.round(interp_ml).astype(int)

    # Filter unlabeled
    valid = gt >= 0
    gt_v, da_v, ml_v = gt[valid], da[valid], interp_ml[valid]

    if len(gt_v) == 0:
        return None

    # Metrics
    f1_da  = f1_score(gt_v, da_v, average='macro', zero_division=0)
    f1_ml  = f1_score(gt_v, ml_v, average='macro', zero_division=0)
    acc_da = accuracy_score(gt_v, da_v)
    acc_ml = accuracy_score(gt_v, ml_v)
    prec_ml = precision_score(gt_v, ml_v, average='macro', zero_division=0)
    rec_ml  = recall_score(gt_v, ml_v, average='macro', zero_division=0)

    # Override analysis
    override_mask = (ml_v != da_v)
    n_overrides = override_mask.sum()
    correct = harmful = 0
    if n_overrides > 0:
        correct = (ml_v[override_mask] == gt_v[override_mask]).sum()
        harmful = (da_v[override_mask] == gt_v[override_mask]).sum()

    return {
        'f1_da': f1_da, 'f1_ml': f1_ml,
        'acc_da': acc_da, 'acc_ml': acc_ml,
        'prec_ml': prec_ml, 'rec_ml': rec_ml,
        'overrides': n_overrides,
        'correct': correct, 'harmful': harmful,
        'net_benefit': correct - harmful,
        'n_samples': len(gt_v),
        'gt_v': gt_v, 'da_v': da_v, 'ml_v': ml_v,
    }


def main():
    print("=" * 90)
    print("COMPARISON: BASELINE  vs  DENOISED")
    print("=" * 90)

    # Discover study IDs from baseline
    baseline_csvs = sorted(BASELINE_RESULTS.glob("*_labeled_segment_detection_results.csv"))
    baseline_ids = [c.name.replace("_labeled_segment_detection_results.csv", "") for c in baseline_csvs]

    if not baseline_ids:
        print(f"No baseline CSVs found in {BASELINE_RESULTS}")
        sys.exit(1)

    print(f"Baseline studies:  {len(baseline_ids)}")
    print(f"Denoised results:  {DENOISED_RESULTS}")
    print()

    rows = []
    all_bl_gt, all_bl_da, all_bl_ml = [], [], []
    all_dn_gt, all_dn_da, all_dn_ml = [], [], []

    for sid in baseline_ids:
        bl_csv = BASELINE_RESULTS / f"{sid}_labeled_segment_detection_results.csv"
        bl_parq = BASELINE_TEST_DATA / f"{sid}_labeled_segment.parquet"
        dn_csv = DENOISED_RESULTS / f"{sid}_labeled_segment_denoised_detection_results.csv"
        dn_parq = DENOISED_TEST_DATA / f"{sid}_labeled_segment_denoised.parquet"

        if not bl_csv.exists() or not bl_parq.exists():
            print(f"  ⚠️  Baseline missing for {sid}")
            continue
        if not dn_csv.exists() or not dn_parq.exists():
            print(f"  ⚠️  Denoised missing for {sid}")
            continue

        bl = load_study_metrics(bl_csv, bl_parq)
        dn = load_study_metrics(dn_csv, dn_parq)

        if bl is None or dn is None:
            continue

        rows.append({
            'study': sid,
            'bl_f1': bl['f1_ml'], 'dn_f1': dn['f1_ml'], 'delta_f1': dn['f1_ml'] - bl['f1_ml'],
            'bl_acc': bl['acc_ml'], 'dn_acc': dn['acc_ml'], 'delta_acc': dn['acc_ml'] - bl['acc_ml'],
            'bl_net': bl['net_benefit'], 'dn_net': dn['net_benefit'], 'delta_net': dn['net_benefit'] - bl['net_benefit'],
            'bl_overrides': bl['overrides'], 'dn_overrides': dn['overrides'],
        })

        all_bl_gt.extend(bl['gt_v']); all_bl_da.extend(bl['da_v']); all_bl_ml.extend(bl['ml_v'])
        all_dn_gt.extend(dn['gt_v']); all_dn_da.extend(dn['da_v']); all_dn_ml.extend(dn['ml_v'])

    if not rows:
        print("No studies matched for comparison.")
        sys.exit(1)

    # ── Per-study table ──
    print(f"\n{'Study':<12} {'BL_F1':>7} {'DN_F1':>7} {'ΔF1':>7}  {'BL_Acc':>7} {'DN_Acc':>7} {'ΔAcc':>7}  {'BL_Net':>7} {'DN_Net':>7} {'ΔNet':>7}")
    print("-" * 100)
    for r in rows:
        marker = "✓" if r['delta_f1'] > 0 else ("✗" if r['delta_f1'] < -0.01 else " ")
        print(f"{r['study']:<12} {r['bl_f1']:7.4f} {r['dn_f1']:7.4f} {r['delta_f1']:+7.4f}  "
              f"{r['bl_acc']:7.4f} {r['dn_acc']:7.4f} {r['delta_acc']:+7.4f}  "
              f"{r['bl_net']:7d} {r['dn_net']:7d} {r['delta_net']:+7d}  {marker}")

    # ── Aggregate ──
    print("\n" + "=" * 90)
    print("AGGREGATE")
    print("=" * 90)

    bl_gt = np.array(all_bl_gt); bl_da = np.array(all_bl_da); bl_ml = np.array(all_bl_ml)
    dn_gt = np.array(all_dn_gt); dn_da = np.array(all_dn_da); dn_ml = np.array(all_dn_ml)

    bl_f1 = f1_score(bl_gt, bl_ml, average='macro', zero_division=0)
    dn_f1 = f1_score(dn_gt, dn_ml, average='macro', zero_division=0)
    bl_acc = accuracy_score(bl_gt, bl_ml)
    dn_acc = accuracy_score(dn_gt, dn_ml)

    bl_net = sum(r['bl_net'] for r in rows)
    dn_net = sum(r['dn_net'] for r in rows)

    n_better  = sum(1 for r in rows if r['delta_f1'] > 0.001)
    n_worse   = sum(1 for r in rows if r['delta_f1'] < -0.001)
    n_same    = len(rows) - n_better - n_worse

    print(f"                Baseline    Denoised    Delta")
    print(f"  F1-macro:     {bl_f1:.4f}      {dn_f1:.4f}      {dn_f1 - bl_f1:+.4f}")
    print(f"  Accuracy:     {bl_acc:.4f}      {dn_acc:.4f}      {dn_acc - bl_acc:+.4f}")
    print(f"  Net benefit:  {bl_net:+d}      {dn_net:+d}      {dn_net - bl_net:+d}")
    print(f"")
    print(f"  Studies improved: {n_better}  |  Same: {n_same}  |  Worse: {n_worse}")
    print(f"")

    verdict = "DENOISED IS BETTER" if dn_f1 > bl_f1 + 0.005 else (
              "DENOISED IS WORSE" if dn_f1 < bl_f1 - 0.005 else
              "NO SIGNIFICANT DIFFERENCE")
    print(f"  VERDICT: {verdict}")
    print("=" * 90)


if __name__ == "__main__":
    main()
