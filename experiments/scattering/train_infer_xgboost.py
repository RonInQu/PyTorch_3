# train_infer_xgboost.py
"""
XGBoost classifier with HYBRID features:
  - 21 hand-crafted features (from ClotFeatureExtractor, clot_wall_focused set)
  - 126 wavelet scattering features (J=6, Q=8)
  = 147 total features per window

Self-contained: trains on training_data/, evaluates on test_data/.

Key differences from GRU approach:
- Per-window classification (no sequence model) — each 5s window classified independently
- XGBoost handles mixed-scale features naturally (tree-based, no normalization needed)
- Temporal smoothing via simple EMA posterior after classification
- SHAP-ready for interpretability

Pipeline:
  1. Extract hybrid features from all windows (train + test)
  2. Train XGBoost with GroupKFold cross-validation
  3. Run inference on test set with EMA temporal smoothing
  4. Compare to DA, produce plots and metrics
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scattering.scattering_features import (
    extract_scattering_features, SCATTERING_DIM, WINDOW_SAMPLES, WINDOW_SEC, SAMPLE_RATE
)
from src.models.gru_torch_V6 import ClotFeatureExtractor, active_idx, active_dim
from scipy.signal import lfilter

HYBRID_DIM = active_dim + SCATTERING_DIM  # 21 + 126 = 147

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
STRIDE_SAMPLES = 30
REPORT_INTERVAL_MS = 200

# EMA smoothing for temporal coherence (applied to XGBoost predictions)
EMA_ALPHA = 0.3  # weight on new prediction (higher = more reactive)

# Temperature scaling: T>1 softens probabilities, T=1 no change, T<1 sharpens
TEMPERATURE = 1.5

# DA override settings (same as GRU pipeline)
DA_LABEL_CONFIDENCE = 0.92
DA_OVERRIDE_THRD = 0.80  # XGBoost must be > this to override DA

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "training_data"
TEST_DIR = PROJECT_ROOT / "test_data"
MODEL_DIR = SCRIPT_DIR / "models"
CACHE_DIR = SCRIPT_DIR / "cache"
OUTPUT_FOLDER = SCRIPT_DIR / "results_xgboost"

for d in [MODEL_DIR, CACHE_DIR, OUTPUT_FOLDER]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['blood', 'clot', 'wall']


def apply_temperature(probs, T=TEMPERATURE):
    """Apply temperature scaling to soften/sharpen probabilities.
    Converts probs → log-space, divides by T, then softmax back.
    T=1: no change. T>1: softer (less extreme). T<1: sharper.
    """
    if T == 1.0:
        return probs
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))
    scaled = log_probs / T
    scaled -= scaled.max()  # numerical stability
    exp_scaled = np.exp(scaled)
    return (exp_scaled / exp_scaled.sum()).astype(np.float32)


# ────────────────────────────────────────────────
# Feature extraction
# ────────────────────────────────────────────────
def extract_dataset_features(data_dir, cache_name):
    """Extract hybrid features (hand-crafted + scattering) from all parquets."""
    cache_file = CACHE_DIR / f"{cache_name}_hybrid_features.npz"

    if cache_file.exists():
        print(f"  Loading cached features: {cache_file.name}")
        data = np.load(cache_file, allow_pickle=True)
        return data['X'], data['y'], data['groups'], data['window_times'], data['study_names']

    parquet_files = sorted(data_dir.glob("*_labeled_segment.parquet"))
    if not parquet_files:
        print(f"  No parquet files in {data_dir}")
        sys.exit(1)

    # Hand-crafted feature extractor + EMA filter coefficients
    hc_extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC,
                                        active_features=active_idx)
    alpha_fast = hc_extractor.alpha_fast
    alpha_slow = hc_extractor.alpha_slow
    B_FAST = np.array([alpha_fast])
    A_FAST = np.array([1.0, -(1.0 - alpha_fast)])
    B_SLOW = np.array([alpha_slow])
    A_SLOW = np.array([1.0, -(1.0 - alpha_slow)])

    all_features = []
    all_labels = []
    all_groups = []
    all_times = []
    all_studies = []

    for file_path in parquet_files:
        study_name = file_path.stem
        df = pd.read_parquet(file_path, engine='pyarrow')

        if 'magRLoadAdjusted' not in df.columns or 'label' not in df.columns:
            continue

        resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
        labels = df['label'].to_numpy(dtype=np.int64)
        times = df['timeInMS'].to_numpy(dtype=np.float64) if 'timeInMS' in df.columns else np.arange(len(df))
        valid_mask = np.isin(labels, [0, 1, 2])

        if len(resistance) < WINDOW_SAMPLES:
            continue

        # Precompute full-run EMA arrays for hand-crafted features
        r0 = float(resistance[0])
        ema_f_all, _ = lfilter(B_FAST, A_FAST, resistance.astype(np.float64),
                               zi=[r0 * (1.0 - alpha_fast)])
        ema_s_all, _ = lfilter(B_SLOW, A_SLOW, resistance.astype(np.float64),
                               zi=[r0 * (1.0 - alpha_slow)])

        n_windows = 0
        for start in range(0, len(resistance) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            end = start + WINDOW_SAMPLES
            if not valid_mask[start:end].all():
                continue

            window = resistance[start:end]

            # Scattering features (126)
            scat_feats = extract_scattering_features(window)
            if scat_feats is None or len(scat_feats) != SCATTERING_DIM:
                continue

            # Hand-crafted features (21)
            hc_feats = hc_extractor.compute_features_from_array(
                window, float(ema_f_all[end - 1]), float(ema_s_all[end - 1]))
            if hc_feats is None or len(hc_feats) != active_dim:
                continue

            # Concatenate: [21 hand-crafted | 126 scattering]
            hybrid = np.concatenate([hc_feats, scat_feats])

            window_label = int(labels[start:end].max())
            all_features.append(hybrid)
            all_labels.append(window_label)
            all_groups.append(study_name)
            all_times.append(times[end - 1])
            all_studies.append(study_name)
            n_windows += 1

        print(f"  {study_name}: {n_windows} windows")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    groups = np.array(all_groups)
    window_times = np.array(all_times)
    study_names = np.array(all_studies)

    np.savez_compressed(cache_file, X=X, y=y, groups=groups,
                        window_times=window_times, study_names=study_names)
    print(f"  Cached → {cache_file.name}")

    return X, y, groups, window_times, study_names


# ────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────
def train_xgboost(X, y, groups):
    """Train XGBoost with GroupKFold CV, return best model."""
    print("\n=== XGBoost Training ===")
    print(f"  Samples: {len(y)}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {CLASS_NAMES[cls]}: {cnt} ({cnt/len(y)*100:.1f}%)")

    # Class weights for imbalanced data
    total = len(y)
    n_classes = len(unique)
    class_counts = np.bincount(y, minlength=3)
    sample_weights = np.array([total / (n_classes * class_counts[yi]) for yi in y])

    # XGBoost params
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'random_state': 456,
        'verbosity': 0,
    }

    # Cross-validation
    gkf = GroupKFold(n_splits=3)
    fold_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = xgb.XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx],
                  sample_weight=sample_weights[train_idx],
                  eval_set=[(X[val_idx], y[val_idx])],
                  verbose=False)

        val_pred = model.predict(X[val_idx])
        f1 = f1_score(y[val_idx], val_pred, average='macro')
        fold_f1s.append(f1)
        print(f"  Fold {fold_idx+1}: F1={f1:.4f}")

    mean_f1 = np.mean(fold_f1s)
    print(f"  Mean CV F1: {mean_f1:.4f}")

    # Final model on all data
    print("\n  Training final model on all data...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, sample_weight=sample_weights, verbose=False)

    model_path = MODEL_DIR / "hybrid_xgboost.json"
    final_model.save_model(str(model_path))
    print(f"  Model saved → {model_path}")

    return final_model, mean_f1


# ────────────────────────────────────────────────
# Inference with EMA smoothing
# ────────────────────────────────────────────────
def infer_study(model, filepath):
    """Run inference on a single study with EMA temporal smoothing."""
    study_name = filepath.stem
    df = pd.read_parquet(filepath, engine='pyarrow')

    resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
    time_ms = df['timeInMS'].to_numpy(dtype=np.float64)
    gt_labels = df['label'].to_numpy(dtype=np.int64) if 'label' in df.columns else None
    da_labels = df['da_label'].to_numpy(dtype=np.int64) if 'da_label' in df.columns else None

    # Hand-crafted feature extractor + precompute EMA
    hc_extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC,
                                        active_features=active_idx)
    alpha_fast = hc_extractor.alpha_fast
    alpha_slow = hc_extractor.alpha_slow
    r0 = float(resistance[0])
    ema_f_all, _ = lfilter(np.array([alpha_fast]),
                           np.array([1.0, -(1.0 - alpha_fast)]),
                           resistance.astype(np.float64),
                           zi=[r0 * (1.0 - alpha_fast)])
    ema_s_all, _ = lfilter(np.array([alpha_slow]),
                           np.array([1.0, -(1.0 - alpha_slow)]),
                           resistance.astype(np.float64),
                           zi=[r0 * (1.0 - alpha_slow)])

    # Extract features at report intervals
    results = []
    posterior = np.array([0.95, 0.025, 0.025], dtype=np.float32)
    last_report = -REPORT_INTERVAL_MS

    # Buffer for streaming
    buf_start = 0
    window_feats_cache = {}  # cache to avoid recomputation

    for i in range(len(time_ms)):
        t = time_ms[i]
        if t - last_report < REPORT_INTERVAL_MS:
            continue
        if i < WINDOW_SAMPLES - 1:
            continue

        last_report = t
        win_start = i - WINDOW_SAMPLES + 1
        window = resistance[win_start:i+1]

        # Scattering features (126)
        scat_feats = extract_scattering_features(window)
        if scat_feats is None:
            continue

        # Hand-crafted features (21)
        hc_feats = hc_extractor.compute_features_from_array(
            window, float(ema_f_all[i]), float(ema_s_all[i]))
        if hc_feats is None or len(hc_feats) != active_dim:
            continue

        # Concatenate: [21 hand-crafted | 126 scattering]
        hybrid = np.concatenate([hc_feats, scat_feats])

        # XGBoost prediction (probability)
        probs = model.predict_proba(hybrid.reshape(1, -1))[0]

        # Temperature scaling to soften extreme probabilities
        probs = apply_temperature(probs)

        # DA override logic
        da_now = int(da_labels[i]) if da_labels is not None else None

        if da_now is not None:
            if da_now == 0:
                # DA says blood → hard reset
                posterior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                results.append({
                    'time': t / 1000.0,
                    'prediction': 0,
                    'resistance': float(resistance[i]),
                    'Nprob': 1.0, 'Cprob': 0.0, 'Wprob': 0.0,
                    'rawN': float(probs[0]), 'rawC': float(probs[1]), 'rawW': float(probs[2]),
                })
                continue
            elif da_now in (1, 2):
                # DA says tissue — trust DA unless XGBoost is very confident otherwise
                xgb_top = np.argmax(probs)
                if xgb_top != da_now and probs[xgb_top] <= DA_OVERRIDE_THRD:
                    # XGBoost not confident enough → use DA
                    probs = np.array([DA_LABEL_CONFIDENCE if j == da_now else
                                      (1-DA_LABEL_CONFIDENCE)/2 for j in range(3)], dtype=np.float32)

        # EMA smoothing
        posterior = (1 - EMA_ALPHA) * posterior + EMA_ALPHA * probs

        status = np.argmax(posterior)
        results.append({
            'time': t / 1000.0,
            'prediction': status,
            'resistance': float(resistance[i]),
            'Nprob': float(posterior[0]),
            'Cprob': float(posterior[1]),
            'Wprob': float(posterior[2]),
            'rawN': float(probs[0]),
            'rawC': float(probs[1]),
            'rawW': float(probs[2]),
        })

    results_df = pd.DataFrame(results)
    return results_df, gt_labels, da_labels, time_ms, resistance


def plot_and_score(study_name, results_df, gt_labels, da_labels, time_ms, resistance,
                   all_gt, all_da, all_ml):
    """Generate plots and compute metrics for one study."""
    colors = {0: 'black', 1: 'red', 2: 'blue'}
    lbl_names = {0: 'blood', 1: 'clot', 2: 'wall'}

    # ── Plot 1: Probability traces ──
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    for lbl, name in [(0, 'blood'), (1, 'clot'), (2, 'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(f'{study_name} — XGBoost + Hybrid Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(results_df['time'], results_df['rawC'], 'r-', label='raw P(clot)', lw=1.2, alpha=0.8)
    ax2.plot(results_df['time'], results_df['rawW'], 'b-', label='raw P(wall)', lw=1.2, alpha=0.8)
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{study_name} — Raw XGBoost Probs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(results_df['time'], results_df['Cprob'], 'r-', label='P(clot)', lw=1.8)
    ax3.plot(results_df['time'], results_df['Wprob'], 'b-', label='P(wall)', lw=1.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    ax3.set_title(f'{study_name} — EMA-Smoothed Posterior')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"{study_name}_xgboost_probs.png", dpi=250, bbox_inches='tight')
    plt.close()

    # Metrics + three-panel plot
    if gt_labels is not None and da_labels is not None:
        full_times = time_ms / 1000.0
        interp_ml = np.interp(full_times, results_df['time'], results_df['prediction'])
        interp_ml = np.round(interp_ml).astype(int)

        valid = gt_labels >= 0
        gt_v = gt_labels[valid]
        da_v = da_labels[valid]
        ml_v = interp_ml[valid]

        all_gt.extend(gt_v.tolist())
        all_da.extend(da_v.tolist())
        all_ml.extend(ml_v.tolist())

        da_f1 = f1_score(gt_v, da_v, average='macro')
        ml_f1 = f1_score(gt_v, ml_v, average='macro')
        print(f"  {study_name}: DA F1={da_f1:.4f}  ML F1={ml_f1:.4f}  Δ={ml_f1-da_f1:+.4f}")

        override_mask = ml_v != da_v
        n_ov = override_mask.sum()
        if n_ov > 0:
            correct = (ml_v[override_mask] == gt_v[override_mask]).sum()
            harmful = (da_v[override_mask] == gt_v[override_mask]).sum()
            print(f"    Overrides: {n_ov}, correct={correct}, harmful={harmful}, net={correct-harmful:+d}")

        # ── Plot 2: Three-panel ML / DA / GT ──
        fig2, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True,
                                  gridspec_kw={'height_ratios': [1, 1, 1]})

        # Top: ML predictions with override highlights
        ax = axes[0]
        for lbl in [0, 1, 2]:
            mask = results_df['prediction'] == lbl
            ax.scatter(results_df['time'][mask], results_df['resistance'][mask],
                       c=colors[lbl], s=5, label=lbl_names[lbl], alpha=0.85)

        # Highlight override regions (ML ≠ DA)
        ml_da_interp = np.interp(results_df['time'], full_times, da_labels)
        ml_da_interp = np.round(ml_da_interp).astype(int)
        diff = (results_df['prediction'].values != ml_da_interp)
        diff_diff = np.diff(diff.astype(int))
        starts = np.where(diff_diff == 1)[0] + 1
        ends = np.where(diff_diff == -1)[0] + 1
        if diff.size > 0 and diff[0]:
            starts = np.insert(starts, 0, 0)
        if diff.size > 0 and diff[-1]:
            ends = np.append(ends, len(diff))
        for s, e in zip(starts, ends):
            ax.axvspan(results_df['time'].iloc[s],
                       results_df['time'].iloc[min(e - 1, len(results_df) - 1)],
                       facecolor='#e8e8e8', alpha=0.55,
                       label='ML ≠ DA' if s == starts[0] else None)

        ax.set_title(f'{study_name} — ML Predictions (XGBoost Hybrid, 200 ms)')
        ax.set_ylabel('Resistance (Ω)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Middle: DA labels, Bottom: GT labels
        panel_data = [("DA Labels (full 150 Hz)", da_labels),
                      ("Ground Truth Labels (full 150 Hz)", gt_labels)]
        for ax_idx, (title, data) in enumerate(panel_data):
            ax = axes[ax_idx + 1]
            unlabeled_mask = data == -1
            if unlabeled_mask.any():
                ax.scatter(full_times[unlabeled_mask], resistance[unlabeled_mask],
                           c='black', s=2, label='unlabeled', alpha=0.4, zorder=1)
            for lbl in [0, 1, 2]:
                mask = data == lbl
                ax.scatter(full_times[mask], resistance[mask], c=colors[lbl], s=2,
                           label=lbl_names[lbl], alpha=0.7, zorder=2)
            ax.set_title(f'{study_name} — {title}')
            ax.set_ylabel('Resistance (Ω)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            if ax_idx == 1:
                ax.set_xlabel('Time (seconds)')

        plt.tight_layout(h_pad=0.8)
        plt.savefig(OUTPUT_FOLDER / f"{study_name}_ml_da_gt_three_panel.png",
                    dpi=300, bbox_inches='tight')
        plt.close()


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  HYBRID XGBoost (21 hand-crafted + 126 scattering = 147 features)")
    print("=" * 60)
    print(f"  Hand-crafted: {active_dim} features (clot_wall_focused)")
    print(f"  Scattering:   {SCATTERING_DIM} features (J=6, Q=8)")
    print(f"  Total:        {HYBRID_DIM} features per window")
    print(f"  Window: {WINDOW_SEC}s ({WINDOW_SAMPLES} samples)")
    print(f"  Stride: {STRIDE_SAMPLES} samples")
    print(f"  Temperature: {TEMPERATURE}")
    print()

    # ── Step 1: Extract training features ──
    print("Extracting training features...")
    X_train, y_train, groups_train, _, _ = extract_dataset_features(DATA_DIR, "train")

    # ── Step 2: Train XGBoost ──
    model, cv_f1 = train_xgboost(X_train, y_train, groups_train)

    # ── Step 3: Feature importance ──
    importances = model.feature_importances_
    top_k = 20
    top_idx = np.argsort(importances)[::-1][:top_k]
    print(f"\n  Top {top_k} features (by importance):")
    for rank, idx in enumerate(top_idx):
        print(f"    #{rank+1}: feature[{idx}] = {importances[idx]:.4f}")

    # ── Step 4: Inference on test set ──
    print("\n" + "=" * 60)
    print("  INFERENCE ON TEST SET")
    print("=" * 60)

    test_files = sorted(TEST_DIR.glob("*_labeled_segment.parquet"))
    if not test_files:
        print(f"No test files in {TEST_DIR}")
        sys.exit(1)

    print(f"  {len(test_files)} test files\n")

    all_gt, all_da, all_ml = [], [], []

    for filepath in test_files:
        results_df, gt_labels, da_labels, time_ms, resistance = infer_study(model, filepath)
        plot_and_score(filepath.stem, results_df, gt_labels, da_labels, time_ms, resistance,
                       all_gt, all_da, all_ml)

    # ── Global summary ──
    if all_gt:
        gt = np.array(all_gt)
        da = np.array(all_da)
        ml = np.array(all_ml)

        print("\n" + "=" * 60)
        print("GLOBAL SUMMARY — Feats + Scattering + XGBoost (Hybrid)")
        print("=" * 60)
        da_f1 = f1_score(gt, da, average='macro')
        ml_f1 = f1_score(gt, ml, average='macro')
        da_acc = accuracy_score(gt, da)
        ml_acc = accuracy_score(gt, ml)
        da_prec = precision_score(gt, da, average='macro')
        ml_prec = precision_score(gt, ml, average='macro')
        da_rec = recall_score(gt, da, average='macro')
        ml_rec = recall_score(gt, ml, average='macro')

        print(f"DA  Accuracy: {da_acc:.4f}    F1-macro: {da_f1:.4f}")
        print(f"ML  Accuracy: {ml_acc:.4f}    F1-macro: {ml_f1:.4f}")
        print(f"Improvement: Acc {ml_acc-da_acc:+.4f}   F1 {ml_f1-da_f1:+.4f}")
        print()
        print(f"DA  Precision: {da_prec:.4f}    Recall: {da_rec:.4f}")
        print(f"ML  Precision: {ml_prec:.4f}    Recall: {ml_rec:.4f}")
        print(f"Improvement: Precision {ml_prec-da_prec:+.4f}   Recall {ml_rec-da_rec:+.4f}")

        override_mask = ml != da
        n_ov = override_mask.sum()
        correct = (ml[override_mask] == gt[override_mask]).sum()
        harmful = (da[override_mask] == gt[override_mask]).sum()
        neither = n_ov - correct - harmful
        net = correct - harmful

        print(f"\nOverride Analysis:")
        print(f"  Total overrides: {n_ov}")
        print(f"  Correct (ML right, DA wrong): {correct}")
        print(f"  Harmful (DA right, ML wrong): {harmful}")
        print(f"  Neither: {neither}")
        print(f"  Override Precision: {correct/n_ov:.4f}" if n_ov > 0 else "")
        print(f"  Net benefit: {net:+d} samples")

        # Save
        summary_path = OUTPUT_FOLDER / "global_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("GLOBAL SUMMARY — Feats + Scattering + XGBoost (Hybrid)\n")
            f.write(f"CV F1-macro: {cv_f1:.4f}\n\n")
            f.write(f"DA  Accuracy: {da_acc:.4f}    F1-macro: {da_f1:.4f}\n")
            f.write(f"ML  Accuracy: {ml_acc:.4f}    F1-macro: {ml_f1:.4f}\n")
            f.write(f"Improvement: Acc {ml_acc-da_acc:+.4f}   F1 {ml_f1-da_f1:+.4f}\n\n")
            f.write(f"Overrides: {n_ov}, Correct: {correct}, Harmful: {harmful}, Net: {net:+d}\n")
            f.write(f"Override Precision: {correct/n_ov:.4f}\n" if n_ov > 0 else "")
        print(f"\nSaved summary → {summary_path}")


if __name__ == '__main__':
    main()
