"""
train_feature_search.py — Test multiple feature sets on denoised data.

Extracts ALL 46 features once, caches them, then trains with different
feature subsets to find which set works best on denoised signals.

Hypothesis: The clot_wall_focused set relies on pulse-dependent features
(variability, Hjorth, 2nd derivative) that lose discriminative power after
cardiac pulse removal. Slope and EMA features should benefit from cleaner
signals.
"""

import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

import joblib
from scipy.signal import lfilter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_torch_V6 import ClotFeatureExtractor, ClotGRU, WINDOW_SEC, SEQ_LEN

# ── Training config (same as baseline) ──
SEED = 456
STRIDE_SAMPLES = 30
BATCH_SIZE = 1024
N_EPOCHS = 100
PATIENCE = 15
LR = 0.0001
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ──
DATA_DIR = EXPERIMENT_DIR / "train_data"
CACHE_DIR = EXPERIMENT_DIR / "cache"
MODELS_DIR = EXPERIMENT_DIR / "models"

# ══════════════════════════════════════════════════
# PROPOSED FEATURE SETS FOR DENOISED DATA
# ══════════════════════════════════════════════════
#
# Rationale:
# - Denoising (2s moving median + 0.3s smooth) removes cardiac pulse oscillations
# - Features capturing variability/oscillation (std, range, Hjorth, 2nd deriv)
#   lose pulse-dependent discriminative power
# - Features capturing trends/levels (mean, slopes, EMA) become CLEANER
# - The original clot_wall_focused had ZERO slope features — a big miss for
#   denoised data where slopes are now much more reliable
#
# Feature groups affected:
#   DEGRADED: f1(std), f5(range), f9(windowed_diff), f17(deriv_std),
#     f21(deriv_kurt), f28-29(detrend_std), f32(detrend_std_diff),
#     f38(p95_p5_range), f39(frac>p95), f40-42(Hjorth/2nd_deriv),
#     f43-45(pulse feats → near-zero on denoised!)
#   PRESERVED/IMPROVED: f0(mean), f3(min), f4(max), f6(median),
#     f10-15(slopes!), f22-27(EMA), f16(deriv_mean), f20(deriv_skew)
#

FEATURE_SETS_TO_TEST = {
    # ── Set A: Replace pulse-sensitive features with all 6 slopes ──
    # Keep robust features from CWF, swap degraded ones for slopes + more EMA
    "dn_slopes": [
        0, 1, 3, 4, 5, 23, 27, 20,       # 8 robust from clot_wall_focused
        10, 11, 12, 13, 14, 15,            # 6 slopes (ALL — key addition)
        22, 24, 26,                         # 3 more EMA features
        6, 16, 19, 28,                      # 4 supplementary
    ],  # 21 features — same count as CWF

    # ── Set B: Lean set — levels + multi-scale slopes + EMA ──
    # Only features that should genuinely benefit from denoising
    "dn_level_slope": [
        0, 3, 4, 6,                         # 4 level features (mean, min, max, median)
        10, 11, 12, 13, 14, 15,             # 6 slopes (the big win)
        22, 23, 24, 27,                      # 4 EMA
        16, 5,                               # deriv_mean, range
    ],  # 16 features

    # ── Set C: Broad search — all non-degenerate features ──
    # Let the model sort out what's useful from a wider pool
    "dn_broad": [
        0, 1, 3, 4, 5, 6, 7, 9,            # 8 basic stats
        10, 11, 12, 13, 14, 15,             # 6 slopes
        16, 17, 19, 20, 21,                  # 5 derivative
        22, 23, 24, 26, 27,                  # 5 EMA
        28, 29, 34,                           # 3 detrended
        36, 37, 38, 39, 40,                   # 5 percentiles + Hjorth mob
    ],  # 30 features

    # ── Baseline reference: original CWF on denoised data ──
    "clot_wall_focused": [
        39, 21, 4, 19, 41, 9, 5, 23, 0, 34,
        28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40,
    ],  # 21 features — control
}

ALL_46 = list(range(46))

# ── Extractor for all 46 features ──
_extractor = ClotFeatureExtractor(sample_rate=150, window_sec=WINDOW_SEC,
                                  active_features=ALL_46)
WINDOW_SAMPLES = _extractor.window_size
ALPHA_FAST = _extractor.alpha_fast
ALPHA_SLOW = _extractor.alpha_slow

_B_FAST = np.array([ALPHA_FAST])
_A_FAST = np.array([1.0, -(1.0 - ALPHA_FAST)])
_B_SLOW = np.array([ALPHA_SLOW])
_A_SLOW = np.array([1.0, -(1.0 - ALPHA_SLOW)])


# ────────────────────────────────────────────────
# Extract ALL 46 features once, cache them
# ────────────────────────────────────────────────

def load_or_extract_all_features():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"features_denoised_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}_all46.npz"

    if cache_file.exists():
        print(f"Loading cached ALL-46 features: {cache_file}")
        data = np.load(cache_file)
        return data['X_seq'], data['y'].astype(np.int64), data['groups']

    print("Extracting ALL 46 features from denoised training data...")
    data_files = sorted(DATA_DIR.glob("*.parquet"))
    if not data_files:
        print(f"No parquet files in {DATA_DIR}")
        sys.exit(1)

    all_data = [pd.read_parquet(f).assign(run_id=f.stem) for f in data_files]
    df_all = pd.concat(all_data, ignore_index=True)

    seq_list, labels_list, groups_list = [], [], []

    for run_id, group in df_all.groupby('run_id'):
        resistance = group['magRLoadAdjusted'].to_numpy(dtype=np.float32)
        label_array = group['label'].values.astype(np.int64)
        valid_mask = np.isin(label_array, [0, 1, 2])

        if len(resistance) < WINDOW_SAMPLES:
            continue

        r0 = float(resistance[0])
        ema_f_all, _ = lfilter(_B_FAST, _A_FAST, resistance.astype(np.float64),
                               zi=[r0 * (1.0 - ALPHA_FAST)])
        ema_s_all, _ = lfilter(_B_SLOW, _A_SLOW, resistance.astype(np.float64),
                               zi=[r0 * (1.0 - ALPHA_SLOW)])

        extraction_indices = np.arange(WINDOW_SAMPLES - 1, len(resistance), STRIDE_SAMPLES)

        run_features, run_labels = [], []
        for idx in extraction_indices:
            win_start = idx - WINDOW_SAMPLES + 1
            if not valid_mask[win_start:idx + 1].all():
                continue
            window_data = resistance[win_start:idx + 1]
            feats = _extractor.compute_features_from_array(
                window_data, float(ema_f_all[idx]), float(ema_s_all[idx]))
            if feats is not None and len(feats) == 46:
                run_features.append(feats)
                run_labels.append(int(label_array[win_start:idx + 1].max()))

        for i in range(SEQ_LEN - 1, len(run_features)):
            seq = np.array(run_features[i - SEQ_LEN + 1: i + 1])
            seq_list.append(seq)
            labels_list.append(run_labels[i])
            groups_list.append(run_id)

    X_seq = np.array(seq_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    groups = np.array(groups_list)
    print(f"Extracted: {X_seq.shape[0]} sequences | shape={X_seq.shape} from {len(np.unique(groups))} runs")

    np.savez_compressed(cache_file, X_seq=X_seq, y=y, groups=groups)
    print(f"Saved cache → {cache_file}")
    return X_seq, y, groups


# ────────────────────────────────────────────────
# Subset features + fit per-set scaler
# ────────────────────────────────────────────────

def subset_and_scale(X_all, feature_indices):
    """Select feature columns from all-46 data, fit scaler, return scaled data."""
    idx = np.array(feature_indices)
    X_sub = X_all[:, :, idx]  # (N, SEQ_LEN, n_features)
    N, S, F = X_sub.shape

    X_flat = X_sub.reshape(-1, F)
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled_flat = np.nan_to_num(X_scaled_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = X_scaled_flat.reshape(N, S, F)

    return X_scaled, scaler


# ────────────────────────────────────────────────
# Train one configuration
# ────────────────────────────────────────────────

def train_one_set(set_name, feature_indices, X_all, y, groups):
    n_features = len(feature_indices)
    print(f"\n{'='*70}")
    print(f"TRAINING: {set_name}  ({n_features} features)")
    print(f"  Indices: {feature_indices}")
    print(f"{'='*70}")

    # Subset + scale
    X_scaled, scaler = subset_and_scale(X_all, feature_indices)

    # Class weights
    unique = np.unique(y)
    balanced = compute_class_weight('balanced', classes=unique, y=y)
    weights = np.ones(3)
    for cls, w in zip(unique, balanced):
        weights[cls] = w
    cw_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: blood={weights[0]:.2f} clot={weights[1]:.2f} wall={weights[2]:.2f}")

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # GroupKFold
    gkf = GroupKFold(n_splits=5)
    unique_groups = np.array(sorted(set(groups)))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups])

    best_f1_overall = 0.0
    best_model_state = None
    fold_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, group_ids)):
        X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

        model = ClotGRU(input_size=n_features).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_f1_fold = 0.0
        best_state_fold = None
        wait = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(DEVICE)
                    logits, _ = model(xb)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(yb.numpy())

            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            scheduler.step(f1)

            if f1 > best_f1_fold:
                best_f1_fold = f1
                best_state_fold = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
                marker = "  ← best"
            else:
                wait += 1
                marker = ""

            if epoch <= 3 or f1 > best_f1_fold - 0.001 or epoch == 1:
                print(f"    Fold {fold_idx+1} Ep {epoch:3d} | F1 {f1:.4f}{marker}")

            if wait >= PATIENCE:
                print(f"    Fold {fold_idx+1} early stop at epoch {epoch}")
                break

        fold_f1s.append(best_f1_fold)
        print(f"    Fold {fold_idx+1} best F1: {best_f1_fold:.4f}")

        if best_f1_fold > best_f1_overall:
            best_f1_overall = best_f1_fold
            best_model_state = best_state_fold

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f"\n  {set_name} RESULT — Mean F1: {mean_f1:.4f} ± {std_f1:.4f}  "
          f"Best fold: {best_f1_overall:.4f}")
    print(f"  Per-fold: {[f'{f:.4f}' for f in fold_f1s]}")

    # Save model + scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"clot_gru_dn_{set_name}.pt"
    scaler_path = MODELS_DIR / f"scaler_dn_{set_name}.pkl"
    torch.save(best_model_state, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved: {model_path.name}, {scaler_path.name}")

    return {
        'name': set_name,
        'n_features': n_features,
        'indices': feature_indices,
        'fold_f1s': fold_f1s,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'best_f1': best_f1_overall,
    }


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FEATURE SET SEARCH — DENOISED DATA")
    print("=" * 70)

    # Step 1: Extract all 46 features
    X_all, y, groups = load_or_extract_all_features()
    print(f"\nData: {X_all.shape[0]} sequences, {len(np.unique(groups))} runs, 46 features")

    # Step 2: Train each feature set
    results = []
    for set_name, indices in FEATURE_SETS_TO_TEST.items():
        res = train_one_set(set_name, indices, X_all, y, groups)
        results.append(res)

    # Step 3: Summary
    print("\n" + "=" * 70)
    print("SUMMARY — FEATURE SET COMPARISON")
    print("=" * 70)
    print(f"\n{'Set':<22} {'#Feat':>5}  {'Mean F1':>8}  {'± Std':>7}  {'Best Fold':>9}")
    print("-" * 60)

    results.sort(key=lambda r: r['mean_f1'], reverse=True)
    for r in results:
        marker = " ★" if r == results[0] else ""
        print(f"{r['name']:<22} {r['n_features']:5d}  {r['mean_f1']:8.4f}  {r['std_f1']:7.4f}  "
              f"{r['best_f1']:9.4f}{marker}")

    best = results[0]
    print(f"\nBEST: {best['name']} — Mean F1 = {best['mean_f1']:.4f}")
    print(f"  Features ({best['n_features']}): {best['indices']}")
    print(f"\nBaseline reference (CWF on denoised): "
          f"Mean F1 = {next(r['mean_f1'] for r in results if r['name'] == 'clot_wall_focused'):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
