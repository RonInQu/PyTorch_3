"""
train_dn.py — Train GRU on denoised features.

Same architecture, hyperparams, and logic as train_gru_V6.py but reads from
experiments/denoised/ paths and saves model to experiments/denoised/models/.
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import from existing codebase ──
from src.models.gru_torch_V6 import (
    ClotFeatureExtractor, ClotGRU,
    FEATURE_SET, SEQ_LEN, WINDOW_SEC,
    active_idx, active_dim, dim_str,
)

# ── Configuration (same as train_gru_V6.py) ──
SEEDS_TO_TRY = [456]
STRIDE_SAMPLES = 30
BATCH_SIZE = 1024
N_EPOCHS = 100
PATIENCE = 15
LR = 0.0001
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
PIN_MEMORY = True

# ── Denoised experiment paths ──
DATA_DIR = EXPERIMENT_DIR / "train_data"
SCALER_PATH = EXPERIMENT_DIR / "models" / "scaler_denoised.pkl"
CACHE_DIR = EXPERIMENT_DIR / "cache"
MODELS_DIR = EXPERIMENT_DIR / "models"

CLASS_NAMES = ['blood', 'clot', 'wall']
CLINICAL_WEIGHTS = [1.0, 1.0, 1.0]

# ── Extractor + lfilter setup ──
_extractor = ClotFeatureExtractor(sample_rate=150, window_sec=WINDOW_SEC,
                                  active_features=active_idx)
WINDOW_SAMPLES = _extractor.window_size
ALPHA_FAST = _extractor.alpha_fast
ALPHA_SLOW = _extractor.alpha_slow

_B_FAST = np.array([ALPHA_FAST])
_A_FAST = np.array([1.0, -(1.0 - ALPHA_FAST)])
_B_SLOW = np.array([ALPHA_SLOW])
_A_SLOW = np.array([1.0, -(1.0 - ALPHA_SLOW)])


# ────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────

def print_cm_text(y_true, y_pred, title, class_names=CLASS_NAMES):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title}")
    print(" " * 10 + " ".join(f"{name:>8}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<10}" + " ".join(f"{v:8}" for v in row))


# ────────────────────────────────────────────────
# Cached Feature Extraction
# ────────────────────────────────────────────────

def load_or_extract_features(force_extract=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_filename = f"features_denoised_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}_{FEATURE_SET}.npz"
    CACHE_FILE = CACHE_DIR / cache_filename

    if CACHE_FILE.exists() and not force_extract:
        print(f"Loading cached features from: {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        X_seq = data['X_seq']
        y = data['y'].astype(np.int64)
        groups = data['groups']
        print(f"Loaded: {X_seq.shape[0]} sequences | shape={X_seq.shape}")
    else:
        print("Extracting features from denoised data...")
        data_files = sorted(DATA_DIR.glob("*.parquet"))
        if not data_files:
            print(f"No parquet files in {DATA_DIR}")
            sys.exit(1)

        all_data = [pd.read_parquet(f).assign(run_id=f.stem) for f in data_files]
        df_all = pd.concat(all_data, ignore_index=True)

        seq_list = []
        labels_list = []
        groups_list = []

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

            run_features = []
            run_labels = []

            for idx in extraction_indices:
                win_start = idx - WINDOW_SAMPLES + 1

                if not valid_mask[win_start : idx + 1].all():
                    continue

                window_data = resistance[win_start : idx + 1]

                feats = _extractor.compute_features_from_array(
                    window_data, float(ema_f_all[idx]), float(ema_s_all[idx]))
                if feats is not None and len(feats) == active_dim:
                    run_features.append(feats)
                    window_label = int(label_array[win_start : idx + 1].max())
                    run_labels.append(window_label)

            for i in range(SEQ_LEN - 1, len(run_features)):
                seq = np.array(run_features[i - SEQ_LEN + 1 : i + 1])
                seq_list.append(seq)
                labels_list.append(run_labels[i])
                groups_list.append(run_id)

        X_seq = np.array(seq_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        groups = np.array(groups_list)

        print(f"Extracted: {X_seq.shape[0]} sequences | shape={X_seq.shape} from {len(np.unique(groups))} runs")
        np.savez_compressed(CACHE_FILE, X_seq=X_seq, y=y, groups=groups)
        print(f"Saved cache → {CACHE_FILE}")

    # Scaling
    print(f"Loading scaler: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    N, S, F = X_seq.shape
    X_flat = X_seq.reshape(-1, F)
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled_flat = np.nan_to_num(X_scaled_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = X_scaled_flat.reshape(N, S, F)

    assert np.abs(X_scaled.mean()) < 0.1, f"Scaling failed — mean={X_scaled.mean():.4f}"
    assert 0.6 < X_scaled.std() < 1.4, f"Scaling failed — std={X_scaled.std():.4f}"
    print("Scaling check passed")

    return X_scaled, y, groups, scaler


# ────────────────────────────────────────────────
# Class Weights
# ────────────────────────────────────────────────

def compute_final_class_weights(y):
    unique = np.unique(y)
    balanced = compute_class_weight('balanced', classes=unique, y=y)
    w_dict = dict(zip(unique, balanced))

    weights = np.ones(3)
    for cls, w in w_dict.items():
        weights[cls] = w

    clinical = np.array(CLINICAL_WEIGHTS)
    final_w = np.where(clinical > 1.0, clinical, weights)

    print(f"Class weights: blood={final_w[0]:.2f} clot={final_w[1]:.2f} wall={final_w[2]:.2f}")
    return torch.tensor(final_w, dtype=torch.float32)


# ────────────────────────────────────────────────
# Training fold
# ────────────────────────────────────────────────

def train_fold(model, train_loader, val_loader, class_weights):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    best_f1 = 0
    best_state = None
    patience_cnt = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, _ = model(xb, None)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits, _ = model(xb.to(DEVICE), None)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(yb.cpu().numpy())

        f1_macro = f1_score(trues, preds, average='macro', zero_division=0)
        scheduler.step(f1_macro)

        print(f"Epoch {epoch:3d} | F1 {f1_macro:.4f}", end="")
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_state = model.state_dict().copy()
            patience_cnt = 0
            print("  ← best")
        else:
            patience_cnt += 1
            print("")

        if patience_cnt >= PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    return best_state, best_f1


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING — DENOISED EXPERIMENT")
    print("=" * 60)
    print(f"  Device:        {DEVICE}")
    print(f"  Feature set:   {FEATURE_SET} ({active_dim} features)")
    print(f"  SEQ_LEN:       {SEQ_LEN}")
    print(f"  Seeds:         {SEEDS_TO_TRY}")
    print(f"  Data dir:      {DATA_DIR}")
    print(f"  Scaler:        {SCALER_PATH}")
    print("-" * 60)

    force_extract = "--force-extract" in sys.argv or "-f" in sys.argv
    X_scaled, y, groups, scaler = load_or_extract_features(force_extract=force_extract)

    # Drop unlabeled sequences
    valid = (y != -1)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} unlabeled sequences")
        X_scaled, y, groups = X_scaled[valid], y[valid], groups[valid]

    class_weights = compute_final_class_weights(y)

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    print(f"\nGroupKFold: {n_splits} splits on {n_groups} groups")

    gkf = GroupKFold(n_splits=n_splits)

    best_global_f1 = 0.0
    best_state_global = None
    best_seed = None

    for seed in SEEDS_TO_TRY:
        print(f"\n{'='*60}")
        print(f"Seed = {seed}")
        print(f"{'='*60}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        best_f1_this_seed = 0.0
        best_state_this_seed = None

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups), 1):
            print(f"\nFold {fold}/{n_splits}")

            tr_X, va_X = X_scaled[train_idx], X_scaled[val_idx]
            tr_y, va_y = y[train_idx], y[val_idx]

            # Mild clot oversampling (same as baseline)
            clot_idx = np.where(tr_y == 1)[0]
            if len(clot_idx) > 0:
                tr_X = np.concatenate([tr_X, tr_X[clot_idx]])
                tr_y = np.concatenate([tr_y, tr_y[clot_idx]])

            train_ds = TensorDataset(torch.from_numpy(tr_X).float(), torch.from_numpy(tr_y).long())
            val_ds   = TensorDataset(torch.from_numpy(va_X).float(), torch.from_numpy(va_y).long())

            g = torch.Generator().manual_seed(seed + fold)

            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                  persistent_workers=NUM_WORKERS > 0, generator=g)
            val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                  persistent_workers=NUM_WORKERS > 0)

            model = ClotGRU().to(DEVICE)
            state, fold_f1 = train_fold(model, train_dl, val_dl, class_weights)

            if fold_f1 > best_f1_this_seed:
                best_f1_this_seed = fold_f1
                best_state_this_seed = state

        print(f"\nSeed {seed} best F1-macro: {best_f1_this_seed:.4f}")

        if best_f1_this_seed > best_global_f1:
            best_global_f1 = best_f1_this_seed
            best_state_global = best_state_this_seed
            best_seed = seed

    # Save best model
    if best_state_global is not None:
        model_path = MODELS_DIR / "clot_gru_denoised.pt"
        torch.save(best_state_global, model_path)
        print(f"\n✅ Model saved → {model_path}")
        print(f"   Best F1-macro: {best_global_f1:.4f} (seed {best_seed})")

        # Also save with F1 in filename
        detailed_path = MODELS_DIR / f"clot_gru_denoised_seed{best_seed}_f1{best_global_f1:.4f}.pt"
        torch.save(best_state_global, detailed_path)

    print("\n" + "=" * 60)
    print(f"DONE — Best F1-macro: {best_global_f1:.4f} (Seed {best_seed})")
    print("=" * 60)


if __name__ == "__main__":
    main()
