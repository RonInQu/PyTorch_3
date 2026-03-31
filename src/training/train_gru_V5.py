# train_gru_V5.py
"""
Training script for clot detection — V5 (GRU on features, REDUCE_DIM compatible)
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

import joblib
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix

# Add project root to Python path so "src" can be found
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # goes up from src/data/ → PyTorch_3
sys.path.insert(0, str(PROJECT_ROOT))

# Import from gru_torch_V5 (single source of truth)
from src.models.gru_torch_V5 import ClotFeatureExtractor, ClotGRU, REDUCE_DIM

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

# SEEDS_TO_TRY = [456, 123, 42]        # ← Add more seeds here if desired
SEEDS_TO_TRY = [456, 127]

WINDOW_SEC = 5.0
STRIDE_SAMPLES = 30

BATCH_SIZE = 1024
N_EPOCHS = 100
PATIENCE = 15
LR = 0.0001
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For DataLoader
NUM_WORKERS = 8
PIN_MEMORY = True

# Reproducibility will be set per seed
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "training_data"
TEST_DIR = PROJECT_ROOT / "test_data"

# Dynamic scaler path (matches fit_scaler_V5 and gru_torch_V5)
active_dim = 40 - len(ClotFeatureExtractor().zero_idx) if REDUCE_DIM else 40
dim_str = f"red{active_dim}" if REDUCE_DIM else "40"

SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_{dim_str}.pkl"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_FILE = CACHE_DIR / f"features_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_red{REDUCE_DIM}.npz"

CLASS_NAMES = ['blood', 'clot', 'wall']
CLINICAL_WEIGHTS = [1.0, 1.0, 1.0]

# ────────────────────────────────────────────────
# Utility Functions (from your V3)
# ────────────────────────────────────────────────

def set_print_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


def print_cm_text(y_true, y_pred, title, class_names=CLASS_NAMES, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title}")
    print(" " * 10 + " ".join(f"{name:>8}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<10}" + " ".join(f"{v:8}" for v in row))

    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"\nNormalized {title}:")
        print(" " * 10 + " ".join(f"{name:>8}" for name in class_names))
        for i, row in enumerate(cm_norm):
            print(f"{class_names[i]:<10}" + " ".join(f"{v:8.3f}" for v in row))


def print_label_stats_table(y_true, y_pred, title):
    class_names = ['Bld', 'Clt', 'Wall']
    print(f"\n{title}")
    print("-" * 82)
    print(f"{'Label':<6} {'TP':>6} {'FN':>6} {'FP':>6} {'TN':>6} {'Prec':>8} {'Sens':>8} {'Spec':>8} {'#Pos':>8} {'#Neg':>8}")
    print("-" * 82)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for i, label in enumerate(class_names):
        tp = np.sum((y_true == i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        tn = np.sum((y_true != i) & (y_pred != i))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        pos = tp + fn
        neg = tn + fp

        print(f"{label:<6} {tp:6d} {fn:6d} {fp:6d} {tn:6d} {prec:8.2f} {sens:8.2f} {spec:8.2f} {pos:8d} {neg:8d}")

    print("-" * 82)


# ────────────────────────────────────────────────
# Cached Feature Extraction (updated for REDUCE_DIM)
# ────────────────────────────────────────────────

def load_or_extract_features(force_extract: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not force_extract:
        print(f"Loading cached features from: {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        X_full = data['X_full']      # always save full 40-dim
        y = data['y']
        groups = data['groups']
        print(f"Loaded: {X_full.shape[0]} windows")
    else:
        print("Extracting features (this may take a while)...")
        data_files = sorted(DATA_DIR.glob("*.parquet"))
        all_data = [pd.read_parquet(f).assign(run_id=f.stem) for f in data_files]
        df_all = pd.concat(all_data, ignore_index=True)

        extractor = ClotFeatureExtractor(sample_rate=150, window_sec=WINDOW_SEC)
        features_list, labels_list, groups_list = [], [], []

        for run_id, group in df_all.groupby('run_id'):
            resistance = group['magRLoadAdjusted'].to_numpy(dtype=np.float32)
            labels = group['label'].values
            extractor.reset()

            window_samples = extractor.window_size
            step = STRIDE_SAMPLES
            for start in range(0, len(resistance) - window_samples + 1, step):
                window_res = resistance[start:start + window_samples]
                window_label = np.max(labels[start:start + window_samples])

                for r in window_res:
                    extractor.update(r)

                feats_40 = extractor.compute_features()
                features_list.append(feats_40)
                labels_list.append(window_label)
                groups_list.append(run_id)

        X_full = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        groups = np.array(groups_list)

        print(f"Extracted: {X_full.shape[0]} windows from {len(np.unique(groups))} runs")

        print("Caching features...")
        np.savez_compressed(
            CACHE_FILE,
            X_full=X_full,
            y=y,
            groups=groups
        )
        print(f"Saved cache → {CACHE_FILE}")

    # Drop unused features BEFORE scaling
    if REDUCE_DIM:
        zero_idx = ClotFeatureExtractor().zero_idx
        active_idx = [i for i in range(40) if i not in zero_idx]
        X = X_full[:, active_idx]
        print(f"Dropped {len(zero_idx)} unused features → training on {X.shape[1]} active features")
    else:
        X = X_full

    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    assert np.abs(X_scaled.mean()) < 0.1, "Scaling failed — mean should be ~0"
    assert 0.6 < X_scaled.std() < 1.3, "Scaling failed — std should be ~1"
    print("Scaling check passed")

    return X_scaled, y, groups, scaler


# ────────────────────────────────────────────────
# Class Weights
# ────────────────────────────────────────────────

def compute_final_class_weights(y):
    print("Computing class weights...")
    unique = np.unique(y)
    balanced = compute_class_weight('balanced', classes=unique, y=y)
    w_dict = dict(zip(unique, balanced))

    weights = np.ones(3)
    for cls, w in w_dict.items():
        weights[cls] = w

    clinical = np.array(CLINICAL_WEIGHTS)
    final_w = np.where(clinical > 1.0, clinical, weights)

    print(f"APPLIED weights: blood={final_w[0]:.2f} clot={final_w[1]:.2f} wall={final_w[2]:.2f}")
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

        acc = (np.array(preds) == trues).mean()
        f1_macro = f1_score(trues, preds, average='macro', zero_division=0)
        scheduler.step(f1_macro)

        print(f"Epoch {epoch:3d} | Acc {acc:.4f} | F1 {f1_macro:.4f}", end="")
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_state = model.state_dict().copy()
            patience_cnt = 0
            print("  ← new best")
        else:
            patience_cnt += 1
            print("")

        if patience_cnt >= PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    return best_state, best_f1


# ────────────────────────────────────────────────
# Main — Multiple Seeds Loop
# ────────────────────────────────────────────────

def main():
    set_print_options()
    
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Device               : {DEVICE}")
    print(f"REDUCE_DIM           : {REDUCE_DIM}")
    print(f"Active features      : {active_dim}")
    print(f"SEEDS_TO_TRY         : {SEEDS_TO_TRY}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Num workers          : {NUM_WORKERS}")
    print(f"Patience             : {PATIENCE}")
    print("-" * 70)

    force_extract = "--force-extract" in sys.argv or "-f" in sys.argv
    if force_extract:
        print("Force re-extraction requested.")

    X_scaled, y, groups, scaler = load_or_extract_features(force_extract=force_extract)
    class_weights = compute_final_class_weights(y)

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    print(f"\nGroupKFold: {n_splits} splits on {n_groups} groups")

    gkf = GroupKFold(n_splits=n_splits)

    best_global_f1 = 0
    best_state_global = None
    best_seed = None

    for seed in SEEDS_TO_TRY:
        print(f"\n{'='*60}")
        print(f"Training with SEED = {seed}")
        print(f"{'='*60}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        best_f1_this_seed = 0
        best_state_this_seed = None

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups), 1):
            print(f"\nFold {fold}/{n_splits}")

            tr_X, va_X = X_scaled[train_idx], X_scaled[val_idx]
            tr_y, va_y = y[train_idx], y[val_idx]

            # ── Mild oversampling / repetition of rare classes ────────────────
            clot_idx = np.where(tr_y == 1)[0]
            wall_idx = np.where(tr_y == 2)[0]
    
            repeat_clot = 1   # total 2×
            repeat_wall = 0   # total 1× (disabled)
    
            if len(clot_idx) > 0 and repeat_clot > 0:
                tr_X = np.concatenate([tr_X] + [tr_X[clot_idx]] * repeat_clot)
                tr_y = np.concatenate([tr_y] + [tr_y[clot_idx]] * repeat_clot)
    
            if len(wall_idx) > 0 and repeat_wall > 0:
                tr_X = np.concatenate([tr_X] + [tr_X[wall_idx]] * repeat_wall)
                tr_y = np.concatenate([tr_y] + [tr_y[wall_idx]] * repeat_wall)

            train_ds = TensorDataset(torch.from_numpy(tr_X).float(), torch.from_numpy(tr_y).long())
            val_ds = TensorDataset(torch.from_numpy(va_X).float(), torch.from_numpy(va_y).long())

            train_dl = DataLoader(
                train_ds, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                num_workers=NUM_WORKERS,       # ← Start with 4 or 8 (match your CPU cores / 2)
                pin_memory=PIN_MEMORY,         # Helps even on CPU
                persistent_workers=True if NUM_WORKERS > 0 else False
                )
            
            val_dl = DataLoader(
                val_ds, 
                batch_size=BATCH_SIZE, 
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=True if NUM_WORKERS > 0 else False
                )
            
            model = ClotGRU().to(DEVICE)
            state, fold_f1 = train_fold(model, train_dl, val_dl, class_weights)

            if fold_f1 > best_f1_this_seed:
                best_f1_this_seed = fold_f1
                best_state_this_seed = state

        print(f"\nSeed {seed} best F1-macro: {best_f1_this_seed:.4f}")

        # ====================== SAVE EVERY SEED ======================
        if best_state_this_seed is not None:
            model_filename = f"clot_gru_trained_seed{seed}_f1{best_f1_this_seed:.4f}.pt"
            save_path = PROJECT_ROOT / "src" / "training" / model_filename
            
            torch.save(best_state_this_seed, save_path)
            
            print(f"   ✅ Saved model for seed {seed}")
            print(f"      Filename: {model_filename}")
            print(f"      F1-macro: {best_f1_this_seed:.4f}")
            print(f"      Path: {save_path}")
        else:
            print(f"   ⚠️  No model saved for seed {seed} (best_state was None)")

        # Update global best (still useful for reference)
        if best_f1_this_seed > best_global_f1:
            best_global_f1 = best_f1_this_seed
            best_state_global = best_state_this_seed
            best_seed = seed
            print(f"   → New global best! (Seed {seed})")

    # Optional: also save the overall best as a simple name for quick inference
    if best_state_global is not None:
        latest_path = PROJECT_ROOT / "src" / "training" / "clot_gru_trained.pt"
        torch.save(best_state_global, latest_path)
        print(f"\n✅ Also saved overall best as: clot_gru_trained.pt")

    print("\n" + "="*70)
    print("ALL SEEDS FINISHED")
    print("="*70)
    print(f"Global best F1-macro: {best_global_f1:.4f} (Seed {best_seed})")

if __name__ == "__main__":
    main()