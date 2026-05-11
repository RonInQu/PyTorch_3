# train_gru_scattering.py
"""
Train a GRU model on wavelet scattering features.
Same architecture as ClotGRU but with input_dim = SCATTERING_DIM (126).
Same training procedure as train_gru_V6.py.
"""

import os
import sys
import random
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
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scattering.scattering_features import (
    extract_scattering_features, SCATTERING_DIM, WINDOW_SAMPLES, WINDOW_SEC, SAMPLE_RATE
)

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
SEEDS_TO_TRY = [456]
SEQ_LEN = 8
STRIDE_SAMPLES = 30

BATCH_SIZE = 1024
N_EPOCHS = 100
PATIENCE = 15
LR = 0.0001
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # scattering extraction is CPU-bound; avoid multiprocess issues on Windows
PIN_MEMORY = False

# Paths
DATA_DIR = PROJECT_ROOT / "training_data"
SCALER_PATH = Path(__file__).resolve().parent / f"scattering_scaler_J6_Q8_{SCATTERING_DIM}f.pkl"
MODEL_DIR = Path(__file__).resolve().parent / "models"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

CLASS_NAMES = ['blood', 'clot', 'wall']
CLINICAL_WEIGHTS = [1.0, 1.0, 1.0]

# ────────────────────────────────────────────────
# GRU Model (same architecture as ClotGRU, different input_dim)
# ────────────────────────────────────────────────
class ScatteringGRU(nn.Module):
    def __init__(self, input_dim=SCATTERING_DIM, hidden_dim=64, num_layers=1,
                 fc_dim=32, num_classes=3, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        logits = self.classifier(out[:, -1, :])
        return logits, hidden


# ────────────────────────────────────────────────
# Feature extraction with caching
# ────────────────────────────────────────────────
def load_or_extract_features(force_extract=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"scattering_seqs_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}.npz"

    if cache_file.exists() and not force_extract:
        print(f"Loading cached scattering features from: {cache_file}")
        data = np.load(cache_file)
        return data['X_seq'], data['y'].astype(np.int64), data['groups']

    print("Extracting scattering features from training data...")
    data_files = sorted(DATA_DIR.glob("*.parquet"))
    if not data_files:
        print(f"No parquet files in {DATA_DIR}")
        sys.exit(1)

    print(f"  {len(data_files)} files found")

    seq_list = []
    labels_list = []
    groups_list = []

    for file_path in data_files:
        run_id = file_path.stem
        df = pd.read_parquet(file_path, engine='pyarrow')

        if 'magRLoadAdjusted' not in df.columns or 'label' not in df.columns:
            continue

        resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
        label_array = df['label'].to_numpy(dtype=np.int64)
        valid_mask = np.isin(label_array, [0, 1, 2])

        if len(resistance) < WINDOW_SAMPLES:
            continue

        # Extract per-window scattering features
        run_features = []
        run_labels = []

        for start in range(0, len(resistance) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            end = start + WINDOW_SAMPLES
            if not valid_mask[start:end].all():
                continue

            window = resistance[start:end]
            feats = extract_scattering_features(window)
            if feats is not None and len(feats) == SCATTERING_DIM:
                run_features.append(feats)
                window_label = int(label_array[start:end].max())
                run_labels.append(window_label)

        # Build sequences of length SEQ_LEN
        for i in range(SEQ_LEN - 1, len(run_features)):
            seq = np.array(run_features[i - SEQ_LEN + 1: i + 1])
            seq_list.append(seq)
            labels_list.append(run_labels[i])
            groups_list.append(run_id)

        if len(run_features) > 0:
            print(f"  {run_id}: {len(run_features)} windows → "
                  f"{max(0, len(run_features) - SEQ_LEN + 1)} sequences")

    X_seq = np.array(seq_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    groups = np.array(groups_list)

    print(f"\nExtracted: {X_seq.shape[0]} sequences | shape={X_seq.shape}")
    np.savez_compressed(cache_file, X_seq=X_seq, y=y, groups=groups)
    print(f"Cached → {cache_file}")

    return X_seq, y, groups


# ────────────────────────────────────────────────
# Training
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

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits, _ = model(xb, None)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(yb.numpy())

        f1 = f1_score(trues, preds, average='macro')
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or patience_cnt == 0:
            print(f"    Epoch {epoch:3d} | F1={f1:.4f} | best={best_f1:.4f} | patience={patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    return best_state, best_f1


def main():
    print(f"=== Scattering GRU Training ===")
    print(f"  Device: {DEVICE}")
    print(f"  Scattering dim: {SCATTERING_DIM}")
    print(f"  SEQ_LEN: {SEQ_LEN}")
    print(f"  Window: {WINDOW_SEC}s, Stride: {STRIDE_SAMPLES}")
    print()

    # Load or extract features
    X_seq, y, groups = load_or_extract_features()

    # Scale
    if not SCALER_PATH.exists():
        print(f"ERROR: Scaler not found at {SCALER_PATH}")
        print("Run fit_scaler_scattering.py first.")
        sys.exit(1)

    scaler = joblib.load(SCALER_PATH)
    N, S, F = X_seq.shape
    X_scaled = scaler.transform(X_seq.reshape(-1, F)).reshape(N, S, F)
    print(f"Scaled: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {CLASS_NAMES[cls]}: {cnt} ({cnt/len(y)*100:.1f}%)")

    # Class weights
    balanced = compute_class_weight('balanced', classes=unique, y=y)
    clinical = np.array(CLINICAL_WEIGHTS)
    weights = np.where(clinical > 1.0, clinical, balanced)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"\nClass weights: {weights}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_overall_f1 = 0
    best_overall_seed = None

    for seed in SEEDS_TO_TRY:
        print(f"\n{'='*60}")
        print(f"  SEED = {seed}")
        print(f"{'='*60}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # GroupKFold (3 folds)
        gkf = GroupKFold(n_splits=5)
        fold_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
            print(f"\n  Fold {fold_idx + 1}/3:")
            X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
            y_train = torch.tensor(y[train_idx], dtype=torch.long)
            X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
            y_val = torch.tensor(y[val_idx], dtype=torch.long)

            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

            model = ScatteringGRU(input_dim=SCATTERING_DIM).to(DEVICE)
            best_state, best_f1 = train_fold(model, train_loader, val_loader, class_weights)
            fold_f1s.append(best_f1)

        mean_f1 = np.mean(fold_f1s)
        print(f"\n  Seed {seed}: mean F1 across folds = {mean_f1:.4f} ({fold_f1s})")

        if mean_f1 > best_overall_f1:
            best_overall_f1 = mean_f1
            best_overall_seed = seed

    # Final training on all data with best seed
    print(f"\n{'='*60}")
    print(f"  Final training with seed={best_overall_seed} (F1={best_overall_f1:.4f})")
    print(f"{'='*60}")

    random.seed(best_overall_seed)
    np.random.seed(best_overall_seed)
    torch.manual_seed(best_overall_seed)

    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    y_all = torch.tensor(y, dtype=torch.long)
    full_ds = TensorDataset(X_all, y_all)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = ScatteringGRU(input_dim=SCATTERING_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for xb, yb in full_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, _ = model(xb, None)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save model
    model_path = MODEL_DIR / "scattering_gru_trained.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved → {model_path}")
    print(f"Best cross-val F1-macro: {best_overall_f1:.4f}")


if __name__ == '__main__':
    main()
