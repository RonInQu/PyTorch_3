# train_cnn1d.py
"""
Train a 1D CNN (ResNet-style) directly on raw resistance waveforms.
No feature engineering — the network learns what matters from raw 750-sample windows.

Architecture:
  - Input normalization (per-window z-score inside the network)
  - 3 residual blocks with increasing channels (32→64→128)
  - Global average pooling → FC → 3-class softmax

Trained with GroupKFold CV, same data loading as train_gru_V6.
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
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
SAMPLE_RATE = 150
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 750
STRIDE_SAMPLES = 30

SEEDS_TO_TRY = [456]
BATCH_SIZE = 1024
N_EPOCHS = 100
PATIENCE = 15
LR = 0.001
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == 'cuda'
NUM_WORKERS = 0

DATA_DIR = PROJECT_ROOT / "training_data"
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "models"
CACHE_DIR = SCRIPT_DIR / "cache"

for d in [MODEL_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['blood', 'clot', 'wall']
CLINICAL_WEIGHTS = [1.0, 1.0, 1.0]


# ────────────────────────────────────────────────
# 1D ResNet Architecture
# ────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    """Residual block with two 1D convolutions + skip connection."""
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class ClotCNN1D(nn.Module):
    """
    1D ResNet for raw waveform classification.
    Input: (batch, 750) raw resistance values
    Output: (batch, 3) logits for [blood, clot, wall]
    """
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()

        # Initial conv: 1 channel → 32 channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),  # 750 → 375
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 375 → 187
        )

        # Residual blocks with downsampling
        self.block1 = ResBlock1D(32, 32, kernel_size=7)       # 187 → 187
        self.block2 = ResBlock1D(32, 64, kernel_size=7, stride=2)   # 187 → 94
        self.block3 = ResBlock1D(64, 128, kernel_size=5, stride=2)  # 94 → 47

        # Global average pooling → classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, 750)
        # Normalize per-sample: z-score each window
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        x = (x - mu) / sigma

        # Reshape to (batch, 1, 750) for Conv1d
        x = x.unsqueeze(1)

        x = self.input_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.gap(x).squeeze(-1)  # (batch, 128)
        return self.classifier(x)


# ────────────────────────────────────────────────
# Data Loading (raw windows, no feature extraction)
# ────────────────────────────────────────────────
def load_or_extract_windows(force_extract=False):
    cache_file = CACHE_DIR / f"raw_windows_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}.npz"

    if cache_file.exists() and not force_extract:
        print(f"Loading cached raw windows from: {cache_file.name}")
        data = np.load(cache_file)
        return data['X'], data['y'], data['groups']

    print("Extracting raw windows from training data...")
    data_files = sorted(DATA_DIR.glob("*_labeled_segment.parquet"))
    if not data_files:
        print(f"No parquet files in {DATA_DIR}")
        sys.exit(1)

    print(f"  {len(data_files)} files found")

    all_windows = []
    all_labels = []
    all_groups = []

    for file_path in data_files:
        run_id = file_path.stem
        df = pd.read_parquet(file_path, engine='pyarrow')

        if 'magRLoadAdjusted' not in df.columns or 'label' not in df.columns:
            continue

        resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
        labels = df['label'].to_numpy(dtype=np.int64)
        valid_mask = np.isin(labels, [0, 1, 2])

        if len(resistance) < WINDOW_SAMPLES:
            continue

        n_windows = 0
        for start in range(0, len(resistance) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            end = start + WINDOW_SAMPLES
            if not valid_mask[start:end].all():
                continue

            window = resistance[start:end]
            window_label = int(labels[start:end].max())

            all_windows.append(window)
            all_labels.append(window_label)
            all_groups.append(run_id)
            n_windows += 1

        print(f"  {run_id}: {n_windows} windows")

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    groups = np.array(all_groups)

    print(f"\nExtracted: {X.shape[0]} windows | shape={X.shape}")
    np.savez_compressed(cache_file, X=X, y=y, groups=groups)
    print(f"Cached → {cache_file.name}")

    return X, y, groups


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
        epoch_loss = 0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(yb.numpy())

        f1 = f1_score(trues, preds, average='macro')
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or patience_cnt == 0:
            print(f"    Epoch {epoch:3d} | loss={epoch_loss/n_batches:.4f} | "
                  f"F1={f1:.4f} | best={best_f1:.4f} | patience={patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    return best_state, best_f1


def main():
    print("=" * 60)
    print("  1D CNN (ResNet) on Raw Waveforms")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Window: {WINDOW_SEC}s ({WINDOW_SAMPLES} samples)")
    print(f"  Stride: {STRIDE_SAMPLES} samples")
    print()

    # Count parameters
    test_model = ClotCNN1D()
    n_params = sum(p.numel() for p in test_model.parameters())
    print(f"  Model parameters: {n_params:,}")
    del test_model

    # Load data
    X, y, groups = load_or_extract_windows()

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
    print(f"\nClass weights: blood={weights[0]:.2f} clot={weights[1]:.2f} wall={weights[2]:.2f}")

    best_overall_f1 = 0
    best_overall_seed = None

    for seed in SEEDS_TO_TRY:
        print(f"\n{'='*60}")
        print(f"  SEED = {seed}")
        print(f"{'='*60}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        gkf = GroupKFold(n_splits=3)
        fold_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"\n  Fold {fold_idx + 1}/3:")

            X_train = torch.tensor(X[train_idx], dtype=torch.float32)
            y_train = torch.tensor(y[train_idx], dtype=torch.long)
            X_val = torch.tensor(X[val_idx], dtype=torch.float32)
            y_val = torch.tensor(y[val_idx], dtype=torch.long)

            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

            model = ClotCNN1D().to(DEVICE)
            best_state, best_f1 = train_fold(model, train_loader, val_loader, class_weights)
            fold_f1s.append(best_f1)

        mean_f1 = np.mean(fold_f1s)
        print(f"\n  Seed {seed}: mean F1 = {mean_f1:.4f}  ({fold_f1s})")

        if mean_f1 > best_overall_f1:
            best_overall_f1 = mean_f1
            best_overall_seed = seed

    # ── Final training on all data ──
    print(f"\n{'='*60}")
    print(f"  Final training: seed={best_overall_seed} (CV F1={best_overall_f1:.4f})")
    print(f"{'='*60}")

    random.seed(best_overall_seed)
    np.random.seed(best_overall_seed)
    torch.manual_seed(best_overall_seed)

    X_all = torch.tensor(X, dtype=torch.float32)
    y_all = torch.tensor(y, dtype=torch.long)
    full_ds = TensorDataset(X_all, y_all)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = ClotCNN1D().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for xb, yb in full_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0:
            print(f"  Final epoch {epoch:3d} | loss={epoch_loss/n_batches:.4f}")

    model_path = MODEL_DIR / "cnn1d_trained.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved → {model_path}")
    print(f"Best CV F1-macro: {best_overall_f1:.4f}")


if __name__ == '__main__':
    main()
