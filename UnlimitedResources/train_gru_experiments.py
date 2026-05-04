"""
Experiment runner for spike-vs-sustained ML architecture exploration.
Run with: python train_gru_experiments.py --exp A|B|C|D

Experiments:
  A: Train only on events ≥ 5s (filter spikes from training)
  B: SEQ_LEN=16 (more temporal context, same window)
  C: SEQ_LEN=32 (even more context)
  D: Larger model (hidden=128, 2-layer GRU) — PC-only, no M4 constraint
"""
import os, sys, argparse
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument('--exp', required=True, choices=['A', 'B', 'C', 'D', 'E'])
args = parser.parse_args()

# ═══════════════════════════════════════════
# EXPERIMENT CONFIGS
# ═══════════════════════════════════════════
EXPERIMENTS = {
    'A': {
        'desc': 'Train on sustained events only (≥5s)',
        'SEQ_LEN': 8, 'WINDOW_SEC': 5.0, 'HIDDEN': 32, 'N_LAYERS': 1,
        'MIN_EVENT_DURATION_SEC': 5.0,  # Filter: skip windows from events <5s
    },
    'B': {
        'desc': 'Longer sequence (SEQ_LEN=16)',
        'SEQ_LEN': 16, 'WINDOW_SEC': 5.0, 'HIDDEN': 32, 'N_LAYERS': 1,
        'MIN_EVENT_DURATION_SEC': None,
    },
    'C': {
        'desc': 'Very long sequence (SEQ_LEN=32)',
        'SEQ_LEN': 32, 'WINDOW_SEC': 5.0, 'HIDDEN': 32, 'N_LAYERS': 1,
        'MIN_EVENT_DURATION_SEC': None,
    },
    'D': {
        'desc': 'Large model (hidden=128, 2-layer GRU) — PC only',
        'SEQ_LEN': 8, 'WINDOW_SEC': 5.0, 'HIDDEN': 128, 'N_LAYERS': 2,
        'MIN_EVENT_DURATION_SEC': None,
    },
    'E': {
        'desc': 'Best combo: sustained-only + SEQ_LEN=16 + large model',
        'SEQ_LEN': 16, 'WINDOW_SEC': 5.0, 'HIDDEN': 128, 'N_LAYERS': 2,
        'MIN_EVENT_DURATION_SEC': 5.0,
    },
}

cfg = EXPERIMENTS[args.exp]
print(f"{'='*70}")
print(f"EXPERIMENT {args.exp}: {cfg['desc']}")
print(f"{'='*70}")
print(f"  SEQ_LEN       = {cfg['SEQ_LEN']}")
print(f"  WINDOW_SEC    = {cfg['WINDOW_SEC']}")
print(f"  HIDDEN        = {cfg['HIDDEN']}")
print(f"  N_LAYERS      = {cfg['N_LAYERS']}")
print(f"  MIN_DURATION  = {cfg['MIN_EVENT_DURATION_SEC']}")
print()

import random
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

from src.models.gru_torch_V6 import ClotFeatureExtractor, \
    FEATURE_SET, active_idx, active_dim, dim_str

# ═══════════════════════════════════════════
# EXPERIMENT-SPECIFIC MODEL
# ═══════════════════════════════════════════
class ClotGRU_Exp(nn.Module):
    def __init__(self, input_size=active_dim, hidden_size=32, output_size=3, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=n_layers,
                          batch_first=True, dropout=0.1 if n_layers > 1 else 0)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        out = out[:, -1]
        out = torch.relu(self.fc1(out))
        logits = self.fc2(out)
        return logits, hidden

# ═══════════════════════════════════════════
# FEATURE EXTRACTION WITH DURATION FILTER
# ═══════════════════════════════════════════
SAMPLE_RATE = 150
STRIDE_SAMPLES = 30
WINDOW_SAMPLES = int(cfg['WINDOW_SEC'] * SAMPLE_RATE)
SEQ_LEN = cfg['SEQ_LEN']
MIN_DUR = cfg['MIN_EVENT_DURATION_SEC']

DATA_DIR = PROJECT_ROOT / "training_data"
SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_seq8_{dim_str}.pkl"
CACHE_DIR = PROJECT_ROOT / "cache"

_extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=cfg['WINDOW_SEC'],
                                  active_features=active_idx)
ALPHA_FAST = _extractor.alpha_fast
ALPHA_SLOW = _extractor.alpha_slow
_B_FAST = np.array([ALPHA_FAST])
_A_FAST = np.array([1.0, -(1.0 - ALPHA_FAST)])
_B_SLOW = np.array([ALPHA_SLOW])
_A_SLOW = np.array([1.0, -(1.0 - ALPHA_SLOW)])


def compute_event_duration_mask(labels, min_duration_sec):
    """Return boolean mask: True for samples that belong to events ≥ min_duration."""
    if min_duration_sec is None:
        return np.ones(len(labels), dtype=bool)

    min_samples = int(min_duration_sec * SAMPLE_RATE)
    mask = np.zeros(len(labels), dtype=bool)

    # Blood is always included (label=0)
    mask[labels == 0] = True

    # For clot(1) and wall(2), only include events ≥ min_duration
    for cls in [1, 2]:
        cls_mask = (labels == cls).astype(int)
        diff = np.diff(np.concatenate([[0], cls_mask, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) >= min_samples:
                mask[s:e] = True

    return mask


def extract_features_exp():
    """Extract features with optional duration filtering."""
    cache_name = f"features_exp{args.exp}_w{cfg['WINDOW_SEC']:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}_{FEATURE_SET}.npz"
    cache_path = CACHE_DIR / cache_name

    if cache_path.exists():
        print(f"Loading cached: {cache_path}")
        data = np.load(cache_path)
        return data['X_seq'], data['y'].astype(np.int64), data['groups']

    print("Extracting features...")
    data_files = sorted(DATA_DIR.glob("*.parquet"))

    seq_list, labels_list, groups_list = [], [], []

    for fpath in data_files:
        df = pd.read_parquet(fpath)
        resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
        label_array = df['label'].values.astype(np.int64)
        run_id = fpath.stem
        valid_mask = np.isin(label_array, [0, 1, 2])

        # Duration filter
        dur_mask = compute_event_duration_mask(label_array, MIN_DUR)
        combined_mask = valid_mask & dur_mask

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
            if not combined_mask[win_start:idx+1].all():
                continue
            window_data = resistance[win_start:idx+1]
            feats = _extractor.compute_features_from_array(
                window_data, float(ema_f_all[idx]), float(ema_s_all[idx]))
            if feats is not None and len(feats) == active_dim:
                run_features.append(feats)
                run_labels.append(int(label_array[win_start:idx+1].max()))

        for i in range(SEQ_LEN - 1, len(run_features)):
            seq = np.array(run_features[i - SEQ_LEN + 1:i + 1])
            seq_list.append(seq)
            labels_list.append(run_labels[i])
            groups_list.append(run_id)

    X_seq = np.array(seq_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    groups = np.array(groups_list)

    print(f"Extracted: {X_seq.shape[0]} sequences | shape={X_seq.shape}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X_seq=X_seq, y=y, groups=groups)
    return X_seq, y, groups


# ═══════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
N_EPOCHS = 100
PATIENCE = 15
LR = 0.0001
SEED = 456

def main():
    X_seq, y, groups = extract_features_exp()

    # Scale
    scaler = joblib.load(SCALER_PATH)
    N, S, F = X_seq.shape
    X_scaled = scaler.transform(X_seq.reshape(-1, F)).reshape(N, S, F)

    valid = (y != -1)
    X_scaled, y, groups = X_scaled[valid], y[valid], groups[valid]

    # Class weights
    unique = np.unique(y)
    balanced = compute_class_weight('balanced', classes=unique, y=y)
    weights = np.ones(3)
    for cls, w in zip(unique, balanced):
        weights[cls] = w
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print(f"\nData: {len(y)} sequences, {len(np.unique(groups))} groups")
    for c in [0,1,2]:
        print(f"  Class {c}: {(y==c).sum()} ({100*(y==c).mean():.1f}%)")

    # Train
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    best_f1 = 0; best_state = None

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups), 1):
        print(f"\nFold {fold}/{n_splits}")
        tr_X, va_X = X_scaled[train_idx], X_scaled[val_idx]
        tr_y, va_y = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.from_numpy(tr_X).float(), torch.from_numpy(tr_y).long())
        val_ds = TensorDataset(torch.from_numpy(va_X).float(), torch.from_numpy(va_y).long())
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = ClotGRU_Exp(hidden_size=cfg['HIDDEN'], n_layers=cfg['N_LAYERS']).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        fold_best_f1 = 0; fold_best_state = None; patience_cnt = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits, _ = model(xb, None)
                loss = criterion(logits, yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    logits, _ = model(xb.to(DEVICE), None)
                    preds.extend(logits.argmax(1).cpu().numpy())
                    trues.extend(yb.numpy())

            f1 = f1_score(trues, preds, average='macro', zero_division=0)
            scheduler.step(f1)

            if f1 > fold_best_f1:
                fold_best_f1 = f1; fold_best_state = model.state_dict().copy(); patience_cnt = 0
                if epoch % 10 == 0 or epoch <= 5:
                    print(f"  Epoch {epoch:3d} | F1 {f1:.4f} ← best")
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch} | Best F1 {fold_best_f1:.4f}")
                break

        if fold_best_f1 > best_f1:
            best_f1 = fold_best_f1; best_state = fold_best_state

    # Save
    save_path = PROJECT_ROOT / "src" / "training" / f"clot_gru_exp{args.exp}_{FEATURE_SET}_f1{best_f1:.4f}.pt"
    torch.save(best_state, save_path)
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {args.exp} COMPLETE")
    print(f"Best validation F1-macro: {best_f1:.4f}")
    print(f"Saved: {save_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()