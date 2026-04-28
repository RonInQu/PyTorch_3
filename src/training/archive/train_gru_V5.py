# train_gru_V5.py
"""
Training script for clot detection — V5 (GRU on features, REDUCE_DIM compatible)
"""

import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend - prevents Tkinter crash

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

import joblib
from scipy.signal import lfilter
from scipy import stats as sp_stats
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix

# Add project root to Python path so "src" can be found
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # goes up from src/data/ → PyTorch_3
sys.path.insert(0, str(PROJECT_ROOT))

# Import from gru_torch_V5 (single source of truth)
from src.models.gru_torch_V5 import ClotFeatureExtractor, ClotGRU, \
    FEATURE_SET, SEQ_LEN, WINDOW_SEC, \
    active_idx, active_dim, dim_str

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

"""
To set up the different configurations:
Config	                STRIDE_SAMPLES (train)	REPORT_INTERVAL_MS (inference)
Current	                      30	                      200
Option A — wider training	  150	                      200
Option B — matched at 500ms	  75	                      500

The reporting interval (200–500ms) and the training stride don't have to match. They serve different purposes:

Reporting interval = how often inference makes a prediction. 200ms is good for real-time responsiveness.
Training stride = how much feature evolution the GRU learns to interpret between timesteps.
Recommended Approach
Train with a larger stride (e.g., 150 samples = 1s) to teach the GRU meaningful temporal patterns. 
Then at inference, keep reporting every 200ms — the GRU will still work because:

The feat_history deque collects scaled features every 200ms
The GRU sees 8 features spaced 200ms apart (1.4s span)
The features change less between consecutive inference steps than between training steps
But the GRU learned what "rising std" or "changing slope" looks like from training — it just sees a smoother, 
smaller version of those patterns at inference
This is a form of data augmentation — the model learns from coarser-grained temporal patterns and generalizes 
to finer-grained ones. It's analogous to training an image model on 224px and running inference on 256px.

Alternatively: match training stride to inference
If you want strict train/inference parity, use STRIDE_SAMPLES=75 (500ms) as a compromise:

Stride	Overlap	Seq span	Distinct info per step
30 (200ms)	96%	1.4s	4% new data
75 (500ms)	90%	3.5s	10% new data
150 (1.0s)	80%	7.0s	20% new data
"""

# SEEDS_TO_TRY = [456, 123, 42]        # ← Add more seeds here if desired
SEEDS_TO_TRY = [456]

# WINDOW_SEC = 5.0
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
DATA_DIR = PROJECT_ROOT / "training_data"
TEST_DIR = PROJECT_ROOT / "test_data"

# Dynamic scaler path (matches fit_scaler_V5 and gru_torch_V5)
SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_seq{SEQ_LEN}_{dim_str}.pkl"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_FILE = CACHE_DIR / f"features_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}_{FEATURE_SET}.npz"

CLASS_NAMES = ['blood', 'clot', 'wall']
CLINICAL_WEIGHTS = [1.0, 1.0, 1.0]

# ────────────────────────────────────────────────
# Fast feature extractor (active features only — skips FFT, slopes, sample entropy)
# ────────────────────────────────────────────────

_SUPPORTED_FEATURES = {0, 1, 3, 4, 5, 9, 17, 19, 20, 21, 23, 27, 28, 29, 32, 34, 38, 39, 44, 45, 52}
_unsupported = set(active_idx) - _SUPPORTED_FEATURES
if _unsupported:
    print(f"ERROR: active_idx contains features {_unsupported} not supported by fast extractor.")
    print(f"       Update compute_active_features_fast() or use full ClotFeatureExtractor.")
    sys.exit(1)

# Read EMA constants from canonical source
_ext = ClotFeatureExtractor(sample_rate=150, window_sec=WINDOW_SEC)
WINDOW_SAMPLES = _ext.window_size
ALPHA_FAST = _ext.alpha_fast
ALPHA_SLOW = _ext.alpha_slow
del _ext

# lfilter coefficients (reused across all runs)
_B_FAST = np.array([ALPHA_FAST])
_A_FAST = np.array([1.0, -(1.0 - ALPHA_FAST)])
_B_SLOW = np.array([ALPHA_SLOW])
_A_SLOW = np.array([1.0, -(1.0 - ALPHA_SLOW)])


def compute_active_features_fast(data, ema_fast, ema_slow):
    """
    Compute ONLY the 21 clot_wall_focused features directly.
    No FFT, no slopes, no sample entropy, no 53-element array.
    Returns array of shape (active_dim,) in active_idx order.
    """
    n = len(data)
    if n < 100:
        return None

    # Shared intermediates (computed once, reused)
    data_mean = data.mean()
    data_std = data.std()
    deriv = np.diff(data)
    ddx = np.diff(deriv)
    data_var = data.var() + 1e-8
    dx_var = deriv.var() + 1e-8
    ddx_var = ddx.var() + 1e-8
    hjorth_mob = np.sqrt(dx_var / data_var)

    # Detrended residual
    kernel = 450
    if n >= kernel:
        trend = np.convolve(data, np.ones(kernel) / kernel, 'valid')
        detr = data[-len(trend):] - trend
        r600 = detr[-min(600, len(detr)):]
    else:
        r600 = np.zeros(1, dtype=np.float32)

    # Diff of last 500
    if n >= 500:
        diff_500 = np.diff(data[-500:])
    else:
        diff_500 = None

    p95 = np.percentile(data, 95)

    # Build output in active_idx order:
    # [39, 21, 4, 19, 45, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 52, 27, 1, 20, 44]
    f = np.array([
        np.sum(data > p95) / n,                                      # f39: frac above p95
        sp_stats.kurtosis(deriv) if len(deriv) >= 4 else 0,          # f21: deriv kurtosis
        data.max(),                                                   # f4:  max
        np.mean(np.abs(deriv)) if len(deriv) > 10 else 0,            # f19: mean abs deriv
        np.sqrt(ddx_var / dx_var) / (hjorth_mob + 1e-8),             # f45: Hjorth complexity
        np.mean(np.abs(diff_500)) if diff_500 is not None else 0,    # f9:  mean abs diff last 500
        np.ptp(data),                                                 # f5:  range (max-min)
        ema_slow,                                                     # f23: EMA slow
        data_mean,                                                    # f0:  mean
        sp_stats.skew(r600) if len(r600) >= 3 else 0,                # f34: detrended skew
        np.std(r600),                                                 # f28: detrended std
        np.std(r600[:300]) if len(r600) >= 300 else 0,               # f29: detrended std first 300
        data.min(),                                                   # f3:  min
        p95 - np.percentile(data, 5),                                 # f38: p95 - p5
        deriv.std() if len(deriv) > 10 else 0,                       # f17: deriv std
        np.std(diff_500) if diff_500 is not None else 0,             # f32: std diff last 500
        np.mean(np.abs(ddx)) if len(ddx) > 0 else 0,                # f52: mean abs 2nd deriv
        np.abs(ema_fast - ema_slow),                                  # f27: abs(EMA fast - slow)
        data_std,                                                     # f1:  std
        sp_stats.skew(deriv) if len(deriv) >= 3 else 0,              # f20: deriv skew
        hjorth_mob,                                                   # f44: Hjorth mobility
    ], dtype=np.float32)

    return f

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
# Cached Feature Extraction (updated for FEATURE_SET)
# ────────────────────────────────────────────────

def load_or_extract_features(force_extract: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_filename = f"features_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}_{FEATURE_SET}.npz"
    CACHE_FILE = CACHE_DIR / cache_filename

    if CACHE_FILE.exists() and not force_extract:
        print(f"Loading cached features from: {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        X_seq = data['X_seq']
        y = data['y'].astype(np.int64)
        groups = data['groups']
        print(f"Loaded: {X_seq.shape[0]} sequences | shape={X_seq.shape}")
    else:
        print("Extracting features (fast path — active features only)...")
        data_files = sorted(DATA_DIR.glob("*.parquet"))
        all_data = [pd.read_parquet(f).assign(run_id=f.stem) for f in data_files]
        df_all = pd.concat(all_data, ignore_index=True)

        seq_list = []
        labels_list = []
        groups_list = []

        for run_id, group in df_all.groupby('run_id'):
            resistance = group['magRLoadAdjusted'].to_numpy(dtype=np.float32)
            label_array = group['label'].values.astype(np.int64)

            if len(resistance) < WINDOW_SAMPLES:
                continue

            # Precompute full-run EMA arrays (cumulative from run start)
            r0 = float(resistance[0])
            ema_f_all, _ = lfilter(_B_FAST, _A_FAST, resistance.astype(np.float64),
                                   zi=[r0 * (1.0 - ALPHA_FAST)])
            ema_s_all, _ = lfilter(_B_SLOW, _A_SLOW, resistance.astype(np.float64),
                                   zi=[r0 * (1.0 - ALPHA_SLOW)])

            # Extraction indices (same as original streaming logic)
            extraction_indices = np.arange(WINDOW_SAMPLES - 1, len(resistance), STRIDE_SAMPLES)

            run_features = []
            run_labels = []

            for idx in extraction_indices:
                win_start = idx - WINDOW_SAMPLES + 1
                window_data = resistance[win_start : idx + 1]

                feats = compute_active_features_fast(
                    window_data, float(ema_f_all[idx]), float(ema_s_all[idx])
                )
                if feats is not None:
                    run_features.append(feats)
                    window_label = int(label_array[win_start : idx + 1].max())
                    run_labels.append(window_label)

            # Build sequences
            for i in range(SEQ_LEN - 1, len(run_features)):
                seq = np.array(run_features[i - SEQ_LEN + 1 : i + 1])
                seq_list.append(seq)
                labels_list.append(run_labels[i])
                groups_list.append(run_id)

        X_seq = np.array(seq_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        groups = np.array(groups_list)

        print(f"Extracted: {X_seq.shape[0]} sequences | shape={X_seq.shape} from {len(np.unique(groups))} runs")

        print("Caching features...")
        np.savez_compressed(CACHE_FILE, X_seq=X_seq, y=y, groups=groups)
        print(f"Saved cache → {CACHE_FILE}")

    # Scaling
    print("Loading scaler and scaling sequences...")
    scaler = joblib.load(SCALER_PATH)
    N, S, F = X_seq.shape
    X_flat = X_seq.reshape(-1, F)
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(N, S, F)

    assert np.abs(X_scaled.mean()) < 0.1, f"Scaling failed — mean should be ~0 (got {X_scaled.mean():.4f})"
    assert 0.6 < X_scaled.std() < 1.4, f"Scaling failed — std should be ~1 (got {X_scaled.std():.4f})"
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
    print(f"FEATURE_SET          : {FEATURE_SET}")
    print(f"Active features      : {active_dim}  ({FEATURE_SET})")
    print(f"SEQ_LEN              : {SEQ_LEN}")
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

    best_global_f1 = 0.0
    best_state_global = None
    best_seed = None

    for seed in SEEDS_TO_TRY:
        print(f"\n{'='*60}")
        print(f"Training with SEED = {seed}")
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
            val_ds   = TensorDataset(torch.from_numpy(va_X).float(), torch.from_numpy(va_y).long())

            g = torch.Generator().manual_seed(seed + fold)

            train_dl = DataLoader(
                train_ds, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=True if NUM_WORKERS > 0 else False,
                generator=g,
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
            model_filename = f"clot_gru_trained_seq{SEQ_LEN}_{FEATURE_SET}_seed{seed}_f1{best_f1_this_seed:.4f}.pt"
            save_path = PROJECT_ROOT / "src" / "training" / model_filename
            
            torch.save(best_state_this_seed, save_path)
            
            print(f"   ✅ Saved model for seed {seed}")
            print(f"      Filename: {model_filename}")
            print(f"      F1-macro: {best_f1_this_seed:.4f}")
            print(f"      Path: {save_path}")
        else:
            print(f"   ⚠️  No model saved for seed {seed} (best_state was None)")

        # Update global best
        if best_f1_this_seed > best_global_f1:
            best_global_f1 = best_f1_this_seed
            best_state_global = best_state_this_seed
            best_seed = seed
            print(f"   → New global best! (Seed {seed})")

    # Optional: save overall best as generic name for easy inference
    if best_state_global is not None:
        latest_path = PROJECT_ROOT / "src" / "training" / f"clot_gru_trained_seq{SEQ_LEN}_{FEATURE_SET}.pt"
        torch.save(best_state_global, latest_path)
        print(f"\n✅ Also saved overall best as: clot_gru_trained_seq{SEQ_LEN}_{FEATURE_SET}.pt")

        generic_path = PROJECT_ROOT / "src" / "training" / "clot_gru_trained.pt"
        torch.save(best_state_global, generic_path)
        print(f"✅ Also saved as: clot_gru_trained.pt")

    print("\n" + "="*70)
    print("ALL SEEDS FINISHED")
    print("="*70)
    print(f"Global best F1-macro: {best_global_f1:.4f} (Seed {best_seed})")

if __name__ == "__main__":
    main()