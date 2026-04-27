# analyze_feature_importance.py
# Standalone — computes permutation importance (f1_macro) on TRAINING data
# Uses V6 pipeline with efficient batch feature extraction (compute_features_from_array + lfilter EMA)

import os
import sys
import torch
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_torch_V6 import ClotGRU, SEQ_LEN, \
    FEATURE_SET, active_idx, active_dim, dim_str, WINDOW_SEC

# V6 feature descriptions (f0-f50)
feature_descriptions = {
    0: "mean of window",
    1: "standard deviation",
    2: "variance",
    3: "minimum value",
    4: "maximum value",
    5: "peak-to-peak (max - min)",
    6: "median",
    7: "std of recent 500 samples",
    8: "var of recent 500 samples",
    9: "mean abs diff recent 500",
    10: "slope over 1s",
    11: "slope over 2s",
    12: "slope over 3s",
    13: "slope over 4s",
    14: "slope over 5s",
    15: "slope over 6s",
    16: "mean of derivative",
    17: "std of derivative",
    18: "var of derivative",
    19: "mean abs derivative",
    20: "skew of derivative",
    21: "kurtosis of derivative",
    22: "ema_fast (α=0.2)",
    23: "ema_slow (α=0.01)",
    24: "ema_fast - ema_slow",
    25: "forced to 0.0 (unused)",
    26: "ema_fast / ema_slow",
    27: "abs(ema_fast - ema_slow)",
    28: "std detrended ~6s",
    29: "std detrended ~3s",
    30: "mean abs detrended ~6s",
    31: "forced to 0.0 (unused)",
    32: "std recent diff",
    33: "mean abs recent diff",
    34: "skew detrended recent",
    35: "kurtosis detrended recent",
    36: "90th percentile - mean",
    37: "IQR (75th - 25th)",
    38: "95th - 5th percentile range",
    39: "fraction above 95th percentile",
    40: "Hjorth mobility",
    41: "Hjorth complexity",
    42: "mean abs 2nd derivative",
    43: "pulse amplitude (cardiac std)",
    44: "pulse-to-signal ratio",
    45: "pulse rate (peaks/sec)",
    46: "coeff of variation (std/mean)",
    47: "plateau fraction",
    48: "settling time ratio",
    49: "trend stationarity (Q4/Q1)",
    50: "R level relative to baseline",
    51: "short slope 0.1s (15 samples)",
    52: "short slope 0.2s (30 samples)",
    53: "short slope 0.3s (45 samples)",
    54: "short slope 0.4s (60 samples)",
    55: "short slope 0.5s (75 samples)",
    56: "short slope 0.6s (90 samples)",
    57: "norm max rise rate",
    58: "rise time fraction",
    59: "rise linearity (R²)",
    60: "peak sharpness",
    61: "descent smoothness",
    62: "shape asymmetry (skew)",
    63: "plateau ratio",
}

# CONFIG
SAMPLE_RATE    = 150
STRIDE_SAMPLES = 30

SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_seq{SEQ_LEN}_{dim_str}.pkl"
MODEL_PATH  = PROJECT_ROOT / "src" / "training" / "clot_gru_trained.pt"
TRAIN_DIR   = PROJECT_ROOT / "training_data"

print(f"Feature set: {FEATURE_SET} ({active_dim} features)")
print(f"Active indices: {active_idx}")

# Check model compatibility — if model input_size != active_dim, warn and abort
print("Loading scaler and model...")
if not SCALER_PATH.exists():
    print(f"ERROR: Scaler not found: {SCALER_PATH.name}")
    print(f"  Run fit_scaler_V6.py first to create it for feature set '{FEATURE_SET}'")
    sys.exit(1)

scaler = joblib.load(SCALER_PATH)

# Load model and verify input size matches current feature set
model = ClotGRU(input_size=active_dim)
state = torch.load(MODEL_PATH, map_location='cpu')
# Check input size from saved weights
saved_input_size = state['gru.weight_ih_l0'].shape[1]
if saved_input_size != active_dim:
    print(f"\nERROR: Model input_size={saved_input_size} but FEATURE_SET='{FEATURE_SET}' has {active_dim} features.")
    print(f"  Either train a model for '{FEATURE_SET}' first (run train_gru_V6.py),")
    print(f"  or change FEATURE_SET in gru_torch_V6.py to match the trained model.")
    sys.exit(1)

model.load_state_dict(state)
model.eval()

# ── Load features from training cache (built by train_gru_V6.py) ──
CACHE_DIR = PROJECT_ROOT / "cache"
cache_filename = f"features_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}_{FEATURE_SET}.npz"
CACHE_FILE = CACHE_DIR / cache_filename

if CACHE_FILE.exists():
    print(f"\nLoading cached features from: {CACHE_FILE.name}")
    data = np.load(CACHE_FILE)
    X_seq = data['X_seq']       # shape (N, SEQ_LEN, active_dim)
    y_train = data['y'].astype(np.int64)
    # Use the last timestep features for permutation importance (2D)
    X_train = X_seq[:, -1, :]   # shape (N, active_dim)
    del X_seq, data              # free the full sequence array
    print(f"Loaded: {len(y_train)} sequences, using last-timestep features {X_train.shape}")

    # Subsample to avoid MemoryError during permutation importance
    MAX_SAMPLES = 100_000
    if len(X_train) > MAX_SAMPLES:
        print(f"Subsampling {MAX_SAMPLES} from {len(X_train)} (stratified by class)...")
        rng = np.random.RandomState(42)
        idx_all = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            n_cls = max(1, int(MAX_SAMPLES * len(cls_idx) / len(y_train)))
            chosen = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
            idx_all.append(chosen)
        idx_sub = np.sort(np.concatenate(idx_all))
        X_train = X_train[idx_sub]
        y_train = y_train[idx_sub]
        print(f"Subsampled: {len(X_train)} windows")
else:
    print(f"\nERROR: Cache not found: {CACHE_FILE}")
    print(f"  Run train_gru_V6.py first to build the feature cache.")
    sys.exit(1)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_train_scaled = scaler.transform(X_train)

print(f"\nTraining set size: {len(X_train)} windows")
print(f"Class distribution: {np.bincount(y_train)}")
print(f"  blood={np.sum(y_train==0)}, clot={np.sum(y_train==1)}, wall={np.sum(y_train==2)}")

# Shape-safe wrapper
class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, seq_len=SEQ_LEN):
        self.model = model
        self.seq_len = seq_len

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_seq = np.tile(X[:, np.newaxis, :], (1, self.seq_len, 1))
        X_tensor = torch.from_numpy(X_seq).float()

        with torch.no_grad():
            logits, _ = self.model(X_tensor, None)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

wrapped_model = TorchModelWrapper(model, seq_len=SEQ_LEN)

# Permutation importance
print("\nComputing permutation importance (f1_macro) — this may take 5–20 min...")
perm_result = permutation_importance(
    wrapped_model,
    X_train_scaled,
    y_train,
    scoring='f1_macro',
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

# Results table
sorted_idx = perm_result.importances_mean.argsort()[::-1]

print(f"\n" + "="*110)
print(f"PERMUTATION IMPORTANCE (drop in f1_macro) — {active_dim} ACTIVE FEATURES (set: {FEATURE_SET})")
print(f"Computed on TRAINING data ({len(X_train)} windows)")
print("="*110)
print(f"{'Rank':>4} {'Feature':>8} {'Mean drop':>12} {'Std':>10} {'Description':<45} {'Hint':<35}")
print("-"*110)

for rank, idx in enumerate(sorted_idx, 1):
    mean_drop = perm_result.importances_mean[idx]
    std_drop  = perm_result.importances_std[idx]
    global_idx = active_idx[idx]
    desc = feature_descriptions.get(global_idx, "unknown")
    hint = ""
    if mean_drop < 0.001:
        hint = "→ likely safe to remove"
    elif mean_drop < 0.005:
        hint = "→ low value, test removal"
    print(f"{rank:4d}   f{global_idx:3d}       {mean_drop:9.5f}   ± {std_drop:7.5f}  {desc:<45} {hint}")

# Save results to text file
output_txt = PROJECT_ROOT / "src" / "data" / f"feature_importance_{FEATURE_SET}.txt"
with open(output_txt, 'w') as fout:
    fout.write(f"PERMUTATION IMPORTANCE (drop in f1_macro) — {active_dim} ACTIVE FEATURES (set: {FEATURE_SET})\n")
    fout.write(f"Computed on TRAINING data ({len(X_train)} windows)\n\n")
    fout.write(f"{'Rank':>4} {'Feature':>8} {'Mean drop':>12} {'Std':>10} {'Description':<45} {'Hint':<35}\n")
    fout.write("-"*110 + "\n")
    for rank, idx in enumerate(sorted_idx, 1):
        mean_drop = perm_result.importances_mean[idx]
        std_drop  = perm_result.importances_std[idx]
        global_idx = active_idx[idx]
        desc = feature_descriptions.get(global_idx, "unknown")
        hint = ""
        if mean_drop < 0.001:
            hint = "→ likely safe to remove"
        elif mean_drop < 0.005:
            hint = "→ low value, test removal"
        fout.write(f"{rank:4d}   f{global_idx:3d}       {mean_drop:9.5f}   ± {std_drop:7.5f}  {desc:<45} {hint}\n")
print(f"\nSaved importance table to {output_txt.name}")

# Plots
plt.figure(figsize=(14, 10))
n_top = min(30, len(sorted_idx))
labels_top = [f"f{active_idx[i]}" for i in sorted_idx[:n_top]]
plt.boxplot(
    perm_result.importances[sorted_idx[:n_top]].T,
    vert=False,
    labels=labels_top
)
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.title(f"Permutation Feature Importance — Top {n_top} (training data, f1_macro, set: {FEATURE_SET})", fontsize=14)
plt.xlabel("Drop in f1_macro when feature is permuted")
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "src" / "data" / f"permutation_importance_top30_{FEATURE_SET}.png", dpi=200, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 14))
plt.boxplot(
    perm_result.importances[sorted_idx].T,
    vert=False,
    labels=[f"f{active_idx[i]}" for i in sorted_idx]
)
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.title(f"Permutation Feature Importance — All {active_dim} Features (training, set: {FEATURE_SET})", fontsize=14)
plt.xlabel("Drop in f1_macro when feature is permuted")
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "src" / "data" / f"permutation_importance_all_{FEATURE_SET}.png", dpi=200, bbox_inches='tight')
plt.close()

# Correlation heatmap of top 20
important_indices = sorted_idx[:20]
print("\nTop 20 features:", [f"f{active_idx[i]}" for i in important_indices])
top_features_data = pd.DataFrame(X_train_scaled[:, important_indices],
                                 columns=[f"f{active_idx[i]}" for i in important_indices])

corr = top_features_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title(f"Correlation Heatmap — Top 20 Important Features (training, set: {FEATURE_SET})")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "src" / "data" / f"top20_correlation_heatmap_{FEATURE_SET}.png", dpi=300)
plt.close()

print("\nPlots saved to src/data/")
print("Done!")