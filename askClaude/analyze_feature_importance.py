# analyze_feature_importance.py
# Standalone — computes permutation importance (f1_macro) on test data

import torch
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from src.models.gru_torch_V5 import ClotFeatureExtractor, ClotGRU, SEQ_LEN, \
    TOTAL_FEATURES, FEATURE_SET, active_idx, active_dim, dim_str

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
    31: "std diff decimated deriv",
    32: "std recent diff",
    33: "mean abs recent diff",
    34: "skew detrended recent",
    35: "kurtosis detrended recent",
    36: "90th percentile - mean",
    37: "IQR (75th - 25th)",
    38: "95th - 5th percentile range",
    39: "fraction above 95th percentile",
    40: "spectral centroid",
    41: "spectral flatness",
    42: "band power ratio (low/high)",
    43: "dominant frequency",
    44: "Hjorth mobility",
    45: "Hjorth complexity",
    46: "sample entropy (decimated)",
    47: "zero-crossing rate",
    48: "variance ratio (1s/4s)",
    49: "CUSUM max",
    50: "autocorrelation lag-1",
    51: "energy ratio (last 1s / full)",
    52: "mean abs 2nd derivative",
}

# CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_seq{SEQ_LEN}_{dim_str}.pkl"

MODEL_PATH  = PROJECT_ROOT / "src" / "training" / "clot_gru_trained.pt"
TEST_DIR     = PROJECT_ROOT / "test_data"

STRIDE_SAMPLES = 30  # 75
WINDOW_SEC     = 5.0 # 6.0
SAMPLE_RATE    = 150

print(f"Feature set: {FEATURE_SET} ({active_dim} features)")
print("Loading scaler and model...")
scaler = joblib.load(SCALER_PATH)
model  = ClotGRU()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC)

print("Processing test files to build X_test and y_test...")
X_test_list = []
y_test_list = []

test_files = list(TEST_DIR.glob("*.parquet"))
for file in test_files:
    df = pd.read_parquet(file)
    resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
    labels     = df['label'].values
    extractor.reset()

    window_samples = extractor.window_size
    for start in range(0, len(resistance) - window_samples, STRIDE_SAMPLES):
        window_res = resistance[start:start + window_samples]
        window_label = np.max(labels[start:start + window_samples])
        for r in window_res:
            extractor.update(r)
        feats_all = extractor.compute_features()
        feats = feats_all[active_idx]
        X_test_list.append(feats)
        y_test_list.append(window_label)

X_test = np.array(X_test_list)
y_test = np.array(y_test_list, dtype=np.int64)

X_test_scaled = scaler.transform(X_test)

print(f"Test set size: {len(X_test)} windows")
print(f"Class distribution: {np.bincount(y_test)}")

# Shape-safe wrapper – send 2D only, let model unsqueeze
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

        # Force 2D input [n_samples, n_features]
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Create pseudo-sequences by repeating each feature vector SEQ_LEN times
        # Shape: (batch, SEQ_LEN, features)
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
    X_test_scaled,
    y_test,
    scoring='f1_macro',
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

# Results table
sorted_idx = perm_result.importances_mean.argsort()[::-1]

print(f"\n" + "="*100)
print(f"PERMUTATION IMPORTANCE (drop in f1_macro) — {active_dim} ACTIVE FEATURES (set: {FEATURE_SET})")
print("="*100)
print(f"{'Rank':>4} {'Feature':>8} {'Mean drop':>12} {'Std':>10} {'Description':<40} {'Hint':<35}")
print("-"*100)

for rank, idx in enumerate(sorted_idx, 1):
    mean_drop = perm_result.importances_mean[idx]
    std_drop  = perm_result.importances_std[idx]
    # Map local index back to global feature index
    global_idx = active_idx[idx]
    desc = feature_descriptions.get(global_idx, "unknown")
    hint = ""
    if mean_drop < 0.001:
        hint = "→ likely safe to remove"
    elif mean_drop < 0.005:
        hint = "→ low value, test removal"
    print(f"{rank:4d}   f{global_idx:3d}       {mean_drop:9.5f}   ± {std_drop:7.5f}  {desc:<40} {hint}")

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
plt.title(f"Permutation Feature Importance \u2013 Top {n_top} (test set, f1_macro, set: {FEATURE_SET})", fontsize=14)
plt.xlabel("Drop in f1_macro when feature is permuted")
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("permutation_importance_f1_macro_top30.png", dpi=200, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 14))
plt.boxplot(
    perm_result.importances[sorted_idx].T,
    vert=False,
    labels=[f"f{active_idx[i]}" for i in sorted_idx]
)
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.title(f"Permutation Feature Importance – All {active_dim} Features (set: {FEATURE_SET})", fontsize=14)
plt.xlabel("Drop in f1_macro when feature is permuted")
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"permutation_importance_f1_macro_all_{FEATURE_SET}.png", dpi=200, bbox_inches='tight')
plt.show()

important_indices = sorted_idx[:20]  # top 20 for example

# ────────────────────────────────────────────────
# ADD CORRELATION HEATMAP HERE
# ────────────────────────────────────────────────
print("Top 20 features:", [f"f{active_idx[i]}" for i in important_indices])
top_features_data = pd.DataFrame(X_test_scaled[:, important_indices],
                                 columns=[f"f{active_idx[i]}" for i in important_indices])

corr = top_features_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title(f"Correlation Heatmap - Top 20 Important Features (set: {FEATURE_SET})")
plt.tight_layout()
plt.savefig(f"top20_features_correlation_heatmap_{FEATURE_SET}.png", dpi=300)
plt.show()  # or plt.close() if you don't want to display

print("\nPlots saved.")
print("Done!")