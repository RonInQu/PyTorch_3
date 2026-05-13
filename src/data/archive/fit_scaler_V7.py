# fit_scaler_V7.py
"""
Fits the StandardScaler for clot detection features — V7 (DA-as-feature).
21 impedance features + 2 DA fraction features = 23 total.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
os.environ["PYARROW_HOTFIX_DISABLED"] = "1"

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ================= CONSTANTS =================
SAMPLE_RATE = 150

# ================= IMPORT FROM V7 =================
from src.models.gru_torch_V7 import (
    IMPEDANCE_IDX, TOTAL_INPUT_DIM, WINDOW_SEC, SEQ_LEN, WINDOW_SAMPLES,
    dim_str, compute_da_fractions_from_array
)
from src.models.gru_torch_V6 import ClotFeatureExtractor

STRIDE_SAMPLES = 30  # same as train_gru_V7

# ================= Derived CONFIG =================
OUTPUT_SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_seq{SEQ_LEN}_{dim_str}.pkl"

print("Scaler config (V7 — DA-as-feature):")
print(f"   SEQ_LEN         = {SEQ_LEN}")
print(f"   SAMPLE_RATE     = {SAMPLE_RATE} Hz")
print(f"   WINDOW_SEC      = {WINDOW_SEC} s")
print(f"   STRIDE_SAMPLES  = {STRIDE_SAMPLES}")
print(f"   Impedance feats = {len(IMPEDANCE_IDX)}")
print(f"   DA feats        = 2 (da_clot_frac, da_wall_frac)")
print(f"   Total features  = {TOTAL_INPUT_DIM}")
print(f"   Output scaler   = {OUTPUT_SCALER_PATH.name}")

# ================= Extractor + lfilter setup =================
extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC,
                                 active_features=IMPEDANCE_IDX)
ALPHA_FAST = extractor.alpha_fast
ALPHA_SLOW = extractor.alpha_slow

_B_FAST = np.array([ALPHA_FAST])
_A_FAST = np.array([1.0, -(1.0 - ALPHA_FAST)])
_B_SLOW = np.array([ALPHA_SLOW])
_A_SLOW = np.array([1.0, -(1.0 - ALPHA_SLOW)])

# ================= Main Scaler Fitting =================
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"

print(f"\nLooking for parquet files in: {TRAINING_DATA_DIR}")

parquet_files = list(TRAINING_DATA_DIR.glob("*_labeled_segment.parquet"))
if not parquet_files:
    print("No *_labeled_segment.parquet files found!")
    sys.exit(1)

print(f"Found {len(parquet_files)} parquet files.")

global_features = []

for file_path in tqdm(parquet_files, desc="Processing files"):
    print(f"\nReading: {file_path.name}")
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
    except Exception as e:
        print(f"  Failed to read: {e}")
        continue

    required_cols = ['timeInMS', 'magRLoadAdjusted', 'label', 'da_label']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  Missing columns: {missing} — skipping")
        continue

    df = df[required_cols].copy().dropna().reset_index(drop=True)
    resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
    labels = df['label'].to_numpy(dtype=np.int64)
    da_labels = df['da_label'].to_numpy(dtype=np.int64)

    # Boolean mask: True where label is blood(0), clot(1), or wall(2)
    valid_mask = np.isin(labels, [0, 1, 2])
    n_valid = valid_mask.sum()
    n_total = len(labels)
    if n_valid < n_total:
        print(f"  Filtering: {n_total - n_valid}/{n_total} unlabeled samples excluded")

    if len(resistance) < SAMPLE_RATE * WINDOW_SEC:
        print(f"  Too short ({len(resistance)} samples) — skipping")
        continue

    run_features = []
    skipped_unlabeled = 0

    for start in range(0, len(resistance) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        end = start + WINDOW_SAMPLES
        if not valid_mask[start:end].all():
            skipped_unlabeled += 1
            continue

        window_res = resistance[start:end]
        window_da = da_labels[start:end]

        # Vectorized EMA via lfilter (per-window reset)
        r0 = float(window_res[0])
        ema_f, _ = lfilter(_B_FAST, _A_FAST, window_res.astype(np.float64),
                           zi=[r0 * (1.0 - ALPHA_FAST)])
        ema_s, _ = lfilter(_B_SLOW, _A_SLOW, window_res.astype(np.float64),
                           zi=[r0 * (1.0 - ALPHA_SLOW)])

        # 21 impedance features
        imp_feats = extractor.compute_features_from_array(
            window_res, float(ema_f[-1]), float(ema_s[-1]))
        if imp_feats is None or len(imp_feats) != len(IMPEDANCE_IDX):
            continue

        # 2 DA fraction features
        da_clot_frac, da_wall_frac = compute_da_fractions_from_array(window_da)

        # Concatenate: 23-dim
        full_feats = np.concatenate([imp_feats,
                                     np.array([da_clot_frac, da_wall_frac], dtype=np.float32)])
        run_features.append(full_feats)

    global_features.extend(run_features)
    print(f"  → {len(run_features)} feature vectors extracted"
          + (f" ({skipped_unlabeled} windows skipped — unlabeled)" if skipped_unlabeled else ""))

if not global_features:
    print("No features extracted!")
    sys.exit(1)

X = np.array(global_features, dtype=np.float32)
print(f"\nCollected {X.shape[0]} feature vectors | shape={X.shape}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Scaling
print("Fitting scaler on feature vectors...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nAfter scaling ({X.shape[1]} features = 21 impedance + 2 DA):")
print(f"   Mean: {X_scaled.mean():.6f}   (should be very close to 0)")
print(f"   Std : {X_scaled.std():.6f}   (should be very close to 1)")

# DA feature stats
print(f"\n   DA feature means (unscaled): da_clot_frac={X[:, -2].mean():.4f}, da_wall_frac={X[:, -1].mean():.4f}")
print(f"   DA feature stds  (unscaled): da_clot_frac={X[:, -2].std():.4f}, da_wall_frac={X[:, -1].std():.4f}")

# Save the fitted scaler
joblib.dump(scaler, OUTPUT_SCALER_PATH)
print(f"\n✅ Scaler fitted and saved → {OUTPUT_SCALER_PATH}")

# ================= Feature Correlation Heatmap =================
print("\nGenerating feature correlation heatmap...")
feat_names = [f"imp_{i}" for i in range(len(IMPEDANCE_IDX))] + ['da_clot_frac', 'da_wall_frac']
df_feat = pd.DataFrame(X_scaled, columns=feat_names)
corr_matrix = df_feat.corr().abs()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1, square=True)
plt.title(f"Feature Correlation Heatmap (V7: {X_scaled.shape[1]} features)")
plt.tight_layout()
plt.savefig("feature_correlation_heatmap_V7.png", dpi=300, bbox_inches='tight')
plt.close()
print("   Saved → feature_correlation_heatmap_V7.png")

print("\nScaler fitting complete.")
