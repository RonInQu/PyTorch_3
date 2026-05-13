# plot_feature_distributions.py
"""
Plot feature distributions by class (blood/clot/wall) using training data.
Shows overlapping histograms for each feature — good features show clear class separation.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_torch_V6 import ClotFeatureExtractor, WINDOW_SEC, \
    FEATURE_SET, SEQ_LEN, active_idx, active_dim

from src.training.train_gru_V6 import STRIDE_SAMPLES

# ── Feature descriptions (for plot titles) ──
feature_descriptions = {
    0: "mean", 1: "std", 2: "variance", 3: "min", 4: "max",
    5: "peak-to-peak", 6: "median", 7: "std recent 500",
    8: "var recent 500", 9: "mean abs diff 500",
    10: "slope 1s", 11: "slope 2s", 12: "slope 3s",
    13: "slope 4s", 14: "slope 5s", 15: "slope 6s",
    16: "mean deriv", 17: "std deriv", 18: "var deriv",
    19: "mean abs deriv", 20: "skew deriv", 21: "kurtosis deriv",
    22: "ema_fast", 23: "ema_slow", 24: "ema_fast-slow",
    25: "unused (0)", 26: "ema_fast/slow", 27: "abs(ema_fast-slow)",
    28: "std detrended 6s", 29: "std detrended 3s",
    30: "mean abs detrended", 31: "unused (0)",
    32: "std recent diff", 33: "mean abs recent diff",
    34: "skew detrended", 35: "kurtosis detrended",
    36: "90th pctl-mean", 37: "IQR", 38: "95th-5th range",
    39: "frac above 95th",
    40: "Hjorth mobility", 41: "Hjorth complexity",
    42: "mean abs 2nd deriv",
    43: "pulse amplitude", 44: "pulse-to-signal ratio",
    45: "pulse rate",
    46: "coeff of variation", 47: "plateau fraction",
    48: "settling time ratio", 49: "trend stationarity (Q4/Q1)",
    50: "R level rel baseline",
    51: "short slope 0.1s", 52: "short slope 0.2s",
    53: "short slope 0.3s", 54: "short slope 0.4s",
    55: "short slope 0.5s", 56: "short slope 0.6s",
    57: "norm max rise rate", 58: "rise time fraction",
    59: "rise linearity (R²)", 60: "peak sharpness",
    61: "descent smoothness", 62: "shape asymmetry (skew)",
    63: "plateau ratio",
    64: "dir changes/sec",
    65: "peaks/sec",
    66: "mean abs curvature",
    67: "longest mono run frac",
    68: "zero crossing rate (detr)",
}

# ── Config ──
SAMPLE_RATE = 150
DATA_DIR = PROJECT_ROOT / "training_data"
CACHE_DIR = PROJECT_ROOT / "cache"
CLASS_NAMES = {0: 'blood', 1: 'clot', 2: 'wall'}
CLASS_COLORS = {0: 'black', 1: 'red', 2: 'blue'}

# ── Load from cache if available, else extract ──
print(f"Feature set: {FEATURE_SET} ({active_dim} active features)")

cache_filename = f"features_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}_{FEATURE_SET}.npz"
cache_path = CACHE_DIR / cache_filename

if cache_path.exists():
    print(f"Loading cached features from: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    X_seq = data['X_seq']       # (N, SEQ_LEN, active_dim)
    y = data['y'].astype(np.int64)
    groups = data['groups']     # run_id per window
    # Use last timestep of each sequence for distribution plots
    X = X_seq[:, -1, :]
    print(f"Loaded {len(X)} windows from cache (last timestep of sequences)")
else:
    print(f"Cache not found: {cache_path}")
    print(f"Extracting features from training data...")

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} files")

    all_feats = []
    all_labels = []
    all_groups = []

    extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC)

    for file_path in tqdm(parquet_files, desc="Processing"):
        df = pd.read_parquet(file_path)
        resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
        labels = df['label'].values.astype(int)
        run_id = file_path.stem

        extractor.reset()
        window_size = extractor.window_size

        for idx, r in enumerate(resistance):
            extractor.update(r)
            samples_in = idx + 1
            if samples_in >= window_size and (samples_in - window_size) % STRIDE_SAMPLES == 0:
                feats = extractor.compute_features()
                feats_active = feats[active_idx]
                win_start = idx - window_size + 1
                window_label = int(np.max(labels[win_start:idx + 1]))
                all_feats.append(feats_active)
                all_labels.append(window_label)
                all_groups.append(run_id)

    X = np.array(all_feats, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    groups = np.array(all_groups)

print(f"\n{len(X)} windows total")
for cls in [0, 1, 2]:
    print(f"  {CLASS_NAMES[cls]}: {(y == cls).sum()}")

# ── Plot helper ──
def plot_distributions(X, y, classes_to_plot, suffix=""):
    """Plot feature distributions for the given classes."""
    n_features = X.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    class_label = "_".join(CLASS_NAMES[c] for c in classes_to_plot)

    for feat_idx in range(n_features):
        ax = axes[feat_idx]
        global_idx = active_idx[feat_idx]
        desc = feature_descriptions.get(global_idx, f"f{global_idx}")

        for cls in classes_to_plot:
            mask = y == cls
            vals = X[mask, feat_idx]
            p1, p99 = np.percentile(vals, [1, 99])
            vals_clipped = vals[(vals >= p1) & (vals <= p99)]
            ax.hist(vals_clipped, bins=50, alpha=0.7, density=True,
                    color=CLASS_COLORS[cls], label=CLASS_NAMES[cls],
                    histtype='step', linewidth=1.5)

        ax.set_title(f"f{global_idx}: {desc}", fontsize=9)
        ax.tick_params(labelsize=7)
        if feat_idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    title = f"Feature Distributions — {class_label} — {FEATURE_SET} ({n_features} features)"
    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    fname = f"feature_distributions_{FEATURE_SET}_{class_label}{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {fname}")

# ── Generate both plots ──
plot_distributions(X, y, classes_to_plot=[0, 1, 2])       # all three classes
plot_distributions(X, y, classes_to_plot=[1, 2])           # clot vs wall only

# ── Per-run AUC helper ──
def auc_binary(class_a, class_b):
    """AUC for separating two classes using a single feature. Direction-invariant."""
    labels = np.concatenate([np.zeros(len(class_a)), np.ones(len(class_b))])
    scores = np.concatenate([class_a, class_b])
    auc = roc_auc_score(labels, scores)
    return max(auc, 1.0 - auc)

def mean_per_run_auc(feat_vals, labels, run_ids, cls_a=1, cls_b=2):
    """Mean AUC across runs. Each run's clot/wall windows scored independently."""
    unique_runs = np.unique(run_ids)
    aucs = []
    for run in unique_runs:
        run_mask = run_ids == run
        run_labels = labels[run_mask]
        run_vals = feat_vals[run_mask]
        a_vals = run_vals[run_labels == cls_a]
        b_vals = run_vals[run_labels == cls_b]
        if len(a_vals) < 2 or len(b_vals) < 2:
            continue
        # Skip if feature is constant within this run
        if a_vals.std() == 0 and b_vals.std() == 0:
            aucs.append(0.5)
            continue
        aucs.append(auc_binary(a_vals, b_vals))
    return np.mean(aucs) if aucs else 0.5, len(aucs)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std()**2 + (nb - 1) * b.std()**2) / (na + nb - 2) + 1e-8)
    return abs(a.mean() - b.mean()) / pooled_std

# ── Class separability summary ──
print(f"\n{'='*110}")
print(f"CLASS SEPARABILITY SUMMARY — ranked by mean per-run AUC(clot vs wall)")
print(f"{'='*110}")
print(f"{'Rank':>4} {'Feature':>10} {'AUC c/w':>9} {'#runs':>6} {'global':>8} {'d(clt-wall)':>12} {'d(bld-clt)':>12} {'d(bld-wall)':>12} {'Description':<30}")
print("-" * 110)

separability = []
n_features = X.shape[1]
for feat_idx in range(n_features):
    global_idx = active_idx[feat_idx]
    blood = X[y == 0, feat_idx]
    clot = X[y == 1, feat_idx]
    wall = X[y == 2, feat_idx]

    d_bc = cohens_d(blood, clot)
    d_bw = cohens_d(blood, wall)
    d_cw = cohens_d(clot, wall)
    auc_cw, n_runs = mean_per_run_auc(X[:, feat_idx], y, groups)
    auc_global = auc_binary(clot, wall)
    separability.append((auc_cw, n_runs, auc_global, feat_idx, global_idx, d_bc, d_bw, d_cw))

# Sort by per-run AUC (descending)
separability.sort(reverse=True)

summary_lines = []
header = f"CLASS SEPARABILITY SUMMARY — ranked by mean per-run AUC(clot vs wall)"
col_header = f"{'Rank':>4} {'Feature':>10} {'AUC c/w':>9} {'#runs':>6} {'global':>8} {'d(clt-wall)':>12} {'d(bld-clt)':>12} {'d(bld-wall)':>12} {'Description':<30}"
summary_lines.append("=" * 110)
summary_lines.append(header)
summary_lines.append(f"Feature set: {FEATURE_SET} ({active_dim} active features)")
summary_lines.append("=" * 110)
summary_lines.append(col_header)
summary_lines.append("-" * 110)

for rank, (auc_cw, n_runs, auc_global, feat_idx, global_idx, d_bc, d_bw, d_cw) in enumerate(separability, 1):
    desc = feature_descriptions.get(global_idx, "")
    line = f"{rank:4d}   f{global_idx:3d}      {auc_cw:7.4f}   {n_runs:4d}   {auc_global:7.4f}   {d_cw:10.3f}   {d_bc:10.3f}   {d_bw:10.3f}   {desc}"
    summary_lines.append(line)

summary_lines.append("")
summary_lines.append("AUC: per-run mean (within-run discriminability). 'global' = pooled across all runs.")
summary_lines.append("AUC interpretation: 0.50=useless, 0.60=weak, 0.70=useful, 0.80=strong, 0.90+=excellent")
summary_lines.append("Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large, >1.2=very large")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

# Save to file
output_txt = PROJECT_ROOT / "src" / "data" / f"class_separability_{FEATURE_SET}.txt"
output_txt.write_text(summary_text, encoding="utf-8")
print(f"\nSaved separability table to {output_txt.name}")
print("Done!")
