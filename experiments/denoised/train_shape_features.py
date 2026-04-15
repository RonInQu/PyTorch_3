"""
train_shape_features.py — Train GRU with NEW shape-based features on denoised data.

Shape features capture signal morphology (direction changes, curvature, monotonic
runs, peak density, zero crossings) that discriminate clot vs wall — the hardest
and most important distinction. Blood is always accepted from DA.

Shape feature analysis results (Cohen's d, C-W separation):
  dir_chg_per_s  0.726   ← best C-W
  peaks_per_s    0.596
  curvature      0.576
  longest_mono   0.574
  zero_cross     0.511
  waveform_len   0.228
  R2_linear      0.195
  mean_seg_slope 0.167
  rise_frac      0.157
  level_change   0.110

Strategy: combine top shape features with best original-46 features,
prioritizing C-W separation over B-C or B-W.
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
from scipy.signal import lfilter, find_peaks
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_torch_V6 import ClotFeatureExtractor, ClotGRU, WINDOW_SEC, SEQ_LEN

# ── Training config ──
SEED = 456
STRIDE_SAMPLES = 30
BATCH_SIZE = 1024
N_EPOCHS = 100
PATIENCE = 15
LR = 0.0001
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 150
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 750

# ── Paths ──
DATA_DIR = EXPERIMENT_DIR / "train_data"
CACHE_DIR = EXPERIMENT_DIR / "cache"
MODELS_DIR = EXPERIMENT_DIR / "models"


# ════════════════════════════════════════════════════
# SHAPE FEATURE EXTRACTION
# ════════════════════════════════════════════════════

def compute_shape_features(window):
    """Compute 10 shape features from a 750-sample window.

    Returns array of 10 floats:
      s0: dir_chg_per_s   — direction changes per second
      s1: peaks_per_s     — peaks per second
      s2: curvature       — mean absolute curvature
      s3: longest_mono    — longest monotonic run fraction
      s4: zero_cross      — zero crossing rate (detrended)
      s5: waveform_len    — mean |diff|
      s6: R2_linear       — linearity (R² of linear fit)
      s7: mean_seg_slope  — mean slope of monotonic segments
      s8: rise_frac       — fraction of rising samples
      s9: level_change    — normalized level change (Q4 - Q1)
    """
    w = window
    n = len(w)
    d = np.diff(w)
    dd = np.diff(d)
    w_std = np.std(w)

    # Direction changes
    signs = np.sign(d)
    signs[signs == 0] = 1
    dir_changes = np.sum(np.diff(signs) != 0)

    # s0: direction changes per second
    s0 = dir_changes / (n / SAMPLE_RATE)

    # s1: peaks per second
    if w_std > 0.01:
        peaks, _ = find_peaks(w, prominence=w_std * 0.3)
        s1 = len(peaks) / (n / SAMPLE_RATE)
    else:
        s1 = 0.0

    # s2: mean absolute curvature
    denom = (1 + d[:-1] ** 2) ** 1.5
    s2 = np.mean(np.abs(dd) / denom)

    # s3: longest monotonic run fraction
    changes = np.where(np.diff(signs) != 0)[0]
    runs = np.diff(np.concatenate([[0], changes, [len(signs)]]))
    s3 = np.max(runs) / len(d)

    # s4: zero crossing rate (detrended)
    kernel = 150  # 1s smoothing
    if n > kernel * 2:
        trend = np.convolve(w, np.ones(kernel) / kernel, 'same')
        detr = w[kernel // 2: -(kernel // 2)] - trend[kernel // 2: -(kernel // 2)]
        s4 = np.sum(np.diff(np.sign(detr)) != 0) / max(len(detr), 1)
    else:
        s4 = 0.0

    # s5: waveform length
    s5 = np.mean(np.abs(d))

    # s6: R² of linear fit
    x = np.arange(n, dtype=np.float64)
    p = np.polyfit(x, w, 1)
    resid = w - np.polyval(p, x)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((w - w.mean()) ** 2) + 1e-10
    s6 = max(0.0, 1 - ss_res / ss_tot)

    # s7: mean segment slope magnitude
    if len(runs) > 1:
        slope_mags = []
        pos = 0
        for rl in runs[:50]:
            seg = w[pos:pos + rl + 1]
            if len(seg) >= 2:
                slope_mags.append(abs(seg[-1] - seg[0]) / len(seg))
            pos += rl
        s7 = np.mean(slope_mags) if slope_mags else 0.0
    else:
        s7 = 0.0

    # s8: rise fraction
    s8 = np.sum(d > 0) / len(d)

    # s9: level change (normalized)
    first_q = np.mean(w[:n // 4])
    last_q = np.mean(w[-n // 4:])
    s9 = (last_q - first_q) / (w_std + 1e-6)

    return np.array([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9], dtype=np.float32)


SHAPE_NAMES = [
    'dir_chg_per_s', 'peaks_per_s', 'curvature', 'longest_mono',
    'zero_cross', 'waveform_len', 'R2_linear', 'mean_seg_slope',
    'rise_frac', 'level_change'
]
N_SHAPE = len(SHAPE_NAMES)


# ════════════════════════════════════════════════════
# LOAD ALL-46 CACHE + EXTRACT SHAPE FEATURES
# ════════════════════════════════════════════════════

# Pre-compute EMA filter coefficients
_ext = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC, active_features=[0])
ALPHA_FAST = _ext.alpha_fast
ALPHA_SLOW = _ext.alpha_slow
_B_FAST = np.array([ALPHA_FAST])
_A_FAST = np.array([1.0, -(1.0 - ALPHA_FAST)])
_B_SLOW = np.array([ALPHA_SLOW])
_A_SLOW = np.array([1.0, -(1.0 - ALPHA_SLOW)])


def load_all46_cache():
    """Load the already-cached all-46 features (from train_feature_search.py run)."""
    cache_file = CACHE_DIR / f"features_denoised_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}_all46.npz"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"All-46 cache not found: {cache_file}\n"
            "Run train_feature_search.py first to build it."
        )
    data = np.load(cache_file)
    return data['X_seq'], data['y'].astype(np.int64), data['groups']


def extract_shape_features():
    """Extract shape features from denoised training data, matching the all-46 extraction."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"features_denoised_w{WINDOW_SEC:.1f}s_s{STRIDE_SAMPLES}_seq{SEQ_LEN}_shape{N_SHAPE}.npz"

    if cache_file.exists():
        print(f"Loading cached shape features: {cache_file}")
        data = np.load(cache_file)
        return data['X_seq'], data['y'].astype(np.int64), data['groups']

    print(f"Extracting {N_SHAPE} shape features from denoised training data...")
    data_files = sorted(DATA_DIR.glob("*.parquet"))
    if not data_files:
        print(f"No parquet files in {DATA_DIR}")
        sys.exit(1)

    seq_list, labels_list, groups_list = [], [], []

    for fi, f in enumerate(data_files):
        df = pd.read_parquet(f)
        resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
        label_array = df['label'].values.astype(np.int64)
        valid_mask = np.isin(label_array, [0, 1, 2])
        run_id = f.stem

        if len(resistance) < WINDOW_SAMPLES:
            continue

        extraction_indices = np.arange(WINDOW_SAMPLES - 1, len(resistance), STRIDE_SAMPLES)

        run_features, run_labels = [], []
        for idx in extraction_indices:
            win_start = idx - WINDOW_SAMPLES + 1
            if not valid_mask[win_start:idx + 1].all():
                continue
            window_data = resistance[win_start:idx + 1]
            feats = compute_shape_features(window_data)
            if feats is not None and np.all(np.isfinite(feats)):
                run_features.append(feats)
                run_labels.append(int(label_array[win_start:idx + 1].max()))

        for i in range(SEQ_LEN - 1, len(run_features)):
            seq = np.array(run_features[i - SEQ_LEN + 1: i + 1])
            seq_list.append(seq)
            labels_list.append(run_labels[i])
            groups_list.append(run_id)

        if (fi + 1) % 10 == 0 or fi == len(data_files) - 1:
            print(f"  [{fi+1}/{len(data_files)}] {run_id}: {len(run_features)} windows → "
                  f"{max(0, len(run_features) - SEQ_LEN + 1)} seqs")

    X_seq = np.array(seq_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    groups = np.array(groups_list)
    print(f"Shape features: {X_seq.shape[0]} sequences | shape={X_seq.shape} from {len(np.unique(groups))} runs")

    np.savez_compressed(cache_file, X_seq=X_seq, y=y, groups=groups)
    print(f"Saved cache → {cache_file}")
    return X_seq, y, groups


# ════════════════════════════════════════════════════
# COMBINE ORIGINAL + SHAPE FEATURES
# ════════════════════════════════════════════════════

def combine_features(X_orig, X_shape, orig_indices):
    """Select orig_indices from all-46 and append shape features.

    Returns (N, SEQ_LEN, len(orig_indices) + N_SHAPE)
    """
    X_sub = X_orig[:, :, orig_indices]
    return np.concatenate([X_sub, X_shape], axis=2)


# ════════════════════════════════════════════════════
# FEATURE SET DEFINITIONS
# ════════════════════════════════════════════════════
# Naming: orig features are f0-f45, shape features are s0-s9
#
# Focus: C-W separation. Blood is handled by DA.
# The top C-W shape features: s0(dir_chg), s1(peaks), s2(curv), s3(longest_mono), s4(zero_cross)
# From original-46, best C-W features (from clot_wall_focused analysis):
#   f0(mean), f3(min), f4(max), f23(ema_slow), f27(|ema_diff|) — level features
#   f10-f15 — slopes (clean after denoising)
#   f16(deriv_mean), f19(mean_abs_deriv), f20(deriv_skew) — derivative

FEATURE_SETS = {
    # ── Shape only: just the 10 new morphology features ──
    "shape_only": {
        'orig_indices': [],
        'description': "10 shape features only — pure morphology"
    },

    # ── Top 5 shape (C-W focus) + slopes ──
    "shape5_slopes": {
        'orig_indices': [10, 11, 12, 13, 14, 15],  # 6 slopes
        'description': "Top 5 shape (C-W) + 6 slopes = 16 features"
    },

    # ── All shape + slopes + levels ──
    "shape_slopes_level": {
        'orig_indices': [
            0, 3, 4, 6,                # mean, min, max, median
            10, 11, 12, 13, 14, 15,    # 6 slopes
            23, 27,                     # ema_slow, |ema_diff|
        ],  # 12 orig + 10 shape = 22 features
        'description': "All shape + slopes + level/EMA = 22 features"
    },

    # ── Shape features + C-W focused original subset ──
    "shape_cw_focused": {
        'orig_indices': [
            0, 3, 4,                    # mean, min, max (level)
            10, 11, 12, 13, 14, 15,    # 6 slopes (clean on denoised)
            16, 19, 20,                 # deriv_mean, mean_abs_deriv, deriv_skew
            23, 27,                     # ema_slow, |ema_diff|
            28,                         # detrend_std
        ],  # 15 orig + 10 shape = 25 features
        'description': "All shape + C-W focused originals = 25 features"
    },

    # ── Top 5 shape only (minimal C-W model) ──
    "shape_top5_only": {
        'orig_indices': [],
        'shape_indices': [0, 1, 2, 3, 4],  # top 5 C-W shape features only
        'description': "5 best C-W shape features — minimal model"
    },

    # ── Baseline: original CWF (no shape features, for comparison) ──
    "cwf_baseline": {
        'orig_indices': [39, 21, 4, 19, 41, 9, 5, 23, 0, 34,
                         28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40],
        'no_shape': True,
        'description': "Original CWF on denoised — control (21 features)"
    },
}


# ════════════════════════════════════════════════════
# TRAINER
# ════════════════════════════════════════════════════

def scale_data(X):
    """Fit StandardScaler on 3D array (N, SEQ_LEN, F)."""
    N, S, F = X.shape
    X_flat = X.reshape(-1, F)
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled_flat = np.nan_to_num(X_scaled_flat, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled_flat.reshape(N, S, F), scaler


def train_one_set(set_name, cfg, X_orig, X_shape, y, groups):
    """Train a single feature set configuration."""
    no_shape = cfg.get('no_shape', False)
    orig_idx = cfg['orig_indices']
    shape_idx = cfg.get('shape_indices', list(range(N_SHAPE)))  # default: all 10 shape

    # Build feature matrix
    if no_shape:
        X = X_orig[:, :, orig_idx]
        feat_desc = f"{len(orig_idx)} orig features"
    elif len(orig_idx) == 0:
        X = X_shape[:, :, shape_idx]
        feat_desc = f"{len(shape_idx)} shape features"
    else:
        X_o = X_orig[:, :, orig_idx]
        X_s = X_shape[:, :, shape_idx]
        X = np.concatenate([X_o, X_s], axis=2)
        feat_desc = f"{len(orig_idx)} orig + {len(shape_idx)} shape = {X.shape[2]} features"

    n_features = X.shape[2]
    print(f"\n{'='*70}")
    print(f"TRAINING: {set_name}  ({feat_desc})")
    print(f"  {cfg['description']}")
    print(f"{'='*70}")

    X_scaled, scaler = scale_data(X)

    # Class weights
    unique = np.unique(y)
    balanced = compute_class_weight('balanced', classes=unique, y=y)
    weights = np.ones(3)
    for cls, w in zip(unique, balanced):
        weights[cls] = w
    cw_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: blood={weights[0]:.2f} clot={weights[1]:.2f} wall={weights[2]:.2f}")

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # GroupKFold
    gkf = GroupKFold(n_splits=5)
    unique_groups = np.array(sorted(set(groups)))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups])

    best_f1_overall = 0.0
    best_model_state = None
    fold_f1s = []
    fold_cw_f1s = []  # Track clot-wall F1 specifically

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, group_ids)):
        X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

        model = ClotGRU(input_size=n_features).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_f1_fold = 0.0
        best_cw_fold = 0.0
        best_state_fold = None
        wait = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(DEVICE)
                    logits, _ = model(xb)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(yb.numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            f1_per = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)

            # C-W F1: average of clot and wall F1 (what matters most)
            cw_f1 = (f1_per[1] + f1_per[2]) / 2.0 if len(f1_per) == 3 else 0.0

            scheduler.step(f1)

            if f1 > best_f1_fold:
                best_f1_fold = f1
                best_cw_fold = cw_f1
                best_state_fold = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
                marker = f"  ← best (C-W: {cw_f1:.4f})"
            else:
                wait += 1
                marker = ""

            if epoch <= 3 or f1 > best_f1_fold - 0.001:
                print(f"    Fold {fold_idx+1} Ep {epoch:3d} | F1 {f1:.4f} | "
                      f"B:{f1_per[0]:.3f} C:{f1_per[1]:.3f} W:{f1_per[2]:.3f}{marker}")

            if wait >= PATIENCE:
                print(f"    Fold {fold_idx+1} early stop at epoch {epoch}")
                break

        fold_f1s.append(best_f1_fold)
        fold_cw_f1s.append(best_cw_fold)
        print(f"    Fold {fold_idx+1} best F1: {best_f1_fold:.4f} | C-W F1: {best_cw_fold:.4f}")

        if best_f1_fold > best_f1_overall:
            best_f1_overall = best_f1_fold
            best_model_state = best_state_fold

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    mean_cw = np.mean(fold_cw_f1s)
    print(f"\n  {set_name} RESULT")
    print(f"    Mean macro-F1:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"    Mean C-W F1:    {mean_cw:.4f}  ← PRIMARY METRIC")
    print(f"    Best fold F1:   {best_f1_overall:.4f}")
    print(f"    Per-fold macro: {[f'{f:.4f}' for f in fold_f1s]}")
    print(f"    Per-fold C-W:   {[f'{f:.4f}' for f in fold_cw_f1s]}")

    # Save model + scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"clot_gru_dn_shape_{set_name}.pt"
    scaler_path = MODELS_DIR / f"scaler_dn_shape_{set_name}.pkl"
    torch.save(best_model_state, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved: {model_path.name}, {scaler_path.name}")

    return {
        'name': set_name,
        'n_features': n_features,
        'feat_desc': feat_desc,
        'fold_f1s': fold_f1s,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'best_f1': best_f1_overall,
        'mean_cw_f1': mean_cw,
        'fold_cw_f1s': fold_cw_f1s,
    }


# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SHAPE FEATURE TRAINING — DENOISED DATA")
    print("Focus: clot-wall separation (blood accepted from DA)")
    print("=" * 70)

    # Step 1: Load cached all-46 features
    print("\n[1] Loading all-46 feature cache...")
    X_orig, y_orig, groups_orig = load_all46_cache()
    print(f"    All-46: {X_orig.shape[0]} sequences, {len(np.unique(groups_orig))} runs")

    # Step 2: Extract shape features (or load from cache)
    print("\n[2] Extracting shape features...")
    X_shape, y_shape, groups_shape = extract_shape_features()
    print(f"    Shape:  {X_shape.shape[0]} sequences, {len(np.unique(groups_shape))} runs")

    # Verify alignment
    assert X_orig.shape[0] == X_shape.shape[0], \
        f"Mismatch: orig={X_orig.shape[0]} vs shape={X_shape.shape[0]}"
    assert np.array_equal(y_orig, y_shape), "Label mismatch between orig and shape"
    assert np.array_equal(groups_orig, groups_shape), "Group mismatch between orig and shape"

    y = y_orig
    groups = groups_orig

    # Print class distribution
    classes, counts = np.unique(y, return_counts=True)
    print(f"\n    Class distribution:")
    for c, cnt in zip(classes, counts):
        pct = 100 * cnt / len(y)
        name = ['blood', 'clot', 'wall'][c]
        print(f"      {name}: {cnt:,} ({pct:.1f}%)")

    # Step 3: Train each feature set
    print("\n[3] Training feature sets...")
    results = []
    for set_name, cfg in FEATURE_SETS.items():
        res = train_one_set(set_name, cfg, X_orig, X_shape, y, groups)
        results.append(res)

    # Step 4: Summary — ranked by C-W F1 (the metric that matters)
    print("\n" + "=" * 70)
    print("SUMMARY — RANKED BY CLOT-WALL F1")
    print("(Blood is accepted from DA — C-W separation is what matters)")
    print("=" * 70)

    results.sort(key=lambda r: r['mean_cw_f1'], reverse=True)

    print(f"\n{'Set':<22} {'#Feat':>5}  {'CW-F1':>7}  {'Macro-F1':>8}  {'± Std':>7}  {'Description'}")
    print("-" * 90)
    for i, r in enumerate(results):
        marker = " ★" if i == 0 else ""
        desc = FEATURE_SETS[r['name']]['description'][:30]
        print(f"{r['name']:<22} {r['n_features']:5d}  {r['mean_cw_f1']:7.4f}  "
              f"{r['mean_f1']:8.4f}  {r['std_f1']:7.4f}  {desc}{marker}")

    best = results[0]
    cwf_ref = next((r for r in results if r['name'] == 'cwf_baseline'), None)

    print(f"\nBEST (C-W):  {best['name']} — C-W F1 = {best['mean_cw_f1']:.4f}, "
          f"Macro F1 = {best['mean_f1']:.4f}")
    if cwf_ref:
        delta_cw = best['mean_cw_f1'] - cwf_ref['mean_cw_f1']
        delta_f1 = best['mean_f1'] - cwf_ref['mean_f1']
        print(f"vs CWF:      C-W F1 {delta_cw:+.4f}, Macro F1 {delta_f1:+.4f}")
        print(f"CWF baseline: C-W F1 = {cwf_ref['mean_cw_f1']:.4f}, Macro F1 = {cwf_ref['mean_f1']:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
