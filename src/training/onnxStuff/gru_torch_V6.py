# gru_torch_V6.py
"""
Real-time clot detection — V6
Stripped feature set: original 40 + Hjorth mobility, Hjorth complexity, mean abs 2nd derivative.
Total features = 43.  No FFT, no sample entropy, no zero-crossing, no transition features.

Modular compute_features(): skips feature groups not needed by the selected FEATURE_SET.
compute_features_from_array(): batch-mode method for scaler/training (no streaming state).

Feature index map (V6 vs V5):
  f0-f39:  same as V5 (original 40)
  f40:     Hjorth mobility      (was V5 f44)
  f41:     Hjorth complexity     (was V5 f45)
  f42:     Mean abs 2nd deriv   (was V5 f52)
  V5 f40-f43 (spectral), f46 (SampEn), f47 (ZCR), f48-f51 (transition): REMOVED
"""

import os
import warnings
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ────────────────────────────────────────────────
# CONFIG — Single source of truth
# ────────────────────────────────────────────────
SEQ_LEN = 8

WINDOW_SEC = 5.0
REPORT_INTERVAL_MS = 200

GRU_OVERRIDE_THRD_CLOT = 0.80
GRU_OVERRIDE_THRD_WALL = 0.92

# Feature set selection
FEATURE_SET = "clot_wall_focused"

TOTAL_FEATURES = 43

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# ────────────────────────────────────────────────
# FEATURE_SETS — index-based selection
# ────────────────────────────────────────────────
# f0-f9:   Basic stats (mean, std, var, min, max, range, median, windowed std/var/diff)
# f10-f15: Slopes (1-6 sec polyfit)
# f16-f21: Derivative stats (mean, std, var, mean_abs, skew, kurtosis)
# f22-f27: EMA (fast, slow, diff, zero, ratio, abs_diff)
# f28-f35: Detrended (std, std300, mean_abs, zero, std_diff500, mean_abs_diff500, skew, kurtosis)
# f36-f39: Percentiles (p90-mean, IQR, p95-p5, frac_above_p95)
# f40:     Hjorth mobility
# f41:     Hjorth complexity
# f42:     Mean absolute 2nd derivative

FEATURE_SETS = {
    "all":               list(range(TOTAL_FEATURES)),
    "original_40":       list(range(40)),
    "clean_36":          [i for i in range(40) if i not in [14, 25, 31, 33]],
    "top20":             [4, 0, 1, 9, 23, 3, 21, 19, 30, 32, 15, 36, 27, 24, 16, 12, 8, 34, 20, 10],
    # d(clt-wall) > 0.15 — clot vs wall distinguishing features
    "clot_wall_focused": [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40],
    # per-run AUC >= 0.65 (clot vs wall), spectral removed → 20 features
    "auc_cw_20":         [0, 1, 3, 4, 5, 6, 9, 17, 18, 19, 21, 22, 23, 32, 33, 38, 39, 40, 41, 42],
}

# ────────────────────────────────────────────────
# ClotFeatureExtractor — modular, skips unused groups
# ────────────────────────────────────────────────
class ClotFeatureExtractor:
    """
    Computes up to 43 signal features from a resistance buffer.

    If ``active_features`` is given (list of indices), only the feature groups
    that overlap with those indices are computed.  ``compute_features()`` and
    ``compute_features_from_array()`` then return an array of length
    ``len(active_features)`` in the order specified.  Groups that are skipped
    leave their slots at zero, but those slots are never selected.
    """

    # Feature-group index ranges
    _STATS       = set(range(0, 10))
    _SLOPES      = set(range(10, 16))
    _DERIV       = set(range(16, 22))
    _EMA         = set(range(22, 28))
    _DETRENDED   = set(range(28, 36))
    _PERCENTILES = set(range(36, 40))
    _HJORTH      = {40, 41}
    _DERIV2      = {42}

    def __init__(self, sample_rate=150, window_sec=5.0, active_features=None):
        self.fs = sample_rate
        self.window_size = int(sample_rate * window_sec)
        self.buffer = deque(maxlen=self.window_size)
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.alpha_fast = 0.2
        self.alpha_slow = 0.01

        # Active-feature config
        if active_features is not None:
            self._active_features = list(active_features)
            aset = set(active_features)
        else:
            self._active_features = None
            aset = set(range(TOTAL_FEATURES))

        # Per-group flags (evaluated once at init — zero overhead at compute time)
        self._need_stats       = bool(aset & self._STATS)
        self._need_slopes      = bool(aset & self._SLOPES)
        self._need_deriv       = bool(aset & self._DERIV)
        self._need_ema         = bool(aset & self._EMA)
        self._need_detrended   = bool(aset & self._DETRENDED)
        self._need_percentiles = bool(aset & self._PERCENTILES)
        self._need_hjorth      = bool(aset & self._HJORTH)
        self._need_deriv2      = bool(aset & self._DERIV2)
        # Shared dependency: first derivative needed by deriv, hjorth, or deriv2
        self._need_deriv_data  = self._need_deriv or self._need_hjorth or self._need_deriv2

    def reset(self):
        self.buffer.clear()
        self.ema_fast = self.ema_slow = 0.0

    def update(self, r):
        self.buffer.append(float(r))
        if len(self.buffer) == 1:
            self.ema_fast = self.ema_slow = self.buffer[0]
        else:
            self.ema_fast = self.alpha_fast * self.buffer[-1] + (1 - self.alpha_fast) * self.ema_fast
            self.ema_slow = self.alpha_slow * self.buffer[-1] + (1 - self.alpha_slow) * self.ema_slow

    # ── Public API ──────────────────────────────────

    def compute_features(self):
        """Streaming mode: compute from internal buffer + EMA state."""
        n_out = len(self._active_features) if self._active_features else TOTAL_FEATURES
        if len(self.buffer) < 100:
            return np.zeros(n_out, dtype=np.float32)
        data = np.array(self.buffer, dtype=np.float32)
        return self._compute(data, self.ema_fast, self.ema_slow)

    def compute_features_from_array(self, data, ema_fast, ema_slow):
        """Batch mode: compute from a numpy window + precomputed EMA values."""
        n_out = len(self._active_features) if self._active_features else TOTAL_FEATURES
        if len(data) < 100:
            return np.zeros(n_out, dtype=np.float32)
        return self._compute(np.asarray(data, dtype=np.float32), ema_fast, ema_slow)

    # ── Internal compute (modular) ──────────────────

    def _compute(self, data, ema_fast, ema_slow):
        n = len(data)
        f = np.zeros(TOTAL_FEATURES, dtype=np.float32)

        # Shared: first derivative (reused by deriv, hjorth, deriv2 groups)
        deriv = np.diff(data) if self._need_deriv_data else None

        # ── f0-f9: Basic stats ──
        if self._need_stats:
            f[0] = data.mean()
            f[1] = data.std()
            f[2] = data.var()
            f[3] = data.min()
            f[4] = data.max()
            f[5] = np.ptp(data)
            f[6] = np.median(data)
            if n >= 500:
                tail = data[-500:]
                f[7] = np.std(tail)
                f[8] = np.var(tail)
                f[9] = np.mean(np.abs(np.diff(tail)))

        # ── f10-f15: Slopes (most expensive group — 6 polyfits) ──
        if self._need_slopes:
            for j, secs in enumerate([1, 2, 3, 4, 5, 6]):
                ns = min(int(secs * self.fs), n)
                if ns >= 2:
                    slope = np.polyfit(np.arange(ns), data[-ns:], 1)[0]
                    f[10 + j] = np.abs(slope) if np.isfinite(slope) else 0.0

        # ── f16-f21: Derivative stats ──
        if self._need_deriv and deriv is not None and len(deriv) > 10:
            f[16] = deriv.mean()
            f[17] = deriv.std()
            f[18] = deriv.var()
            f[19] = np.mean(np.abs(deriv))
            f[20] = stats.skew(deriv) if len(deriv) >= 3 else 0
            f[21] = stats.kurtosis(deriv) if len(deriv) >= 4 else 0

        # ── f22-f27: EMA ──
        if self._need_ema:
            f[22] = ema_fast
            f[23] = ema_slow
            f[24] = ema_fast - ema_slow
            f[25] = 0.0
            f[26] = ema_fast / (ema_slow + 1e-6)
            f[27] = np.abs(ema_fast - ema_slow)

        # ── f28-f35: Detrended ──
        if self._need_detrended:
            kernel = 450
            if n >= kernel:
                trend = np.convolve(data, np.ones(kernel) / kernel, 'valid')
                detr = data[-len(trend):] - trend
                r600 = detr[-min(600, len(detr)):]
                f[28] = np.std(r600)
                f[29] = np.std(r600[:300])
                f[30] = np.mean(np.abs(r600))
                f[31] = 0.0
                if n >= 500:
                    d500 = np.diff(data[-500:])
                    f[32] = np.std(d500)
                    f[33] = np.mean(np.abs(d500))
                f[34] = stats.skew(r600) if len(r600) >= 3 else 0
                f[35] = stats.kurtosis(r600) if len(r600) >= 4 else 0

        # ── f36-f39: Percentiles ──
        if self._need_percentiles:
            f[36] = np.percentile(data, 90) - data.mean()
            f[37] = np.percentile(data, 75) - np.percentile(data, 25)
            p95 = np.percentile(data, 95)
            f[38] = p95 - np.percentile(data, 5)
            f[39] = np.sum(data > p95) / n

        # ── f40-f42: Hjorth + mean abs 2nd derivative (shared ddx) ──
        if (self._need_hjorth or self._need_deriv2) and deriv is not None and len(deriv) > 1:
            ddx = np.diff(deriv)
            if self._need_hjorth:
                data_var = data.var() + 1e-8
                dx_var = deriv.var() + 1e-8
                ddx_var = ddx.var() + 1e-8
                mob = np.sqrt(dx_var / data_var)
                f[40] = mob
                f[41] = np.sqrt(ddx_var / dx_var) / (mob + 1e-8)
            if self._need_deriv2:
                f[42] = np.mean(np.abs(ddx))

        if self._active_features is not None:
            return f[self._active_features]
        return f.copy()


# ────────────────────────────────────────────────
# Dynamic dimension & paths
# ────────────────────────────────────────────────
active_idx = FEATURE_SETS[FEATURE_SET]
active_dim = len(active_idx)
_idx_hash = hash(tuple(active_idx)) % 0xFFFF
dim_str = f"{FEATURE_SET}_{active_dim}_{_idx_hash:04x}"

SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_seq{SEQ_LEN}_{dim_str}.pkl"
MODEL_PATH = PROJECT_ROOT / "src" / "training" / "clot_gru_trained.pt"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FOLDER = PROJECT_ROOT / "inference_deploy" / "Results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────
# ClotGRU Model
# ────────────────────────────────────────────────
class ClotGRU(nn.Module):
    def __init__(self, input_size=None, hidden_size=32, output_size=3):
        super().__init__()
        if input_size is None:
            input_size = active_dim

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

        self.fc1 = nn.Linear(hidden_size, 24)
        self.fc2 = nn.Linear(24, output_size)

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


# ────────────────────────────────────────────────
# LiveClotDetector
# ────────────────────────────────────────────────
class LiveClotDetector:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.scaler = joblib.load(scaler_path)
        self.model = ClotGRU().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        self.hidden = None
        self.posterior = np.array([0.95, 0.025, 0.025], dtype=np.float32)
        self.feat_history = deque(maxlen=SEQ_LEN)

    @torch.no_grad()
    def predict(self, active_feats, da_label=None):
        """
        active_feats: already-selected active features (active_dim,).
        The extractor returns these directly when active_features is set.
        """
        scaled = self.scaler.transform(active_feats.reshape(1, -1))[0]

        self.feat_history.append(scaled)

        if len(self.feat_history) < SEQ_LEN:
            pad = list(self.feat_history)[0] if self.feat_history else scaled
            seq_list = [pad] * (SEQ_LEN - len(self.feat_history)) + list(self.feat_history)
        else:
            seq_list = list(self.feat_history)

        seq = np.array(seq_list, dtype=np.float32)
        x = torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)

        logits, self.hidden = self.model(x, self.hidden)
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

        probs = torch.softmax(logits, 1).squeeze(0).cpu().numpy()

        prior_idx = np.argmax(self.posterior)

        if da_label is not None:
            if da_label == 0:
                self.posterior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self.hidden = None
                self.feat_history.clear()
                return self.posterior.copy()

            elif da_label in (1, 2):
                da_probs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                da_probs[da_label] = 0.92
                da_probs[0] = 0.04
                da_probs[3 - da_label] = 0.04

                gru_top_idx = np.argmax(probs)
                if gru_top_idx != da_label:
                    thrd = GRU_OVERRIDE_THRD_CLOT if da_label == 1 else GRU_OVERRIDE_THRD_WALL
                    if probs[gru_top_idx] <= thrd:
                        probs = da_probs

        if prior_idx == 0:
            alpha_history, alpha_new = 0.78, 0.22
        else:
            new_idx = np.argmax(probs)
            if new_idx == 0:
                alpha_history, alpha_new = 0.35, 0.65      # quick exit to blood
            elif new_idx == prior_idx:
                alpha_history, alpha_new = 0.96, 0.04      # confirm same class
            else:
                alpha_history, alpha_new = 0.995, 0.005      # resist clot↔wall flicker

        self.posterior = alpha_history * self.posterior + alpha_new * probs

        if da_label in (1, 2):
            final_idx = np.argmax(self.posterior)
            if final_idx != da_label:
                gru_top_idx = np.argmax(probs)
                thrd = GRU_OVERRIDE_THRD_CLOT if da_label == 1 else GRU_OVERRIDE_THRD_WALL
                if probs[gru_top_idx] < thrd:
                    da_probs = np.array([0.04, 0.04, 0.04], dtype=np.float32)
                    da_probs[da_label] = 0.92
                    self.posterior = da_probs

        return self.posterior.copy()

# ────────────────────────────────────────────────
#  Main Processing
# ────────────────────────────────────────────────

def process_file(filepath: Path,
                 all_gt_labels: list,
                 all_da_labels: list,
                 all_ml_preds: list,
                 all_override_times: list):

    study_name = filepath.stem
    print(f"\nProcessing: {study_name}")

    df = pd.read_parquet(filepath)
    time_ms = df['timeInMS'].values
    resistance = df['magRLoadAdjusted'].values.astype(np.float32)
    gt_labels = df.get('label', None)
    da_labels = df.get('da_label', None) if 'da_label' in df.columns else None

    extractor = ClotFeatureExtractor(active_features=active_idx)
    detector = LiveClotDetector()

    results = []
    last_report = -REPORT_INTERVAL_MS

    for i, (t, r) in enumerate(zip(time_ms, resistance)):
        extractor.update(float(r))

        if t - last_report >= REPORT_INTERVAL_MS:
            feats = extractor.compute_features()
            da_now = da_labels[i] if da_labels is not None else None
            post = detector.predict(feats, da_now)
            status = np.argmax(post)
            entropy = -np.sum(post * np.log(post + 1e-12))

            results.append({
                'time': t/1000.0,
                'prediction': status,
                'resistance': float(r),
                'Nprob': float(post[0]),
                'Cprob': float(post[1]),
                'Wprob': float(post[2]),
                'entropy': float(entropy)
            })
            last_report = t

    results_df = pd.DataFrame(results)

    # Save detection_results parquet (and csv)
    results_df.to_parquet(OUTPUT_FOLDER / f"{study_name}_detection_results.parquet", index=False)
    results_df.to_csv(OUTPUT_FOLDER / f"{study_name}_detection_results.csv", index=False)
    print(f"  Saved detection_results.parquet")

    # ── Probability plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    colors = {0:'black', 1:'red', 2:'blue'}
    for lbl, name in [(0,'blood'),(1,'clot'),(2,'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(f'{study_name} — Detected Labels')
    ax1.grid(True, alpha=0.3)

    ax2.plot(results_df['time'], results_df['Cprob'], color='red',   label='P(clot)', linewidth=1.8)
    ax2.plot(results_df['time'], results_df['Wprob'], color='blue',  label='P(wall)', linewidth=1.8)

    blood_dom = (results_df['Nprob'] > results_df['Cprob']) & (results_df['Nprob'] > results_df['Wprob'])
    ax2.fill_between(results_df['time'], 0, 1, where=blood_dom, color='gray', alpha=0.12, label='Blood dominant')

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{study_name} — Clot & Wall Detection Probabilities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"{study_name}_detected_vs_clot_wall_probs.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved probability plot")

    # ── Three-panel plot ──
    if gt_labels is not None and da_labels is not None:
        gt = gt_labels.values.astype(int)
        da = da_labels.values.astype(int)
        full_times = time_ms / 1000.0

        interp_ml = np.interp(full_times, results_df['time'], results_df['prediction'])
        interp_ml = np.round(interp_ml).astype(int)

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True,
                                 gridspec_kw={'height_ratios': [1,1,1]})

        lbl_names = {0:'blood', 1:'clot', 2:'wall'}

        ax = axes[0]
        for lbl in [0,1,2]:
            mask = results_df['prediction'] == lbl
            ax.scatter(results_df['time'][mask], results_df['resistance'][mask],
                       c=colors[lbl], s=5, label=lbl_names[lbl], alpha=0.85)

        ml_da = np.interp(results_df['time'], full_times, da)
        ml_da = np.round(ml_da).astype(int)
        diff = (results_df['prediction'].values != ml_da)
        diff_diff = np.diff(diff.astype(int))
        starts = np.where(diff_diff == 1)[0] + 1
        ends = np.where(diff_diff == -1)[0] + 1
        if diff[0]: starts = np.insert(starts, 0, 0)
        if diff[-1]: ends = np.append(ends, len(diff))

        for s, e in zip(starts, ends):
            ax.axvspan(results_df['time'].iloc[s], results_df['time'].iloc[e-1],
                       facecolor='#e8e8e8', alpha=0.55, label='ML ≠ DA' if s==starts[0] else None)

        ax.set_title(f'{study_name} — ML Predictions (200 ms reporting)')
        ax.set_ylabel('Resistance (Ω)')
        ax.grid(True, alpha=0.3)

        for ax_idx, (title, data) in enumerate([("DA Labels (full 150 Hz)", da), ("Ground Truth Labels (full 150 Hz)", gt)]):
            ax = axes[ax_idx+1]
            for lbl in [0,1,2]:
                mask = data == lbl
                ax.scatter(full_times[mask], resistance[mask], c=colors[lbl], s=2, label=lbl_names[lbl], alpha=0.7)
            ax.set_title(f'{study_name} — {title}')
            ax.set_ylabel('Resistance (Ω)')
            ax.grid(True, alpha=0.3)
            if ax_idx == 1:
                ax.set_xlabel('Time (seconds)')

        plt.tight_layout(h_pad=0.8)
        plt.savefig(OUTPUT_FOLDER / f"{study_name}_ml_da_gt_three_panel.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved three-panel plot")

        # Metrics
        print(f"\n{study_name} metrics:")
        print(f"DA  Acc: {accuracy_score(gt, da):.4f}  F1: {f1_score(gt, da, average='macro'):.4f} "
              f"Prec: {precision_score(gt, da, average='macro'):.4f}  Rec: {recall_score(gt, da, average='macro'):.4f}")
        print(f"ML  Acc: {accuracy_score(gt, interp_ml):.4f}  F1: {f1_score(gt, interp_ml, average='macro'):.4f} "
              f"Prec: {precision_score(gt, interp_ml, average='macro'):.4f}  Rec: {recall_score(gt, interp_ml, average='macro'):.4f}")
        print(f"Improvement: Acc {accuracy_score(gt, interp_ml)-accuracy_score(gt, da):+.4f}   "
              f"F1 {f1_score(gt, interp_ml, average='macro')-f1_score(gt, da, average='macro'):+.4f}")

        # Override analysis
        override_mask = (interp_ml != da)
        n_overrides = override_mask.sum()
        if n_overrides > 0:
            correct_overrides = ((interp_ml[override_mask] == gt[override_mask]).sum())
            harmful_overrides = ((da[override_mask] == gt[override_mask]).sum())
            override_prec = correct_overrides / n_overrides

            da_cw_errors = ((da != gt) & ((gt == 1) | (gt == 2))).sum()
            override_rec = correct_overrides / da_cw_errors if da_cw_errors > 0 else 0.0

            print(f"\n  Override analysis ({study_name}):")
            print(f"    Total overrides:   {n_overrides}")
            print(f"    Correct (ML right, DA wrong): {correct_overrides}")
            print(f"    Harmful (DA right, ML wrong): {harmful_overrides}")
            print(f"    Override Precision: {override_prec:.4f}")
            print(f"    Override Recall:    {override_rec:.4f}  (of {da_cw_errors} DA clot/wall errors)")
        else:
            print(f"\n  No overrides in {study_name}")

        if gt_labels is not None and da_labels is not None:
            all_gt_labels.extend(gt)
            all_da_labels.extend(da)
            all_ml_preds.extend(interp_ml)

        overrides = np.where(interp_ml != da)[0]
        all_override_times.extend(full_times[overrides])

    print(f"Finished {study_name}\n")


# ────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────

def main():
    files = sorted(TEST_DATA_DIR.glob("*_labeled_segment.parquet"))
    print(f"Found {len(files)} labeled_segment files.\n")

    all_gt_labels     = []
    all_da_labels     = []
    all_ml_preds      = []
    all_override_times = []

    for f in files:
        process_file(f,
                     all_gt_labels=all_gt_labels,
                     all_da_labels=all_da_labels,
                     all_ml_preds=all_ml_preds,
                     all_override_times=all_override_times)

    # ── Global summary ──
    if all_gt_labels:
        print("\n" + "="*70)
        print("GLOBAL SUMMARY ACROSS ALL STUDIES")
        print("="*70)

        acc_da  = accuracy_score(all_gt_labels, all_da_labels)
        f1_da   = f1_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        prec_da = precision_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        rec_da  = recall_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)

        acc_ml  = accuracy_score(all_gt_labels, all_ml_preds)
        f1_ml   = f1_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        prec_ml = precision_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        rec_ml  = recall_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)

        print(f"DA  Accuracy: {acc_da:.4f}    F1-macro: {f1_da:.4f}")
        print(f"ML  Accuracy: {acc_ml:.4f}    F1-macro: {f1_ml:.4f}")
        print(f"Improvement: Acc {acc_ml - acc_da:+.4f}   F1 {f1_ml - f1_da:+.4f}")
        print("")
        print(f"DA  Precision: {prec_da:.4f}    Recall: {rec_da:.4f}")
        print(f"ML  Precision: {prec_ml:.4f}    Recall: {rec_ml:.4f}")
        print(f"Improvement: Precision {prec_ml - prec_da:+.4f}   Recall {rec_ml - rec_da:+.4f}")

        gt_arr = np.array(all_gt_labels)
        da_arr = np.array(all_da_labels)
        ml_arr = np.array(all_ml_preds)

        g_override_mask = (ml_arr != da_arr)
        g_n_overrides = g_override_mask.sum()

        print(f"\n{'─'*70}")
        print(f"GLOBAL OVERRIDE ANALYSIS")
        print(f"{'─'*70}")
        print(f"Total overrides across all studies: {g_n_overrides}")

        if g_n_overrides > 0:
            g_correct = (ml_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_harmful = (da_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_neither = g_n_overrides - g_correct - g_harmful
            g_override_prec = g_correct / g_n_overrides

            g_da_cw_errors = ((da_arr != gt_arr) & ((gt_arr == 1) | (gt_arr == 2))).sum()
            g_override_rec = g_correct / g_da_cw_errors if g_da_cw_errors > 0 else 0.0

            print(f"  Correct overrides (ML right, DA wrong): {g_correct}")
            print(f"  Harmful overrides (DA right, ML wrong): {g_harmful}")
            print(f"  Neither correct (both wrong differently): {g_neither}")
            print(f"")
            print(f"  Override Precision: {g_override_prec:.4f}  (target: >0.85)")
            print(f"  Override Recall:    {g_override_rec:.4f}  (of {g_da_cw_errors} DA clot/wall errors)")
            print(f"  Net benefit:        {g_correct - g_harmful:+d} samples")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
