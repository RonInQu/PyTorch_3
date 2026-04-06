# gru_torch_V5.py
"""
Real-time clot detection — V5 (cleaned, REDUCE_DIM compatible version of working V2)
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

# Feature set selection — replaces REDUCE_DIM
# "all"          = all features (including new spectral/complexity/transition)
# "original_40"  = original 40 features only (backward compatible)
# "top20"        = top 20 from permutation importance analysis
# "clean_36"     = original 40 minus 4 dead features (f14, f25, f31, f33)
FEATURE_SET = "all"

TOTAL_FEATURES = 53  # Total features computed by ClotFeatureExtractor

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# ────────────────────────────────────────────────
# FEATURE_SETS — index-based selection
# ────────────────────────────────────────────────
# Features 0-39:  original features
# Features 40-43: spectral (centroid, flatness, low/high band power ratio, dominant freq)
# Features 44-46: complexity (Hjorth mobility, Hjorth complexity, SampEn decimated)
# Features 47:    zero-crossing rate
# Features 48-52: transition (variance ratio 1s/4s, CUSUM max, autocorr lag-1,
#                              energy ratio last1s/full, mean abs 2nd derivative)

FEATURE_SETS = {
    "all":          list(range(TOTAL_FEATURES)),
    "original_40":  list(range(40)),
    "clean_36":     [i for i in range(40) if i not in [14, 25, 31, 33]],
    "top20":        [4, 0, 1, 9, 23, 3, 21, 19, 30, 32, 15, 36, 27, 24, 16, 12, 8, 34, 20, 10],
    # d(clt-wall) > 0.15 — only features that help distinguish clot from wall
    "clot_wall_focused": [39, 21, 4, 19, 45, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 52, 27, 1, 20, 44],
    # per-run AUC >= 0.65 (clot vs wall), minus f42, plus f1/f38 — 24 features
    "auc_cw_24": [0, 1, 3, 4, 5, 6, 9, 17, 18, 19, 21, 22, 23, 32, 33, 38, 39, 40, 41, 44, 45, 46, 47, 52],
}

# ────────────────────────────────────────────────
# ClotFeatureExtractor
# ────────────────────────────────────────────────
class ClotFeatureExtractor:
    def __init__(self, sample_rate=150, window_sec=5.0):
        self.fs = sample_rate
        self.window_size = int(sample_rate * window_sec)
        self.buffer = deque(maxlen=self.window_size)
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.alpha_fast = 0.2
        self.alpha_slow = 0.01
        self.zero_idx = [3, 6, 7, 25, 28, 29, 30, 31, 32, 33, 34, 35]  # kept for backward compat

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

    def compute_features(self):
        if len(self.buffer) < 100:
            return np.zeros(TOTAL_FEATURES, dtype=np.float32)

        data = np.array(self.buffer, dtype=np.float32)
        n = len(data)
        f = np.zeros(TOTAL_FEATURES, dtype=np.float32)
        i = 0

        # ── Original 40 features (f0-f39) ──────────────────

        # Basic stats (f0-f9)
        f[i:i+10] = [data.mean(), data.std(), data.var(), data.min(), data.max(),
                     np.ptp(data), np.median(data),
                     np.std(data[-500:]) if n >= 500 else 0,
                     np.var(data[-500:]) if n >= 500 else 0,
                     np.mean(np.abs(np.diff(data[-500:]))) if n >= 500 else 0]
        i += 10

        # Slopes (f10-f15)
        for secs in [1,2,3,4,5,6]:
            ns = min(int(secs * self.fs), n)
            if ns >= 2:
                slope = np.polyfit(np.arange(ns), data[-ns:], 1)[0]
                slope = np.abs(slope)
                f[i] = slope if np.isfinite(slope) else 0.0
            i += 1

        # Derivative (f16-f21)
        deriv = np.diff(data)
        if len(deriv) > 10:
            f[i:i+6] = [deriv.mean(), deriv.std(), deriv.var(),
                        np.mean(np.abs(deriv)),
                        stats.skew(deriv) if len(deriv)>=3 else 0,
                        stats.kurtosis(deriv) if len(deriv)>=4 else 0]
        i += 6

        # EMA (f22-f27)
        f[i:i+6] = [self.ema_fast, self.ema_slow, self.ema_fast-self.ema_slow,
                    0.0, self.ema_fast/(self.ema_slow+1e-6),
                    np.abs(self.ema_fast-self.ema_slow)]
        i += 6

        # Detrended (f28-f35)
        kernel = 450
        if n >= kernel:
            trend = np.convolve(data, np.ones(kernel)/kernel, 'valid')
            detr = data[-len(trend):] - trend
            r600 = detr[-min(600,len(detr)):]
            f[i:i+8] = [np.std(r600), np.std(r600[:300]), np.mean(np.abs(r600)), 0.0,
                        np.std(np.diff(data[-500:])) if n>=500 else 0,
                        np.mean(np.abs(np.diff(data[-500:]))) if n>=500 else 0,
                        stats.skew(r600) if len(r600)>=3 else 0,
                        stats.kurtosis(r600) if len(r600)>=4 else 0]
        i += 8

        # Percentiles (f36-f39)
        f[i:i+4] = [np.percentile(data,90)-data.mean(),
                    np.percentile(data,75)-np.percentile(data,25),
                    np.percentile(data,95)-np.percentile(data,5),
                    np.sum(data > np.percentile(data,95))/len(data)]
        i += 4

        # ── New spectral features (f40-f43) ────────────────

        mag = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        mag_sum = mag.sum() + 1e-8
        # f40: spectral centroid
        f[i] = (freqs * mag).sum() / mag_sum
        i += 1
        # f41: spectral flatness
        mag_safe = mag + 1e-10
        f[i] = np.exp(np.log(mag_safe).mean()) / (mag_safe.mean() + 1e-10)
        i += 1
        # f42: band power ratio (low 0-15 Hz / high 40-75 Hz)
        low_mask = (freqs >= 0) & (freqs < 15)
        high_mask = (freqs >= 40) & (freqs < 75)
        power_low = (mag[low_mask]**2).sum()
        power_high = (mag[high_mask]**2).sum() + 1e-8
        f[i] = power_low / power_high
        i += 1
        # f43: dominant frequency
        f[i] = freqs[np.argmax(mag)]
        i += 1

        # ── New complexity features (f44-f47) ──────────────

        # f44: Hjorth mobility
        dx = np.diff(data)
        data_var = data.var() + 1e-8
        dx_var = dx.var() + 1e-8
        f[i] = np.sqrt(dx_var / data_var)
        i += 1
        # f45: Hjorth complexity
        ddx = np.diff(dx)
        ddx_var = ddx.var() + 1e-8
        mob_data = np.sqrt(dx_var / data_var)
        mob_dx = np.sqrt(ddx_var / dx_var)
        f[i] = mob_dx / (mob_data + 1e-8)
        i += 1
        # f46: Sample entropy (decimated to ~75 samples)
        decimate_factor = max(1, n // 75)
        data_dec = data[::decimate_factor]
        n_dec = len(data_dec)
        r_tol = 0.2 * (data_dec.std() + 1e-8)
        m = 2
        count_m, count_m1 = 0, 0
        for j in range(n_dec - m):
            tmpl_m = data_dec[j:j+m]
            tmpl_m1 = data_dec[j:j+m+1] if j+m+1 <= n_dec else None
            for k in range(j+1, n_dec - m):
                if np.max(np.abs(tmpl_m - data_dec[k:k+m])) < r_tol:
                    count_m += 1
                    if tmpl_m1 is not None and k+m+1 <= n_dec:
                        if np.max(np.abs(tmpl_m1 - data_dec[k:k+m+1])) < r_tol:
                            count_m1 += 1
        f[i] = -np.log((count_m1 + 1e-8) / (count_m + 1e-8))
        i += 1
        # f47: zero-crossing rate
        data_centered = data - data.mean()
        f[i] = ((data_centered[:-1] * data_centered[1:]) < 0).sum() / n
        i += 1

        # ── New transition features (f48-f52) ──────────────

        # f48: variance ratio (recent 1s / older 4s)
        n_1s = min(int(1.0 * self.fs), n)
        n_4s = min(int(4.0 * self.fs), n)
        if n >= n_1s + n_4s:
            var_recent = data[-n_1s:].var() + 1e-8
            var_older = data[-(n_1s + n_4s):-n_1s].var() + 1e-8
            f[i] = var_recent / var_older
        else:
            f[i] = 1.0
        i += 1
        # f49: CUSUM max (cumulative sum of deviations from running mean)
        mu = data.mean()
        cusum = np.cumsum(data - mu)
        f[i] = np.max(np.abs(cusum)) / (n + 1e-8)
        i += 1
        # f50: autocorrelation at lag 1
        if n > 1:
            data_dm = data - data.mean()
            c0 = np.dot(data_dm, data_dm)
            c1 = np.dot(data_dm[:-1], data_dm[1:])
            f[i] = c1 / (c0 + 1e-8)
        i += 1
        # f51: energy ratio (last 1s energy / full 5s energy)
        energy_full = (data**2).sum() + 1e-8
        energy_recent = (data[-n_1s:]**2).sum()
        f[i] = energy_recent / energy_full
        i += 1
        # f52: mean absolute second derivative
        if len(deriv) > 1:
            deriv2 = np.diff(deriv)
            f[i] = np.mean(np.abs(deriv2))
        i += 1

        return f.copy()


# ────────────────────────────────────────────────
# Dynamic dimension & paths
# ────────────────────────────────────────────────
active_idx = FEATURE_SETS[FEATURE_SET]
active_dim = len(active_idx)
# Include hash of actual indices so reordering invalidates cache/scaler
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
            input_size = active_dim                     # Use dynamic size

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
        # x shape: (batch, SEQ_LEN, input_size)
        out, hidden = self.gru(x, hidden)
        out = out[:, -1]
        out = torch.relu(self.fc1(out))
        logits = self.fc2(out)
        return logits, hidden


# ────────────────────────────────────────────────
# LiveClotDetector — FIXED
# ────────────────────────────────────────────────
class LiveClotDetector:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.scaler = joblib.load(scaler_path)
        self.model = ClotGRU().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        self.hidden = None
        self.posterior = np.array([0.95, 0.025, 0.025], dtype=np.float32)
        
        # NEW: Feature history buffer for SEQ_LEN
        self.feat_history = deque(maxlen=SEQ_LEN)   # will store scaled active features

    @torch.no_grad()
    def predict(self, raw_feats, da_label=None):
        # Select active features
        feats_active = raw_feats[active_idx]

        scaled = self.scaler.transform(feats_active.reshape(1, -1))[0]

        # Add current scaled features to history buffer
        self.feat_history.append(scaled)

        # If history is not full yet, pad with the first available feature
        if len(self.feat_history) < SEQ_LEN:
            pad = list(self.feat_history)[0] if self.feat_history else scaled
            seq_list = [pad] * (SEQ_LEN - len(self.feat_history)) + list(self.feat_history)
        else:
            seq_list = list(self.feat_history)

        seq = np.array(seq_list, dtype=np.float32)   # (SEQ_LEN, feat_size)
        x = torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)  # (1, SEQ_LEN, feat_size)

        logits, self.hidden = self.model(x, self.hidden)
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

        probs = torch.softmax(logits, 1).squeeze(0).cpu().numpy()
        
        prior_idx = np.argmax(self.posterior)
        
        if da_label is not None:
            if da_label == 0:  # DA says blood → force blood and reset everything
                self.posterior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self.hidden = None
                self.feat_history.clear()          # ← Important: reset history
                return self.posterior.copy()
            
            elif da_label in (1, 2):  # DA says clot or wall
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
            if np.argmax(probs) == 0:
                alpha_history, alpha_new = 0.35, 0.65
            else:
                alpha_history, alpha_new = 0.96, 0.04
        
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
    time_ms = df['timeInMS'].values          # ← exact column from your old code
    resistance = df['magRLoadAdjusted'].values.astype(np.float32)
    gt_labels = df.get('label', None)
    da_labels = df.get('da_label', None) if 'da_label' in df.columns else None

    extractor = ClotFeatureExtractor()
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

    # ── Probability plot (exact style from your image) ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top: Detected Labels scatter
    colors = {0:'black', 1:'red', 2:'blue'}
    for lbl, name in [(0,'blood'),(1,'clot'),(2,'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(f'{study_name} — Detected Labels')
    ax1.grid(True, alpha=0.3)

    # Bottom: Only P(clot) & P(wall) with grey blood dominant shading
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
    print(f"  Saved probability plot (exact style)")

    # ── Three-panel plot (exact style from your image) ──
    if gt_labels is not None and da_labels is not None:
        gt = gt_labels.values.astype(int)
        da = da_labels.values.astype(int)
        full_times = time_ms / 1000.0

        interp_ml = np.interp(full_times, results_df['time'], results_df['prediction'])
        interp_ml = np.round(interp_ml).astype(int)

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True,
                                 gridspec_kw={'height_ratios': [1,1,1]})

        colors = {0:'black', 1:'red', 2:'blue'}
        lbl_names = {0:'blood', 1:'clot', 2:'wall'}

        # Top: ML sparse with grey shading
        ax = axes[0]
        for lbl in [0,1,2]:
            mask = results_df['prediction'] == lbl
            ax.scatter(results_df['time'][mask], results_df['resistance'][mask],
                       c=colors[lbl], s=5, label=lbl_names[lbl], alpha=0.85)

        # Grey ML ≠ DA shading
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

        # Middle & Bottom (DA & GT)
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
        print(f"  Saved three-panel plot (exact style)")

        # Metrics
        print(f"\n{study_name} metrics:")
        print(f"DA  Acc: {accuracy_score(gt, da):.4f}  F1: {f1_score(gt, da, average='macro'):.4f} "
              f"Prec: {precision_score(gt, da, average='macro'):.4f}  Rec: {recall_score(gt, da, average='macro'):.4f}")
        print(f"ML  Acc: {accuracy_score(gt, interp_ml):.4f}  F1: {f1_score(gt, interp_ml, average='macro'):.4f} "
              f"Prec: {precision_score(gt, interp_ml, average='macro'):.4f}  Rec: {recall_score(gt, interp_ml, average='macro'):.4f}")
        print(f"Improvement: Acc {accuracy_score(gt, interp_ml)-accuracy_score(gt, da):+.4f}   "
              f"F1 {f1_score(gt, interp_ml, average='macro')-f1_score(gt, da, average='macro'):+.4f}")

        # ── Override analysis ──
        override_mask = (interp_ml != da)  # ML changed DA's call
        n_overrides = override_mask.sum()
        if n_overrides > 0:
            correct_overrides = ((interp_ml[override_mask] == gt[override_mask]).sum())
            harmful_overrides = ((da[override_mask] == gt[override_mask]).sum())
            override_prec = correct_overrides / n_overrides

            # DA errors on clot/wall that ML could have caught
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

        # Collect for global summary (only if ground truth & DA exist)
        if gt_labels is not None and da_labels is not None:
            # Append to global lists
            all_gt_labels.extend(gt)
            all_da_labels.extend(da)
            all_ml_preds.extend(interp_ml)

        # Track override times
        overrides = np.where(interp_ml != da)[0]
        all_override_times.extend(full_times[overrides])
        
    print(f"Finished {study_name}\n")


# ────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────

def main():
    files = sorted(TEST_DATA_DIR.glob("*_labeled_segment.parquet"))
    print(f"Found {len(files)} labeled_segment files.\n")
    
    # ── Global aggregation lists ──
    all_gt_labels     = []
    all_da_labels     = []
    all_ml_preds      = []
    all_override_times = []  # optional — only if you track override moments

    for f in files:
        process_file(f,
                     all_gt_labels=all_gt_labels,
                     all_da_labels=all_da_labels,
                     all_ml_preds=all_ml_preds,
                     all_override_times=all_override_times)
        
    # ────────────────────────────────────────────────
    # Global summary – after ALL files processed
    # ────────────────────────────────────────────────
    if all_gt_labels:  # only if we have any ground truth data
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

        # ── Global Override Analysis ──
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
            g_neither = g_n_overrides - g_correct - g_harmful  # both wrong, but differently
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