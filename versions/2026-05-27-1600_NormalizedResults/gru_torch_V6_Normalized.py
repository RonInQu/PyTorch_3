# gru_torch_V6_Normalized.py
"""
Real-time clot detection — V6 Normalized
Variant of gru_torch_V6 for division-normalized data: R_norm = (R - baseline) / baseline.

Key differences from gru_torch_V6:
  - Input column: 'R_normalized' (dimensionless, blood≈0, clot≈0.5-3, wall≈1-5+)
  - f50: Just the mean (data is already baseline-relative)
  - f47: Plateau threshold ±0.005 (was ±2 Ω)
  - f48: Settling threshold ±0.01 (was ±5 Ω)
  - Rise-shape gate: r_range > 0.01 (was > 5.0 Ω)
  - All other features (stats, slopes, derivatives, EMA, detrended, percentiles,
    Hjorth, short-slopes, shape) work on the normalized signal as-is since they
    are relative/ratio-based or get StandardScaled anyway.
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
from scipy.signal import medfilt, find_peaks, butter, sosfiltfilt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
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

TEMPERATURE = 1.5

EMA_BLOOD_PRIOR_HISTORY = 0.78
EMA_BLOOD_PRIOR_NEW     = 1 - EMA_BLOOD_PRIOR_HISTORY
EMA_EXIT_TO_BLOOD_HISTORY = 0.35
EMA_EXIT_TO_BLOOD_NEW     = 1 - EMA_EXIT_TO_BLOOD_HISTORY
EMA_SAME_CLASS_HISTORY  = 0.97
EMA_SAME_CLASS_NEW      = 1 - EMA_SAME_CLASS_HISTORY
EMA_CROSS_CLASS_HISTORY = 0.99
EMA_CROSS_CLASS_NEW     = 1 - EMA_CROSS_CLASS_HISTORY

DA_LABEL_CONFIDENCE = 0.92
DA_OTHER_CONFIDENCE = (1.0 - DA_LABEL_CONFIDENCE) / 2

INIT_BLOOD_PROB = 0.95
INIT_CLOT_PROB  = (1 - INIT_BLOOD_PROB) / 2
INIT_WALL_PROB  = (1 - INIT_BLOOD_PROB) / 2

# Feature set selection
FEATURE_SET = "clot_wall_focused"

TOTAL_FEATURES = 65

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# ────────────────────────────────────────────────
# FEATURE_SETS — index-based selection
# ────────────────────────────────────────────────
FEATURE_SETS = {
    "all":               list(range(TOTAL_FEATURES)),
    "original_40":       list(range(40)),
    "clot_wall_focused": [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40],
}

# ────────────────────────────────────────────────
# ClotFeatureExtractor — adapted for normalized data
# ────────────────────────────────────────────────
class ClotFeatureExtractor:
    """
    Computes signal features from a normalized resistance buffer.
    Input signal is dimensionless: R_norm = (R_raw - baseline) / baseline.
    blood ≈ 0, clot ≈ 0.5-3, wall ≈ 1-5+.
    """

    _STATS       = set(range(0, 10))
    _SLOPES      = set(range(10, 16))
    _DERIV       = set(range(16, 22))
    _EMA         = set(range(22, 28))
    _DETRENDED   = set(range(28, 36))
    _PERCENTILES = set(range(36, 40))
    _HJORTH      = {40, 41}
    _DERIV2      = {42}
    _PULSE       = {43, 44, 45}
    _CLOT_WALL   = {46, 47, 48, 49, 50}
    _SHORT_SLOPES = {51, 52, 53, 54, 55, 56}
    _RISE_SHAPE  = {57, 58, 59, 60, 61, 62, 63}
    _TEXTURE     = {64}

    def __init__(self, sample_rate=150, window_sec=5.0, active_features=None):
        self.fs = sample_rate
        self.window_size = int(sample_rate * window_sec)
        self.buffer = deque(maxlen=self.window_size)
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.alpha_fast = 0.2
        self.alpha_slow = 0.01

        if active_features is not None:
            self._active_features = list(active_features)
            aset = set(active_features)
        else:
            self._active_features = None
            aset = set(range(TOTAL_FEATURES))

        self._need_stats       = bool(aset & self._STATS)
        self._need_slopes      = bool(aset & self._SLOPES)
        self._need_deriv       = bool(aset & self._DERIV)
        self._need_ema         = bool(aset & self._EMA)
        self._need_detrended   = bool(aset & self._DETRENDED)
        self._need_percentiles = bool(aset & self._PERCENTILES)
        self._need_hjorth      = bool(aset & self._HJORTH)
        self._need_deriv2      = bool(aset & self._DERIV2)
        self._need_pulse       = bool(aset & self._PULSE)
        self._need_clot_wall   = bool(aset & self._CLOT_WALL)
        self._need_short_slopes = bool(aset & self._SHORT_SLOPES)
        self._need_rise_shape  = bool(aset & self._RISE_SHAPE)
        self._need_texture     = bool(aset & self._TEXTURE)
        self._need_deriv_data  = self._need_deriv or self._need_hjorth or self._need_deriv2

        if self._need_texture:
            nyq = 0.5 * self.fs
            self._texture_sos = butter(4, [5.0 / nyq, 50.0 / nyq], btype='bandpass', output='sos')

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
        n_out = len(self._active_features) if self._active_features else TOTAL_FEATURES
        if len(self.buffer) < 100:
            return np.zeros(n_out, dtype=np.float32)
        data = np.array(self.buffer, dtype=np.float32)
        return self._compute(data, self.ema_fast, self.ema_slow)

    def compute_features_from_array(self, data, ema_fast, ema_slow):
        n_out = len(self._active_features) if self._active_features else TOTAL_FEATURES
        if len(data) < 100:
            return np.zeros(n_out, dtype=np.float32)
        return self._compute(np.asarray(data, dtype=np.float32), ema_fast, ema_slow)

    def _compute(self, data, ema_fast, ema_slow):
        n = len(data)
        f = np.zeros(TOTAL_FEATURES, dtype=np.float32)

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

        # ── f10-f15: Slopes ──
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

        # ── f40-f42: Hjorth + mean abs 2nd derivative ──
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

        # ── f43-f45: Pulse features ──
        if self._need_pulse:
            _MED_KERNEL = int(self.fs * 1.5) | 1
            if n >= _MED_KERNEL:
                trend = medfilt(data, kernel_size=_MED_KERNEL)
                pulse = data - trend
                pulse_std = np.std(pulse)
                f[43] = pulse_std
                f[44] = pulse_std / (f[1] + 1e-8) if self._need_stats else pulse_std / (np.std(data) + 1e-8)
                if pulse_std > 0.001:  # Adjusted threshold for normalized data
                    _min_dist = int(0.15 * self.fs)
                    peaks, _ = find_peaks(pulse, height=pulse_std * 0.4, distance=_min_dist)
                    window_sec = n / self.fs
                    f[45] = len(peaks) / window_sec
                else:
                    f[45] = 0.0
            else:
                f[43] = f[44] = f[45] = 0.0

        # ── f46-f50: Clot-vs-wall discriminative features ──
        # ADAPTED for normalized data where blood≈0, clot≈0.5-3, wall≈1-5+
        if self._need_clot_wall:
            mean_val = f[0] if self._need_stats else data.mean()
            std_val = f[1] if self._need_stats else data.std()

            # f46: Coefficient of variation (std / |mean|)
            f[46] = std_val / (abs(mean_val) + 1e-6)

            # f47: Plateau fraction — tight band = ±0.005 in normalized units
            # (equivalent to ±2 Ω when baseline is ~400 Ω)
            chunk_size = min(50, n // 4)
            if chunk_size >= 10:
                n_chunks = n // chunk_size
                plateau_count = 0
                for ci in range(n_chunks):
                    chunk = data[ci * chunk_size:(ci + 1) * chunk_size]
                    chunk_med = np.median(chunk)
                    if np.all(np.abs(chunk - chunk_med) < 0.005):
                        plateau_count += 1
                f[47] = plateau_count / n_chunks if n_chunks > 0 else 0.0
            else:
                f[47] = 0.0

            # f48: Settling time ratio — threshold ±0.01 in normalized units
            # (equivalent to ±5 Ω when baseline is ~500 Ω)
            peak_idx = np.argmax(data)
            post_peak = data[peak_idx:]
            if len(post_peak) >= 20:
                post_med = np.median(post_peak)
                settled = np.abs(post_peak - post_med) < 0.01
                f[48] = settled.sum() / len(post_peak)
            else:
                f[48] = 0.0

            # f49: Trend stationarity — ratio of Q4 mean to Q1 mean
            q_len = n // 4
            if q_len >= 10:
                q1_mean = data[:q_len].mean()
                q4_mean = data[-q_len:].mean()
                f[49] = q4_mean / (q1_mean + 1e-6)
            else:
                f[49] = 1.0

            # f50: R level relative to baseline — for normalized data, this IS the mean
            # (since data = (R-baseline)/baseline, the mean already encodes R level)
            f[50] = mean_val

        # ── f51-f56: Short-timescale slopes ──
        if self._need_short_slopes:
            for j, secs in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
                ns = min(int(secs * self.fs), n)
                if ns >= 2:
                    segment = data[-ns:]
                    slope = np.polyfit(np.arange(ns), segment, 1)[0]
                    f[51 + j] = np.abs(slope) if np.isfinite(slope) else 0.0

        # ── f57-f63: Rise-shape features (amplitude-normalized) ──
        if self._need_rise_shape:
            r_range = np.ptp(data)
            if r_range > 0.01 and n >= 50:  # Adjusted gate for normalized data
                d_norm = (data - data.min()) / r_range

                kern = min(15, n // 10)
                if kern >= 3:
                    smooth = np.convolve(data, np.ones(kern)/kern, 'valid')
                    smooth_deriv = np.diff(smooth)
                else:
                    smooth_deriv = np.diff(data)

                # f57: Normalized max rise rate
                if len(smooth_deriv) > 0:
                    f[57] = np.max(smooth_deriv) / r_range
                else:
                    f[57] = 0.0

                # f58: Rise time fraction
                lo_thresh = 0.10
                hi_thresh = 0.90
                cross_lo = np.where(d_norm >= lo_thresh)[0]
                cross_hi = np.where(d_norm >= hi_thresh)[0]
                if len(cross_lo) > 0 and len(cross_hi) > 0:
                    rise_samples = cross_hi[0] - cross_lo[0]
                    f[58] = max(rise_samples, 0) / n
                else:
                    f[58] = 0.0

                # f59: Rise linearity
                if len(cross_lo) > 0 and len(cross_hi) > 0:
                    rise_start = cross_lo[0]
                    rise_end = cross_hi[0]
                    rise_seg = data[rise_start:rise_end + 1]
                    if len(rise_seg) >= 5:
                        x_rise = np.arange(len(rise_seg))
                        coeffs = np.polyfit(x_rise, rise_seg, 1)
                        fitted = np.polyval(coeffs, x_rise)
                        ss_res = np.sum((rise_seg - fitted) ** 2)
                        ss_tot = np.sum((rise_seg - rise_seg.mean()) ** 2) + 1e-8
                        f[59] = max(0.0, 1.0 - ss_res / ss_tot)
                    else:
                        f[59] = 0.0
                else:
                    f[59] = 0.0

                # f60: Peak sharpness
                peak_idx = np.argmax(data)
                hood = max(15, n // 20)
                p_lo = max(0, peak_idx - hood)
                p_hi = min(n, peak_idx + hood)
                seg_peak = data[p_lo:p_hi]
                if len(seg_peak) >= 4:
                    d2_peak = np.diff(seg_peak, n=2)
                    f[60] = np.max(np.abs(d2_peak)) / r_range
                else:
                    f[60] = 0.0

                # f61: Descent smoothness
                post = data[peak_idx:]
                if len(post) >= 10:
                    post_deriv = np.diff(post)
                    f[61] = np.std(post_deriv) / r_range
                else:
                    f[61] = 0.0

                # f62: Shape asymmetry
                f[62] = float(stats.skew(d_norm)) if n >= 8 else 0.0

                # f63: Plateau ratio
                f[63] = np.sum(d_norm >= 0.90) / n

            else:
                f[57] = f[58] = f[59] = f[60] = f[61] = f[62] = f[63] = 0.0

        # ── f64: Texture RMS (bandpass 5-50 Hz) ──
        if self._need_texture:
            _TEX_SAMPLES = int(self.fs * 1.0)
            seg = data[-_TEX_SAMPLES:] if n >= _TEX_SAMPLES else data
            if len(seg) >= 60:
                z_texture = sosfiltfilt(self._texture_sos, seg)
                f[64] = np.sqrt(np.mean(z_texture ** 2))
            else:
                f[64] = 0.0

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

SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_normalized_5s_seq{SEQ_LEN}_{dim_str}.pkl"
MODEL_PATH = PROJECT_ROOT / "src" / "training" / "clot_gru_trained_normalized.pt"

ENSEMBLE_SEEDS = None
SAVE_PARQUET = True
SAVE_CSV = False
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FOLDER = PROJECT_ROOT / "inference_deploy" / "Results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────
# ClotGRU Model (identical architecture)
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
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH,
                 ensemble_seeds=ENSEMBLE_SEEDS):
        self.scaler = joblib.load(scaler_path)

        if ensemble_seeds:
            self.models = []
            self.hiddens = []
            model_dir = PROJECT_ROOT / "src" / "training"
            for seed in ensemble_seeds:
                pattern = f"clot_gru_trained_normalized_seq{SEQ_LEN}_{FEATURE_SET}_seed{seed}_f1*.pt"
                candidates = sorted(model_dir.glob(pattern))
                if not candidates:
                    print(f"  WARNING: No model found for seed {seed} ({pattern})")
                    continue
                best_path = candidates[-1]
                m = ClotGRU().to(DEVICE)
                m.load_state_dict(torch.load(best_path, map_location=DEVICE))
                m.eval()
                self.models.append(m)
                self.hiddens.append(None)
            if not self.models:
                raise FileNotFoundError(f"No ensemble models found for seeds {ensemble_seeds}")
            self.ensemble = True
        else:
            self.models = [ClotGRU().to(DEVICE)]
            self.models[0].load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.models[0].eval()
            self.hiddens = [None]
            self.ensemble = False

        self.posterior = np.array([INIT_BLOOD_PROB, INIT_CLOT_PROB, INIT_WALL_PROB],
                                  dtype=np.float32)
        self.feat_history = deque(maxlen=SEQ_LEN)

    def _make_da_probs(self, da_label):
        da_probs = np.array([DA_OTHER_CONFIDENCE] * 3, dtype=np.float32)
        da_probs[da_label] = DA_LABEL_CONFIDENCE
        return da_probs

    def _da_should_override_gru(self, probs, da_label, strict=False):
        gru_top_idx = np.argmax(probs)
        if gru_top_idx == da_label:
            return False
        threshold = GRU_OVERRIDE_THRD_CLOT if da_label == 1 else GRU_OVERRIDE_THRD_WALL
        if strict:
            return probs[gru_top_idx] < threshold
        return probs[gru_top_idx] <= threshold

    @torch.no_grad()
    def predict(self, active_feats, da_label=None):
        scaled = self.scaler.transform(active_feats.reshape(1, -1))[0]
        self.feat_history.append(scaled)

        if len(self.feat_history) < SEQ_LEN:
            pad = list(self.feat_history)[0] if self.feat_history else scaled
            seq_list = [pad] * (SEQ_LEN - len(self.feat_history)) + list(self.feat_history)
        else:
            seq_list = list(self.feat_history)

        seq = np.array(seq_list, dtype=np.float32)
        x = torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)

        all_logits = []
        for i, model in enumerate(self.models):
            logits, h = model(x, self.hiddens[i])
            self.hiddens[i] = h.detach() if h is not None else None
            all_logits.append(logits)

        avg_logits = torch.stack(all_logits).mean(dim=0)
        probs = torch.softmax(avg_logits / TEMPERATURE, 1).squeeze(0).cpu().numpy()
        self.raw_probs = probs.copy()

        prior_idx = np.argmax(self.posterior)

        if da_label is not None:
            if da_label == 0:
                self.posterior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self.hiddens = [None] * len(self.models)
                self.feat_history.clear()
                return self.posterior.copy()
            elif da_label in (1, 2):
                if self._da_should_override_gru(probs, da_label):
                    probs = self._make_da_probs(da_label)

        if prior_idx == 0:
            alpha_history = EMA_BLOOD_PRIOR_HISTORY
            alpha_new     = EMA_BLOOD_PRIOR_NEW
        else:
            new_idx = np.argmax(probs)
            if new_idx == 0:
                alpha_history = EMA_EXIT_TO_BLOOD_HISTORY
                alpha_new     = EMA_EXIT_TO_BLOOD_NEW
            elif new_idx == prior_idx:
                alpha_history = EMA_SAME_CLASS_HISTORY
                alpha_new     = EMA_SAME_CLASS_NEW
            else:
                alpha_history = EMA_CROSS_CLASS_HISTORY
                alpha_new     = EMA_CROSS_CLASS_NEW

        self.posterior = alpha_history * self.posterior + alpha_new * probs

        if da_label in (1, 2):
            final_idx = np.argmax(self.posterior)
            if final_idx != da_label and self._da_should_override_gru(probs, da_label, strict=True):
                self.posterior = self._make_da_probs(da_label)

        return self.posterior.copy()


# ────────────────────────────────────────────────
#  Main Processing
# ────────────────────────────────────────────────

def process_file(filepath: Path,
                 all_gt_labels: list,
                 all_da_labels: list,
                 all_ml_preds: list,
                 all_override_times: list,
                 save_parquet: bool = True,
                 save_csv: bool = False):

    study_name = filepath.stem
    print(f"\nProcessing: {study_name}")

    df = pd.read_parquet(filepath)
    time_ms = df['timeInMS'].values
    resistance = df['R_normalized'].values.astype(np.float32)
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
            raw = detector.raw_probs
            status = np.argmax(post)
            entropy = -np.sum(post * np.log(post + 1e-12))

            results.append({
                'time': t/1000.0,
                'prediction': status,
                'resistance': float(r),
                'Nprob': float(post[0]),
                'Cprob': float(post[1]),
                'Wprob': float(post[2]),
                'rawN': float(raw[0]),
                'rawC': float(raw[1]),
                'rawW': float(raw[2]),
                'entropy': float(entropy)
            })
            last_report = t

    results_df = pd.DataFrame(results)

    if save_parquet:
        results_df.to_parquet(OUTPUT_FOLDER / f"{study_name}_detection_results.parquet", index=False)
    if save_csv:
        results_df.to_csv(OUTPUT_FOLDER / f"{study_name}_detection_results.csv", index=False)

    # ── Probability plot ──
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    colors = {0:'black', 1:'red', 2:'blue'}
    for lbl, name in [(0,'blood'),(1,'clot'),(2,'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('R_normalized')
    ax1.set_title(f'{study_name} — Detected Labels')
    ax1.grid(True, alpha=0.3)

    ax2.plot(results_df['time'], results_df['rawC'], color='red',   label='raw P(clot)', linewidth=1.2, alpha=0.8)
    ax2.plot(results_df['time'], results_df['rawW'], color='blue',  label='raw P(wall)', linewidth=1.2, alpha=0.8)
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{study_name} — Raw GRU Probabilities (T={TEMPERATURE})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(results_df['time'], results_df['Cprob'], color='red',   label='P(clot)', linewidth=1.8)
    ax3.plot(results_df['time'], results_df['Wprob'], color='blue',  label='P(wall)', linewidth=1.8)
    blood_dom = (results_df['Nprob'] > results_df['Cprob']) & (results_df['Nprob'] > results_df['Wprob'])
    ax3.fill_between(results_df['time'], 0, 1, where=blood_dom, color='gray', alpha=0.12, label='Blood dominant')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    ax3.set_title(f'{study_name} — Smoothed Posterior')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"{study_name}_detected_vs_clot_wall_probs.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Three-panel plot ──
    if gt_labels is not None and da_labels is not None:
        gt = gt_labels.values.astype(int)
        da = da_labels.values.astype(int)
        full_times = time_ms / 1000.0

        interp_ml = np.interp(full_times, results_df['time'], results_df['prediction'])
        interp_ml = np.round(interp_ml).astype(int)

        valid = (gt >= 0)
        gt_valid = gt[valid]
        da_valid = da[valid]
        ml_valid = interp_ml[valid]

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True)
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
        ax.set_ylabel('R_normalized')
        ax.grid(True, alpha=0.3)

        for ax_idx, (title, data_arr) in enumerate([("DA Labels", da), ("Ground Truth Labels", gt)]):
            ax = axes[ax_idx+1]
            unlabeled_mask = data_arr == -1
            if unlabeled_mask.any():
                ax.scatter(full_times[unlabeled_mask], resistance[unlabeled_mask],
                           c='black', s=2, label='unlabeled', alpha=0.4, zorder=1)
            for lbl in [0,1,2]:
                mask = data_arr == lbl
                ax.scatter(full_times[mask], resistance[mask], c=colors[lbl], s=2,
                           label=lbl_names[lbl], alpha=0.7, zorder=2)
            ax.set_title(f'{study_name} — {title}')
            ax.set_ylabel('R_normalized')
            ax.grid(True, alpha=0.3)
            if ax_idx == 1:
                ax.set_xlabel('Time (seconds)')

        plt.tight_layout(h_pad=0.8)
        plt.savefig(OUTPUT_FOLDER / f"{study_name}_ml_da_gt_three_panel.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Metrics
        print(f"\n{study_name} metrics:")
        print(f"DA  Acc: {accuracy_score(gt_valid, da_valid):.4f}  F1: {f1_score(gt_valid, da_valid, average='macro'):.4f}")
        print(f"ML  Acc: {accuracy_score(gt_valid, ml_valid):.4f}  F1: {f1_score(gt_valid, ml_valid, average='macro'):.4f}")
        print(f"Improvement: Acc {accuracy_score(gt_valid, ml_valid)-accuracy_score(gt_valid, da_valid):+.4f}   "
              f"F1 {f1_score(gt_valid, ml_valid, average='macro')-f1_score(gt_valid, da_valid, average='macro'):+.4f}")

        # Override analysis
        override_mask = (ml_valid != da_valid)
        n_overrides = override_mask.sum()
        if n_overrides > 0:
            correct_overrides = ((ml_valid[override_mask] == gt_valid[override_mask]).sum())
            harmful_overrides = ((da_valid[override_mask] == gt_valid[override_mask]).sum())
            override_prec = correct_overrides / n_overrides
            da_cw_errors = ((da_valid != gt_valid) & ((gt_valid == 1) | (gt_valid == 2))).sum()
            override_rec = correct_overrides / da_cw_errors if da_cw_errors > 0 else 0.0

            print(f"  Override Precision: {override_prec:.4f}")
            print(f"  Net benefit: {correct_overrides - harmful_overrides:+d} samples")

        if gt_labels is not None and da_labels is not None:
            all_gt_labels.extend(gt_valid)
            all_da_labels.extend(da_valid)
            all_ml_preds.extend(ml_valid)

        overrides = np.where(ml_valid != da_valid)[0]
        all_override_times.extend(full_times[valid][overrides])

    print(f"Finished {study_name}\n")


def main():
    files = sorted(TEST_DATA_DIR.glob("*_labeled_segment.parquet"))
    print(f"Found {len(files)} files in {TEST_DATA_DIR.name}/\n")

    all_gt_labels     = []
    all_da_labels     = []
    all_ml_preds      = []
    all_override_times = []

    for f in files:
        process_file(f,
                     all_gt_labels=all_gt_labels,
                     all_da_labels=all_da_labels,
                     all_ml_preds=all_ml_preds,
                     all_override_times=all_override_times,
                     save_parquet=SAVE_PARQUET,
                     save_csv=SAVE_CSV)

    # ── Global summary ──
    if all_gt_labels:
        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("GLOBAL SUMMARY ACROSS ALL STUDIES")
        summary_lines.append("=" * 70)

        acc_da  = accuracy_score(all_gt_labels, all_da_labels)
        f1_da   = f1_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        prec_da = precision_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        rec_da  = recall_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)

        acc_ml  = accuracy_score(all_gt_labels, all_ml_preds)
        f1_ml   = f1_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        prec_ml = precision_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        rec_ml  = recall_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)

        summary_lines.append(f"DA  Accuracy: {acc_da:.4f}    F1-macro: {f1_da:.4f}")
        summary_lines.append(f"ML  Accuracy: {acc_ml:.4f}    F1-macro: {f1_ml:.4f}")
        summary_lines.append(f"Improvement: Acc {acc_ml - acc_da:+.4f}   F1 {f1_ml - f1_da:+.4f}")
        summary_lines.append("")
        summary_lines.append(f"DA  Precision: {prec_da:.4f}    Recall: {rec_da:.4f}")
        summary_lines.append(f"ML  Precision: {prec_ml:.4f}    Recall: {rec_ml:.4f}")
        summary_lines.append(f"Improvement: Precision {prec_ml - prec_da:+.4f}   Recall {rec_ml - rec_da:+.4f}")

        gt_arr = np.array(all_gt_labels)
        da_arr = np.array(all_da_labels)
        ml_arr = np.array(all_ml_preds)

        g_override_mask = (ml_arr != da_arr)
        g_n_overrides = g_override_mask.sum()

        summary_lines.append(f"\n{'─'*70}")
        summary_lines.append(f"GLOBAL OVERRIDE ANALYSIS")
        summary_lines.append(f"{'─'*70}")
        summary_lines.append(f"Total overrides across all studies: {g_n_overrides}")

        if g_n_overrides > 0:
            g_correct = (ml_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_harmful = (da_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_neither = g_n_overrides - g_correct - g_harmful
            g_override_prec = g_correct / g_n_overrides
            g_da_cw_errors = ((da_arr != gt_arr) & ((gt_arr == 1) | (gt_arr == 2))).sum()
            g_override_rec = g_correct / g_da_cw_errors if g_da_cw_errors > 0 else 0.0

            summary_lines.append(f"  Correct overrides (ML right, DA wrong): {g_correct}")
            summary_lines.append(f"  Harmful overrides (DA right, ML wrong): {g_harmful}")
            summary_lines.append(f"  Neither correct (both wrong differently): {g_neither}")
            summary_lines.append(f"")
            summary_lines.append(f"  Override Precision: {g_override_prec:.4f}  (target: >0.85)")
            summary_lines.append(f"  Override Recall:    {g_override_rec:.4f}  (of {g_da_cw_errors} DA clot/wall errors)")
            summary_lines.append(f"  Net benefit:        {g_correct - g_harmful:+d} samples")

        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)

        summary_path = OUTPUT_FOLDER / "global_summary.txt"
        summary_path.write_text(summary_text, encoding="utf-8")
        print(f"\nSaved global summary to {summary_path.name}")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
