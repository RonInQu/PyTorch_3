# gru_torch_V6.py
"""
Real-time clot detection — V6
Stripped feature set: original 40 + Hjorth mobility, Hjorth complexity, mean abs 2nd derivative, flatness.
Total features = 57.  No FFT, no sample entropy, no zero-crossing, no transition features.

Modular compute_features(): skips feature groups not needed by the selected FEATURE_SET.
compute_features_from_array(): batch-mode method for scaler/training (no streaming state).

Feature index map (V6 vs V5):
  f0-f39:  same as V5 (original 40)
  f40:     Hjorth mobility      (was V5 f44)
  f41:     Hjorth complexity     (was V5 f45)
  f42:     Mean abs 2nd deriv   (was V5 f52)
  f43:     Flatness             (fraction of window with near-zero local slope)
  V5 f40-f43 (spectral), f46 (SampEn), f47 (ZCR), f48-f51 (transition): REMOVED
  f46:     Coefficient of Variation (std/mean) — clot=high, wall=low
  f47:     Plateau fraction     — fraction of window in stable bands; wall=high, clot=low
  f48:     Settling time ratio  — how quickly signal settles after max; wall=fast, clot=slow
  f49:     Trend stationarity   — last-quarter / first-quarter mean ratio; wall≈1, clot≠1
  f50:     R level relative to baseline — (mean-800)/800; blood≈0, clot=moderate, wall=high
  f51-f56: Short-timescale slopes (abs linear reg over 0.1s, 0.2s, ..., 0.6s of window end)
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
from scipy.signal import medfilt, find_peaks
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no GUI windows
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ────────────────────────────────────────────────
# CONFIG — Single source of truth
# ────────────────────────────────────────────────
SEQ_LEN = 8

WINDOW_SEC = 5.0
REPORT_INTERVAL_MS = 200

GRU_OVERRIDE_THRD_CLOT = 0.82
GRU_OVERRIDE_THRD_WALL = 0.92

# Temperature scaling for softmax (T>1 = less confident, T=1 = no change)
TEMPERATURE = 1.8

# ── Posterior EMA (exponential moving average) blending weights ──
# Controls how fast the smoothed posterior responds to new GRU outputs.
# alpha_history = weight on previous posterior, alpha_new = weight on new probs.
# Higher alpha_history → slower/more stable; higher alpha_new → faster/more reactive.
EMA_BLOOD_PRIOR_HISTORY = 0.78   # when prior state is blood: moderate reactivity
EMA_BLOOD_PRIOR_NEW     = 1 - EMA_BLOOD_PRIOR_HISTORY
EMA_EXIT_TO_BLOOD_HISTORY = 0.35 # leaving clot/wall back to blood: fast transition
EMA_EXIT_TO_BLOOD_NEW     = 1 - EMA_EXIT_TO_BLOOD_HISTORY
EMA_SAME_CLASS_HISTORY  = 0.97   # non-blood transitions: unified rate (no ratchet)
EMA_SAME_CLASS_NEW      = 1 - EMA_SAME_CLASS_HISTORY
EMA_CROSS_CLASS_HISTORY = 0.99   # same as SAME_CLASS — eliminates asymmetric lock-in
EMA_CROSS_CLASS_NEW     = 1 - EMA_CROSS_CLASS_HISTORY

# ── DA (device-assisted) label override confidence ──
# When the device provides a label, we construct a probability vector
# with this much confidence on the labeled class.
DA_LABEL_CONFIDENCE = 0.92   # confidence assigned to the DA-labeled class
DA_OTHER_CONFIDENCE = (1.0 - DA_LABEL_CONFIDENCE) / 2  # 0.04   # split equally among the other two classes

# ── Initial posterior (blood-dominant prior) ──
# Starting belief before any data: mostly blood.
# Increase INIT_BLOOD_PROB to make the detector more conservative (slower to leave blood).
INIT_BLOOD_PROB = 0.95
INIT_CLOT_PROB  = (1 - INIT_BLOOD_PROB) /2
INIT_WALL_PROB  = (1 - INIT_BLOOD_PROB) /2

# Feature set selection
FEATURE_SET = "clot_wall_v4"

TOTAL_FEATURES = 57

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
# f43:     Pulse amplitude (std of extracted cardiac pulse component)
# f44:     Pulse-to-signal ratio (pulse_std / signal_std)
# f45:     Pulse rate (peaks per second in pulse component)
# f46:     Coefficient of variation (std/mean) — clot=high(noisy), wall=low(stable)
# f47:     Plateau fraction — fraction of window in tight band; wall=high, clot=low
# f48:     Settling time ratio — post-peak settling speed; wall=fast, clot=slow
# f49:     Trend stationarity — Q4/Q1 mean ratio; wall≈1.0, clot deviates
# f50:     R level relative to baseline — (mean-800)/800; blood≈0, clot=mod, wall=high
# f51-f56: Short-timescale slopes (abs linear regression over 0.1s, 0.2s, ..., 0.6s)
#          Captures fast dynamics: clot=steep/variable, wall=flat/stable

FEATURE_SETS = {
    "all":               list(range(TOTAL_FEATURES)),
    "original_40":       list(range(40)),
    "clean_36":          [i for i in range(40) if i not in [14, 25, 31, 33]],
    "top20":             [4, 0, 1, 9, 23, 3, 21, 19, 30, 32, 15, 36, 27, 24, 16, 12, 8, 34, 20, 10],
    # d(clt-wall) > 0.15 — clot vs wall distinguishing features + pulse
    "clot_wall_focused": [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40],
    "clot_wall_focused_pulse": [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40, 43, 44, 45],
    # per-run AUC >= 0.65 (clot vs wall), spectral removed → 20 features + pulse
    "auc_cw_20":         [0, 1, 3, 4, 5, 6, 9, 17, 18, 19, 21, 22, 23, 32, 33, 38, 39, 40, 41, 42, 43, 44, 45],
    # New clot-vs-wall discriminative features (dynamic vs stable plateau)
    "clot_wall_v2":      [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40,
                          46, 47, 48, 49, 50],
    # v3: clot_wall_focused (21) + short-timescale slopes (6) — fast dynamics for clot/wall
    "clot_wall_v3":      [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40,
                          51, 52, 53, 54, 55, 56],
    "clot_wall_v4":      [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40,
                           46, 47, 48, 49, 50,51, 52, 53, 54, 55, 56],
    "clot_wall_v5":      [39, 21, 4, 19, 41, 9, 5, 23, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40,
                           46, 47, 48, 49, 51, 52, 53, 54, 55, 56],                      
}

# ────────────────────────────────────────────────
# ClotFeatureExtractor — modular, skips unused groups
# ────────────────────────────────────────────────
class ClotFeatureExtractor:
    """
    Computes up to 44 signal features from a resistance buffer.

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
    _PULSE       = {43, 44, 45}
    _CLOT_WALL   = {46, 47, 48, 49, 50}  # New clot-vs-wall discriminative features
    _SHORT_SLOPES = {51, 52, 53, 54, 55, 56}  # Short-timescale slopes (0.1s-0.6s)

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
        self._need_pulse       = bool(aset & self._PULSE)
        self._need_clot_wall   = bool(aset & self._CLOT_WALL)
        self._need_short_slopes = bool(aset & self._SHORT_SLOPES)
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

        # ── f43-f45: Pulse features ──
        # Extract cardiac pulse component via in-window median filter,
        # then compute features from the noise itself.
        # f43: Pulse amplitude — std of the pulse component.
        #      Blood → high (strong coupling), Clot → low (damped), Wall → variable.
        # f44: Pulse-to-signal ratio — pulse_std / signal_std.
        #      Normalizes for baseline level; high = signal is mostly pulse.
        # f45: Pulse rate — detected peaks per second in the pulse component.
        #      Regular cardiac contact → ~1-3 Hz; no coupling → 0.
        if self._need_pulse:
            _MED_KERNEL = int(self.fs * 1.5) | 1  # 1.5s median — fits in 5s window
            if n >= _MED_KERNEL:
                trend = medfilt(data, kernel_size=_MED_KERNEL)
                pulse = data - trend
                pulse_std = np.std(pulse)
                f[43] = pulse_std
                f[44] = pulse_std / (f[1] + 1e-8) if self._need_stats else pulse_std / (np.std(data) + 1e-8)
                # Count peaks in pulse component
                if pulse_std > 0.05:
                    _min_dist = int(0.15 * self.fs)  # 200 BPM ceiling
                    peaks, _ = find_peaks(pulse, height=pulse_std * 0.4, distance=_min_dist)
                    window_sec = n / self.fs
                    f[45] = len(peaks) / window_sec  # peaks per second
                else:
                    f[45] = 0.0
            else:
                f[43] = f[44] = f[45] = 0.0

        # ── f46-f50: Clot-vs-wall discriminative features ──
        # Physical basis: clot = dynamic/noisy signal, wall = stable plateau at high R.
        # These features exploit that fundamental difference.
        if self._need_clot_wall:
            mean_val = f[0] if self._need_stats else data.mean()
            std_val = f[1] if self._need_stats else data.std()

            # f46: Coefficient of variation (std / |mean|)
            # Clot: high CV (noisy/dynamic), Wall: low CV (stable plateau)
            f[46] = std_val / (abs(mean_val) + 1e-6)

            # f47: Plateau fraction — fraction of window where consecutive samples
            # stay within a tight band (±2 Ω of local median over 50-sample chunks).
            # Wall = long stable plateaus → high fraction; Clot = few → low fraction.
            chunk_size = min(50, n // 4)
            if chunk_size >= 10:
                n_chunks = n // chunk_size
                plateau_count = 0
                for ci in range(n_chunks):
                    chunk = data[ci * chunk_size:(ci + 1) * chunk_size]
                    chunk_med = np.median(chunk)
                    if np.all(np.abs(chunk - chunk_med) < 2.0):
                        plateau_count += 1
                f[47] = plateau_count / n_chunks if n_chunks > 0 else 0.0
            else:
                f[47] = 0.0

            # f48: Settling time ratio — how quickly signal stabilizes after its peak.
            # Find the peak, then measure the fraction of post-peak samples that are
            # within ±5 Ω of the post-peak median.  Wall settles fast (high ratio),
            # Clot stays dynamic (low ratio).
            peak_idx = np.argmax(data)
            post_peak = data[peak_idx:]
            if len(post_peak) >= 20:
                post_med = np.median(post_peak)
                settled = np.abs(post_peak - post_med) < 5.0
                f[48] = settled.sum() / len(post_peak)
            else:
                f[48] = 0.0

            # f49: Trend stationarity — ratio of last-quarter mean to first-quarter mean.
            # Wall ≈ 1.0 (flat plateau); Clot deviates (rising/falling trends).
            q_len = n // 4
            if q_len >= 10:
                q1_mean = data[:q_len].mean()
                q4_mean = data[-q_len:].mean()
                f[49] = q4_mean / (q1_mean + 1e-6)
            else:
                f[49] = 1.0

            # f50: R level relative to baseline — (mean - 800) / 800.
            # Blood ≈ 0 (R ≈ 800), Clot = moderate positive, Wall = high positive.
            f[50] = (mean_val - 800.0) / 800.0

        # ── f51-f56: Short-timescale slopes (0.1s through 0.6s) ──
        # Linear regression over the LAST 0.1s, 0.2s, ..., 0.6s of the window.
        # Captures fast transient dynamics: clot events have steep/variable slopes,
        # wall events are flat/stable at these timescales.
        if self._need_short_slopes:
            for j, secs in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
                ns = min(int(secs * self.fs), n)
                if ns >= 2:
                    segment = data[-ns:]
                    slope = np.polyfit(np.arange(ns), segment, 1)[0]
                    f[51 + j] = np.abs(slope) if np.isfinite(slope) else 0.0

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
USE_DENOISED = False   # Set True to use pulse-subtracted data from test_data_denoised/
SAVE_CSV_PARQUET = True  # Set True to save detection_results .csv and .parquet files
TEST_DATA_DIR = PROJECT_ROOT / ("test_data_denoised" if USE_DENOISED else "test_data")
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
        self.posterior = np.array([INIT_BLOOD_PROB, INIT_CLOT_PROB, INIT_WALL_PROB],
                                  dtype=np.float32)
        self.feat_history = deque(maxlen=SEQ_LEN)

    def _make_da_probs(self, da_label):
        """Build a probability vector heavily favoring the DA-labeled class.
        Tune DA_LABEL_CONFIDENCE (default 0.92) to control how strongly
        the device-assisted label overrides the GRU output.
        """
        da_probs = np.array([DA_OTHER_CONFIDENCE] * 3, dtype=np.float32)
        da_probs[da_label] = DA_LABEL_CONFIDENCE
        return da_probs

    def _da_should_override_gru(self, probs, da_label, strict=False):
        """Return True if the GRU is not confident enough to contradict the DA label.
        Tune GRU_OVERRIDE_THRD_CLOT / GRU_OVERRIDE_THRD_WALL to control
        how easily the DA label wins over the GRU prediction.
        Higher threshold → DA label wins more often.
        strict=False: use <=  (pre-EMA check, slightly more permissive)
        strict=True:  use <   (post-EMA safety net, slightly less permissive)
        """
        gru_top_idx = np.argmax(probs)
        if gru_top_idx == da_label:
            return False  # GRU already agrees with DA
        threshold = GRU_OVERRIDE_THRD_CLOT if da_label == 1 else GRU_OVERRIDE_THRD_WALL
        if strict:
            return probs[gru_top_idx] < threshold
        return probs[gru_top_idx] <= threshold

    @torch.no_grad()
    def predict(self, active_feats, da_label=None):
        """
        Run one prediction step.  Returns a 3-element posterior [P(blood), P(clot), P(wall)].

        Pipeline:
          1. Scale features, build sequence, run GRU → raw probs
          2. If DA label present, optionally override GRU probs
          3. EMA-blend new probs into the running posterior
          4. Post-EMA safety check: force DA label if GRU still disagrees
        """

        # ── Step 1: Scale features & run GRU ──
        scaled = self.scaler.transform(active_feats.reshape(1, -1))[0]
        self.feat_history.append(scaled)

        # Pad the sequence with the earliest available frame if we don't have SEQ_LEN yet
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

        # Temperature-scaled softmax.  Tune TEMPERATURE (>1 → softer/less peaky probs)
        probs = torch.softmax(logits / TEMPERATURE, 1).squeeze(0).cpu().numpy()
        self.raw_probs = probs.copy()  # store for diagnostics

        prior_idx = np.argmax(self.posterior)  # class the posterior currently favors

        # ── Step 2: DA (device-assisted) label override ──
        # When the device provides a ground-truth label, trust it unless the
        # GRU is extremely confident in a different class.
        if da_label is not None:
            if da_label == 0:
                # DA says blood → hard reset: clear history and return certain blood.
                self.posterior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self.hidden = None
                self.feat_history.clear()
                return self.posterior.copy()

            elif da_label in (1, 2):
                # DA says clot or wall → override GRU if GRU isn't confident enough
                if self._da_should_override_gru(probs, da_label):
                    probs = self._make_da_probs(da_label)

        # ── Step 3: EMA blending ──
        # Blend new probs into the running posterior.  The blend weights depend
        # on what transition is happening, to control responsiveness vs stability.
        #
        # Tuning guide:
        #   - EMA_BLOOD_PRIOR:   when currently in blood. More NEW → faster clot/wall detection.
        #   - EMA_EXIT_TO_BLOOD: leaving clot/wall back to blood. More NEW → faster recovery.
        #   - EMA_SAME_CLASS:    staying in same non-blood class. More HISTORY → more stable.
        #   - EMA_CROSS_CLASS:   clot↔wall switch. More HISTORY → resist flicker.
        if prior_idx == 0:
            # Currently in blood — moderately reactive to new evidence
            alpha_history = EMA_BLOOD_PRIOR_HISTORY
            alpha_new     = EMA_BLOOD_PRIOR_NEW
        else:
            new_idx = np.argmax(probs)
            if new_idx == 0:
                # Transitioning back to blood — respond quickly
                alpha_history = EMA_EXIT_TO_BLOOD_HISTORY
                alpha_new     = EMA_EXIT_TO_BLOOD_NEW
            elif new_idx == prior_idx:
                # Confirming same non-blood class — stay very stable
                alpha_history = EMA_SAME_CLASS_HISTORY
                alpha_new     = EMA_SAME_CLASS_NEW
            else:
                # Clot↔wall switch — resist flicker, change very slowly
                alpha_history = EMA_CROSS_CLASS_HISTORY
                alpha_new     = EMA_CROSS_CLASS_NEW

        self.posterior = alpha_history * self.posterior + alpha_new * probs

        # ── Step 4: Post-EMA DA safety net ──
        # After blending, if the posterior still disagrees with the DA label
        # and the GRU wasn't confident enough, force the posterior to the DA label.
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
                 save_csv_parquet: bool = False):

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

    # Save detection_results parquet and csv (optional)
    if save_csv_parquet:
        results_df.to_parquet(OUTPUT_FOLDER / f"{study_name}_detection_results.parquet", index=False)
        results_df.to_csv(OUTPUT_FOLDER / f"{study_name}_detection_results.csv", index=False)
        print(f"  Saved detection_results.parquet and .csv")

    # ── Probability plot (3 panels: labels, raw GRU, smoothed posterior) ──
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    colors = {0:'black', 1:'red', 2:'blue'}
    for lbl, name in [(0,'blood'),(1,'clot'),(2,'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(f'{study_name} — Detected Labels')
    ax1.grid(True, alpha=0.3)

    # Raw GRU probabilities (temperature-scaled)
    ax2.plot(results_df['time'], results_df['rawC'], color='red',   label='raw P(clot)', linewidth=1.2, alpha=0.8)
    ax2.plot(results_df['time'], results_df['rawW'], color='blue',  label='raw P(wall)', linewidth=1.2, alpha=0.8)
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{study_name} — Raw GRU Probabilities (T={TEMPERATURE})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Smoothed posterior
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
    print(f"  Saved probability plot")

    # ── Three-panel plot ──
    if gt_labels is not None and da_labels is not None:
        gt = gt_labels.values.astype(int)
        da = da_labels.values.astype(int)
        full_times = time_ms / 1000.0

        interp_ml = np.interp(full_times, results_df['time'], results_df['prediction'])
        interp_ml = np.round(interp_ml).astype(int)

        # Filter out unlabeled samples (label == -1) before metrics
        valid = (gt >= 0)
        gt_valid = gt[valid]
        da_valid = da[valid]
        ml_valid = interp_ml[valid]
        n_unlabeled = (~valid).sum()
        if n_unlabeled > 0:
            print(f"  Excluding {n_unlabeled} unlabeled samples (label == -1) from metrics")

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
            # Plot unlabeled samples (label == -1) in black first (behind labeled)
            unlabeled_mask = data == -1
            if unlabeled_mask.any():
                ax.scatter(full_times[unlabeled_mask], resistance[unlabeled_mask],
                           c='black', s=2, label='unlabeled', alpha=0.4, zorder=1)
            for lbl in [0,1,2]:
                mask = data == lbl
                ax.scatter(full_times[mask], resistance[mask], c=colors[lbl], s=2,
                           label=lbl_names[lbl], alpha=0.7, zorder=2)
            ax.set_title(f'{study_name} — {title}')
            ax.set_ylabel('Resistance (Ω)')
            ax.grid(True, alpha=0.3)
            if ax_idx == 1:
                ax.set_xlabel('Time (seconds)')

        plt.tight_layout(h_pad=0.8)
        plt.savefig(OUTPUT_FOLDER / f"{study_name}_ml_da_gt_three_panel.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved three-panel plot")

        # Metrics (on labeled samples only)
        print(f"\n{study_name} metrics:")
        print(f"DA  Acc: {accuracy_score(gt_valid, da_valid):.4f}  F1: {f1_score(gt_valid, da_valid, average='macro'):.4f} "
              f"Prec: {precision_score(gt_valid, da_valid, average='macro'):.4f}  Rec: {recall_score(gt_valid, da_valid, average='macro'):.4f}")
        print(f"ML  Acc: {accuracy_score(gt_valid, ml_valid):.4f}  F1: {f1_score(gt_valid, ml_valid, average='macro'):.4f} "
              f"Prec: {precision_score(gt_valid, ml_valid, average='macro'):.4f}  Rec: {recall_score(gt_valid, ml_valid, average='macro'):.4f}")
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

            print(f"\n  Override analysis ({study_name}):")
            print(f"    Total overrides:   {n_overrides}")
            print(f"    Correct (ML right, DA wrong): {correct_overrides}")
            print(f"    Harmful (DA right, ML wrong): {harmful_overrides}")
            print(f"    Override Precision: {override_prec:.4f}")
            print(f"    Override Recall:    {override_rec:.4f}  (of {da_cw_errors} DA clot/wall errors)")
        else:
            print(f"\n  No overrides in {study_name}")

        if gt_labels is not None and da_labels is not None:
            all_gt_labels.extend(gt_valid)
            all_da_labels.extend(da_valid)
            all_ml_preds.extend(ml_valid)

        overrides = np.where(ml_valid != da_valid)[0]
        all_override_times.extend(full_times[valid][overrides])

    print(f"Finished {study_name}\n")


# ────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────

def main():
    glob_pattern = "*_labeled_segment_denoised.parquet" if USE_DENOISED else "*_labeled_segment.parquet"
    files = sorted(TEST_DATA_DIR.glob(glob_pattern))
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
                     save_csv_parquet=SAVE_CSV_PARQUET)

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

        # Save summary to file for use by save_version.py
        summary_path = OUTPUT_FOLDER / "global_summary.txt"
        summary_path.write_text(summary_text, encoding="utf-8")
        print(f"\nSaved global summary to {summary_path.name}")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
