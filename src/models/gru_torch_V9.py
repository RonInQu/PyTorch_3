# gru_torch_V9.py
"""
Real-time clot detection — V9 (2-class: clot vs wall)

Architecture:
  - GRU model outputs 2 logits → softmax → [P(clot), P(wall)].
  - Blood is NEVER predicted by the model. It is emitted only when the
    detector-agnostic (DA) logic tree says blood (unconditional V6 rule).
  - Downstream code still receives a 3-vec [P(blood), P(clot), P(wall)]
    for backward compatibility:
        • DA=blood        → emit [1, 0, 0] (hard reset)
        • DA=clot/wall    → DA wins by default; ML overrides only when
                             the raw-stability gate is satisfied
        • DA=None         → emit [0, P(clot), P(wall)] from posterior

Stability gate (ML override rights):
  Raw GRU must predict the SAME non-DA class for at least
  ML_STABILITY_STREAK consecutive samples AND the mean raw confidence over
  that stable run must exceed ML_STABILITY_MEAN_CONF.

    Optional V9.1 hybrid gate:
    After the raw-stability gate passes, an auxiliary real-time override policy
    (trained from DA-vs-GT disagreement windows) can be required to confirm the
    same non-DA class at high confidence before an override is allowed.

  Thresholds are tuned for the 2-class model, which peaks at lower confidence
  than a 3-class model (typical peak ~0.75–0.85 on test data vs ~0.95 for
  the 3-class variant). Setting the mean-conf floor too high blocks all ML
  overrides; too low lets noisy files (PALM0507, 9CB4378D) do harm.

Feature set: same 57-dim clot_wall_focused set used since V6. Scaler, cache,
and model artifacts are all V9-tagged so V6 files are untouched.
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

# Temperature scaling for softmax (T=1 = raw calibrated logits, T>1 softens).
# V9 keeps raw logits: stability gate already discounts single-sample spikes,
# so softening at the softmax fights the gate rather than helping it.
TEMPERATURE = 1.0

# ── Posterior EMA (single smoothing rate) ──
# alpha_history = weight on previous posterior, alpha_new = weight on new probs.
# One rate for all transitions — asymmetric rates gave no measurable benefit
# and created hidden lock-in behavior.
EMA_HISTORY = 0.98
EMA_NEW     = 1 - EMA_HISTORY

# ── DA (device-assisted) label override confidence ──
# V9 rule: DA wins clot/wall by default. ML overrides only through the
# stability gate below.
DA_LABEL_CONFIDENCE = 0.97
DA_OTHER_CONFIDENCE = 1.0 - DA_LABEL_CONFIDENCE   # 2-class: goes to the other class

# ── V9 ML-override stability gate ──
# ML must present a STABLE, CONFIDENT, SUSTAINED disagreement before it is
# allowed to override DA on clot/wall. Single-sample confidence spikes are not
# enough. Three checks (down from four — removed PEAK_CONF because it was
# redundant with MEAN_CONF given the CONF_RANGE cap, and its 0.75 floor was
# above the softmax's practical ceiling when TEMPERATURE was 1.5):
#   1. streak      : raw GRU argmax stable for N samples
#   2. mean_conf   : mean raw confidence over that run ≥ threshold
#   3. conf_range  : max-min of raw confidence over run ≤ threshold
#                    (rejects "argmax-stable but confidence oscillating" runs
#                    like 9CB4378D while preserving steady-high runs like
#                    5DFFDF16).
ML_STABILITY_STREAK      = 4
ML_STABILITY_MEAN_CONF   = 0.70
ML_STABILITY_CONF_RANGE  = 0.25

# Feature set selection
FEATURE_SET = "clot_wall_focused"

TOTAL_FEATURES = 65

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
# f57:     Normalized max rise rate — max(smoothed_deriv)/range; clot=high, wall=low
# f58:     Rise time fraction — samples from 10% to 90% of range / window_len; clot=short, wall=long
# f59:     Rise linearity — R² of linear fit during rise phase; wall=high(linear), clot=low(curved)
# f60:     Peak sharpness — max |2nd deriv| near peak / range; clot=sharp, wall=round
# f61:     Descent smoothness — std of deriv in post-peak region / range; wall=smooth, clot=noisy
# f62:     Shape asymmetry — skewness of normalized signal; clot=right-skewed, wall=left-skewed
# f63:     Plateau ratio — fraction of window within 10% of max; wall=high(sustained), clot=low(spike)
# f64:     Texture RMS — RMS of bandpass(5-50 Hz) signal. Clot=high(rough surface),
#          Wall=low(smooth). Butterworth order 4, zero-phase (sosfiltfilt).
#          Based on Joe Zott / Zedtech Nov 2025 physics analysis.

FEATURE_SETS = {
    "all":               list(range(TOTAL_FEATURES)),
    "check_1":           [0, 1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 36, 37, 38, 39],
    "check_2":           [i for i in range(40) if i not in [3,6,7,14,15,19,22,23,25,30,31,32,36,38]],
    "original_40":       list(range(40)),
    "clean_36":          [i for i in range(40) if i not in [14, 25, 31, 33]],
    "top20":             [4, 0, 1, 9, 23, 3, 21, 19, 30, 32, 15, 36, 27, 24, 16, 12, 8, 34, 20, 10],
    # d(clt-wall) > 0.15 — clot vs wall distinguishing features + pulse
    "clot_wall_focused": [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40],
    "clot_wall_focused+": [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40, 44],
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
    # v6: clot_wall_focused (21) + rise-shape features (7) — amplitude-invariant morphology
    "clot_wall_v6":      [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40,
                          57, 58, 59, 60, 61, 62, 63],
    # v7: clot_wall_focused (21) + all new features (v2+v3+rise-shape = 18)
    "clot_wall_v7":      [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40,
                          46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                          57, 58, 59, 60, 61, 62, 63],
    # v8: rise-shape only (7) — purely morphological, no other features
    "rise_shape_only":   [57, 58, 59, 60, 61, 62, 63],
    # Shape features: clot_wall_focused (21) + top 5 shape morphology (f64-f68) — FAILED
    "clot_wall_shape":   [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40,
                          64, 65, 66, 67, 68],
    # Texture RMS: clot_wall_focused (21) + bandpass 5-50Hz RMS (1 feature)
    # Physics: isolates surface roughness from cardiac/drift. Clot>30, Wall<20 (Zott).
    "clot_wall_texture": [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40,
                          64],
}

# ────────────────────────────────────────────────
# ClotFeatureExtractor — modular, skips unused groups
# ────────────────────────────────────────────────
class  ClotFeatureExtractor:
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
    _RISE_SHAPE  = {57, 58, 59, 60, 61, 62, 63}  # Rise-shape features (R-level invariant)
    _TEXTURE     = {64}  # Bandpass texture RMS (Zott-inspired, 5-50 Hz)

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
        self._need_rise_shape  = bool(aset & self._RISE_SHAPE)
        self._need_texture     = bool(aset & self._TEXTURE)
        # Shared dependency: first derivative needed by deriv, hjorth, or deriv2
        self._need_deriv_data  = self._need_deriv or self._need_hjorth or self._need_deriv2

        # Precompute bandpass filter coefficients (5-50 Hz at 150 Hz, order 4)
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

        # ── f57-f63: Rise-shape features (amplitude-normalized) ──
        # All features are normalized by the window's R range, making them
        # invariant to absolute R level.  They capture the SHAPE of the signal:
        # clot = sharp/fast rise, spiky peak, noisy descent
        # wall = gradual/smooth rise, rounded peak, smooth sustained curve
        if self._need_rise_shape:
            r_range = np.ptp(data)
            if r_range > 5.0 and n >= 50:  # meaningful signal, not flat noise
                # Amplitude-normalize to [0, 1]
                d_norm = (data - data.min()) / r_range

                # Smoothed first derivative (15-sample ~ 0.1s moving average)
                kern = min(15, n // 10)
                if kern >= 3:
                    smooth = np.convolve(data, np.ones(kern)/kern, 'valid')
                    smooth_deriv = np.diff(smooth)
                else:
                    smooth_deriv = np.diff(data)

                # f57: Normalized max rise rate
                # Max positive slope / R range.  Clot = high (sharp rise), wall = low.
                if len(smooth_deriv) > 0:
                    f[57] = np.max(smooth_deriv) / r_range
                else:
                    f[57] = 0.0

                # f58: Rise time fraction
                # Samples from first crossing of 10% to first crossing of 90% of range,
                # divided by window length.  Clot = short fraction, wall = long.
                lo_thresh = 0.10
                hi_thresh = 0.90
                cross_lo = np.where(d_norm >= lo_thresh)[0]
                cross_hi = np.where(d_norm >= hi_thresh)[0]
                if len(cross_lo) > 0 and len(cross_hi) > 0:
                    rise_samples = cross_hi[0] - cross_lo[0]
                    f[58] = max(rise_samples, 0) / n
                else:
                    f[58] = 0.0

                # f59: Rise linearity (R² of linear fit during rise phase)
                # Wall rises more linearly; clot has curved/exponential rise.
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
                # Max |2nd derivative| in a neighborhood around the peak, / R range.
                # Clot = sharp peak (high), wall = rounded (low).
                peak_idx = np.argmax(data)
                hood = max(15, n // 20)  # ~0.1s neighborhood
                p_lo = max(0, peak_idx - hood)
                p_hi = min(n, peak_idx + hood)
                seg_peak = data[p_lo:p_hi]
                if len(seg_peak) >= 4:
                    d2_peak = np.diff(seg_peak, n=2)
                    f[60] = np.max(np.abs(d2_peak)) / r_range
                else:
                    f[60] = 0.0

                # f61: Descent smoothness
                # Std of first derivative in post-peak half, / R range.
                # Wall = smooth descent (low std), clot = noisy (high std).
                post = data[peak_idx:]
                if len(post) >= 10:
                    post_deriv = np.diff(post)
                    f[61] = np.std(post_deriv) / r_range
                else:
                    f[61] = 0.0

                # f62: Shape asymmetry (skewness of normalized signal)
                # Clot: sharp rise + slow fall → right-skewed (positive).
                # Wall: gradual rise + sharp drop → left-skewed (negative).
                f[62] = float(stats.skew(d_norm)) if n >= 8 else 0.0

                # f63: Plateau ratio
                # Fraction of window within top 10% of range.
                # Wall = high (sustained near max), clot = low (sharp spike).
                f[63] = np.sum(d_norm >= 0.90) / n

            else:
                # Flat/low-range signal — set all to 0
                f[57] = f[58] = f[59] = f[60] = f[61] = f[62] = f[63] = 0.0

        # ── f64: Texture RMS (bandpass 5-50 Hz) ──
        # Physics: bandpass isolates surface roughness from cardiac (0.5-3 Hz) and drift (<0.1 Hz).
        # Clot = rough surface → high texture RMS (>30 Ω in Zott's data).
        # Wall = smooth surface → low texture RMS (<20 Ω).
        # Blood = moderate (cardiac dominates, little texture).
        # Implementation: 4th-order Butterworth bandpass, zero-phase (sosfiltfilt).
        # Uses only the LAST 1s (150 samples) to match Zott's 1s analysis window.
        if self._need_texture:
            _TEX_SAMPLES = int(self.fs * 1.0)  # 1s = 150 samples
            seg = data[-_TEX_SAMPLES:] if n >= _TEX_SAMPLES else data
            if len(seg) >= 60:  # need enough samples for the filter to be stable
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

SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_V9_2class_seq{SEQ_LEN}_{dim_str}.pkl"
MODEL_PATH = PROJECT_ROOT / "src" / "training" / "clot_gru_trained_V9_2class.pt"

# ── Ensemble configuration ──
# Set ENSEMBLE_SEEDS to a list of seeds to average multiple models' outputs.
# Set to None or [] to use a single model (MODEL_PATH) — original behavior.
ENSEMBLE_SEEDS = None  # Set to [42, 123, 456, 789, 2026] for ensemble inference
USE_DENOISED = False   # Set True to use pulse-subtracted data from test_data_denoised/
SAVE_PARQUET = True    # Set True to save detection_results .parquet files
SAVE_CSV = False       # Set True to save detection_results .csv files
TEST_DATA_DIR = PROJECT_ROOT / ("test_data_denoised" if USE_DENOISED else "test_data")
OUTPUT_FOLDER = PROJECT_ROOT / "inference_deploy" / "Results"

# ── Optional V9.2 hybrid confirmation gate ──
# If enabled, ML override is allowed only when BOTH:
#   1) the native V9 raw-stability gate passes, and
#   2) the learned DA-override policy agrees with the same non-DA class at
#      confidence >= (policy_threshold + OVERRIDE_POLICY_EXTRA_MARGIN).
USE_OVERRIDE_POLICY_CONFIRM = True
OVERRIDE_POLICY_BUNDLE_PATH = PROJECT_ROOT / "analysis_data_drift" / "override_model_v1" / "override_policy_v1.joblib"
OVERRIDE_POLICY_EXTRA_MARGIN = 0.12
OVERRIDE_POLICY_REQUIRE_AGREE_WITH_RAW = True

# Asymmetric safety policy (V9.2): allow only DA clot -> ML wall direction.
# 3-class labels: blood=0, clot=1, wall=2.
ALLOW_ONLY_CLOT_TO_WALL_OVERRIDE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────
# ClotGRU Model
# ────────────────────────────────────────────────
class ClotGRU(nn.Module):
    def __init__(self, input_size=None, hidden_size=32, output_size=2):
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

        # Load model(s)
        if ensemble_seeds:
            # Ensemble mode: load one model per seed
            self.models = []
            self.hiddens = []
            model_dir = PROJECT_ROOT / "src" / "training"
            for seed in ensemble_seeds:
                # V9 (2-class) model naming
                pattern = f"clot_gru_trained_V9_2class_seq{SEQ_LEN}_{FEATURE_SET}_seed{seed}_f1*.pt"
                candidates = sorted(model_dir.glob(pattern))
                if not candidates:
                    print(f"  WARNING: No model found for seed {seed} ({pattern})")
                    continue
                # Pick the one with highest F1 (last alphabetically since f1 is in filename)
                best_path = candidates[-1]
                m = ClotGRU().to(DEVICE)
                m.load_state_dict(torch.load(best_path, map_location=DEVICE))
                m.eval()
                self.models.append(m)
                self.hiddens.append(None)
            if not self.models:
                raise FileNotFoundError(f"No ensemble models found for seeds {ensemble_seeds}")
            self.ensemble = True
            print(f"  Ensemble: loaded {len(self.models)} models (seeds: {ensemble_seeds})")
        else:
            # Single model mode (original behavior)
            self.models = [ClotGRU().to(DEVICE)]
            self.models[0].load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.models[0].eval()
            self.hiddens = [None]
            self.ensemble = False

        # 2-class internal posterior: [P(clot), P(wall)]
        # Initialized near class prior from training data (clot ~66%, wall ~34%).
        # Blood is NEVER in this posterior; blood is emitted only when DA says blood.
        self.posterior = np.array([0.65, 0.35], dtype=np.float32)
        self.feat_history = deque(maxlen=SEQ_LEN)
        # Raw-prediction stability tracking
        # raw_last_idx uses the 2-class model index (0=clot, 1=wall).
        self.raw_last_idx = None
        self.raw_stable_streak = 0
        self.raw_conf_window = deque(maxlen=ML_STABILITY_STREAK)
        # raw_probs is stored as a 3-vec for downstream diagnostic compatibility:
        # [0.0, P(clot), P(wall)]. Blood slot is always 0 because model is 2-class.
        self.raw_probs = np.array([0.0, 0.65, 0.35], dtype=np.float32)

        # Optional learned override-policy confirmation model.
        self.override_policy_enabled = False
        self.override_policy_model = None
        self.override_policy_threshold = 0.50
        self.override_policy_last_pred = None
        self.override_policy_last_conf = np.nan
        if USE_OVERRIDE_POLICY_CONFIRM:
            self._load_override_policy()

    def _load_override_policy(self):
        """Load learned DA-override policy bundle if available and compatible."""
        if not OVERRIDE_POLICY_BUNDLE_PATH.exists():
            warnings.warn(
                f"Override policy bundle missing ({OVERRIDE_POLICY_BUNDLE_PATH}); "
                "falling back to native V9 stability gate only."
            )
            return

        try:
            bundle = joblib.load(OVERRIDE_POLICY_BUNDLE_PATH)
            model = bundle.get("model")
            feature_cols = bundle.get("feature_cols", [])
            threshold = float(bundle.get("best_threshold", 0.50))

            if model is None:
                warnings.warn("Override policy bundle has no model; disabled.")
                return

            if len(feature_cols) != active_dim:
                warnings.warn(
                    "Override policy feature count does not match active_dim "
                    f"({len(feature_cols)} != {active_dim}); disabled."
                )
                return

            self.override_policy_model = model
            self.override_policy_threshold = threshold
            self.override_policy_enabled = True
            print(
                "  Override policy confirm: enabled "
                f"(thr={self.override_policy_threshold:.2f}, +margin={OVERRIDE_POLICY_EXTRA_MARGIN:.2f})"
            )
        except Exception as exc:
            warnings.warn(f"Failed loading override policy bundle: {exc}")

    def _override_policy_confirms(self, active_feats, da_label):
        """Second gate for override: learned DA-error policy must agree.

        Returns True only when the policy predicts the same non-DA class as the
        raw GRU disagreement and does so at high confidence.
        """
        if not self.override_policy_enabled:
            return True
        if da_label not in (1, 2):
            return False

        # V9.2 asymmetric policy: only allow clot -> wall overrides.
        # DA=clot means 2-class da_idx=0 and override target must be wall idx=1.
        if ALLOW_ONLY_CLOT_TO_WALL_OVERRIDE:
            if da_label != 1:
                return False
            if self.raw_last_idx is not None and self.raw_last_idx != 1:
                return False

        x = np.asarray(active_feats, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != active_dim or not np.all(np.isfinite(x)):
            return False

        da_idx = 0 if da_label == 1 else 1
        da_onehot = np.zeros((1, 2), dtype=np.float32)
        da_onehot[0, da_idx] = 1.0
        x_all = np.hstack([x, da_onehot])

        probs = self.override_policy_model.predict_proba(x_all)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(np.max(probs))
        self.override_policy_last_pred = pred_idx
        self.override_policy_last_conf = conf

        # Must disagree with DA to justify an override.
        if pred_idx == da_idx:
            return False

        if ALLOW_ONLY_CLOT_TO_WALL_OVERRIDE and not (da_idx == 0 and pred_idx == 1):
            return False

        # Optional consistency check: learned policy and raw GRU must propose
        # the same class direction for the override.
        if OVERRIDE_POLICY_REQUIRE_AGREE_WITH_RAW and self.raw_last_idx is not None:
            if pred_idx != self.raw_last_idx:
                return False

        req_conf = min(0.999, self.override_policy_threshold + OVERRIDE_POLICY_EXTRA_MARGIN)
        if conf < req_conf:
            return False
        return True

    def _emit(self):
        """Convert the 2-class posterior to the 3-vec [blood, clot, wall] format
        that downstream code expects. Blood is 0 here because this method is only
        called when we are NOT in DA-blood; DA-blood emits [1,0,0] directly.
        """
        return np.array([0.0, float(self.posterior[0]), float(self.posterior[1])],
                        dtype=np.float32)

    def _make_da_probs_2class(self, da_label):
        """Build a 2-class probability vector [P(clot), P(wall)] heavily favoring
        the DA-labeled class. da_label is 1 (clot) or 2 (wall) in the 3-class
        namespace; map to 2-class index (0=clot, 1=wall).
        """
        da_idx = 0 if da_label == 1 else 1
        p = np.empty(2, dtype=np.float32)
        p[da_idx] = DA_LABEL_CONFIDENCE
        p[1 - da_idx] = DA_OTHER_CONFIDENCE
        return p

    def _update_raw_stability(self, raw_probs_2c):
        """Track how many consecutive samples the raw GRU has predicted the same
        2-class label (0=clot or 1=wall), and keep a rolling window of raw
        confidence for that stable run.
        """
        idx = int(np.argmax(raw_probs_2c))
        conf = float(np.max(raw_probs_2c))
        if idx == self.raw_last_idx:
            self.raw_stable_streak += 1
        else:
            self.raw_last_idx = idx
            self.raw_stable_streak = 1
            self.raw_conf_window.clear()
        self.raw_conf_window.append(conf)

    def _ml_should_override_da(self, da_label):
        """Return True when ML has EARNED the right to override DA on clot/wall.

        V9 refined policy: DA wins clot/wall by default. ML only overrides when
        ALL of the following are true over the last ML_STABILITY_STREAK samples:
          1. Raw GRU has predicted the same non-DA 2-class label the whole time
             (argmax stability).
          2. Mean raw confidence  >= ML_STABILITY_MEAN_CONF.
          3. Confidence range (max - min) <= ML_STABILITY_CONF_RANGE (steady
             confidence, not oscillating; filters 9CB4378D-style failures where
             the argmax is stable but the confidence swings wildly).

        da_label is in the 3-class namespace: 1=clot, 2=wall. Map to 2-class:
        0=clot, 1=wall.
        """
        if da_label not in (1, 2):
            return False
        da_idx_2c = 0 if da_label == 1 else 1

        # V9.2 asymmetric policy: only permit clot -> wall overrides.
        if ALLOW_ONLY_CLOT_TO_WALL_OVERRIDE and da_label != 1:
            return False

        if self.raw_last_idx is None or self.raw_last_idx == da_idx_2c:
            return False

        if ALLOW_ONLY_CLOT_TO_WALL_OVERRIDE and self.raw_last_idx != 1:
            return False

        if self.raw_stable_streak < ML_STABILITY_STREAK:
            return False
        if len(self.raw_conf_window) < ML_STABILITY_STREAK:
            return False

        conf_arr = np.asarray(self.raw_conf_window, dtype=np.float32)
        mean_conf = float(conf_arr.mean())
        conf_range = float(conf_arr.max() - conf_arr.min())

        if mean_conf < ML_STABILITY_MEAN_CONF:
            return False
        if conf_range > ML_STABILITY_CONF_RANGE:
            return False
        return True

    @torch.no_grad()
    def predict(self, active_feats, da_label=None):
        """
        Run one prediction step. Returns a 3-element vector [P(blood), P(clot), P(wall)]
        for downstream compatibility, even though the internal model is 2-class.

        Pipeline:
          1. Scale features, build sequence, run GRU(s) → raw 2-class probs [P(clot), P(wall)]
          2. EMA-blend into the 2-class posterior
          3. DA guardrail:
             - da_label == 0  → hard blood emit [1,0,0], clear state, return
             - da_label ∈ {1,2} → DA wins by default; ML overrides only when the
                             raw stability gate is satisfied; optional learned policy can
                             require a second confirmation before override is allowed
             - da_label is None → emit posterior directly
          4. Final safety snap: on da_label ∈ {1,2}, if posterior drifted away
             from DA and gate not satisfied, snap posterior to DA.
        """

        # ── Step 1: Scale features & run GRU(s) ──
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

        # 2-class temperature-scaled softmax → [P(clot), P(wall)]
        probs_2c = torch.softmax(avg_logits / TEMPERATURE, 1).squeeze(0).cpu().numpy()

        # Store raw probs as a 3-vec for diagnostic compatibility.
        self.raw_probs = np.array([0.0, float(probs_2c[0]), float(probs_2c[1])],
                                  dtype=np.float32)

        # ── Step 2: EMA blending (2-class, single rate) ──
        self.posterior = EMA_HISTORY * self.posterior + EMA_NEW * probs_2c

        # ── Step 3: DA guardrail ──
        self._update_raw_stability(probs_2c)

        if da_label is not None:
            if da_label == 0:
                # DA says blood → hard reset. Unconditional, no gate.
                # Reset 2-class posterior to neutral prior and clear model state.
                self.posterior = np.array([0.65, 0.35], dtype=np.float32)
                self.hiddens = [None] * len(self.models)
                self.feat_history.clear()
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)

            elif da_label in (1, 2):
                # DA says clot/wall → DA wins by default. ML only takes over
                # with stable, sustained, confident disagreement.
                can_override = self._ml_should_override_da(da_label)
                if can_override and USE_OVERRIDE_POLICY_CONFIRM:
                    can_override = self._override_policy_confirms(active_feats, da_label)

                if not can_override:
                    self.posterior = self._make_da_probs_2class(da_label)

        # ── Step 4: Final safety check ──
        if da_label in (1, 2):
            da_idx_2c = 0 if da_label == 1 else 1
            final_idx = int(np.argmax(self.posterior))
            can_override = self._ml_should_override_da(da_label)
            if can_override and USE_OVERRIDE_POLICY_CONFIRM:
                can_override = self._override_policy_confirms(active_feats, da_label)

            if final_idx != da_idx_2c and not can_override:
                self.posterior = self._make_da_probs_2class(da_label)

        return self._emit()

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

    # Save detection_results (optional)
    if save_parquet:
        results_df.to_parquet(OUTPUT_FOLDER / f"{study_name}_detection_results.parquet", index=False)
        print(f"  Saved detection_results.parquet")
    if save_csv:
        results_df.to_csv(OUTPUT_FOLDER / f"{study_name}_detection_results.csv", index=False)
        print(f"  Saved detection_results.csv")

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
    _prob_plot_path = OUTPUT_FOLDER / f"{study_name}_detected_vs_clot_wall_probs.png"
    try:
        plt.savefig(_prob_plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved probability plot")
    except OSError as e:
        # Typical on Windows/OneDrive when the target PNG is locked by an image viewer
        # or is in a bad sync state. Skip so the batch keeps going.
        print(f"  WARNING: could not save probability plot ({e}); continuing")
    plt.close()

    # ── Three-panel plot ──
    if gt_labels is not None and da_labels is not None:
        gt = gt_labels.values.astype(int)
        da = da_labels.values.astype(int)
        full_times = time_ms / 1000.0

        # Categorical labels must be aligned with stepwise indexing, not linear
        # interpolation. Linear interpolation between class IDs fabricates
        # non-existent transitions and inflates override counts.
        ml_report_times = results_df['time'].to_numpy(dtype=np.float64)
        ml_report_preds = results_df['prediction'].to_numpy(dtype=np.int16)

        idx_full = np.searchsorted(ml_report_times, full_times, side='right') - 1
        idx_full = np.clip(idx_full, 0, len(ml_report_preds) - 1)
        interp_ml = ml_report_preds[idx_full]

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

        # For diagnostic shading at report points, use nearest DA sample.
        idx_rep = np.searchsorted(full_times, ml_report_times, side='left')
        idx_rep = np.clip(idx_rep, 0, len(da) - 1)
        ml_da = da[idx_rep]
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
        _three_panel_path = OUTPUT_FOLDER / f"{study_name}_ml_da_gt_three_panel.png"
        try:
            plt.savefig(_three_panel_path, dpi=300, bbox_inches='tight')
            print(f"  Saved three-panel plot")
        except OSError as e:
            print(f"  WARNING: could not save three-panel plot ({e}); continuing")
        plt.close()

        # Metrics (on labeled samples only)
        print(f"\n{study_name} metrics:")
        print(f"DA  Acc: {accuracy_score(gt_valid, da_valid):.4f}  F1: {f1_score(gt_valid, da_valid, average='macro'):.4f} "
              f"Prec: {precision_score(gt_valid, da_valid, average='macro'):.4f}  Rec: {recall_score(gt_valid, da_valid, average='macro'):.4f}")
        print(f"ML  Acc: {accuracy_score(gt_valid, ml_valid):.4f}  F1: {f1_score(gt_valid, ml_valid, average='macro'):.4f} "
              f"Prec: {precision_score(gt_valid, ml_valid, average='macro'):.4f}  Rec: {recall_score(gt_valid, ml_valid, average='macro'):.4f}")
        print(f"Improvement: Acc {accuracy_score(gt_valid, ml_valid)-accuracy_score(gt_valid, da_valid):+.4f}   "
              f"F1 {f1_score(gt_valid, ml_valid, average='macro')-f1_score(gt_valid, da_valid, average='macro'):+.4f}")

        # Override analysis (clot/wall only)
        # Exclude any sample where GT, DA, or ML is blood (0) when judging
        # override usefulness for clot-vs-wall behavior.
        cw_mask = (gt_valid != 0) & (da_valid != 0) & (ml_valid != 0)
        override_mask = cw_mask & (ml_valid != da_valid)
        n_overrides = int(override_mask.sum())
        if n_overrides > 0:
            correct_overrides = int((ml_valid[override_mask] == gt_valid[override_mask]).sum())
            harmful_overrides = int((da_valid[override_mask] == gt_valid[override_mask]).sum())
            override_prec = correct_overrides / n_overrides

            da_cw_errors = ((da_valid != gt_valid) & ((gt_valid == 1) | (gt_valid == 2))).sum()
            override_rec = correct_overrides / da_cw_errors if da_cw_errors > 0 else 0.0

            print(f"\n  Override analysis ({study_name}):")
            print(f"    Total overrides:   {n_overrides}")
            print(f"    Correct (ML right, DA wrong): {correct_overrides}")
            print(f"    Harmful (DA right, ML wrong): {harmful_overrides}")
            print(f"    Override Precision: {override_prec:.4f}")
            print(f"    Override Recall:    {override_rec:.4f}  (of {da_cw_errors} DA clot/wall errors)")

            # ── Direction breakdown (clot/wall-only) ──
            # An override is characterised by (DA_class -> ML_class). For each
            # direction we print how many were correct vs harmful. If one
            # direction is consistently good and the other consistently bad,
            # we can gate them asymmetrically.
            def _dir_stats(da_from, ml_to, name):
                mask = override_mask & (da_valid == da_from) & (ml_valid == ml_to)
                n = int(mask.sum())
                if n == 0:
                    return f"    {name:<20} n=0"
                correct = int((ml_valid[mask] == gt_valid[mask]).sum())
                harmful = int((da_valid[mask] == gt_valid[mask]).sum())
                prec = correct / n if n else 0.0
                return (f"    {name:<20} n={n:6d}  correct={correct:6d}  "
                        f"harmful={harmful:6d}  prec={prec:.3f}")

            print(f"\n  Override direction breakdown ({study_name}):")
            print(_dir_stats(1, 2, "clot -> wall"))   # DA said clot, ML said wall
            print(_dir_stats(2, 1, "wall -> clot"))   # DA said wall, ML said clot
            # Track excluded blood-involved disagreements for transparency.
            n_blood_dir = int(((ml_valid != da_valid) & ~cw_mask).sum())
            if n_blood_dir > 0:
                print(f"    {'(blood involved)':<20} n={n_blood_dir:6d}  "
                      "(excluded from override metrics)")
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
        try:
            process_file(f,
                         all_gt_labels=all_gt_labels,
                         all_da_labels=all_da_labels,
                         all_ml_preds=all_ml_preds,
                         all_override_times=all_override_times,
                         save_parquet=SAVE_PARQUET,
                         save_csv=SAVE_CSV)
        except Exception as e:
            print(f"  ERROR processing {f.name}: {type(e).__name__}: {e}")
            print(f"  Skipping and continuing with remaining files.")
            import matplotlib.pyplot as _plt
            _plt.close('all')

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
