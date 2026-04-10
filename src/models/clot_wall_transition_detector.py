# clot_wall_transition_detector.py
"""
Deterministic clot-to-wall transition detector with pulse subtraction.

Two algorithms:
  1. Pulse subtraction — removes cardiac artifact from resistance signal
     using adaptive moving-median detrending.
  2. Clot-to-wall pattern detector — finds the signature shape:
     sharp vertical rise (clot) → smooth flat plateau (wall/latch).

Both are streaming-capable (designed for real-time use on STM32).
"""

import numpy as np
from collections import deque
from scipy.signal import medfilt, find_peaks
import matplotlib.pyplot as plt
from pathlib import Path


# ════════════════════════════════════════════════
#  1. PULSE SUBTRACTION
# ════════════════════════════════════════════════

class PulseSubtractor:
    """
    Removes cardiac pulse artifact from a resistance signal.

    Approach: moving median extracts the slow trend (immune to pulse spikes),
    then a small moving-average smooths the residual jitter.

    The "clean" signal = moving_average( moving_median(raw) ).
    The "pulse" signal = raw - clean.

    Why moving median and not moving average?
      - Moving average smears pulse peaks into the trend.
        e.g. avg([100, 100, 180, 100, 100]) = 116 (contaminated)
      - Moving median ignores the pulse spikes entirely because they're
        short outliers within each window.
        e.g. median([100, 100, 180, 100, 100]) = 100 (clean)

    The median works as long as pulse spikes occupy <50% of each window.
    A pulse spike is typically 0.05-0.15s wide.  At 200 BPM (0.3s period),
    a 2.0s window contains ~7 spikes × 0.1s = 0.7s of spike in 2.0s = 35%,
    well under 50%.  At 40 BPM (1.5s period), the window contains ~1.3
    spikes — still safe.

    Tuning:
      CARDIAC_WINDOW_SEC:  Must be longer than the longest expected pulse
        period.  Rule of thumb: 2× period at slowest expected BPM.
        At 40 BPM → period = 1.5s → window = 2.0s (covers 1+ full cycle).
        At 200 BPM → period = 0.3s → window = 2.0s (covers 6-7 cycles).
        Default 2.0s works for the full 40-200 BPM range.
        WARNING: too short a window at low BPM will smear the pulse into
        the trend; too long adds latency to real signal changes.

      SMOOTH_WINDOW_SEC:  Post-median smoothing to remove the staircase
        artifact inherent in median filters.  The median output can have
        small jumps when the window slides past a value.  The moving average
        smooths these.  Smaller = more responsive, larger = smoother.
        Default 0.3s is a good balance.

    For embedded (STM32): replace median with a ring-buffer insertion-sort
    median, which is O(n) per sample for small windows (~300 samples).
    """

    def __init__(self, sample_rate=150, cardiac_window_sec=2.0, smooth_window_sec=0.3):
        self.fs = sample_rate
        # Median window must be odd
        self.med_win = int(cardiac_window_sec * sample_rate) | 1  # force odd
        self.smooth_win = max(1, int(smooth_window_sec * sample_rate))

        # Streaming buffers
        self.raw_buf = deque(maxlen=self.med_win + self.smooth_win)
        self.clean_buf = deque(maxlen=self.smooth_win)

    def reset(self):
        self.raw_buf.clear()
        self.clean_buf.clear()

    def update(self, r):
        """Feed one raw sample, return the pulse-subtracted (clean) value.
        Returns None until enough samples have been buffered."""
        self.raw_buf.append(float(r))

        if len(self.raw_buf) < self.med_win:
            return None

        # Streaming median: take the last med_win samples
        window = list(self.raw_buf)[-self.med_win:]
        med_val = float(np.median(window))

        # Moving average of median values for additional smoothing
        self.clean_buf.append(med_val)
        if len(self.clean_buf) < self.smooth_win:
            return med_val

        return float(np.mean(self.clean_buf))

    def process_batch(self, signal):
        """Batch mode: process entire array, return pulse-subtracted signal.
        Faster than calling update() in a loop."""
        # Median filter (removes pulse spikes)
        med_win = min(self.med_win, len(signal))
        if med_win % 2 == 0:
            med_win -= 1
        if med_win < 3:
            return signal.copy()

        trend = medfilt(signal, kernel_size=med_win)

        # Moving average for extra smoothing
        if self.smooth_win > 1 and len(trend) >= self.smooth_win:
            kernel = np.ones(self.smooth_win) / self.smooth_win
            clean = np.convolve(trend, kernel, mode='same')
        else:
            clean = trend

        return clean.astype(np.float32)

    def extract_pulse(self, signal):
        """Return the pulse component: raw - clean."""
        clean = self.process_batch(signal)
        return signal - clean

    def estimate_bpm(self, signal, time_sec=None, window_sec=5.0, step_sec=1.0):
        """Estimate heart rate (BPM) over time from the pulse component.

        Approach: in sliding windows, find peaks in the pulse signal and
        compute BPM from the median inter-peak interval.

        Args:
            signal:     raw resistance signal (pulse will be extracted)
            time_sec:   time array (seconds).  If None, derived from sample_rate.
            window_sec: sliding window length for BPM estimation.
                        Longer = smoother BPM, shorter = more responsive.
            step_sec:   step size between BPM estimates.

        Returns:
            bpm_times:  array of time points (center of each window)
            bpm_values: array of BPM estimates (NaN where no pulse detected)
        """
        pulse = self.extract_pulse(signal)
        n = len(signal)

        if time_sec is None:
            time_sec = np.arange(n) / self.fs

        win_samples = int(window_sec * self.fs)
        step_samples = int(step_sec * self.fs)

        # Minimum peak distance: at 200 BPM, period = 0.3s = 45 samples at 150Hz
        min_peak_dist = int(0.15 * self.fs)  # half of fastest expected period

        bpm_times = []
        bpm_values = []

        for start in range(0, n - win_samples + 1, step_samples):
            end = start + win_samples
            chunk = pulse[start:end]
            t_center = time_sec[start + win_samples // 2]

            # Adaptive peak height: use the std of this chunk's pulse component.
            # Peaks must be at least 0.5× std above zero to count.
            chunk_std = np.std(chunk)
            if chunk_std < 0.1:
                # Signal is too flat — no detectable pulse (e.g. during wall contact)
                bpm_times.append(t_center)
                bpm_values.append(np.nan)
                continue

            height_threshold = chunk_std * 0.5

            peaks, _ = find_peaks(chunk, height=height_threshold, distance=min_peak_dist)

            if len(peaks) >= 2:
                # BPM from median inter-peak interval
                intervals = np.diff(peaks) / self.fs  # seconds between peaks
                median_interval = np.median(intervals)
                if median_interval > 0:
                    bpm = 60.0 / median_interval
                    # Sanity check: reject physiologically impossible rates
                    if 30 <= bpm <= 220:
                        bpm_times.append(t_center)
                        bpm_values.append(bpm)
                    else:
                        bpm_times.append(t_center)
                        bpm_values.append(np.nan)
                else:
                    bpm_times.append(t_center)
                    bpm_values.append(np.nan)
            else:
                bpm_times.append(t_center)
                bpm_values.append(np.nan)

        return np.array(bpm_times), np.array(bpm_values)


# ════════════════════════════════════════════════
#  2. CLOT-TO-WALL TRANSITION DETECTOR
# ════════════════════════════════════════════════

class ClotWallTransitionDetector:
    """
    Detects the clot→wall signature: sharp vertical rise followed by
    a smooth flat plateau.

    The detection has two phases:
      Phase 1 — SPIKE detection:
        A rapid increase in resistance over a short window.
        Triggered when slope exceeds SPIKE_SLOPE_THRESHOLD.

      Phase 2 — PLATEAU confirmation:
        After a spike, monitor for a sustained flat region.
        Confirmed when the signal stays flat (low slope, low std)
        for at least PLATEAU_MIN_SEC seconds.

    The detector outputs a confidence score (0-1) that ramps up
    during plateau confirmation.

    Tuning guide:
      SPIKE_SLOPE_THRESHOLD:  Minimum resistance change per second to
        qualify as a "spike".  Higher = fewer false positives, but may
        miss gentle clot attachments.  Units: Ω/sec.

      SPIKE_WINDOW_SEC:  The time window to measure the spike slope.
        Shorter = catches sharper spikes, longer = catches gentler rises.

      PLATEAU_MIN_SEC:  How long the signal must stay flat after a spike
        to confirm wall contact.  Shorter = faster detection, but more
        false positives on transient flatness.

      PLATEAU_MAX_SLOPE:  Maximum |slope| (Ω/sec) for a region to count
        as "flat".  Set relative to your signal's noise floor.

      PLATEAU_MAX_STD:  Maximum std within the plateau window.
        Catches cases where slope is near zero but signal is jittery.

    Works on either raw or pulse-subtracted signal (better on clean).
    """

    def __init__(self, sample_rate=150,
                 spike_slope_threshold=50.0,    # Ω/sec — minimum spike steepness
                 spike_window_sec=0.5,          # window to measure spike
                 plateau_min_sec=2.0,           # minimum flat duration to confirm
                 plateau_max_slope=5.0,         # Ω/sec — max slope to count as flat
                 plateau_max_std=3.0,           # Ω — max std to count as flat
                 plateau_window_sec=1.0):       # rolling window for plateau stats

        self.fs = sample_rate
        self.spike_slope_thrd = spike_slope_threshold
        self.spike_win = max(2, int(spike_window_sec * sample_rate))
        self.plateau_min_samples = int(plateau_min_sec * sample_rate)
        self.plateau_max_slope = plateau_max_slope
        self.plateau_max_std = plateau_max_std
        self.plateau_win = max(2, int(plateau_window_sec * sample_rate))

        # State
        self.buffer = deque(maxlen=max(self.spike_win, self.plateau_win) + 10)
        self.state = "idle"         # idle → spike_detected → plateau_confirming → confirmed
        self.plateau_count = 0      # samples in flat region since spike
        self.spike_time = None      # sample index when spike was detected
        self.spike_peak = None      # resistance value at spike peak
        self.sample_idx = 0
        self.confidence = 0.0

        # Output log
        self.detections = []        # list of (time_sec, spike_value, confidence)

    def reset(self):
        self.buffer.clear()
        self.state = "idle"
        self.plateau_count = 0
        self.spike_time = None
        self.spike_peak = None
        self.sample_idx = 0
        self.confidence = 0.0
        self.detections = []

    def update(self, r, time_sec=None):
        """Feed one sample (preferably pulse-subtracted). Returns confidence 0-1.

        confidence > 0 means a clot→wall transition is being tracked.
        confidence ≈ 1.0 means the transition is confirmed.
        """
        self.buffer.append(float(r))
        self.sample_idx += 1
        t = time_sec if time_sec is not None else self.sample_idx / self.fs

        if len(self.buffer) < self.spike_win:
            return 0.0

        buf = np.array(self.buffer)

        if self.state == "idle":
            # Check for a spike: large positive slope over spike_win
            spike_seg = buf[-self.spike_win:]
            slope = (spike_seg[-1] - spike_seg[0]) / (self.spike_win / self.fs)

            if slope >= self.spike_slope_thrd:
                self.state = "spike_detected"
                self.spike_time = t
                self.spike_peak = float(spike_seg[-1])
                self.plateau_count = 0
                self.confidence = 0.1
                return self.confidence

        elif self.state == "spike_detected":
            # Transition: check if we've moved past the spike peak
            # (allow a small settling period)
            if len(self.buffer) >= self.plateau_win:
                plat_seg = buf[-self.plateau_win:]
                slope = np.abs(np.polyfit(np.arange(len(plat_seg)), plat_seg, 1)[0] * self.fs)
                std = np.std(plat_seg)

                if slope <= self.plateau_max_slope and std <= self.plateau_max_std:
                    self.state = "plateau_confirming"
                    self.plateau_count = self.plateau_win
                    self.confidence = 0.2
                elif t - self.spike_time > 3.0:
                    # Timeout: if no plateau within 3s of spike, reset
                    self.state = "idle"
                    self.confidence = 0.0

            return self.confidence

        elif self.state == "plateau_confirming":
            # Keep checking that the signal stays flat
            if len(self.buffer) >= self.plateau_win:
                plat_seg = buf[-self.plateau_win:]
                slope = np.abs(np.polyfit(np.arange(len(plat_seg)), plat_seg, 1)[0] * self.fs)
                std = np.std(plat_seg)

                if slope <= self.plateau_max_slope and std <= self.plateau_max_std:
                    self.plateau_count += 1
                    progress = min(1.0, self.plateau_count / self.plateau_min_samples)
                    self.confidence = 0.2 + 0.8 * progress

                    if self.plateau_count >= self.plateau_min_samples:
                        self.state = "confirmed"
                        self.confidence = 1.0
                        self.detections.append({
                            'spike_time': self.spike_time,
                            'confirm_time': t,
                            'spike_peak': self.spike_peak,
                            'plateau_mean': float(np.mean(plat_seg)),
                        })
                else:
                    # Signal became non-flat — reset
                    self.state = "idle"
                    self.confidence = 0.0
                    self.plateau_count = 0

            return self.confidence

        elif self.state == "confirmed":
            # Stay confirmed until signal leaves the plateau
            if len(self.buffer) >= self.plateau_win:
                plat_seg = buf[-self.plateau_win:]
                slope = np.abs(np.polyfit(np.arange(len(plat_seg)), plat_seg, 1)[0] * self.fs)
                std = np.std(plat_seg)

                if slope > self.plateau_max_slope * 2 or std > self.plateau_max_std * 2:
                    self.state = "idle"
                    self.confidence = 0.0
                    self.plateau_count = 0

            return self.confidence

        return self.confidence

    def detect_batch(self, signal, time=None):
        """Batch mode: process entire signal array.
        Returns array of confidence values (same length as signal)."""
        self.reset()
        confidences = np.zeros(len(signal), dtype=np.float32)
        for i in range(len(signal)):
            t = time[i] if time is not None else None
            confidences[i] = self.update(signal[i], t)
        return confidences


# ════════════════════════════════════════════════
#  DEMO / VISUALIZATION
# ════════════════════════════════════════════════

def demo(parquet_path, start_sec=None, end_sec=None):
    """Run both algorithms on a parquet file and plot results."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    time_ms = df['timeInMS'].values
    resistance = df['magRLoadAdjusted'].values.astype(np.float32)
    time_sec = time_ms / 1000.0

    # Optionally crop to a region of interest
    if start_sec is not None:
        mask = time_sec >= start_sec
        if end_sec is not None:
            mask &= time_sec <= end_sec
        time_sec = time_sec[mask]
        time_ms = time_ms[mask]
        resistance = resistance[mask]

    print(f"Processing {len(resistance)} samples ({time_sec[0]:.1f}s - {time_sec[-1]:.1f}s)")

    # ── Step 1: Pulse subtraction ──
    ps = PulseSubtractor(sample_rate=150, cardiac_window_sec=2.0, smooth_window_sec=0.3)
    clean = ps.process_batch(resistance)
    pulse = resistance - clean

    # ── Step 1b: Estimate BPM from pulse component ──
    bpm_times, bpm_values = ps.estimate_bpm(resistance, time_sec, window_sec=5.0, step_sec=1.0)
    valid_bpm = ~np.isnan(bpm_values)
    if np.any(valid_bpm):
        mean_bpm = np.nanmean(bpm_values)
        print(f"Mean heart rate: {mean_bpm:.0f} BPM  "
              f"(range: {np.nanmin(bpm_values):.0f}-{np.nanmax(bpm_values):.0f})")
    else:
        print("No pulse detected")

    # ── Step 2: Transition detection (on clean signal) ──
    det = ClotWallTransitionDetector(
        sample_rate=150,
        spike_slope_threshold=50.0,   # Ω/sec — adjust based on your signal range
        spike_window_sec=0.5,
        plateau_min_sec=2.0,
        plateau_max_slope=5.0,        # Ω/sec
        plateau_max_std=3.0,          # Ω
    )
    confidence = det.detect_batch(clean, time_sec)

    print(f"\nDetected {len(det.detections)} clot→wall transitions:")
    for d in det.detections:
        print(f"  Spike at {d['spike_time']:.1f}s (peak={d['spike_peak']:.1f}Ω) → "
              f"Confirmed at {d['confirm_time']:.1f}s (plateau={d['plateau_mean']:.1f}Ω)")

    # ── Plot ──
    fig, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1.5, 2, 1.5, 1]})

    # Panel 1: Raw + Clean overlay
    ax = axes[0]
    ax.plot(time_sec, resistance, color='gray', alpha=0.5, linewidth=0.5, label='Raw')
    ax.plot(time_sec, clean, color='black', linewidth=1.2, label='Pulse-subtracted')
    ax.set_ylabel('Resistance (Ω)')
    ax.set_title('Raw vs Pulse-subtracted signal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Extracted pulse component + BPM annotation
    ax = axes[1]
    ax.plot(time_sec, pulse, color='red', linewidth=0.5, alpha=0.7)
    ax.set_ylabel('Pulse component (Ω)')
    if np.any(valid_bpm):
        ax.set_title(f'Extracted cardiac pulse  (mean {mean_bpm:.0f} BPM)')
    else:
        ax.set_title('Extracted cardiac pulse')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    # Panel 3: Clean signal with transition markers
    ax = axes[2]
    ax.plot(time_sec, clean, color='black', linewidth=1.0)
    for d in det.detections:
        ax.axvline(d['spike_time'], color='red', linestyle='--', alpha=0.7, label='Spike')
        ax.axvline(d['confirm_time'], color='blue', linestyle='--', alpha=0.7, label='Confirmed')
        ax.axvspan(d['spike_time'], d['confirm_time'], color='orange', alpha=0.15)
    ax.set_ylabel('Resistance (Ω)')
    ax.set_title('Clot→Wall transition detection')
    ax.grid(True, alpha=0.3)

    # Panel 4: Confidence trace
    ax = axes[3]
    ax.plot(time_sec, confidence, color='green', linewidth=1.5)
    ax.fill_between(time_sec, 0, confidence, color='green', alpha=0.15)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylabel('Confidence')
    ax.set_title('Transition confidence (0=idle, 1=confirmed wall)')
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: Heart rate over time
    ax = axes[4]
    if np.any(valid_bpm):
        ax.plot(bpm_times[valid_bpm], bpm_values[valid_bpm], color='purple',
                linewidth=1.5, marker='.', markersize=3)
        # Shade regions where no pulse was detected (e.g. wall contact)
        no_pulse = ~valid_bpm
        if np.any(no_pulse):
            for i in range(len(bpm_times)):
                if no_pulse[i]:
                    ax.axvspan(bpm_times[i] - 0.5, bpm_times[i] + 0.5,
                               color='gray', alpha=0.1)
        ax.set_ylim(max(0, np.nanmin(bpm_values) - 10),
                    min(220, np.nanmax(bpm_values) + 10))
    else:
        ax.text(0.5, 0.5, 'No pulse detected', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.set_ylabel('BPM')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Estimated Heart Rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name = Path(parquet_path).stem + "_transition_detection.png"
    out_path = Path(__file__).parent / out_name
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {out_path}")

    # ── Separate figure: pulse subtraction + BPM (no transition indicators) ──
    fig2, axes2 = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1.5, 2, 1]})
    study_name = Path(parquet_path).stem

    # Panel 1: Raw + Clean overlay
    ax = axes2[0]
    ax.plot(time_sec, resistance, color='gray', alpha=0.5, linewidth=0.5, label='Raw')
    ax.plot(time_sec, clean, color='black', linewidth=1.2, label='Pulse-subtracted')
    ax.set_ylabel('Resistance (Ω)')
    ax.set_title(f'{study_name} — Raw vs Pulse-subtracted signal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Extracted pulse component
    ax = axes2[1]
    ax.plot(time_sec, pulse, color='red', linewidth=0.5, alpha=0.7)
    ax.set_ylabel('Pulse component (Ω)')
    if np.any(valid_bpm):
        ax.set_title(f'Extracted cardiac pulse  (mean {mean_bpm:.0f} BPM)')
    else:
        ax.set_title('Extracted cardiac pulse')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    # Panel 3: Clean signal only (no transition markers)
    ax = axes2[2]
    ax.plot(time_sec, clean, color='black', linewidth=1.0)
    ax.set_ylabel('Resistance (Ω)')
    ax.set_title(f'{study_name} — Pulse-subtracted signal')
    ax.grid(True, alpha=0.3)

    # Panel 4: Heart rate over time
    ax = axes2[3]
    if np.any(valid_bpm):
        ax.plot(bpm_times[valid_bpm], bpm_values[valid_bpm], color='purple',
                linewidth=1.5, marker='.', markersize=3)
        no_pulse = ~valid_bpm
        if np.any(no_pulse):
            for i in range(len(bpm_times)):
                if no_pulse[i]:
                    ax.axvspan(bpm_times[i] - 0.5, bpm_times[i] + 0.5,
                               color='gray', alpha=0.1)
        ax.set_ylim(max(0, np.nanmin(bpm_values) - 10),
                    min(220, np.nanmax(bpm_values) + 10))
    else:
        ax.text(0.5, 0.5, 'No pulse detected', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.set_ylabel('BPM')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Estimated Heart Rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name2 = Path(parquet_path).stem + "_pulse_and_bpm.png"
    out_path2 = Path(__file__).parent / out_name2
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path2}")


if __name__ == "__main__":
    import sys

    # Default: STCLD001, full file
    test_dir = Path(__file__).resolve().parent.parent.parent / "test_data"
    parquet = test_dir / "33CFB812_labeled_segment.parquet"

    if not parquet.exists():
        files = sorted(test_dir.glob("*.parquet"))
        if files:
            parquet = files[0]
            print(f"Using: {parquet.name}")
        else:
            print(f"No parquet files found in {test_dir}")
            sys.exit(1)

    demo(str(parquet))
