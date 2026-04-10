# cardiac_pulse_removal.py
"""
Cardiac pulse removal from impedance/resistance signals.

Reads a parquet file, removes pulse artifact using moving-median filtering,
estimates heart rate (BPM), and outputs:
  1. A denoised parquet file with the clean signal replacing the original.
  2. A 4-panel plot: raw vs clean, extracted pulse, clean signal, BPM trace.

Usage:
  python cardiac_pulse_removal.py                          # process all parquet files in test_data/
  python cardiac_pulse_removal.py path/to/file.parquet     # process a single file
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import medfilt, find_peaks


# ════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════
SAMPLE_RATE = 150           # Hz
CARDIAC_WINDOW_SEC = 2.0    # Median filter window — covers 40-200 BPM
SMOOTH_WINDOW_SEC = 0.3     # Post-median moving average
BPM_WINDOW_SEC = 5.0        # Sliding window for BPM estimation
BPM_STEP_SEC = 1.0          # Step between BPM estimates

SIGNAL_COL = 'magRLoadAdjusted'   # column name of the resistance signal
TIME_COL = 'timeInMS'             # column name of the time (milliseconds)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_DIR = SCRIPT_DIR / "pulse_removal_results"


# ════════════════════════════════════════════════
#  PULSE SUBTRACTION
# ════════════════════════════════════════════════

def remove_pulse(signal, sample_rate=SAMPLE_RATE,
                 cardiac_window_sec=CARDIAC_WINDOW_SEC,
                 smooth_window_sec=SMOOTH_WINDOW_SEC):
    """Remove cardiac pulse artifact using moving-median filtering.

    The moving median is immune to brief pulse spikes because they
    occupy <50% of the window.  A post-median moving average removes
    the staircase artifact inherent in median filters.

    Args:
        signal:             1-D float array of resistance values
        sample_rate:        samples per second
        cardiac_window_sec: median filter window (must span >1 pulse period)
        smooth_window_sec:  moving average window for final smoothing

    Returns:
        clean:  pulse-subtracted signal (same length as input)
        pulse:  extracted pulse component (raw - clean)
    """
    med_win = int(cardiac_window_sec * sample_rate) | 1  # force odd
    smooth_win = max(1, int(smooth_window_sec * sample_rate))

    med_win = min(med_win, len(signal))
    if med_win % 2 == 0:
        med_win -= 1
    if med_win < 3:
        return signal.copy(), np.zeros_like(signal)

    # Step 1: Median filter — extracts slow trend, ignores pulse spikes
    trend = medfilt(signal, kernel_size=med_win)

    # Step 2: Moving average — smooths staircase artifact from median
    if smooth_win > 1 and len(trend) >= smooth_win:
        kernel = np.ones(smooth_win) / smooth_win
        clean = np.convolve(trend, kernel, mode='same')
    else:
        clean = trend

    clean = clean.astype(np.float32)
    pulse = signal - clean
    return clean, pulse


# ════════════════════════════════════════════════
#  BPM ESTIMATION
# ════════════════════════════════════════════════

def estimate_bpm(pulse, time_sec, sample_rate=SAMPLE_RATE,
                 window_sec=BPM_WINDOW_SEC, step_sec=BPM_STEP_SEC):
    """Estimate heart rate over time from the pulse component.

    In each sliding window, finds peaks and computes BPM from the
    median inter-peak interval.  Returns 0 BPM where no pulse is detected.

    Args:
        pulse:      extracted pulse component (from remove_pulse)
        time_sec:   time array in seconds
        sample_rate: samples per second
        window_sec: sliding window length
        step_sec:   step between estimates

    Returns:
        bpm_times:  array of time points (center of each window)
        bpm_values: array of BPM values (0 where no pulse detected)
    """
    n = len(pulse)
    win_samples = int(window_sec * sample_rate)
    step_samples = int(step_sec * sample_rate)
    min_peak_dist = int(0.15 * sample_rate)  # 200 BPM ceiling

    bpm_times = []
    bpm_values = []

    for start in range(0, n - win_samples + 1, step_samples):
        end = start + win_samples
        chunk = pulse[start:end]
        t_center = time_sec[start + win_samples // 2]

        chunk_std = np.std(chunk)
        if chunk_std < 0.1:
            bpm_times.append(t_center)
            bpm_values.append(0.0)
            continue

        height_threshold = chunk_std * 0.5
        peaks, _ = find_peaks(chunk, height=height_threshold, distance=min_peak_dist)

        if len(peaks) >= 2:
            intervals = np.diff(peaks) / sample_rate
            median_interval = np.median(intervals)
            if median_interval > 0:
                bpm = 60.0 / median_interval
                if 30 <= bpm <= 220:
                    bpm_times.append(t_center)
                    bpm_values.append(bpm)
                else:
                    bpm_times.append(t_center)
                    bpm_values.append(0.0)
            else:
                bpm_times.append(t_center)
                bpm_values.append(0.0)
        else:
            bpm_times.append(t_center)
            bpm_values.append(0.0)

    return np.array(bpm_times), np.array(bpm_values)


# ════════════════════════════════════════════════
#  PROCESS ONE FILE
# ════════════════════════════════════════════════

def process_file(parquet_path, output_dir=OUTPUT_DIR):
    """Process a single parquet file: remove pulse, estimate BPM, save outputs."""
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_name = parquet_path.stem

    print(f"\n{'─'*60}")
    print(f"Processing: {study_name}")
    print(f"{'─'*60}")

    # ── Load data ──
    df = pd.read_parquet(parquet_path)
    time_ms = df[TIME_COL].values
    resistance = df[SIGNAL_COL].values.astype(np.float32)
    time_sec = time_ms / 1000.0

    print(f"  Samples: {len(resistance)}  Duration: {time_sec[-1] - time_sec[0]:.1f}s")

    # ── Remove pulse ──
    clean, pulse = remove_pulse(resistance)

    # ── Estimate BPM ──
    bpm_times, bpm_values = estimate_bpm(pulse, time_sec)
    has_pulse = bpm_values > 0
    if np.any(has_pulse):
        valid_bpm = bpm_values[has_pulse]
        mean_bpm = np.mean(valid_bpm)
        print(f"  Heart rate: mean={mean_bpm:.0f} BPM  "
              f"range={np.min(valid_bpm):.0f}-{np.max(valid_bpm):.0f} BPM")
    else:
        mean_bpm = 0
        print(f"  No pulse detected")

    # ── Save denoised parquet ──
    df_out = df.copy()
    df_out[SIGNAL_COL] = clean
    out_parquet = output_dir / f"{study_name}_denoised.parquet"
    df_out.to_parquet(out_parquet, index=False)
    print(f"  Saved: {out_parquet.name}")

    # ── Plot ──
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1.5, 2, 1]})

    # Panel 1: Raw + Clean overlay
    ax = axes[0]
    ax.plot(time_sec, resistance, color='gray', alpha=0.5, linewidth=0.5, label='Raw')
    ax.plot(time_sec, clean, color='black', linewidth=1.2, label='Pulse-subtracted')
    ax.set_ylabel('Resistance (Ω)')
    ax.set_title(f'{study_name} — Raw vs Pulse-subtracted signal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Extracted pulse component
    ax = axes[1]
    ax.plot(time_sec, pulse, color='red', linewidth=0.5, alpha=0.7)
    ax.set_ylabel('Pulse component (Ω)')
    if mean_bpm > 0:
        ax.set_title(f'Extracted cardiac pulse  (mean {mean_bpm:.0f} BPM)')
    else:
        ax.set_title('Extracted cardiac pulse')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    # Panel 3: Clean signal only
    ax = axes[2]
    ax.plot(time_sec, clean, color='black', linewidth=1.0)
    ax.set_ylabel('Resistance (Ω)')
    ax.set_title(f'{study_name} — Pulse-subtracted signal')
    ax.grid(True, alpha=0.3)

    # Panel 4: Heart rate — show 0 BPM in grayed no-pulse regions
    ax = axes[3]
    ax.plot(bpm_times, bpm_values, color='purple', linewidth=1.5,
            marker='.', markersize=3)
    # Gray shading where BPM = 0 (no pulse detected)
    no_pulse = bpm_values == 0
    if np.any(no_pulse):
        # Find contiguous no-pulse regions
        diff = np.diff(no_pulse.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if no_pulse[0]:
            starts = np.insert(starts, 0, 0)
        if no_pulse[-1]:
            ends = np.append(ends, len(no_pulse))
        for s, e in zip(starts, ends):
            ax.axvspan(bpm_times[s], bpm_times[min(e, len(bpm_times)) - 1],
                       color='gray', alpha=0.15)

    if np.any(has_pulse):
        y_max = min(220, np.max(bpm_values[has_pulse]) + 10)
    else:
        y_max = 100
    ax.set_ylim(-5, y_max)
    ax.set_ylabel('BPM')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Estimated Heart Rate  (gray = no pulse detected, 0 BPM)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = output_dir / f"{study_name}_pulse_removal.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png.name}")


# ════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════

def main():
    if len(sys.argv) > 1:
        # Process specific file(s)
        for path in sys.argv[1:]:
            p = Path(path)
            if p.exists():
                process_file(p)
            else:
                print(f"File not found: {p}")
    else:
        # Process all parquet files in test_data/
        files = sorted(TEST_DATA_DIR.glob("*_labeled_segment.parquet"))
        if not files:
            print(f"No parquet files found in {TEST_DATA_DIR}")
            sys.exit(1)
        print(f"Found {len(files)} files in {TEST_DATA_DIR}")
        for f in files:
            process_file(f)

    print("\nDone.")


if __name__ == "__main__":
    main()
