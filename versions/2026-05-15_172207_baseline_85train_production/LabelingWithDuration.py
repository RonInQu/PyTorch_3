# -*- coding: utf-8 -*-
"""
LabelingWithDuration.py — Duration-filtered labeling for sustained events only.

Based on Labeling_5Names_V6.py with added MIN_TISSUE_DURATION_SEC filter.

Training: ALL non-tissue events are blanked to blood_median+noise (label=0).
Short tissue events (< MIN_TISSUE_DURATION_SEC) are also blanked to blood.
Only sustained clot/wall events keep their real resistance values.

Testing: ALL data kept including artifacts. R>5000 clipped to 5000 (not blanked).
NO duration filter — all tissue events (including short ones) keep original labels.
This gives honest real-world evaluation where the model sees everything.
Labels: blood/non-tissue=0, clot=1, wall=2.

Graphics: all 5 event types shown with distinct colors; contrast spikes shown at full
height (not clipped by the >5000 outlier mask).

Rationale: Short spikes (<5s) are easily handled by DA via height thresholds.
ML should focus on sustained events where DA struggles. By blanking short events
in both train and test, F1 is measured only on the domain ML is responsible for.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no GUI windows
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# === USER CONFIGURATION ===
# input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\merged_expts_with_events_April6\event_files'
# input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\merged_expts_with_events_April10\merged_expts_with_events\event_files'
# input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\ReAnalyisOfExpts\New_baseline93_analysis'
# input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\merged_expts_with_events_v94\event_files'
# input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\ReAnalyisOfExpts\New_April6'
# input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\ReAnalyisOfExpts\New_April10'
input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\14May2026\merged_expts_with_events\parquets'

NOISE_VALUE = 5
SAMPLE_RATE = 150

# Minimum tissue event duration (seconds). Clot/wall events shorter than this
# are blanked to blood_median+noise in BOTH training and testing.
# Rationale: short spikes (<5s) are easily handled by DA via height thresholds;
# ML should focus on sustained events where DA struggles.
# Set to None or 0 to disable this filter.
MIN_TISSUE_DURATION_SEC = 7.0

output_base = os.path.join(input_folder, 'processedResults')
training_folder = os.path.join(output_base, 'training')
graphics_folder = os.path.join(output_base, 'graphics')
testing_folder  = os.path.join(output_base, 'testing')

for folder in [training_folder, graphics_folder, testing_folder]:
    os.makedirs(folder, exist_ok=True)

pattern = os.path.join(input_folder, '_merged_rec_and_event_*.parquet')
parquet_files = glob.glob(pattern)

if not parquet_files:
    raise ValueError(f"No parquet files found in {input_folder}")

print(f"Found {len(parquet_files)} file(s) to process.")
print(f"MIN_TISSUE_DURATION_SEC = {MIN_TISSUE_DURATION_SEC}")

# Event definitions
blood_events    = [6, 12]
clot_events     = [7, 11]
wall_events     = [23]
contrast_events = [8]
saline_events   = [15]

# Artifact events — always blanked in training/testing
artifact_events = contrast_events + saline_events

# Events that represent real tissue contact (used for training/testing highlight logic)
tissue_events = blood_events + clot_events + wall_events

# All events shown in graphics (5 types)
graphics_events = tissue_events + contrast_events + saline_events

event_colors = {
    6:  ('black',   'blood'),
    12: ('black',   'blood'),
    7:  ('red',     'clot'),
    11: ('red',     'clot'),
    23: ('blue',    'wall'),
    8:  ('magenta', 'contrast'),
    15: ('cyan',    'saline'),
}

def crop_to_blood_range(df, event_col='event_type_1', time_col=None):
    blood_mask = df[event_col].isin(blood_events)
    if not blood_mask.any():
        print("  Warning: No blood events — using full file")
        return df, (df[time_col].min(), df[time_col].max()) if time_col else (None, None)

    first_blood_idx = blood_mask.idxmax()
    last_blood_idx = blood_mask[::-1].idxmax()
    cropped = df.loc[first_blood_idx:last_blood_idx].copy()
    first_time = df.loc[first_blood_idx, time_col] if time_col else None
    last_time = df.loc[last_blood_idx, time_col] if time_col else None
    return cropped, (first_time, last_time)


def blank_short_tissue_events(df, resistance_col, label_col, blood_median, min_dur_sec):
    """Blank clot/wall events shorter than min_dur_sec to blood_median+noise.
    
    Modifies df in-place. Returns count of blanked events and samples.
    """
    if not min_dur_sec or min_dur_sec <= 0:
        return 0, 0

    min_samples = int(min_dur_sec * SAMPLE_RATE)
    labels = df[label_col].values
    n_events_blanked = 0
    n_samples_blanked = 0

    for cls in [1, 2]:  # clot, wall
        cls_mask = (labels == cls).astype(int)
        diff = np.diff(np.concatenate([[0], cls_mask, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) < min_samples:
                idx = df.index[s:e]
                noise = NOISE_VALUE * np.random.randn(e - s)
                df.loc[idx, resistance_col] = blood_median + noise
                df.loc[idx, label_col] = 0
                n_events_blanked += 1
                n_samples_blanked += (e - s)

    return n_events_blanked, n_samples_blanked


for file_path in parquet_files:
    study_name = os.path.basename(file_path).split('_merged_rec_and_event_')[1].split('.parquet')[0]
    print(f"\nProcessing: {study_name}")

    df1 = pd.read_parquet(file_path)

    event_col = 'event_type_1'
    time_col = df1.columns[0]
    imp_col_name = df1.columns[2]
    baseline_col_name = df1.columns[3]

    # Baseline subtraction + 800 (consistent with LabelingWithSubtraction)
    df1['magRLoadAdjusted'] = df1[imp_col_name] - df1[baseline_col_name] + 800
    resistance_col = 'magRLoadAdjusted'
    df1[resistance_col] = df1[resistance_col].astype('float64')

    df_cropped, (first_time, last_time) = crop_to_blood_range(df1, event_col, time_col)

    # da_label helper
    def map_da_label(led_state):
        if pd.isna(led_state) or led_state in [0, "", None]:
            return 0
        try:
            led = int(float(led_state))
        except (ValueError, TypeError):
            return 0
        if led == 2: return 0
        elif led in (4, 5): return 1
        elif led == 7: return 2
        else: return 0

    # Numeric label helper
    def assign_numeric_label(event):
        if event in blood_events: return 0
        if event in clot_events: return 1
        if event in wall_events: return 2
        return 0   # unrecognized events (13,25,etc) → blood, keep real R (baseline behavior)

    # ===================================================================
    # 1. TRAINING: Blank ALL non-tissue events + outliers + short events
    # ===================================================================
    df_training = df_cropped.copy()

    if 'curr_led_state' in df_training.columns:
        df_training['da_label'] = df_training['curr_led_state'].apply(map_da_label)
    else:
        df_training['da_label'] = 0

    df_training['label'] = df_training[event_col].apply(assign_numeric_label)

    tissue_mask   = df_training[event_col].isin(tissue_events)
    outlier_mask  = df_training[resistance_col] > 5000
    blank_mask    = (~tissue_mask) | outlier_mask

    n_non_tissue = (~tissue_mask).sum()
    print(f"  Non-tissue events blanked: {n_non_tissue} ({n_non_tissue/len(df_training)*100:.1f}%)")

    if blank_mask.any():
        blood_median = df_training.loc[df_training[event_col].isin(blood_events), resistance_col].median()
        if pd.isna(blood_median):
            blood_median = df_training[resistance_col].median()
        noise = NOISE_VALUE * np.random.randn(blank_mask.sum())
        df_training.loc[blank_mask, resistance_col] = blood_median + noise
        df_training.loc[blank_mask, 'label'] = 0

    # Blank short-duration tissue events (< MIN_TISSUE_DURATION_SEC)
    if MIN_TISSUE_DURATION_SEC:
        blood_median_for_short = df_training.loc[df_training['label'] == 0, resistance_col].median()
        if pd.isna(blood_median_for_short):
            blood_median_for_short = df_training[resistance_col].median()
        n_evt, n_samp = blank_short_tissue_events(
            df_training, resistance_col, 'label', blood_median_for_short, MIN_TISSUE_DURATION_SEC)
        if n_evt > 0:
            print(f"  Short events blanked (training): {n_evt} events, {n_samp} samples "
                  f"(< {MIN_TISSUE_DURATION_SEC}s)")

    df_training_out = pd.DataFrame({
        'timeInMS': (df_training[time_col] * 1000).astype(int),
        'magRLoadAdjusted': df_training[resistance_col],
        'label': df_training['label'],
        'da_label': df_training['da_label']
    }).reset_index(drop=True)

    training_out_path = os.path.join(training_folder, f'{study_name}_labeled_segment.parquet')
    df_training_out.to_parquet(training_out_path, index=True)
    print(f"  Saved training (artifacts + short events blanked)")

    # ===================================================================
    # 2. GRAPHICS: Show all 5 event types with distinct colors
    # ===================================================================
    if first_time is not None:
        df_graphics = df_cropped.copy()
        if 'curr_led_state' in df_graphics.columns:
            df_graphics['da_label'] = df_graphics['curr_led_state'].apply(map_da_label)
        else:
            df_graphics['da_label'] = 0

        # For graphics: highlight all 5 event types; don't blank artifacts
        gfx_highlighted_mask = df_graphics[event_col].isin(graphics_events)
        default_blank_mask = (df_graphics['da_label'] == 0) & (~gfx_highlighted_mask)
        # Outlier mask excludes artifact events so contrast spikes are shown at full height
        outlier_mask = (df_graphics[resistance_col] > 5000) & (~df_graphics[event_col].isin(artifact_events))
        blank_mask = default_blank_mask | outlier_mask

        if blank_mask.any():
            blood_median = df_graphics.loc[df_graphics[event_col].isin(blood_events), resistance_col].median()
            if pd.isna(blood_median):
                blood_median = df_graphics[resistance_col].median()
            noise = NOISE_VALUE * np.random.randn(blank_mask.sum())
            df_graphics.loc[blank_mask, resistance_col] = blood_median + noise

        # Plot: real background + colored events (5 types)
        plt.figure(figsize=(14, 7))
        plt.plot(df_graphics[time_col], df_graphics[resistance_col], 'k-', lw=0.8, alpha=0.7, zorder=1)

        seen_labels = set()
        for ev_type, (color, label) in event_colors.items():
            mask = df_graphics[event_col] == ev_type
            if mask.any():
                plt.scatter(df_graphics.loc[mask, time_col],
                            df_graphics.loc[mask, resistance_col],
                            color=color, s=6,
                            label=label if label not in seen_labels else None,
                            zorder=5)
                seen_labels.add(label)

        plt.title(f"{study_name} — First to Last Blood (5-class view)\n"
                  f"magRLoadAdjusted = raw - baseline + 800 | Artifacts shown, not blanked")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Resistance (Ω)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        fig_path = os.path.join(graphics_folder, f'{study_name}_first_last_blood.png')
        plt.savefig(fig_path, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"  Saved graphics (5-class view)")

        # ── Training-view graphic: colored by label after blanking ──
        # Shows exactly what the model sees during training (including duration filter)
        train_label_colors = {0: ('black', 'blood'), 1: ('red', 'clot'), 2: ('blue', 'wall')}

        plt.figure(figsize=(14, 7))
        time_ms = df_training_out['timeInMS'].values / 1000.0  # back to seconds for plotting
        r_vals  = df_training_out['magRLoadAdjusted'].values
        labels  = df_training_out['label'].values

        plt.plot(time_ms, r_vals, 'k-', lw=0.8, alpha=0.3, zorder=1)

        seen_labels = set()
        for lbl, (color, name) in train_label_colors.items():
            lmask = labels == lbl
            if lmask.any():
                plt.scatter(time_ms[lmask], r_vals[lmask],
                            color=color, s=6,
                            label=name if name not in seen_labels else None,
                            zorder=5)
                seen_labels.add(name)

        plt.title(f"{study_name} — Training View (as seen by model)\n"
                  f"After blanking: artifacts, outliers, short events (<{MIN_TISSUE_DURATION_SEC}s) → blood")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Resistance (Ω)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        fig_path_train = os.path.join(graphics_folder, f'{study_name}_training_view.png')
        plt.savefig(fig_path_train, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"  Saved graphics (training view)")

    # ===================================================================
    # 3. TESTING: Keep ALL data with real R values, clip R>5000 to 5000
    #    Short events blanked same as training for consistent evaluation.
    # ===================================================================
    df_testing = df_cropped.copy()
    if 'curr_led_state' in df_testing.columns:
        df_testing['da_label'] = df_testing['curr_led_state'].apply(map_da_label)
    else:
        df_testing['da_label'] = 0

    print("  da_label distribution:", df_testing['da_label'].value_counts().sort_index().to_dict())

    # Clip R>5000 to 5000 (not blanked, just capped)
    n_outlier = (df_testing[resistance_col] > 5000).sum()
    df_testing[resistance_col] = df_testing[resistance_col].clip(upper=5000)

    tissue_mask = df_testing[event_col].isin(tissue_events)
    n_non_tissue = (~tissue_mask).sum()
    print(f"  Test: {n_non_tissue} non-tissue events KEPT ({n_non_tissue/len(df_testing)*100:.1f}%), "
          f"{n_outlier} outliers clipped to 5000")

    # Assign labels before duration filter
    df_testing['label'] = df_testing[event_col].apply(assign_numeric_label)

    # NO duration filter on test data — full duration for honest evaluation

    df_testing_out = pd.DataFrame({
        'timeInMS': (df_testing[time_col] * 1000).astype(int),
        'magRLoadAdjusted': df_testing[resistance_col],
        'label': df_testing['label'],
        'da_label': df_testing['da_label']
    }).reset_index(drop=True)

    testing_out_path = os.path.join(testing_folder, f'{study_name}_labeled_segment.parquet')
    df_testing_out.to_parquet(testing_out_path, index=True)
    print(f"  Saved testing (R>5000 clipped, full duration — no short-event filter)")

print(f"\nAll files processed!")
print(f"  Training: non-tissue blanked, short events (<{MIN_TISSUE_DURATION_SEC}s) blanked.")
print(f"  Testing: all data kept, R>5000 clipped, NO duration filter (full duration).")
