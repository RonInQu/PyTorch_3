# -*- coding: utf-8 -*-
"""
Corrected LabelingMax.py — Background stays REAL (no over-flattening)
Matches desired Ground Truth / DA Labels behavior
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# === USER CONFIGURATION ===
# input_folder = '.'  
input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\merged_expts_with_events_April6\event_files'
NOISE_VALUE = 5

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

# Event definitions
blood_events = [6, 12]
clot_events = [7, 11]
wall_events = [23]
highlighted_events = blood_events + clot_events + wall_events

event_colors = {
    6: ('black', 'blood'),
    12: ('black', 'blood'),
    7: ('red', 'clot'),
    11: ('red', 'clot'),
    23: ('blue', 'wall')
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
        return -1  # unlabeled / not a recognized event

    # ===================================================================
    # 1. TRAINING: Keep real background. Blank ONLY da_label==0 outside highlights + outliers
    # ===================================================================
    df_training = df_cropped.copy()

    if 'curr_led_state' in df_training.columns:
        df_training['da_label'] = df_training['curr_led_state'].apply(map_da_label)
    else:
        df_training['da_label'] = 0

    df_training['label'] = df_training[event_col].apply(assign_numeric_label)

    highlighted_mask = df_training[event_col].isin(highlighted_events)
    default_blank_mask = (df_training['da_label'] == 0) & (~highlighted_mask)
    outlier_mask = df_training[resistance_col] > 5000
    blank_mask = default_blank_mask | outlier_mask

    if blank_mask.any():
        blood_median = df_training.loc[df_training[event_col].isin(blood_events), resistance_col].median()
        if pd.isna(blood_median):
            blood_median = df_training[resistance_col].median()
        noise = NOISE_VALUE * np.random.randn(blank_mask.sum())
        df_training.loc[blank_mask, resistance_col] = blood_median + noise
        df_training.loc[blank_mask, 'label'] = 0

    df_training_out = pd.DataFrame({
        'timeInMS': (df_training[time_col] * 1000).astype(int),
        'magRLoadAdjusted': df_training[resistance_col],
        'label': df_training['label'],
        'da_label': df_training['da_label']
    }).reset_index(drop=True)

    training_out_path = os.path.join(training_folder, f'{study_name}_labeled_segment.parquet')
    df_training_out.to_parquet(training_out_path, index=True)
    print(f"  Saved training (real background, blanked defaults only)")

    # ===================================================================
    # 2. GRAPHICS: Show real background + colored highlights (no gray flats)
    # ===================================================================
    if first_time is not None:
        df_graphics = df_cropped.copy()
        if 'curr_led_state' in df_graphics.columns:
            df_graphics['da_label'] = df_graphics['curr_led_state'].apply(map_da_label)
        else:
            df_graphics['da_label'] = 0

        highlighted_mask = df_graphics[event_col].isin(highlighted_events)
        default_blank_mask = (df_graphics['da_label'] == 0) & (~highlighted_mask)
        outlier_mask = df_graphics[resistance_col] > 5000
        blank_mask = default_blank_mask | outlier_mask

        if blank_mask.any():
            blood_median = df_graphics.loc[df_graphics[event_col].isin(blood_events), resistance_col].median()
            if pd.isna(blood_median):
                blood_median = df_graphics[resistance_col].median()
            noise = NOISE_VALUE * np.random.randn(blank_mask.sum())
            df_graphics.loc[blank_mask, resistance_col] = blood_median + noise

        # Plot: real background (black/gray) + colored events
        plt.figure(figsize=(14, 7))
        plt.plot(df_graphics[time_col], df_graphics[resistance_col], 'k-', lw=0.8, alpha=0.7, zorder=1)  # real signal

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

        blood_median_for_title = df_graphics.loc[df_graphics[event_col].isin(blood_events), resistance_col].median()
        plt.title(f"{study_name} — First to Last Blood (real background)\n"
                  f"magRLoadAdjusted = raw - baseline + 800 | Blanked only defaults/outliers")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Resistance (Ω)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        fig_path = os.path.join(graphics_folder, f'{study_name}_first_last_blood.png')
        plt.savefig(fig_path, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"  Saved graphics (real background)")

    # ===================================================================
    # 3. TESTING: Keep as before (mostly raw)
    # ===================================================================
    df_testing = df_cropped.copy()
    if 'curr_led_state' in df_testing.columns:
        df_testing['da_label'] = df_testing['curr_led_state'].apply(map_da_label)
    else:
        df_testing['da_label'] = 0

    print("  da_label distribution:", df_testing['da_label'].value_counts().sort_index().to_dict())

    df_testing['label'] = df_testing[event_col].apply(assign_numeric_label)

    highlighted_mask = df_testing[event_col].isin(highlighted_events)
    default_non_highlight_mask = (df_testing['da_label'] == 0) & (~highlighted_mask)
    outlier_mask = df_testing[resistance_col] > 5000
    blank_mask = default_non_highlight_mask | outlier_mask

    if blank_mask.any():
        blood_median = df_testing.loc[df_testing[event_col].isin(blood_events), resistance_col].median() or df_testing[resistance_col].median()
        noise = NOISE_VALUE * np.random.randn(blank_mask.sum())
        df_testing.loc[blank_mask, resistance_col] = blood_median + noise
        df_testing.loc[blank_mask, 'label'] = 0

    df_testing_out = pd.DataFrame({
        'timeInMS': (df_testing[time_col] * 1000).astype(int),
        'magRLoadAdjusted': df_testing[resistance_col],
        'label': df_testing['label'],
        'da_label': df_testing['da_label']
    }).reset_index(drop=True)

    testing_out_path = os.path.join(testing_folder, f'{study_name}_labeled_segment.parquet')
    df_testing_out.to_parquet(testing_out_path, index=True)
    print(f"  Saved testing")

print("\nAll files processed! Check the new graphics — background should now stay real (no gray flats).")