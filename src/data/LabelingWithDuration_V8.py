# -*- coding: utf-8 -*-
"""
LabelingWithDuration_V8.py

Experimental variant of LabelingWithDuration.py.

This version resolves a per-row ground-truth event from event_type_1,
event_type_2, and event_type_3 using curr_led_state as the reference.
The selected event becomes the effective truth used for cropping, labeling,
training blanking, graphics, and test output generation.

Selection rule per row:
1. Map curr_led_state to da_label in {0,1,2}.
2. Convert each available event_type_N value to a numeric class:
   blood/non-tissue=0, clot=1, wall=2.
3. Prefer the first event column whose mapped class exactly matches da_label.
4. If none match exactly, use event_type_1.

This keeps the original duration-filtered train/test pipeline behavior while
changing only how the ground truth event is chosen.
"""

import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# === USER CONFIGURATION ===
# input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\31August2026'
# input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\PostMay_strict'
# input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\31August2026\InputFolder'
input_folder = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\19August2026\InputFolder'
input_folder = os.environ.get('LABELING_V8_INPUT_FOLDER', input_folder)

NOISE_VALUE = 5
SAMPLE_RATE = 150
MIN_TISSUE_DURATION_SEC = 7.0

output_base = os.environ.get(
    'LABELING_V8_OUTPUT_BASE',
    os.path.join(input_folder, 'processedResults_V8_truthmatch')
)
training_folder = os.path.join(output_base, 'training')
graphics_folder = os.path.join(output_base, 'graphics')
testing_folder = os.path.join(output_base, 'testing')

for folder in [training_folder, graphics_folder, testing_folder]:
    os.makedirs(folder, exist_ok=True)

pattern = os.path.join(input_folder, '*.parquet')
parquet_files = glob.glob(pattern)

if not parquet_files:
    raise ValueError(f"No parquet files found in {input_folder}")

print(f"Found {len(parquet_files)} file(s) to process.")
print(f"MIN_TISSUE_DURATION_SEC = {MIN_TISSUE_DURATION_SEC}")
print(f"Output folder = {output_base}")


# Event definitions
blood_events = [6, 12]
clot_events = [7, 11]
wall_events = [23]
contrast_events = [8]
saline_events = [15]

artifact_events = contrast_events + saline_events
tissue_events = blood_events + clot_events + wall_events
graphics_events = tissue_events + contrast_events + saline_events

event_colors = {
    6: ('black', 'blood'),
    12: ('black', 'blood'),
    7: ('red', 'clot'),
    11: ('red', 'clot'),
    23: ('blue', 'wall'),
    8: ('magenta', 'contrast'),
    15: ('cyan', 'saline'),
}


def crop_to_blood_range(df, event_col='resolved_event', time_col=None):
    blood_mask = df[event_col].isin(blood_events)
    if not blood_mask.any():
        print("  Warning: No blood events in resolved truth; using full file")
        return df, (df[time_col].min(), df[time_col].max()) if time_col else (None, None)

    first_blood_idx = blood_mask.idxmax()
    last_blood_idx = blood_mask[::-1].idxmax()
    cropped = df.loc[first_blood_idx:last_blood_idx].copy()
    first_time = df.loc[first_blood_idx, time_col] if time_col else None
    last_time = df.loc[last_blood_idx, time_col] if time_col else None
    return cropped, (first_time, last_time)


def blank_short_tissue_events(df, resistance_col, label_col, blood_median, min_dur_sec):
    if not min_dur_sec or min_dur_sec <= 0:
        return 0, 0

    min_samples = int(min_dur_sec * SAMPLE_RATE)
    labels = df[label_col].values
    n_events_blanked = 0
    n_samples_blanked = 0

    for cls in [1, 2]:
        cls_mask = (labels == cls).astype(int)
        diff = np.diff(np.concatenate([[0], cls_mask, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start, end in zip(starts, ends):
            if (end - start) < min_samples:
                idx = df.index[start:end]
                noise = NOISE_VALUE * np.random.randn(end - start)
                df.loc[idx, resistance_col] = blood_median + noise
                df.loc[idx, label_col] = 0
                n_events_blanked += 1
                n_samples_blanked += (end - start)

    return n_events_blanked, n_samples_blanked


def map_da_label(led_state):
    if pd.isna(led_state) or led_state in [0, "", None]:
        return 0
    try:
        led = int(float(led_state))
    except (ValueError, TypeError):
        return 0

    if led == 2:
        return 0
    if led in (4, 5):
        return 1
    if led == 7:
        return 2
    return 0


def assign_numeric_label(event):
    if event in blood_events:
        return 0
    if event in clot_events:
        return 1
    if event in wall_events:
        return 2
    return 0


def map_da_label_series(led_series):
    led_numeric = pd.to_numeric(led_series, errors='coerce')
    da_label = pd.Series(0, index=led_series.index, dtype=np.int64)
    da_label = da_label.mask(led_numeric.isin([4, 5]), 1)
    da_label = da_label.mask(led_numeric == 7, 2)
    return da_label


def assign_numeric_label_series(event_series):
    label = pd.Series(0, index=event_series.index, dtype=np.int64)
    label = label.mask(event_series.isin(clot_events), 1)
    label = label.mask(event_series.isin(wall_events), 2)
    return label


def resolve_truth_columns(df, event_cols):
    da_label = map_da_label_series(df.get('curr_led_state', pd.Series(0, index=df.index)))

    base_event_col = event_cols[0]
    resolved_event = df[base_event_col].copy()
    resolved_label = assign_numeric_label_series(resolved_event)
    resolved_source = pd.Series(base_event_col, index=df.index, dtype='object')

    for event_col in event_cols:
        candidate_label = assign_numeric_label_series(df[event_col])
        exact_match = candidate_label == da_label
        unresolved = resolved_source == base_event_col
        should_replace = exact_match & unresolved & (candidate_label != resolved_label)

        resolved_event = resolved_event.where(~should_replace, df[event_col])
        resolved_label = resolved_label.where(~should_replace, candidate_label)
        resolved_source = resolved_source.where(~should_replace, event_col)

    first_label = assign_numeric_label_series(df[base_event_col])
    first_exact_match = first_label == da_label
    resolved_source = resolved_source.where(~first_exact_match, base_event_col)
    resolved_event = resolved_event.where(~first_exact_match, df[base_event_col])
    resolved_label = resolved_label.where(~first_exact_match, first_label)

    return pd.DataFrame(
        {
            'resolved_event_source': resolved_source,
            'resolved_event': resolved_event,
            'resolved_label': resolved_label,
            'da_label_resolved': da_label,
        }
    )


for file_path in parquet_files:
    basename = os.path.basename(file_path)
    if '_merged_rec_and_event_' in basename:
        study_name = basename.split('_merged_rec_and_event_')[1].split('.parquet')[0]
    else:
        study_name = os.path.splitext(basename)[0]
    print(f"\nProcessing: {study_name}")

    df1 = pd.read_parquet(file_path)

    time_col = df1.columns[0]

    event_cols = [col for col in ['event_type_1', 'event_type_2', 'event_type_3'] if col in df1.columns]
    if not event_cols:
        raise ValueError(f"No event_type_1/2/3 columns found in {study_name}")

    imp_col_name = 'imp' if 'imp' in df1.columns else df1.columns[2]
    baseline_col_name = 'blood_baseline' if 'blood_baseline' in df1.columns else df1.columns[3]

    df1['magRLoadAdjusted'] = df1[imp_col_name] - df1[baseline_col_name] + 800
    resistance_col = 'magRLoadAdjusted'
    df1[resistance_col] = df1[resistance_col].astype('float64')

    resolved_df = resolve_truth_columns(df1, event_cols)
    df1 = pd.concat([df1, resolved_df], axis=1)

    print("  Resolved event source counts:", df1['resolved_event_source'].value_counts().to_dict())
    print("  Resolved label counts:", df1['resolved_label'].value_counts().sort_index().to_dict())

    df_cropped, (first_time, last_time) = crop_to_blood_range(df1, 'resolved_event', time_col)

    # ================================================================
    # 1. TRAINING
    # ================================================================
    df_training = df_cropped.copy()
    df_training['da_label'] = df_training['da_label_resolved']
    df_training['label'] = df_training['resolved_label']

    tissue_mask = df_training['resolved_event'].isin(tissue_events)
    outlier_mask = df_training[resistance_col] > 5000
    blank_mask = (~tissue_mask) | outlier_mask

    n_non_tissue = (~tissue_mask).sum()
    print(f"  Non-tissue events blanked: {n_non_tissue} ({n_non_tissue/len(df_training)*100:.1f}%)")

    if blank_mask.any():
        blood_median = df_training.loc[df_training['resolved_event'].isin(blood_events), resistance_col].median()
        if pd.isna(blood_median):
            blood_median = df_training[resistance_col].median()
        noise = NOISE_VALUE * np.random.randn(blank_mask.sum())
        df_training.loc[blank_mask, resistance_col] = blood_median + noise
        df_training.loc[blank_mask, 'label'] = 0

    if MIN_TISSUE_DURATION_SEC:
        blood_median_for_short = df_training.loc[df_training['label'] == 0, resistance_col].median()
        if pd.isna(blood_median_for_short):
            blood_median_for_short = df_training[resistance_col].median()
        n_evt, n_samp = blank_short_tissue_events(
            df_training, resistance_col, 'label', blood_median_for_short, MIN_TISSUE_DURATION_SEC)
        if n_evt > 0:
            print(
                f"  Short events blanked (training): {n_evt} events, {n_samp} samples "
                f"(< {MIN_TISSUE_DURATION_SEC}s)"
            )

    df_training_out = pd.DataFrame(
        {
            'timeInMS': (df_training[time_col] * 1000).astype(int),
            'magRLoadAdjusted': df_training[resistance_col],
            'label': df_training['label'],
            'da_label': df_training['da_label'],
        }
    ).reset_index(drop=True)

    training_out_path = os.path.join(training_folder, f'{study_name}_labeled_segment.parquet')
    df_training_out.to_parquet(training_out_path, index=True)
    print("  Saved training (resolved truth + duration filter)")

    # ================================================================
    # 2. GRAPHICS
    # ================================================================
    if first_time is not None:
        df_graphics = df_cropped.copy()
        df_graphics['da_label'] = df_graphics['da_label_resolved']

        gfx_highlighted_mask = df_graphics['resolved_event'].isin(graphics_events)
        default_blank_mask = (df_graphics['da_label'] == 0) & (~gfx_highlighted_mask)
        outlier_mask = (df_graphics[resistance_col] > 5000) & (~df_graphics['resolved_event'].isin(artifact_events))
        blank_mask = default_blank_mask | outlier_mask

        if blank_mask.any():
            blood_median = df_graphics.loc[df_graphics['resolved_event'].isin(blood_events), resistance_col].median()
            if pd.isna(blood_median):
                blood_median = df_graphics[resistance_col].median()
            noise = NOISE_VALUE * np.random.randn(blank_mask.sum())
            df_graphics.loc[blank_mask, resistance_col] = blood_median + noise

        plt.figure(figsize=(14, 7))
        plt.plot(df_graphics[time_col], df_graphics[resistance_col], 'k-', lw=0.8, alpha=0.7, zorder=1)

        seen_labels = set()
        for ev_type, (color, label_name) in event_colors.items():
            mask = df_graphics['resolved_event'] == ev_type
            if mask.any():
                plt.scatter(
                    df_graphics.loc[mask, time_col],
                    df_graphics.loc[mask, resistance_col],
                    color=color,
                    s=6,
                    label=label_name if label_name not in seen_labels else None,
                    zorder=5,
                )
                seen_labels.add(label_name)

        plt.title(
            f"{study_name} - First to Last Blood (resolved truth view)\n"
            "magRLoadAdjusted = raw - baseline + 800 | Event source selected from event_type_1/2/3"
        )
        plt.xlabel('Time (seconds)')
        plt.ylabel('Resistance (Ohms)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        fig_path = os.path.join(graphics_folder, f'{study_name}_first_last_blood.png')
        plt.savefig(fig_path, dpi=250, bbox_inches='tight')
        plt.close()
        print("  Saved graphics (resolved truth view)")

        train_label_colors = {0: ('black', 'blood'), 1: ('red', 'clot'), 2: ('blue', 'wall')}

        plt.figure(figsize=(14, 7))
        time_sec = df_training_out['timeInMS'].values / 1000.0
        r_vals = df_training_out['magRLoadAdjusted'].values
        labels = df_training_out['label'].values

        plt.plot(time_sec, r_vals, 'k-', lw=0.8, alpha=0.3, zorder=1)

        seen_labels = set()
        for lbl, (color, name) in train_label_colors.items():
            label_mask = labels == lbl
            if label_mask.any():
                plt.scatter(
                    time_sec[label_mask],
                    r_vals[label_mask],
                    color=color,
                    s=6,
                    label=name if name not in seen_labels else None,
                    zorder=5,
                )
                seen_labels.add(name)

        plt.title(
            f"{study_name} - Training View (resolved truth)\n"
            f"After blanking: artifacts, outliers, short events (<{MIN_TISSUE_DURATION_SEC}s) -> blood"
        )
        plt.xlabel('Time (seconds)')
        plt.ylabel('Resistance (Ohms)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        fig_path_train = os.path.join(graphics_folder, f'{study_name}_training_view.png')
        plt.savefig(fig_path_train, dpi=250, bbox_inches='tight')
        plt.close()
        print("  Saved graphics (training view)")

    # ================================================================
    # 3. TESTING
    # ================================================================
    df_testing = df_cropped.copy()
    df_testing['da_label'] = df_testing['da_label_resolved']

    print("  da_label distribution:", df_testing['da_label'].value_counts().sort_index().to_dict())

    n_outlier = (df_testing[resistance_col] > 5000).sum()
    df_testing[resistance_col] = df_testing[resistance_col].clip(upper=5000)

    tissue_mask = df_testing['resolved_event'].isin(tissue_events)
    n_non_tissue = (~tissue_mask).sum()
    print(
        f"  Test: {n_non_tissue} non-tissue events KEPT ({n_non_tissue/len(df_testing)*100:.1f}%), "
        f"{n_outlier} outliers clipped to 5000"
    )

    df_testing['label'] = df_testing['resolved_label']

    df_testing_out = pd.DataFrame(
        {
            'timeInMS': (df_testing[time_col] * 1000).astype(int),
            'magRLoadAdjusted': df_testing[resistance_col],
            'label': df_testing['label'],
            'da_label': df_testing['da_label'],
        }
    ).reset_index(drop=True)

    testing_out_path = os.path.join(testing_folder, f'{study_name}_labeled_segment.parquet')
    df_testing_out.to_parquet(testing_out_path, index=True)
    print("  Saved testing (resolved truth, full duration)")

print("\nAll files processed!")
print(f"  Training: resolved truth used; non-tissue blanked; short events (<{MIN_TISSUE_DURATION_SEC}s) blanked.")
print("  Testing: resolved truth used; all data kept; R>5000 clipped; no short-event filter.")