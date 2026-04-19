# -*- coding: utf-8 -*-
"""
Batch processing script:
- Processes all '_merged_rec_and_event_*.parquet' files in a specified folder
- Skips Figure 1 and Figure 2 (no plotting or showing)
- Generates only Figure 3 (flattened plot with NOISY baseline), saves it (no plt.show())
- Saves the labeled parquet in the required format
- All outputs saved to a 'processedResults' subfolder
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np  # Added for random noise

# === USER CONFIGURATION ===
# Change this to your folder containing the parquet files
input_folder = '.'  # Current directory; change to e.g. r'C:\Data\Studies' if needed
NOISE_VALUE = 5

# Output subfolder
output_folder = os.path.join(input_folder, 'processedResults')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Find all relevant parquet files
pattern = os.path.join(input_folder, '_merged_rec_and_event_*.parquet')
parquet_files = glob.glob(pattern)

if not parquet_files:
    raise ValueError(f"No parquet files found matching pattern in {input_folder}")

print(f"Found {len(parquet_files)} file(s) to process:")
for f in parquet_files:
    print(f"  - {os.path.basename(f)}")

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

# Process each file
for file_path in parquet_files:
    study_name = os.path.basename(file_path).split('_merged_rec_and_event_')[1].split('.parquet')[0]
    print(f"\nProcessing: {study_name}")

    df1 = pd.read_parquet(file_path)

    event_col = 'event_type_1'
    time_col = df1.columns[0]
    
    imp_col_name = df1.columns[2]
    baseline_col_name = df1.columns[3]
    df1['magRLoadAdjusted'] = df1[imp_col_name] - df1[baseline_col_name] + 800 # Bring to 800 as in v93
    resistance_col = 'magRLoadAdjusted'

    # Find first and last blood (event 6)
    blood_rows = df1[df1[event_col] == 6]
    if len(blood_rows) == 0:
        print(f"  Warning: No event_type_1 == 6 found in {study_name}. Skipping.")
        continue

    start_idx = blood_rows.index[0]
    end_idx = blood_rows.index[-1]
    df_plot = df1.loc[start_idx:end_idx].copy()
    
    # Compute median blood resistance (from events 6 and 12)
    blood_mask = df_plot[event_col].isin(blood_events)
    
    blood_median_resistance = df_plot.loc[blood_mask, resistance_col].median()

    # === Figure 3: Flattened version with NOISY baseline ===
    df_plot_mod = df_plot.copy()
    # CRITICAL FIX: Convert resistance column to float to accept noisy values
    df_plot_mod[resistance_col] = df_plot_mod[resistance_col].astype('float64')
    
    
    non_highlighted_mask = ~df_plot_mod[event_col].isin(highlighted_events)

    # Apply noisy baseline: median + Gaussian noise (std = 5 Ω)
    if non_highlighted_mask.any():
        num_noise_points = non_highlighted_mask.sum()
        noise = NOISE_VALUE * np.random.randn(num_noise_points)
        noisy_baseline = blood_median_resistance + noise
        df_plot_mod.loc[non_highlighted_mask, resistance_col] = noisy_baseline
        
        

    plt.figure(figsize=(14, 6))

    # Background (noisy flattened non-highlighted)
    if non_highlighted_mask.any():
        plt.scatter(df_plot_mod.loc[non_highlighted_mask, time_col],
                    df_plot_mod.loc[non_highlighted_mask, resistance_col],
                    color='lightgray', s=2, alpha=0.7)

    # Highlighted events (on top)
    handles_labels = []
    plotted_labels = set()

    for event_type, (color, label_name) in event_colors.items():
        mask = df_plot_mod[event_col] == event_type
        if mask.any():
            scat = plt.scatter(df_plot_mod.loc[mask, time_col],
                               df_plot_mod.loc[mask, resistance_col],
                               color=color, s=2,
                               label=label_name if label_name not in plotted_labels else None,
                               edgecolors='none', zorder=5)
            if label_name not in plotted_labels:
                handles_labels.append((scat, label_name, color))
                plotted_labels.add(label_name)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Resistance (Ω)')
    plt.title(f'Resistance vs Time for {study_name}: First To Last Blood\n'
              f'(Non-highlighted points set to noisy median blood resistance ≈ {blood_median_resistance:.1f} Ω ±5)')
    plt.grid(True, alpha=0.3)

    # Colored legend
    if handles_labels:
        legend_handles = [h[0] for h in handles_labels]
        legend_labels = [h[1] for h in handles_labels]
        legend_colors = [h[2] for h in handles_labels]
        leg = plt.legend(legend_handles, legend_labels, fontsize=12)
        for text, color in zip(leg.get_texts(), legend_colors):
            text.set_color(color)

    # Save figure
    fig_path = os.path.join(output_folder, f'{study_name}_first_to_last_blood_noisy_baseline.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # === Create and save labeled parquet (uses the NOISY values) ===
    df_labeled = df_plot_mod.copy()

    def assign_numeric_label(event):
        if event in blood_events:
            return 0
        elif event in clot_events:
            return 1
        elif event in wall_events:
            return 2
        else:
            return 0  # background labeled as blood

    df_labeled['label'] = df_labeled[event_col].apply(assign_numeric_label)

   # ────────────────────────────────────────────────
    # NEW: Map curr_led_state → da_label (0/1/2)
    # ────────────────────────────────────────────────
    def map_da_label(led_state):
        if pd.isna(led_state) or led_state in [0, "", None]:
            return 0
        
        try:
            led = int(float(led_state))
        except (ValueError, TypeError):
            return 0
        
        if led == 2:
            return 0      # blood
        elif led in (4, 5):
            return 1      # clot
        elif led == 7:
            return 2      # wall
        else:
            return 0      # everything else = blood (baseline)

    if 'curr_led_state' in df_labeled.columns:
        df_labeled['da_label'] = df_labeled['curr_led_state'].apply(map_da_label)
        
        # Optional: quick sanity check
        da_counts = df_labeled['da_label'].value_counts().sort_index()
        print(f"  DA label distribution:\n{da_counts}")
    else:
        print(f"  Warning: 'curr_led_state' column not found in {study_name}")
        df_labeled['da_label'] = 0  # or leave missing
        
    # ────────────────────────────────────────────────
    # CRITICAL FIX: Force da_label = 0 in all non-highlighted (baseline) regions
    # This overrides any previous mapping — baseline MUST be blood
    # ────────────────────────────────────────────────
    if non_highlighted_mask.any():
        df_labeled.loc[non_highlighted_mask, 'da_label'] = 0
        
        # Optional: confirm the fix
        fixed_counts = df_labeled.loc[non_highlighted_mask, 'da_label'].value_counts()
        print(f"  Non-highlighted DA labels after forced fix: {fixed_counts.to_dict()}")    

    # ────────────────────────────────────────────────
    # Final output DataFrame – now includes da_label
    # ────────────────────────────────────────────────
    df_output = pd.DataFrame({
        'timeInMS': (df_labeled[time_col] * 1000).astype(int),
        'magRLoadAdjusted': df_labeled[resistance_col],
        'label': df_labeled['label'],
        'da_label': df_labeled['da_label']          # ← new column
    })

    df_output = df_output.reset_index(drop=True)

    parquet_out_path = os.path.join(output_folder, f'{study_name}_labeled_segment.parquet')
    df_output.to_parquet(parquet_out_path, index=True)   # index=True keeps the original row indices if needed

    print(f"  Saved figure: {os.path.basename(fig_path)}")
    print(f"  Saved labeled data (with da_label): {os.path.basename(parquet_out_path)}")

print("\nAll files processed successfully!")
print(f"Outputs saved to: {output_folder}")



plt.plot( df1[time_col],df1['magRLoadAdjusted'])
plt.show()