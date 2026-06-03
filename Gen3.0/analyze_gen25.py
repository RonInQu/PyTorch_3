# -*- coding: utf-8 -*-
"""
Gen2.5 Multi-Frequency + Pressure Analysis
LOG2_solo.parquet

Columns:
  - imp_mag_adj_0_ohm: 50 kHz (same as legacy single-freq)
  - imp_mag_adj_1_ohm: 5 kHz (NEW)
  - imp_mag_adj_2_ohm: 100 kHz (NEW - heavily saturated at 2^20-1)
  - pressure_mmhg: contact pressure (NEW)
  - solo_led_state_i: GT labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ─── Load data ───────────────────────────────────────────────────────────────
parquet_name = sys.argv[1] if len(sys.argv) > 1 else 'LOG6_solo.parquet'
parquet_path = Path(__file__).parent / parquet_name
df = pd.read_parquet(parquet_path)
df['time_sec'] = df['timestamp_ms'] / 1000.0

# Event mapping
# SoloStateVal mapping (state_defs.h) — uses solo_state_i column
state_defs = {
    0:  'start',
    1:  'exit_start',
    2:  'dev_error',
    3:  'imp_err_short',
    4:  'imp_err_open',
    5:  'eval_tip',
    6:  'nvac_air',
    7:  'nvac_blood',
    8:  'nvac_clot',
    9:  'nvac_inject',
    10: 'asp_in_blood',
    11: 'asp_in_clot',
    12: 'asp_show_clot',
    13: 'vac_wall',
    14: 'vac_lollipop',
    15: 'vac_clog',
    16: 'vac_air',
}

# Tissue classification
tissue_map = {
    7: 'blood', 10: 'blood',           # NVAC_BLOOD, ASP_IN_BLOOD
    8: 'clot', 11: 'clot', 12: 'clot', # NVAC_CLOT, ASP_IN_CLOT, ASP_SHOW_CLOT
    13: 'wall', 14: 'wall',            # VAC_WALL, VAC_LOLLIPOP
}

# Colors for plotting
event_colors = {
    7:  ('black',   'blood'),
    10: ('black',   'blood'),
    8:  ('red',     'clot'),
    11: ('red',     'clot'),
    12: ('red',     'clot'),
    13: ('blue',    'wall'),
    14: ('blue',    'wall'),
    6:  ('gray',    'air'),
    9:  ('magenta', 'inject'),
}

df['tissue'] = df['solo_state_i'].map(tissue_map).fillna('other')

# Mark saturated 100kHz readings
SATURATED_100K = 1048575  # 2^20 - 1
df['freq2_saturated'] = df['imp_mag_adj_2_ohm'] >= SATURATED_100K

print("="*60)
print("Gen2.5 Data Summary")
print("="*60)
print(f"Total samples: {len(df):,}")
print(f"Duration: {df.time_sec.max():.1f}s ({df.time_sec.max()/60:.1f} min)")
print(f"Sample rate: ~{len(df)/df.time_sec.max():.0f} Hz")
print(f"\nTissue distribution:")
print(df.tissue.value_counts().to_string())
print(f"\n100kHz saturation: {df.freq2_saturated.sum():,} / {len(df):,} ({100*df.freq2_saturated.mean():.1f}%)")

# ─── PLOT 1: Full time series overview ───────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
fig.suptitle('Gen2.5 LOG2_solo — Full Time Series', fontsize=14)

# GT labels (solo_state_i)
ax = axes[0]
for ev, (color, name) in event_colors.items():
    mask = df.solo_state_i == ev
    if mask.any():
        ax.scatter(df.loc[mask, 'time_sec'], df.loc[mask, 'solo_state_i'],
                   c=color, s=1, label=f'{ev}={name}', alpha=0.5)
# other states not in event_colors
other_states = set(df.solo_state_i.unique()) - set(event_colors.keys())
for ev in sorted(other_states):
    mask = df.solo_state_i == ev
    if mask.any():
        ax.scatter(df.loc[mask, 'time_sec'], df.loc[mask, 'solo_state_i'],
                   c='lightgray', s=1, label=f'{ev}={state_defs.get(ev,"?")}', alpha=0.3)
ax.set_ylabel('State ID')
ax.legend(loc='upper right', markerscale=5, fontsize=7, ncol=2)
ax.set_title('Ground Truth Labels (solo_state_i)')

# 50 kHz impedance
ax = axes[1]
ax.plot(df.time_sec, df.imp_mag_adj_0_ohm, 'k-', lw=0.3, alpha=0.5)
ax.set_ylabel('Impedance (Ω)')
ax.set_title('50 kHz Impedance')
ax.set_ylim(0, np.percentile(df.imp_mag_adj_0_ohm, 99))

# 5 kHz impedance
ax = axes[2]
ax.plot(df.time_sec, df.imp_mag_adj_1_ohm, 'b-', lw=0.3, alpha=0.5)
ax.set_ylabel('Impedance (Ω)')
ax.set_title('5 kHz Impedance')
ax.set_ylim(0, np.percentile(df.imp_mag_adj_1_ohm, 99))

# 100 kHz impedance (mark saturation)
ax = axes[3]
valid_100k = df[~df.freq2_saturated]
ax.plot(valid_100k.time_sec, valid_100k.imp_mag_adj_2_ohm, 'g-', lw=0.3, alpha=0.5)
sat_mask = df.freq2_saturated
ax.scatter(df.loc[sat_mask, 'time_sec'], df.loc[sat_mask, 'imp_mag_adj_2_ohm'],
           c='red', s=0.5, alpha=0.1, label=f'Saturated ({sat_mask.sum():,})')
ax.set_ylabel('Impedance (Ω)')
ax.set_title('100 kHz Impedance (red=saturated at 2^20-1)')
ax.legend(fontsize=8)

# Pressure
ax = axes[4]
ax.plot(df.time_sec, df.pressure_mmhg, 'm-', lw=0.3, alpha=0.5)
ax.set_ylabel('Pressure (mmHg)')
ax.set_title('Contact Pressure')
ax.set_xlabel('Time (sec)')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot1_time_series.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: plot1_time_series.png")

# ─── PLOT 2: Zoom on main clot event (auto-detected) ────────────────────────
# Find the longest contiguous clot segment
clot_mask = df.tissue == 'clot'
clot_segments = (clot_mask != clot_mask.shift()).cumsum()
clot_segments = clot_segments[clot_mask]
if len(clot_segments) > 0:
    longest_clot_id = clot_segments.value_counts().idxmax()
    clot_seg = df.loc[clot_segments[clot_segments == longest_clot_id].index]
    clot_t_start = clot_seg.time_sec.min()
    clot_t_end = clot_seg.time_sec.max()
    # Add 5s padding around the event
    t_start = max(0, clot_t_start - 5)
    t_end = clot_t_end + 5

    zoom = df[(df.time_sec >= t_start) & (df.time_sec <= t_end)]

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Main Clot Event Zoom ({clot_t_start:.0f}-{clot_t_end:.0f}s, {len(clot_seg)} samples)', fontsize=14)

    # Labels
    ax = axes[0]
    ax.plot(zoom.time_sec, zoom.solo_state_i, 'k-', lw=1)
    ax.axvspan(clot_t_start, clot_t_end, alpha=0.15, color='red', label='clot')
    ax.set_ylabel('State ID')
    ax.legend()

    # 50 kHz
    ax = axes[1]
    ax.plot(zoom.time_sec, zoom.imp_mag_adj_0_ohm, 'k-', lw=0.8)
    ax.set_ylabel('50 kHz (Ω)')

    # 5 kHz
    ax = axes[2]
    ax.plot(zoom.time_sec, zoom.imp_mag_adj_1_ohm, 'b-', lw=0.8)
    ax.set_ylabel('5 kHz (Ω)')

    # 100 kHz
    ax = axes[3]
    ax.plot(zoom.time_sec, zoom.imp_mag_adj_2_ohm, 'g-', lw=0.8)
    ax.set_ylabel('100 kHz (Ω)')
    ax.axhline(SATURATED_100K, color='red', ls=':', alpha=0.5, label='Saturation')
    ax.legend()

    # Pressure
    ax = axes[4]
    ax.plot(zoom.time_sec, zoom.pressure_mmhg, 'm-', lw=0.8)
    ax.set_ylabel('Pressure (mmHg)')
    ax.set_xlabel('Time (sec)')

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'plot2_clot_zoom.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot2_clot_zoom.png")
else:
    print("No clot events found — skipping plot2_clot_zoom.png")

# ─── PLOT 2b: Wall region zoom (auto-detected) ──────────────────────────────
# Find a region with multiple wall segments (densest cluster)
wall_mask = df.tissue == 'wall'
wall_segments = (wall_mask != wall_mask.shift()).cumsum()
wall_segments = wall_segments[wall_mask]
if len(wall_segments) > 0:
    # Find the longest wall segment's neighborhood (±30s around center of mass)
    longest_wall_id = wall_segments.value_counts().idxmax()
    wall_seg = df.loc[wall_segments[wall_segments == longest_wall_id].index]
    center = wall_seg.time_sec.mean()
    # Show a wide window to capture multiple wall events if possible
    t_start = max(0, center - 60)
    t_end = min(df.time_sec.max(), center + 60)

    zoom = df[(df.time_sec >= t_start) & (df.time_sec <= t_end)]

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Wall Events Zoom ({t_start:.0f}-{t_end:.0f}s)', fontsize=14)

    # Labels — highlight wall segments
    ax = axes[0]
    ax.plot(zoom.time_sec, zoom.solo_state_i, 'k-', lw=1)
    wall_zoom = zoom[zoom.tissue == 'wall']
    if len(wall_zoom) > 0:
        ax.scatter(wall_zoom.time_sec, wall_zoom.solo_state_i, c='blue', s=3, alpha=0.5, label='wall')
    clot_zoom = zoom[zoom.tissue == 'clot']
    if len(clot_zoom) > 0:
        ax.scatter(clot_zoom.time_sec, clot_zoom.solo_state_i, c='red', s=3, alpha=0.5, label='clot')
    ax.set_ylabel('State ID')
    ax.legend()

    # 50 kHz
    ax = axes[1]
    ax.plot(zoom.time_sec, zoom.imp_mag_adj_0_ohm, 'k-', lw=0.8)
    ax.set_ylabel('50 kHz (Ω)')

    # 5 kHz
    ax = axes[2]
    ax.plot(zoom.time_sec, zoom.imp_mag_adj_1_ohm, 'b-', lw=0.8)
    ax.set_ylabel('5 kHz (Ω)')

    # 100 kHz
    ax = axes[3]
    ax.plot(zoom.time_sec, zoom.imp_mag_adj_2_ohm, 'g-', lw=0.8)
    ax.set_ylabel('100 kHz (Ω)')
    ax.axhline(SATURATED_100K, color='red', ls=':', alpha=0.5, label='Saturation')
    ax.legend()

    # Pressure
    ax = axes[4]
    ax.plot(zoom.time_sec, zoom.pressure_mmhg, 'm-', lw=0.8)
    ax.set_ylabel('Pressure (mmHg)')
    ax.set_xlabel('Time (sec)')

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'plot2b_wall_zoom.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot2b_wall_zoom.png")
else:
    print("No wall events found — skipping plot2b_wall_zoom.png")

# ─── PLOT 3: Distribution comparison blood vs clot vs wall ───────────────────
blood = df[df.tissue == 'blood'].copy()
clot = df[df.tissue == 'clot'].copy()
wall = df[df.tissue == 'wall'].copy()

print(f"\nTissue sample counts: blood={len(blood)}, clot={len(clot)}, wall={len(wall)}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f'Blood vs Clot vs Wall — Distribution Comparison\n(blood={len(blood):,}, clot={len(clot):,}, wall={len(wall):,})', fontsize=13)

# Filter to reasonable range (remove outliers for visualization)
blood_clean = blood[blood.imp_mag_adj_0_ohm < blood.imp_mag_adj_0_ohm.quantile(0.99)]
clot_clean = clot[clot.imp_mag_adj_0_ohm < clot.imp_mag_adj_0_ohm.quantile(0.95)]
wall_clean = wall[wall.imp_mag_adj_0_ohm < wall.imp_mag_adj_0_ohm.quantile(0.95)]

# 50 kHz histogram
ax = axes[0, 0]
ax.hist(blood_clean.imp_mag_adj_0_ohm, bins=80, alpha=0.5, color='black', density=True, label='blood')
ax.hist(clot_clean.imp_mag_adj_0_ohm, bins=40, alpha=0.5, color='red', density=True, label='clot')
ax.hist(wall_clean.imp_mag_adj_0_ohm, bins=40, alpha=0.5, color='blue', density=True, label='wall')
ax.set_xlabel('Impedance (Ω)')
ax.set_ylabel('Density')
ax.set_title('50 kHz')
ax.legend()

# 5 kHz histogram
ax = axes[0, 1]
blood_clean1 = blood[blood.imp_mag_adj_1_ohm < blood.imp_mag_adj_1_ohm.quantile(0.99)]
clot_clean1 = clot[clot.imp_mag_adj_1_ohm < clot.imp_mag_adj_1_ohm.quantile(0.95)]
wall_clean1 = wall[wall.imp_mag_adj_1_ohm < wall.imp_mag_adj_1_ohm.quantile(0.95)]
ax.hist(blood_clean1.imp_mag_adj_1_ohm, bins=80, alpha=0.5, color='black', density=True, label='blood')
ax.hist(clot_clean1.imp_mag_adj_1_ohm, bins=40, alpha=0.5, color='red', density=True, label='clot')
ax.hist(wall_clean1.imp_mag_adj_1_ohm, bins=40, alpha=0.5, color='blue', density=True, label='wall')
ax.set_xlabel('Impedance (Ω)')
ax.set_ylabel('Density')
ax.set_title('5 kHz')
ax.legend()

# 100 kHz histogram (non-saturated only)
ax = axes[0, 2]
blood_100k = blood[~blood.freq2_saturated]
clot_100k = clot[~clot.freq2_saturated]
wall_100k = wall[~wall.freq2_saturated]
if len(blood_100k) > 0:
    ax.hist(blood_100k.imp_mag_adj_2_ohm, bins=80, alpha=0.5, color='black', density=True, label=f'blood (n={len(blood_100k)})')
if len(clot_100k) > 0:
    ax.hist(clot_100k.imp_mag_adj_2_ohm, bins=40, alpha=0.5, color='red', density=True, label=f'clot (n={len(clot_100k)})')
if len(wall_100k) > 0:
    ax.hist(wall_100k.imp_mag_adj_2_ohm, bins=40, alpha=0.5, color='blue', density=True, label=f'wall (n={len(wall_100k)})')
ax.set_xlabel('Impedance (Ω)')
ax.set_ylabel('Density')
ax.set_title('100 kHz (non-saturated only)')
ax.legend()

# Pressure histogram
ax = axes[1, 0]
ax.hist(blood.pressure_mmhg, bins=50, alpha=0.5, color='black', density=True, label='blood')
ax.hist(clot.pressure_mmhg, bins=30, alpha=0.5, color='red', density=True, label='clot')
ax.hist(wall.pressure_mmhg, bins=30, alpha=0.5, color='blue', density=True, label='wall')
ax.set_xlabel('Pressure (mmHg)')
ax.set_ylabel('Density')
ax.set_title('Pressure')
ax.legend()

# Phase 50kHz
ax = axes[1, 1]
blood_pha = blood[blood.imp_pha_0_millideg.abs() < blood.imp_pha_0_millideg.abs().quantile(0.99)]
ax.hist(blood_pha.imp_pha_0_millideg/1000, bins=80, alpha=0.5, color='black', density=True, label='blood')
ax.hist(clot.imp_pha_0_millideg/1000, bins=30, alpha=0.5, color='red', density=True, label='clot')
ax.hist(wall.imp_pha_0_millideg/1000, bins=30, alpha=0.5, color='blue', density=True, label='wall')
ax.set_xlabel('Phase (deg)')
ax.set_ylabel('Density')
ax.set_title('50 kHz Phase')
ax.legend()

# Ratio: 5kHz / 50kHz
ax = axes[1, 2]
blood['ratio_5k_50k'] = blood.imp_mag_adj_1_ohm / blood.imp_mag_adj_0_ohm.replace(0, np.nan)
clot['ratio_5k_50k'] = clot.imp_mag_adj_1_ohm / clot.imp_mag_adj_0_ohm.replace(0, np.nan)
wall['ratio_5k_50k'] = wall.imp_mag_adj_1_ohm / wall.imp_mag_adj_0_ohm.replace(0, np.nan)
blood_ratio = blood.ratio_5k_50k.dropna()
clot_ratio = clot.ratio_5k_50k.dropna()
wall_ratio = wall.ratio_5k_50k.dropna()
blood_ratio_clean = blood_ratio[(blood_ratio > 0) & (blood_ratio < blood_ratio.quantile(0.99))]
clot_ratio_clean = clot_ratio[(clot_ratio > 0) & (clot_ratio < 5)]
wall_ratio_clean = wall_ratio[(wall_ratio > 0) & (wall_ratio < 5)]
ax.hist(blood_ratio_clean, bins=80, alpha=0.5, color='black', density=True, label='blood')
ax.hist(clot_ratio_clean, bins=30, alpha=0.5, color='red', density=True, label='clot')
ax.hist(wall_ratio_clean, bins=30, alpha=0.5, color='blue', density=True, label='wall')
ax.set_xlabel('Ratio')
ax.set_ylabel('Density')
ax.set_title('Impedance Ratio: 5kHz / 50kHz')
ax.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot3_blood_vs_clot_vs_wall_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot3_blood_vs_clot_vs_wall_distributions.png")

# ─── PLOT 4: Frequency dispersion (impedance vs frequency) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Frequency Dispersion: Blood vs Clot vs Wall', fontsize=14)

# Box/violin for each freq
freqs = [5, 50, 100]  # kHz
freq_cols = ['imp_mag_adj_1_ohm', 'imp_mag_adj_0_ohm', 'imp_mag_adj_2_ohm']

# Get median values for blood, clot, and wall at each freq
tissue_stats = {}
for tissue_name, tissue_df in [('blood', blood), ('clot', clot), ('wall', wall)]:
    medians, q25s, q75s = [], [], []
    for col in freq_cols:
        data = tissue_df[col]
        if '2' in col:
            data = data[data < SATURATED_100K]
        medians.append(data.median())
        q25s.append(data.quantile(0.25))
        q75s.append(data.quantile(0.75))
    tissue_stats[tissue_name] = (medians, q25s, q75s)

ax = axes[0]
for name, color, fmt in [('blood', 'black', 'ko-'), ('clot', 'red', 'ro-'), ('wall', 'blue', 'bs-')]:
    meds, q25s, q75s = tissue_stats[name]
    ax.errorbar(freqs, meds,
                yerr=[np.array(meds)-np.array(q25s), np.array(q75s)-np.array(meds)],
                fmt=fmt, capsize=5, label=f'{name} (median ± IQR)')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Impedance (Ω)')
ax.set_title('Impedance Spectrum')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Ratio to 50kHz baseline
ax = axes[1]
for name, color, fmt in [('blood', 'black', 'ko-'), ('clot', 'red', 'ro-'), ('wall', 'blue', 'bs-')]:
    meds = tissue_stats[name][0]
    norm = [m / meds[1] for m in meds]  # normalize to 50kHz
    ax.plot(freqs, norm, fmt, markersize=8, label=name)
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Normalized to 50kHz')
ax.set_title('Normalized Impedance Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot4_frequency_dispersion.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot4_frequency_dispersion.png")

# ─── PLOT 5: Scatter plots — multi-frequency feature space ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Multi-Frequency Feature Space: Blood vs Clot vs Wall', fontsize=14)

# 50kHz vs 5kHz
ax = axes[0, 0]
ax.scatter(blood.imp_mag_adj_0_ohm, blood.imp_mag_adj_1_ohm, c='black', s=1, alpha=0.03, label='blood')
ax.scatter(clot.imp_mag_adj_0_ohm, clot.imp_mag_adj_1_ohm, c='red', s=8, alpha=0.5, label='clot')
ax.scatter(wall.imp_mag_adj_0_ohm, wall.imp_mag_adj_1_ohm, c='blue', s=8, alpha=0.5, label='wall')
ax.set_xlabel('50 kHz (Ω)')
ax.set_ylabel('5 kHz (Ω)')
ax.set_title('50 kHz vs 5 kHz')
xlim = max(blood.imp_mag_adj_0_ohm.quantile(0.99), clot.imp_mag_adj_0_ohm.quantile(0.95), wall.imp_mag_adj_0_ohm.quantile(0.95))
ylim = max(blood.imp_mag_adj_1_ohm.quantile(0.99), clot.imp_mag_adj_1_ohm.quantile(0.95), wall.imp_mag_adj_1_ohm.quantile(0.95))
ax.set_xlim(0, xlim)
ax.set_ylim(0, ylim)
ax.legend()

# 50kHz vs Pressure
ax = axes[0, 1]
ax.scatter(blood.imp_mag_adj_0_ohm, blood.pressure_mmhg, c='black', s=1, alpha=0.03, label='blood')
ax.scatter(clot.imp_mag_adj_0_ohm, clot.pressure_mmhg, c='red', s=8, alpha=0.5, label='clot')
ax.scatter(wall.imp_mag_adj_0_ohm, wall.pressure_mmhg, c='blue', s=8, alpha=0.5, label='wall')
ax.set_xlabel('50 kHz (Ω)')
ax.set_ylabel('Pressure (mmHg)')
ax.set_title('Impedance vs Pressure')
ax.set_xlim(0, xlim)
ax.legend()

# 5kHz/50kHz ratio vs Pressure
ax = axes[1, 0]
ax.scatter(blood.ratio_5k_50k, blood.pressure_mmhg, c='black', s=1, alpha=0.03, label='blood')
ax.scatter(clot.ratio_5k_50k, clot.pressure_mmhg, c='red', s=8, alpha=0.5, label='clot')
ax.scatter(wall.ratio_5k_50k, wall.pressure_mmhg, c='blue', s=8, alpha=0.5, label='wall')
ax.set_xlabel('5kHz / 50kHz Ratio')
ax.set_ylabel('Pressure (mmHg)')
ax.set_title('Frequency Ratio vs Pressure')
ax.set_xlim(0, 2)
ax.legend()

# Phase 50kHz vs magnitude
ax = axes[1, 1]
ax.scatter(blood.imp_mag_adj_0_ohm, blood.imp_pha_0_millideg/1000, c='black', s=1, alpha=0.03, label='blood')
ax.scatter(clot.imp_mag_adj_0_ohm, clot.imp_pha_0_millideg/1000, c='red', s=8, alpha=0.5, label='clot')
ax.scatter(wall.imp_mag_adj_0_ohm, wall.imp_pha_0_millideg/1000, c='blue', s=8, alpha=0.5, label='wall')
ax.set_xlabel('50 kHz Magnitude (Ω)')
ax.set_ylabel('50 kHz Phase (deg)')
ax.set_title('Magnitude vs Phase @ 50 kHz')
ax.set_xlim(0, xlim)
ax.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot5_feature_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot5_feature_space.png")

# ─── PLOT 6: Unlabeled events exploration (4, 5, 9) ─────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle('Unlabeled Events (4, 5, 9) — Impedance Characteristics', fontsize=14)

# ─── PLOT 6: All states overview ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle('All States — Impedance & Pressure by solo_state_i', fontsize=14)

# Color by tissue class
tissue_colors = {'blood': 'black', 'clot': 'red', 'wall': 'blue', 'other': 'gray'}
for tissue_name, color in tissue_colors.items():
    mask = df.tissue == tissue_name
    if mask.any():
        sub = df[mask]
        axes[0].scatter(sub.time_sec, sub.imp_mag_adj_0_ohm, c=color, s=0.5, alpha=0.3, label=tissue_name)
        axes[1].scatter(sub.time_sec, sub.imp_mag_adj_1_ohm, c=color, s=0.5, alpha=0.3, label=tissue_name)
        axes[2].scatter(sub.time_sec, sub.pressure_mmhg, c=color, s=0.5, alpha=0.3, label=tissue_name)

axes[0].set_ylabel('50 kHz (Ω)')
axes[0].set_ylim(0, 5000)
axes[0].legend(markerscale=10, fontsize=9)
axes[1].set_ylabel('5 kHz (Ω)')
axes[1].set_ylim(0, 3000)
axes[2].set_ylabel('Pressure (mmHg)')
axes[2].set_xlabel('Time (sec)')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot6_all_states.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot6_all_states.png")

# ─── PLOT 7: Phase analysis across frequencies ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Phase Angle Analysis: Blood vs Clot vs Wall', fontsize=14)

phase_cols = ['imp_pha_0_millideg', 'imp_pha_1_millideg', 'imp_pha_2_millideg']
phase_titles = ['50 kHz Phase', '5 kHz Phase', '100 kHz Phase']

for i, (col, title) in enumerate(zip(phase_cols, phase_titles)):
    ax = axes[i]
    b_data = blood[col] / 1000  # to degrees
    c_data = clot[col] / 1000
    w_data = wall[col] / 1000
    # Clip outliers
    b_clip = b_data[(b_data > b_data.quantile(0.01)) & (b_data < b_data.quantile(0.99))]
    ax.hist(b_clip, bins=60, alpha=0.5, color='black', density=True, label=f'blood')
    ax.hist(c_data, bins=30, alpha=0.5, color='red', density=True, label=f'clot')
    ax.hist(w_data, bins=30, alpha=0.5, color='blue', density=True, label=f'wall')
    ax.set_xlabel('Phase (degrees)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot7_phase_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot7_phase_analysis.png")

# ─── Statistical Summary ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("STATISTICAL SUMMARY: Blood vs Clot vs Wall")
print("="*60)

print(f"\nSample sizes: blood={len(blood):,}, clot={len(clot):,}, wall={len(wall):,}")

print(f"\n{'Metric':<25} {'Blood (med)':<14} {'Clot (med)':<14} {'Wall (med)':<14} {'Clot sep':<10} {'Wall sep':<10}")
print("-"*87)

metrics = [
    ('50 kHz (Ω)', 'imp_mag_adj_0_ohm'),
    ('5 kHz (Ω)', 'imp_mag_adj_1_ohm'),
    ('50 kHz Phase (deg)', 'imp_pha_0_millideg'),
    ('5 kHz Phase (deg)', 'imp_pha_1_millideg'),
    ('Pressure (mmHg)', 'pressure_mmhg'),
]

for name, col in metrics:
    b_med = blood[col].median()
    c_med = clot[col].median()
    w_med = wall[col].median()
    b_std = blood[col].std()
    if 'pha' in col:
        b_med /= 1000; c_med /= 1000; w_med /= 1000; b_std /= 1000
    clot_sep = abs(c_med - b_med) / b_std if b_std > 0 else 0
    wall_sep = abs(w_med - b_med) / b_std if b_std > 0 else 0
    print(f"{name:<25} {b_med:<14.1f} {c_med:<14.1f} {w_med:<14.1f} {clot_sep:<10.3f} {wall_sep:<10.3f}")

# Frequency ratio
b_ratio = blood.ratio_5k_50k.median()
c_ratio = clot.ratio_5k_50k.median()
w_ratio = wall.ratio_5k_50k.median()
b_ratio_std = blood.ratio_5k_50k.std()
clot_sep = abs(c_ratio - b_ratio) / b_ratio_std if b_ratio_std > 0 else 0
wall_sep = abs(w_ratio - b_ratio) / b_ratio_std if b_ratio_std > 0 else 0
print(f"{'5kHz/50kHz Ratio':<25} {b_ratio:<14.3f} {c_ratio:<14.3f} {w_ratio:<14.3f} {clot_sep:<10.3f} {wall_sep:<10.3f}")

# Clot vs Wall separation (the hard problem)
print(f"\n{'--- CLOT vs WALL ---'}")
print(f"{'Metric':<25} {'Clot (med)':<14} {'Wall (med)':<14} {'Separation':<10}")
print("-"*63)
for name, col in metrics:
    c_med = clot[col].median()
    w_med = wall[col].median()
    pooled_std = pd.concat([clot[col], wall[col]]).std()
    if 'pha' in col:
        c_med /= 1000; w_med /= 1000; pooled_std /= 1000
    sep = abs(c_med - w_med) / pooled_std if pooled_std > 0 else 0
    print(f"{name:<25} {c_med:<14.1f} {w_med:<14.1f} {sep:<10.3f}")
c_ratio_med = clot.ratio_5k_50k.median()
w_ratio_med = wall.ratio_5k_50k.median()
pooled_std = pd.concat([clot.ratio_5k_50k, wall.ratio_5k_50k]).std()
sep = abs(c_ratio_med - w_ratio_med) / pooled_std if pooled_std > 0 else 0
print(f"{'5kHz/50kHz Ratio':<25} {c_ratio_med:<14.3f} {w_ratio_med:<14.3f} {sep:<10.3f}")

print("\n" + "="*60)
print("KEY OBSERVATIONS")
print("="*60)
print("""
1. CORRECTED LABELS: Using solo_state_i column.
   blood={7,10} (88K samples), clot={8,11,12} (6K), wall={13,14} (8.7K)

2. 100 kHz channel: 50% saturated at 2^20-1 — hardware/ADC issue.

3. Frequency ratio (5kHz/50kHz): Key tissue-specific feature.
   Different dispersion slopes suggest different tissue composition.

4. Pressure: Now can compare wall vs clot/blood — wall should show
   higher contact pressure if catheter is pressed against vessel wall.

5. Phase angle: May separate tissues at different frequencies.

6. CLOT vs WALL is the critical comparison — this is what the ML
   model struggles with on single-frequency data.
""")

print("All plots saved to Gen2.5/ folder.")
