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

# ─── Load data ───────────────────────────────────────────────────────────────
df = pd.read_parquet(Path(__file__).parent / 'LOG2_solo.parquet')
df['time_sec'] = df['timestamp_ms'] / 1000.0

# Event mapping
event_colors = {
    6:  ('black',   'blood'),
    12: ('black',   'blood'),
    7:  ('red',     'clot'),
    11: ('red',     'clot'),
    23: ('blue',    'wall'),
    8:  ('magenta', 'contrast'),
    15: ('cyan',    'saline'),
}
df['tissue'] = df['solo_led_state_i'].map({k: v[1] for k, v in event_colors.items()})
df['tissue'] = df['tissue'].fillna('unlabeled')

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

# GT labels
ax = axes[0]
for ev, (color, name) in event_colors.items():
    mask = df.solo_led_state_i == ev
    if mask.any():
        ax.scatter(df.loc[mask, 'time_sec'], df.loc[mask, 'solo_led_state_i'],
                   c=color, s=1, label=f'{ev}={name}', alpha=0.5)
# unlabeled events
for ev in [4, 5, 9]:
    mask = df.solo_led_state_i == ev
    if mask.any():
        ax.scatter(df.loc[mask, 'time_sec'], df.loc[mask, 'solo_led_state_i'],
                   c='gray', s=1, label=f'{ev}=?', alpha=0.3)
ax.set_ylabel('Event ID')
ax.legend(loc='upper right', markerscale=5, fontsize=8)
ax.set_title('Ground Truth Labels')

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

# ─── PLOT 2: Zoom on clot event (576-581s) ──────────────────────────────────
t_start, t_end = 570, 590
#t_start, t_end = 520, 640

zoom = df[(df.time_sec >= t_start) & (df.time_sec <= t_end)]

fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'Clot Event Zoom ({t_start}-{t_end}s)', fontsize=14)

# Labels
ax = axes[0]
ax.plot(zoom.time_sec, zoom.solo_led_state_i, 'k-', lw=1)
ax.axhspan(6.5, 7.5, alpha=0.2, color='red', label='clot region')
ax.set_ylabel('Event ID')
ax.legend()

# 50 kHz
ax = axes[1]
ax.plot(zoom.time_sec, zoom.imp_mag_adj_0_ohm, 'k-', lw=0.8)
ax.set_ylabel('50 kHz (Ω)')
ax.axvline(576.8, color='red', ls='--', alpha=0.5)
ax.axvline(580.8, color='red', ls='--', alpha=0.5)

# 5 kHz
ax = axes[2]
ax.plot(zoom.time_sec, zoom.imp_mag_adj_1_ohm, 'b-', lw=0.8)
ax.set_ylabel('5 kHz (Ω)')
ax.axvline(576.8, color='red', ls='--', alpha=0.5)
ax.axvline(580.8, color='red', ls='--', alpha=0.5)

# 100 kHz
ax = axes[3]
ax.plot(zoom.time_sec, zoom.imp_mag_adj_2_ohm, 'g-', lw=0.8)
ax.set_ylabel('100 kHz (Ω)')
ax.axhline(SATURATED_100K, color='red', ls=':', alpha=0.5, label='Saturation')
ax.axvline(576.8, color='red', ls='--', alpha=0.5)
ax.axvline(580.8, color='red', ls='--', alpha=0.5)
ax.legend()

# Pressure
ax = axes[4]
ax.plot(zoom.time_sec, zoom.pressure_mmhg, 'm-', lw=0.8)
ax.set_ylabel('Pressure (mmHg)')
ax.set_xlabel('Time (sec)')
ax.axvline(576.8, color='red', ls='--', alpha=0.5)
ax.axvline(580.8, color='red', ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot2_clot_zoom.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot2_clot_zoom.png")

# ─── PLOT 3: Distribution comparison blood vs clot ───────────────────────────
blood = df[df.tissue == 'blood'].copy()
clot = df[df.tissue == 'clot'].copy()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Blood vs Clot — Distribution Comparison', fontsize=14)

# Filter blood to reasonable range (remove outliers for visualization)
blood_clean = blood[blood.imp_mag_adj_0_ohm < blood.imp_mag_adj_0_ohm.quantile(0.99)]
clot_clean = clot[clot.imp_mag_adj_0_ohm < clot.imp_mag_adj_0_ohm.quantile(0.95)]

# 50 kHz histogram
ax = axes[0, 0]
ax.hist(blood_clean.imp_mag_adj_0_ohm, bins=80, alpha=0.6, color='black', density=True, label='blood')
ax.hist(clot_clean.imp_mag_adj_0_ohm, bins=30, alpha=0.6, color='red', density=True, label='clot')
ax.set_xlabel('Impedance (Ω)')
ax.set_title('50 kHz')
ax.legend()

# 5 kHz histogram
ax = axes[0, 1]
blood_clean1 = blood[blood.imp_mag_adj_1_ohm < blood.imp_mag_adj_1_ohm.quantile(0.99)]
clot_clean1 = clot[clot.imp_mag_adj_1_ohm < clot.imp_mag_adj_1_ohm.quantile(0.95)]
ax.hist(blood_clean1.imp_mag_adj_1_ohm, bins=80, alpha=0.6, color='black', density=True, label='blood')
ax.hist(clot_clean1.imp_mag_adj_1_ohm, bins=30, alpha=0.6, color='red', density=True, label='clot')
ax.set_xlabel('Impedance (Ω)')
ax.set_title('5 kHz')
ax.legend()

# 100 kHz histogram (non-saturated only)
ax = axes[0, 2]
blood_100k = blood[(~blood.freq2_saturated)]
clot_100k = clot[(~clot.freq2_saturated)]
if len(blood_100k) > 0:
    ax.hist(blood_100k.imp_mag_adj_2_ohm, bins=80, alpha=0.6, color='black', density=True, label=f'blood (n={len(blood_100k)})')
if len(clot_100k) > 0:
    ax.hist(clot_100k.imp_mag_adj_2_ohm, bins=30, alpha=0.6, color='red', density=True, label=f'clot (n={len(clot_100k)})')
ax.set_xlabel('Impedance (Ω)')
ax.set_title('100 kHz (non-saturated only)')
ax.legend()

# Pressure histogram
ax = axes[1, 0]
ax.hist(blood.pressure_mmhg, bins=50, alpha=0.6, color='black', density=True, label='blood')
ax.hist(clot.pressure_mmhg, bins=20, alpha=0.6, color='red', density=True, label='clot')
ax.set_xlabel('Pressure (mmHg)')
ax.set_title('Pressure')
ax.legend()

# Phase 50kHz
ax = axes[1, 1]
blood_pha = blood[blood.imp_pha_0_millideg.abs() < blood.imp_pha_0_millideg.abs().quantile(0.99)]
clot_pha = clot
ax.hist(blood_pha.imp_pha_0_millideg/1000, bins=80, alpha=0.6, color='black', density=True, label='blood')
ax.hist(clot_pha.imp_pha_0_millideg/1000, bins=30, alpha=0.6, color='red', density=True, label='clot')
ax.set_xlabel('Phase (deg)')
ax.set_title('50 kHz Phase')
ax.legend()

# Ratio: 5kHz / 50kHz
ax = axes[1, 2]
blood['ratio_5k_50k'] = blood.imp_mag_adj_1_ohm / blood.imp_mag_adj_0_ohm.replace(0, np.nan)
clot['ratio_5k_50k'] = clot.imp_mag_adj_1_ohm / clot.imp_mag_adj_0_ohm.replace(0, np.nan)
blood_ratio = blood.ratio_5k_50k.dropna()
clot_ratio = clot.ratio_5k_50k.dropna()
blood_ratio_clean = blood_ratio[(blood_ratio > 0) & (blood_ratio < blood_ratio.quantile(0.99))]
clot_ratio_clean = clot_ratio[(clot_ratio > 0) & (clot_ratio < 5)]
ax.hist(blood_ratio_clean, bins=80, alpha=0.6, color='black', density=True, label='blood')
ax.hist(clot_ratio_clean, bins=30, alpha=0.6, color='red', density=True, label='clot')
ax.set_xlabel('Ratio')
ax.set_title('Impedance Ratio: 5kHz / 50kHz')
ax.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot3_blood_vs_clot_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot3_blood_vs_clot_distributions.png")

# ─── PLOT 4: Frequency dispersion (impedance vs frequency) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Frequency Dispersion: Blood vs Clot', fontsize=14)

# Box/violin for each freq
freqs = [5, 50, 100]  # kHz
freq_cols = ['imp_mag_adj_1_ohm', 'imp_mag_adj_0_ohm', 'imp_mag_adj_2_ohm']

# Get median values for blood and clot at each freq
# Use non-saturated data
blood_medians = []
clot_medians = []
blood_q25 = []
blood_q75 = []
clot_q25 = []
clot_q75 = []

for col in freq_cols:
    b = blood[col]
    c = clot[col]
    # Exclude saturated for 100kHz
    if '2' in col:
        b = b[b < SATURATED_100K]
        c = c[c < SATURATED_100K]
    blood_medians.append(b.median())
    blood_q25.append(b.quantile(0.25))
    blood_q75.append(b.quantile(0.75))
    clot_medians.append(c.median())
    clot_q25.append(c.quantile(0.25))
    clot_q75.append(c.quantile(0.75))

ax = axes[0]
ax.errorbar(freqs, blood_medians,
            yerr=[np.array(blood_medians)-np.array(blood_q25), np.array(blood_q75)-np.array(blood_medians)],
            fmt='ko-', capsize=5, label='blood (median ± IQR)')
ax.errorbar(freqs, clot_medians,
            yerr=[np.array(clot_medians)-np.array(clot_q25), np.array(clot_q75)-np.array(clot_medians)],
            fmt='ro-', capsize=5, label='clot (median ± IQR)')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Impedance (Ω)')
ax.set_title('Impedance Spectrum')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Ratio to 50kHz baseline
ax = axes[1]
blood_norm = [m / blood_medians[1] for m in blood_medians]
clot_norm = [m / clot_medians[1] for m in clot_medians]
ax.plot(freqs, blood_norm, 'ko-', markersize=8, label='blood')
ax.plot(freqs, clot_norm, 'ro-', markersize=8, label='clot')
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
fig.suptitle('Multi-Frequency Feature Space: Blood vs Clot', fontsize=14)

# 50kHz vs 5kHz
ax = axes[0, 0]
ax.scatter(blood.imp_mag_adj_0_ohm, blood.imp_mag_adj_1_ohm, c='black', s=1, alpha=0.05, label='blood')
ax.scatter(clot.imp_mag_adj_0_ohm, clot.imp_mag_adj_1_ohm, c='red', s=10, alpha=0.7, label='clot')
ax.set_xlabel('50 kHz (Ω)')
ax.set_ylabel('5 kHz (Ω)')
ax.set_title('50 kHz vs 5 kHz')
ax.set_xlim(0, blood.imp_mag_adj_0_ohm.quantile(0.99))
ax.set_ylim(0, blood.imp_mag_adj_1_ohm.quantile(0.99))
ax.legend()

# 50kHz vs Pressure
ax = axes[0, 1]
ax.scatter(blood.imp_mag_adj_0_ohm, blood.pressure_mmhg, c='black', s=1, alpha=0.05, label='blood')
ax.scatter(clot.imp_mag_adj_0_ohm, clot.pressure_mmhg, c='red', s=10, alpha=0.7, label='clot')
ax.set_xlabel('50 kHz (Ω)')
ax.set_ylabel('Pressure (mmHg)')
ax.set_title('Impedance vs Pressure')
ax.set_xlim(0, blood.imp_mag_adj_0_ohm.quantile(0.99))
ax.legend()

# 5kHz/50kHz ratio vs Pressure
ax = axes[1, 0]
ax.scatter(blood.ratio_5k_50k, blood.pressure_mmhg, c='black', s=1, alpha=0.05, label='blood')
ax.scatter(clot.ratio_5k_50k, clot.pressure_mmhg, c='red', s=10, alpha=0.7, label='clot')
ax.set_xlabel('5kHz / 50kHz Ratio')
ax.set_ylabel('Pressure (mmHg)')
ax.set_title('Frequency Ratio vs Pressure')
ax.set_xlim(0, 2)
ax.legend()

# Phase 50kHz vs magnitude
ax = axes[1, 1]
ax.scatter(blood.imp_mag_adj_0_ohm, blood.imp_pha_0_millideg/1000, c='black', s=1, alpha=0.05, label='blood')
ax.scatter(clot.imp_mag_adj_0_ohm, clot.imp_pha_0_millideg/1000, c='red', s=10, alpha=0.7, label='clot')
ax.set_xlabel('50 kHz Magnitude (Ω)')
ax.set_ylabel('50 kHz Phase (deg)')
ax.set_title('Magnitude vs Phase @ 50 kHz')
ax.set_xlim(0, blood.imp_mag_adj_0_ohm.quantile(0.99))
ax.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot5_feature_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot5_feature_space.png")

# ─── PLOT 6: Unlabeled events exploration (4, 5, 9) ─────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle('Unlabeled Events (4, 5, 9) — Impedance Characteristics', fontsize=14)

# Color by event type
colors_map = {4: 'orange', 5: 'purple', 6: 'black', 7: 'red', 9: 'green'}
labels_map = {4: 'event 4 (?)', 5: 'event 5 (?)', 6: 'blood', 7: 'clot', 9: 'event 9 (?)'}

for ev, color in colors_map.items():
    mask = df.solo_led_state_i == ev
    if mask.any():
        sub = df[mask]
        axes[0].scatter(sub.time_sec, sub.imp_mag_adj_0_ohm, c=color, s=0.5, alpha=0.3, label=labels_map[ev])
        axes[1].scatter(sub.time_sec, sub.imp_mag_adj_1_ohm, c=color, s=0.5, alpha=0.3, label=labels_map[ev])
        axes[2].scatter(sub.time_sec, sub.pressure_mmhg, c=color, s=0.5, alpha=0.3, label=labels_map[ev])

axes[0].set_ylabel('50 kHz (Ω)')
axes[0].set_ylim(0, 5000)
axes[0].legend(markerscale=10, fontsize=9)
axes[1].set_ylabel('5 kHz (Ω)')
axes[1].set_ylim(0, 3000)
axes[2].set_ylabel('Pressure (mmHg)')
axes[2].set_xlabel('Time (sec)')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot6_unlabeled_events.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot6_unlabeled_events.png")

# ─── PLOT 7: Phase analysis across frequencies ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Phase Angle Analysis: Blood vs Clot', fontsize=14)

phase_cols = ['imp_pha_0_millideg', 'imp_pha_1_millideg', 'imp_pha_2_millideg']
phase_titles = ['50 kHz Phase', '5 kHz Phase', '100 kHz Phase']

for i, (col, title) in enumerate(zip(phase_cols, phase_titles)):
    ax = axes[i]
    b_data = blood[col] / 1000  # to degrees
    c_data = clot[col] / 1000
    # Clip outliers
    b_clip = b_data[(b_data > b_data.quantile(0.01)) & (b_data < b_data.quantile(0.99))]
    ax.hist(b_clip, bins=60, alpha=0.6, color='black', density=True, label=f'blood (n={len(b_clip)})')
    ax.hist(c_data, bins=20, alpha=0.6, color='red', density=True, label=f'clot (n={len(c_data)})')
    ax.set_xlabel('Phase (degrees)')
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'plot7_phase_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot7_phase_analysis.png")

# ─── Statistical Summary ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("STATISTICAL SUMMARY: Blood vs Clot")
print("="*60)

print(f"\nSample sizes: blood={len(blood):,}, clot={len(clot):,}")
print(f"NOTE: Only 325 clot samples (~4s). No wall data in this file.")

print(f"\n{'Metric':<30} {'Blood (median)':<18} {'Clot (median)':<18} {'Separation':<12}")
print("-"*78)

metrics = [
    ('50 kHz Impedance (Ω)', 'imp_mag_adj_0_ohm'),
    ('5 kHz Impedance (Ω)', 'imp_mag_adj_1_ohm'),
    ('50 kHz Phase (deg)', 'imp_pha_0_millideg'),
    ('5 kHz Phase (deg)', 'imp_pha_1_millideg'),
    ('Pressure (mmHg)', 'pressure_mmhg'),
]

for name, col in metrics:
    b_med = blood[col].median()
    c_med = clot[col].median()
    b_std = blood[col].std()
    if 'pha' in col:
        b_med /= 1000
        c_med /= 1000
        b_std /= 1000
    # Cohen's d approximation
    sep = abs(c_med - b_med) / b_std if b_std > 0 else 0
    print(f"{name:<30} {b_med:<18.1f} {c_med:<18.1f} {sep:<12.3f}")

# Frequency ratio
b_ratio = (blood.imp_mag_adj_1_ohm / blood.imp_mag_adj_0_ohm.replace(0, np.nan)).median()
c_ratio = (clot.imp_mag_adj_1_ohm / clot.imp_mag_adj_0_ohm.replace(0, np.nan)).median()
b_ratio_std = (blood.imp_mag_adj_1_ohm / blood.imp_mag_adj_0_ohm.replace(0, np.nan)).std()
sep = abs(c_ratio - b_ratio) / b_ratio_std if b_ratio_std > 0 else 0
print(f"{'5kHz/50kHz Ratio':<30} {b_ratio:<18.3f} {c_ratio:<18.3f} {sep:<12.3f}")

print("\n" + "="*60)
print("KEY OBSERVATIONS")
print("="*60)
print("""
1. 100 kHz channel: 50% saturated at 2^20-1 — likely hardware/ADC issue.
   Cannot be used reliably without fixing saturation.

2. Frequency ratio (5kHz/50kHz): May provide tissue-specific dispersion
   signature independent of absolute impedance level.

3. Pressure: Very narrow range (746-756 mmHg for both blood and clot).
   Minimal separation in this single study. Need wall contact data
   to evaluate pressure's discriminative power.

4. Phase angle: Potentially useful — different tissue compositions
   produce different reactive components at each frequency.

5. CRITICAL LIMITATION: Only 1 clot event (4 sec, 325 samples) and
   NO wall data in this file. Cannot draw firm conclusions about
   clot vs wall discrimination from this single recording.

6. Unlabeled events (4, 5, 9): Large portion of data. Need event
   definitions for Gen2.5 protocol to understand these.
""")

print("All plots saved to Gen2.5/ folder.")
