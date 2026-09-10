# -*- coding: utf-8 -*-
"""
Gen360 — Deep-Dive: 50 kHz Phase (imp_pha_0_millideg) by Tissue
=================================================================
Uses ALL raw samples (not medians).

Plots produced:
  1. Overlapping histograms (density) — Blood / Clot / Wall
  2. Box plots per tissue
  3. Violin plots per tissue
  4. CDF (cumulative distribution) per tissue
  5. Time series for each study file, coloured by tissue
  6. Scatter: 50 kHz impedance vs 50 kHz phase, coloured by tissue

Outputs saved to Gen360/combined_plots/phase_50k/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(r'c:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Gen360')
OUT_DIR  = DATA_DIR / 'combined_plots' / 'phase_50k'
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_TO_TISSUE = {4: 'blood', 5: 'clot', 9: 'wall'}
TISSUE_ORDER    = ['blood', 'clot', 'wall']
TISSUE_COLORS   = {'blood': '#2ca02c', 'clot': '#d62728', 'wall': '#1f77b4'}

NEEDED_COLS = [
    'solo_led_state_i', 'timestamp_ms',
    'imp_mag_adj_0_ohm', 'imp_pha_0_millideg',
]

# ─── Load all files ───────────────────────────────────────────────────────────
files = sorted(DATA_DIR.glob('LOG3_solo_*.parquet'))
print(f'Found {len(files)} file(s)')

frames = []
for p in files:
    print(f'  Loading {p.name} ...', end=' ', flush=True)
    df = pd.read_parquet(p, columns=NEEDED_COLS)
    df['tissue'] = df['solo_led_state_i'].map(STATE_TO_TISSUE).fillna('other')
    df['_file']  = p.stem          # short study name for legend
    df['time_s'] = df['timestamp_ms'] / 1000.0
    frames.append(df)
    print(f'{len(df):,} rows')

full = pd.concat(frames, ignore_index=True)

# Keep only labelled tissue rows with valid (non-zero) 50 kHz impedance
full = full[full['tissue'].isin(TISSUE_ORDER)].copy()
full = full[full['imp_mag_adj_0_ohm'] > 0].copy()

# Convert phase to degrees
full['phase_deg'] = full['imp_pha_0_millideg'] / 1000.0
full['imp_ohm']   = full['imp_mag_adj_0_ohm']

print(f'\nValid rows after filtering: {len(full):,}')
print(full['tissue'].value_counts().to_string())

# ─── Per-tissue arrays ────────────────────────────────────────────────────────
tissue_data = {t: full.loc[full['tissue'] == t, 'phase_deg'].values for t in TISSUE_ORDER}

# ─── Descriptive stats ────────────────────────────────────────────────────────
print('\n' + '='*60)
print('50 kHz PHASE — DESCRIPTIVE STATS  (all valid samples)')
print('='*60)
for t in TISSUE_ORDER:
    d = tissue_data[t]
    print(f'\n  {t.upper()}  (n={len(d):,})')
    print(f'    min={d.min():.3f}  max={d.max():.3f}  mean={d.mean():.3f}  '
          f'median={np.median(d):.3f}  std={d.std():.3f}')
    print(f'    p10={np.percentile(d,10):.3f}  p25={np.percentile(d,25):.3f}  '
          f'p75={np.percentile(d,75):.3f}  p90={np.percentile(d,90):.3f}')

# ─── Plot 1: Overlapping histograms ──────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5), dpi=180)
for t in TISSUE_ORDER:
    d = tissue_data[t]
    ax1.hist(d, bins=120, density=True, alpha=0.45,
             color=TISSUE_COLORS[t], label=f'{t.capitalize()} (n={len(d):,})',
             edgecolor='none')
ax1.set_xlabel('Phase  (deg)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('50 kHz Phase Distribution by Tissue\n(all valid samples)', fontsize=11)
ax1.legend(frameon=False, fontsize=10)
ax1.grid(True, alpha=0.25)
fig1.tight_layout()
fig1.savefig(OUT_DIR / 'phase50k_histogram.png', bbox_inches='tight')
print(f'\nSaved: phase50k_histogram.png')
plt.show()

# ─── Plot 2: Box plots ────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 5.5), dpi=180)
bp_data   = [tissue_data[t] for t in TISSUE_ORDER]
bp_labels = [f'{t.capitalize()}\n(n={len(tissue_data[t]):,})' for t in TISSUE_ORDER]
bp = ax2.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                 medianprops=dict(color='white', lw=2),
                 flierprops=dict(marker='.', markersize=1, alpha=0.2))
for patch, t in zip(bp['boxes'], TISSUE_ORDER):
    patch.set_facecolor(TISSUE_COLORS[t])
    patch.set_alpha(0.75)
for element in ['whiskers', 'caps']:
    for line, t in zip(bp[element], [v for v in TISSUE_ORDER for _ in range(2)]):
        line.set_color(TISSUE_COLORS[t])
        line.set_lw(1.8)
ax2.set_ylabel('Phase  (deg)', fontsize=12)
ax2.set_title('50 kHz Phase — Box Plot by Tissue', fontsize=11)
ax2.grid(True, axis='y', alpha=0.25)
fig2.tight_layout()
fig2.savefig(OUT_DIR / 'phase50k_boxplot.png', bbox_inches='tight')
print(f'Saved: phase50k_boxplot.png')
plt.show()

# ─── Plot 3: Violin plots with embedded box plot ──────────────────────────────
# KDE outer shape + IQR box (thick black) + whiskers (thin black) + median (white dot)
fig3, ax3 = plt.subplots(figsize=(7, 5.5), dpi=180)

positions = range(1, len(TISSUE_ORDER) + 1)

# Draw KDE violin bodies (no built-in median/extrema — we overlay manually)
vp = ax3.violinplot(bp_data, positions=list(positions),
                    showmedians=False, showextrema=False)
for body, t in zip(vp['bodies'], TISSUE_ORDER):
    body.set_facecolor(TISSUE_COLORS[t])
    body.set_edgecolor('none')
    body.set_alpha(0.65)

# Overlay miniature box plot: IQR box + whiskers + median dot
BOX_WIDTH = 0.08   # narrow box so violin shape stays visible
for pos, t in zip(positions, TISSUE_ORDER):
    d = tissue_data[t]
    q25, median, q75 = np.percentile(d, [25, 50, 75])
    iqr = q75 - q25
    lo_whisker = max(d.min(), q25 - 1.5 * iqr)
    hi_whisker = min(d.max(), q75 + 1.5 * iqr)

    # IQR box (thick black)
    ax3.add_patch(plt.Rectangle(
        (pos - BOX_WIDTH / 2, q25), BOX_WIDTH, iqr,
        linewidth=1.8, edgecolor='black', facecolor='black', zorder=3))

    # Whiskers (thin black lines)
    ax3.plot([pos, pos], [lo_whisker, q25], color='black', lw=1.2, zorder=3)
    ax3.plot([pos, pos], [q75, hi_whisker], color='black', lw=1.2, zorder=3)
    # Whisker caps
    ax3.plot([pos - BOX_WIDTH / 2, pos + BOX_WIDTH / 2], [lo_whisker, lo_whisker],
             color='black', lw=1.2, zorder=3)
    ax3.plot([pos - BOX_WIDTH / 2, pos + BOX_WIDTH / 2], [hi_whisker, hi_whisker],
             color='black', lw=1.2, zorder=3)

    # Median white dot
    ax3.scatter([pos], [median], color='white', edgecolors='black',
                s=40, zorder=4, lw=1)

ax3.set_xticks(list(positions))
ax3.set_xticklabels(bp_labels)
ax3.set_ylabel('Phase  (deg)', fontsize=12)
ax3.set_title('50 kHz Phase — Violin Plot by Tissue', fontsize=11)
ax3.grid(True, axis='y', alpha=0.25)
fig3.tight_layout()
fig3.savefig(OUT_DIR / 'phase50k_violin.png', bbox_inches='tight')
print(f'Saved: phase50k_violin.png')
plt.show()

# ─── Plot 4: CDF ─────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(9, 5), dpi=180)
for t in TISSUE_ORDER:
    d = np.sort(tissue_data[t])
    cdf = np.arange(1, len(d)+1) / len(d)
    ax4.plot(d, cdf, color=TISSUE_COLORS[t], lw=2,
             label=f'{t.capitalize()} (n={len(d):,})')
ax4.set_xlabel('Phase  (deg)', fontsize=12)
ax4.set_ylabel('Cumulative fraction', fontsize=12)
ax4.set_title('50 kHz Phase — CDF by Tissue', fontsize=11)
ax4.legend(frameon=False, fontsize=10)
ax4.grid(True, alpha=0.25)
fig4.tight_layout()
fig4.savefig(OUT_DIR / 'phase50k_cdf.png', bbox_inches='tight')
print(f'Saved: phase50k_cdf.png')
plt.show()

# ─── Plot 5: Time series per study, coloured by tissue ───────────────────────
study_files = sorted(full['_file'].unique())
fig5, axes5 = plt.subplots(len(study_files), 1,
                            figsize=(13, 3.5 * len(study_files)), dpi=150, sharex=False)
if len(study_files) == 1:
    axes5 = [axes5]

for ax, sf in zip(axes5, study_files):
    sub = full[full['_file'] == sf].sort_values('time_s')
    for t in TISSUE_ORDER:
        ts = sub[sub['tissue'] == t]
        if ts.empty:
            continue
        # Thin down for speed — plot at most 20 k points per tissue
        step = max(1, len(ts) // 20_000)
        ax.scatter(ts['time_s'].iloc[::step], ts['phase_deg'].iloc[::step],
                   c=TISSUE_COLORS[t], s=1, alpha=0.4, label=t.capitalize(), rasterized=True)
    ax.set_title(sf, fontsize=9)
    ax.set_ylabel('Phase (deg)', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=7, markerscale=4, frameon=False)

axes5[-1].set_xlabel('Time (s)', fontsize=10)
fig5.suptitle('50 kHz Phase — Time Series by Study & Tissue', fontsize=12, y=1.01)
fig5.tight_layout()
fig5.savefig(OUT_DIR / 'phase50k_timeseries.png', bbox_inches='tight')
print(f'Saved: phase50k_timeseries.png')
plt.show()

# ─── Plot 6: Scatter — Impedance vs Phase, coloured by tissue ────────────────
fig6, ax6 = plt.subplots(figsize=(9, 6), dpi=180)
for t in TISSUE_ORDER:
    sub = full[full['tissue'] == t]
    step = max(1, len(sub) // 30_000)
    ax6.scatter(sub['imp_ohm'].iloc[::step], sub['phase_deg'].iloc[::step],
                c=TISSUE_COLORS[t], s=3, alpha=0.3, label=f'{t.capitalize()} (n={len(sub):,})',
                rasterized=True)
ax6.set_xlabel('Impedance |Z|  (Ω)  @ 50 kHz', fontsize=12)
ax6.set_ylabel('Phase  (deg)  @ 50 kHz', fontsize=12)
ax6.set_title('50 kHz Impedance vs Phase — All Valid Samples', fontsize=11)
ax6.legend(frameon=False, fontsize=10, markerscale=3)
ax6.grid(True, alpha=0.25)
ax6.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
fig6.tight_layout()
fig6.savefig(OUT_DIR / 'phase50k_vs_impedance.png', bbox_inches='tight')
print(f'Saved: phase50k_vs_impedance.png')
plt.show()

print(f'\nAll plots saved to: {OUT_DIR}')
