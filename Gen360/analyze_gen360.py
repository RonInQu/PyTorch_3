# -*- coding: utf-8 -*-
"""
Gen360 Multi-Frequency Impedance & Phase Analysis
===================================================
Loads all LOG3_solo_*.parquet files from Gen360 folder.
Labels tissue from solo_led_state_i:  4=blood, 5=clot, 9=wall
Channel mapping (confirmed):
    imp_mag_adj_0_ohm / imp_pha_0_millideg  =>  50 kHz
    imp_mag_adj_1_ohm / imp_pha_1_millideg  =>  100 kHz
    imp_mag_adj_2_ohm / imp_pha_2_millideg  =>  12.5 kHz

Outputs:
    gen360_combined_impedance.png
    gen360_combined_phase.png
    gen360_summary_stats.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path(r'c:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Gen360')
OUT_DIR  = DATA_DIR / 'combined_plots'
OUT_DIR.mkdir(exist_ok=True)

STATE_TO_TISSUE = {4: 'blood', 5: 'clot', 9: 'wall'}

# (column_index, frequency_kHz, label)
# Confirmed mapping: _0 = 50 kHz, _1 = 100 kHz, _2 = 12.5 kHz
FREQ_CHANNELS = [
    (2, 12.5,  '12.5 kHz'),
    (0, 50.0,  '50 kHz'),
    (1, 100.0, '100 kHz'),
]

TISSUE_COLORS  = {'blood': '#2ca02c', 'clot': '#d62728', 'wall': '#1f77b4'}
TISSUE_MARKERS = {'blood': 'o',       'clot': 's',       'wall': '^'}
TISSUE_ORDER   = ['blood', 'clot', 'wall']

SATURATION_100K = 1_048_575   # 2^20 - 1  (flag as invalid if hit)

# ─── Load & label all files ───────────────────────────────────────────────────
files = sorted(DATA_DIR.glob('LOG3_solo_*.parquet'))
print(f'Found {len(files)} parquet file(s):\n  ' + '\n  '.join(f.name for f in files))

NEEDED_COLS = [
    'solo_led_state_i',
    'imp_mag_adj_0_ohm', 'imp_pha_0_millideg',
    'imp_mag_adj_1_ohm', 'imp_pha_1_millideg',
    'imp_mag_adj_2_ohm', 'imp_pha_2_millideg',
]

frames = []
for p in files:
    print(f'  Loading {p.name} ...', end=' ', flush=True)
    df = pd.read_parquet(p, columns=NEEDED_COLS)
    df['tissue'] = df['solo_led_state_i'].map(STATE_TO_TISSUE).fillna('other')
    df['_file'] = p.name
    frames.append(df)
    print(f'{len(df):,} rows')

full = pd.concat(frames, ignore_index=True)
full = full[full['tissue'].isin(TISSUE_ORDER)].copy()
print(f'\nTotal labelled rows: {len(full):,}')
print(full['tissue'].value_counts().to_string())

# ─── Per-channel summary (valid samples only, no zeros, no saturation) ────────
rows = []
for idx, freq_khz, freq_label in FREQ_CHANNELS:
    mag_col = f'imp_mag_adj_{idx}_ohm'
    pha_col = f'imp_pha_{idx}_millideg'

    for tissue in TISSUE_ORDER:
        sub = full[full['tissue'] == tissue].copy()

        mag = sub[mag_col]
        pha = sub[pha_col] / 1000.0      # millideg -> deg

        # Keep only valid (non-zero, non-saturated) measurements
        valid = (mag > 0) & (mag < SATURATION_100K)
        mag = mag[valid]
        pha = pha[valid]

        if mag.empty:
            continue

        rows.append({
            'tissue':        tissue,
            'freq_label':    freq_label,
            'freq_khz':      freq_khz,
            'n_total':       len(sub),
            'n_valid':       len(mag),
            'valid_frac':    len(mag) / len(sub),
            'imp_median':    mag.median(),
            'imp_mean':      mag.mean(),
            'imp_q25':       mag.quantile(0.25),
            'imp_q75':       mag.quantile(0.75),
            'imp_p10':       mag.quantile(0.10),
            'imp_p90':       mag.quantile(0.90),
            'phase_median':  pha.median(),
            'phase_mean':    pha.mean(),
            'phase_q25':     pha.quantile(0.25),
            'phase_q75':     pha.quantile(0.75),
        })

summary = pd.DataFrame(rows).sort_values(['tissue', 'freq_khz'])

# ─── Print summary table ──────────────────────────────────────────────────────
print('\n' + '='*70)
print('SUMMARY  (valid samples only — zeros and saturation excluded)')
print('='*70)
pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)
print(summary[['tissue','freq_label','n_valid','imp_median','imp_q25','imp_q75',
               'phase_median','phase_q25','phase_q75']].to_string(index=False))

# ─── Discriminability ratios ──────────────────────────────────────────────────
print('\n' + '-'*50)
print('DISCRIMINABILITY RATIOS  (clot/blood, wall/blood, clot/wall)')
print('-'*50)
pivot = summary.pivot_table(index='freq_label', columns='tissue', values='imp_median')
pivot = pivot.reindex(index=[fl for _, _, fl in FREQ_CHANNELS])
pivot['clot/blood'] = pivot['clot'] / pivot['blood']
pivot['wall/blood'] = pivot['wall'] / pivot['blood']
pivot['clot/wall']  = pivot['clot'] / pivot['wall']
print(pivot[['blood','clot','wall','clot/blood','wall/blood','clot/wall']].round(2).to_string())

# ─── Save CSV ─────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / 'gen360_summary_stats.csv'
summary.to_csv(csv_path, index=False, float_format='%.2f')
print(f'\nSaved summary CSV: {csv_path}')

# ─── Plot 1: Combined Impedance ───────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=180)

for tissue in TISSUE_ORDER:
    sub = summary[summary['tissue'] == tissue].sort_values('freq_khz')
    freq = sub['freq_khz'].values
    ax1.plot(freq, sub['imp_median'], marker=TISSUE_MARKERS[tissue],
             lw=2.4, ms=8, color=TISSUE_COLORS[tissue], label=tissue.capitalize(), zorder=3)
    ax1.fill_between(freq, sub['imp_q25'], sub['imp_q75'],
                     color=TISSUE_COLORS[tissue], alpha=0.15)

ax1.set_xticks([12.4, 50, 100])
ax1.set_xticklabels(['12.5 kHz', '50 kHz', '100 kHz'])
ax1.set_xlim(7, 110)
ax1.set_xlabel('Frequency', fontsize=12)
ax1.set_ylabel('Impedance |Z|  (Ω)', fontsize=12)
ax1.set_title('Gen360 — Impedance by Tissue & Frequency\n(median ± IQR shading, all studies combined)', fontsize=11)
ax1.legend(frameon=False, fontsize=11)
ax1.grid(True, alpha=0.25)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
fig1.tight_layout()
imp_path = OUT_DIR / 'gen360_combined_impedance.png'
fig1.savefig(imp_path, bbox_inches='tight')
print(f'Saved: {imp_path}')
plt.show()

# ─── Plot 2: Combined Phase ───────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8.5, 5.5), dpi=180)

for tissue in TISSUE_ORDER:
    sub = summary[summary['tissue'] == tissue].sort_values('freq_khz')
    freq = sub['freq_khz'].values
    ax2.plot(freq, sub['phase_median'], marker=TISSUE_MARKERS[tissue],
             lw=2.4, ms=8, color=TISSUE_COLORS[tissue], label=tissue.capitalize(), zorder=3)
    ax2.fill_between(freq, sub['phase_q25'], sub['phase_q75'],
                     color=TISSUE_COLORS[tissue], alpha=0.15)

ax2.set_xticks([12.4, 50, 100])
ax2.set_xticklabels(['12.5 kHz', '50 kHz', '100 kHz'])
ax2.set_xlim(7, 110)
ax2.set_xlabel('Frequency', fontsize=12)
ax2.set_ylabel('Phase  (deg)', fontsize=12)
ax2.set_title('Gen360 — Phase by Tissue & Frequency\n(median ± IQR shading, all studies combined)', fontsize=11)
ax2.legend(frameon=False, fontsize=11)
ax2.grid(True, alpha=0.25)
fig2.tight_layout()
pha_path = OUT_DIR / 'gen360_combined_phase.png'
fig2.savefig(pha_path, bbox_inches='tight')
print(f'Saved: {pha_path}')
plt.show()

# ─── Plot 3: Impedance — data points only (no shading) ───────────────────────
fig3, ax3 = plt.subplots(figsize=(8.5, 5.5), dpi=180)

for tissue in TISSUE_ORDER:
    sub = summary[summary['tissue'] == tissue].sort_values('freq_khz')
    ax3.plot(sub['freq_khz'], sub['imp_median'], marker=TISSUE_MARKERS[tissue],
             lw=2.4, ms=9, color=TISSUE_COLORS[tissue], label=tissue.capitalize(), zorder=3)

ax3.set_xticks([12.5, 50, 100])
ax3.set_xticklabels(['12.5 kHz', '50 kHz', '100 kHz'])
ax3.set_xlim(7, 110)
ax3.set_xlabel('Frequency', fontsize=12)
ax3.set_ylabel('Impedance |Z|  (Ω)', fontsize=12)
ax3.set_title('Gen360 — Impedance by Tissue & Frequency\n(median, all studies combined)', fontsize=11)
ax3.legend(frameon=False, fontsize=11)
ax3.grid(True, alpha=0.25)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
fig3.tight_layout()
imp_pts_path = OUT_DIR / 'gen360_impedance_points.png'
fig3.savefig(imp_pts_path, bbox_inches='tight')
print(f'Saved: {imp_pts_path}')
plt.show()

# ─── Plot 4: Phase — data points only (no shading) ───────────────────────────
fig4, ax4 = plt.subplots(figsize=(8.5, 5.5), dpi=180)

for tissue in TISSUE_ORDER:
    sub = summary[summary['tissue'] == tissue].sort_values('freq_khz')
    ax4.plot(sub['freq_khz'], sub['phase_median'], marker=TISSUE_MARKERS[tissue],
             lw=2.4, ms=9, color=TISSUE_COLORS[tissue], label=tissue.capitalize(), zorder=3)

ax4.set_xticks([12.5, 50, 100])
ax4.set_xticklabels(['12.5 kHz', '50 kHz', '100 kHz'])
ax4.set_xlim(7, 110)
ax4.set_xlabel('Frequency', fontsize=12)
ax4.set_ylabel('Phase  (deg)', fontsize=12)
ax4.set_title('Gen360 — Phase by Tissue & Frequency\n(median, all studies combined)', fontsize=11)
ax4.legend(frameon=False, fontsize=11)
ax4.grid(True, alpha=0.25)
fig4.tight_layout()
pha_pts_path = OUT_DIR / 'gen360_phase_points.png'
fig4.savefig(pha_pts_path, bbox_inches='tight')
print(f'Saved: {pha_pts_path}')
plt.show()
