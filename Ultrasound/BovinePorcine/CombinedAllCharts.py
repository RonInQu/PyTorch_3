import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Data – exactly the points shown in the Combined Rm–Q scatter
# ============================================================
data = {
    # Last week
    'Blood_wk':  {'Rm': [390, 382],          'Q': [2300, 2350],
                  'type': 'Blood',  'marker': 's', 'color': 'green'},
    'Clot_wk':   {'Rm': [661, 621, 729],     'Q': [1477, 1625, 1418],
                  'type': 'Clot',   'marker': 'D', 'color': 'red'},
    'PA_wk':     {'Rm': [355, 471, 348],     'Q': [2659, 2030, 2697],
                  'type': 'Tissue', 'marker': '^', 'color': 'blue'},
    # Monday 08.10.2026
    'T2':        {'Rm': [392],  'Q': [2321], 'type': 'Tissue', 'marker': '^', 'color': 'blue'},
    'T2P':       {'Rm': [417],  'Q': [2185], 'type': 'Tissue', 'marker': '^', 'color': 'blue'},
    'C1':        {'Rm': [891],  'Q': [1045], 'type': 'Clot',   'marker': 'D', 'color': 'red'},
    'C2':        {'Rm': [1031], 'Q': [994],  'type': 'Clot',   'marker': 'D', 'color': 'red'},
    'BL1':       {'Rm': [446],  'Q': [2094], 'type': 'Blood',  'marker': 's', 'color': 'green'},
    # Thursday 08.13.2026
    'BBL1':      {'Rm': [345],  'Q': [2519], 'type': 'Blood',  'marker': 's', 'color': 'green'},
    'BC1':       {'Rm': [386],  'Q': [2091], 'type': 'Clot',   'marker': 'D', 'color': 'red'},
    'BC2':       {'Rm': [518],  'Q': [1613], 'type': 'Clot',   'marker': 'D', 'color': 'red'},
    'PCBL1':     {'Rm': [422],  'Q': [2063], 'type': 'Blood',  'marker': 's', 'color': 'green'},
    'PC1':       {'Rm': [439],  'Q': [1997], 'type': 'Clot',   'marker': 'D', 'color': 'red'},
    'PC2':       {'Rm': [640],  'Q': [1338], 'type': 'Clot',   'marker': 'D', 'color': 'red'},
    'PT1':       {'Rm': [317],  'Q': [2713], 'type': 'Tissue', 'marker': '^', 'color': 'blue'},
    'PT2':       {'Rm': [338],  'Q': [2688], 'type': 'Tissue', 'marker': '^', 'color': 'blue'},
    'PT1p':      {'Rm': [449],  'Q': [1878], 'type': 'Tissue', 'marker': '^', 'color': 'blue'},
    'PT2p':      {'Rm': [474],  'Q': [1821], 'type': 'Tissue', 'marker': '^', 'color': 'blue'},
}

# Arrays for box / bar charts (matched to scatter)
blood_rm  = np.array([390, 382, 446, 345, 422])
blood_q   = np.array([2300, 2350, 2094, 2519, 2063])
clot_rm   = np.array([891, 1031, 386, 518, 439, 640])
clot_q    = np.array([1045, 994, 2091, 1613, 1997, 1338])
tissue_rm = np.array([392, 417, 317, 338, 449, 474])
tissue_q  = np.array([2321, 2185, 2713, 2688, 1878, 1821])

type_colors = {'Blood': '#2ca02c', 'Clot': '#d62728', 'Tissue': '#1f77b4'}

# ============================================================
# 1. Combined Rm vs Q scatter
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

plotted = set()
for name, d in data.items():
    for r, q in zip(d['Rm'], d['Q']):
        label = d['type'] if d['type'] not in plotted else None
        ax.scatter(r, q, s=140, c=d['color'], marker=d['marker'],
                   edgecolors='k', linewidths=0.7, zorder=5, label=label)
        plotted.add(d['type'])
        ax.annotate(name, (r, q), textcoords='offset points',
                    xytext=(5, 4), fontsize=7.5, alpha=0.9)

ax.axvline(500, color='gray', ls='--', lw=1.4, alpha=0.8)
ax.axhline(1700, color='gray', ls='--', lw=1.4, alpha=0.8)
ax.text(580, 1200, 'Clot region\n(Rm > 500 Ω)', fontsize=10,
        color='red', fontweight='bold')
ax.text(280, 2900, 'Blood / Tissue region\n(Rm < 500 Ω)', fontsize=10,
        color='blue', fontweight='bold')

ax.set_xlabel('Motional Resistance Rm (Ω)', fontsize=12)
ax.set_ylabel('Quality Factor Q', fontsize=12)
ax.set_title('Combined Results (Last Week + Mon 08.10 + Thu 08.13)\nAll Blood / Clot / Tissue samples',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, ls=':', alpha=0.6)
ax.set_xlim(250, 1150)
ax.set_ylim(800, 3100)
plt.tight_layout()
plt.savefig('Combined_Rm_vs_Q_all.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 2. Grouped mean ± std bars
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))

cats     = ['Blood', 'Clot', 'Tissue']
rm_means = [np.mean(blood_rm), np.mean(clot_rm), np.mean(tissue_rm)]
q_means  = [np.mean(blood_q),  np.mean(clot_q),  np.mean(tissue_q)]
rm_err   = [np.std(blood_rm),  np.std(clot_rm),  np.std(tissue_rm)]
q_err    = [np.std(blood_q),   np.std(clot_q),   np.std(tissue_q)]
cols     = [type_colors[c] for c in cats]

axes[0].bar(cats, rm_means, yerr=rm_err, color=cols, edgecolor='k',
            width=0.55, capsize=5, error_kw=dict(lw=1.2))
axes[0].set_ylabel(r'$R_m$ (Ω)')
axes[0].set_title(r'Motional Resistance $R_m$ (mean ± std)')
axes[0].axhline(500, color='gray', ls='--', lw=1.1)
axes[0].set_ylim(0, 1050)
axes[0].grid(axis='y', ls=':', alpha=0.5)

axes[1].bar(cats, q_means, yerr=q_err, color=cols, edgecolor='k',
            width=0.55, capsize=5, error_kw=dict(lw=1.2))
axes[1].set_ylabel('Q')
axes[1].set_title('Quality Factor Q (mean ± std)')
axes[1].axhline(1700, color='gray', ls='--', lw=1.1)
axes[1].set_ylim(0, 3000)
axes[1].grid(axis='y', ls=':', alpha=0.5)

fig.suptitle('Summary by Media Type – mean ± std (matched to scatter)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('QCM_bar_Rm_Q_matched.png', dpi=160, bbox_inches='tight')
plt.close()

# ============================================================
# 3. Box + strip plots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
fig.suptitle('Summary by Media Type (points from Combined Rm–Q chart only)',
             fontsize=13, fontweight='bold')

rng = np.random.default_rng(42)

# Rm
ax = axes[0]
data_rm = [blood_rm, clot_rm, tissue_rm]
bp = ax.boxplot(data_rm, tick_labels=['Blood', 'Clot', 'Tissue'],
                patch_artist=True, widths=0.55, showfliers=False)
for patch, t in zip(bp['boxes'], ['Blood', 'Clot', 'Tissue']):
    patch.set_facecolor(type_colors[t])
    patch.set_alpha(0.55)
    patch.set_edgecolor('k')
for med in bp['medians']:
    med.set_color('darkorange')
    med.set_linewidth(2)
for i, (vals, t) in enumerate(zip(data_rm, ['Blood', 'Clot', 'Tissue'])):
    x = rng.normal(i + 1, 0.06, size=len(vals))
    ax.scatter(x, vals, c=type_colors[t], s=60, edgecolors='k',
               linewidths=0.6, zorder=3)
ax.set_ylabel(r'$R_m$ (Ω)', fontsize=12)
ax.set_ylim(250, 1150)
ax.axhline(500, color='gray', ls='--', lw=1.1, alpha=0.7)
ax.grid(axis='y', alpha=0.3)
ax.set_title(r'Motional Resistance $R_m$', fontsize=12)

# Q
ax = axes[1]
data_q = [blood_q, clot_q, tissue_q]
bp = ax.boxplot(data_q, tick_labels=['Blood', 'Clot', 'Tissue'],
                patch_artist=True, widths=0.55, showfliers=False)
for patch, t in zip(bp['boxes'], ['Blood', 'Clot', 'Tissue']):
    patch.set_facecolor(type_colors[t])
    patch.set_alpha(0.55)
    patch.set_edgecolor('k')
for med in bp['medians']:
    med.set_color('darkorange')
    med.set_linewidth(2)
for i, (vals, t) in enumerate(zip(data_q, ['Blood', 'Clot', 'Tissue'])):
    x = rng.normal(i + 1, 0.06, size=len(vals))
    ax.scatter(x, vals, c=type_colors[t], s=60, edgecolors='k',
               linewidths=0.6, zorder=3)
ax.set_ylabel('Q', fontsize=12)
ax.set_ylim(800, 3000)
ax.axhline(1700, color='gray', ls='--', lw=1.1, alpha=0.7)
ax.grid(axis='y', alpha=0.3)
ax.set_title('Quality Factor Q', fontsize=12)

plt.tight_layout()
plt.savefig('QCM_boxplot_Rm_Q_matched.png', dpi=160, bbox_inches='tight')
plt.close()

print('All charts saved:')
print('  Combined_Rm_vs_Q_all.png')
print('  QCM_bar_Rm_Q_matched.png')
print('  QCM_boxplot_Rm_Q_matched.png')
print()
print(f'Blood  (n=5)  Rm = {np.mean(blood_rm):.0f} ± {np.std(blood_rm):.0f}   Q = {np.mean(blood_q):.0f} ± {np.std(blood_q):.0f}')
print(f'Clot   (n=6)  Rm = {np.mean(clot_rm):.0f} ± {np.std(clot_rm):.0f}   Q = {np.mean(clot_q):.0f} ± {np.std(clot_q):.0f}')
print(f'Tissue (n=6)  Rm = {np.mean(tissue_rm):.0f} ± {np.std(tissue_rm):.0f}   Q = {np.mean(tissue_q):.0f} ± {np.std(tissue_q):.0f}')