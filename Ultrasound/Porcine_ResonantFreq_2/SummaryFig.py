import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================================================
# Combined data (Air excluded)
# Last week + today (tissue = T2/T2P only)
# ============================================================

data = {
    # Last week
    'Blood_wk':  {'Rm': [390, 382],          'Q': [2300, 2350],
                  'type': 'Blood',  'marker': 's', 'color': 'green'},
    # 'Clot_wk':   {'Rm': [661, 621, 729],     'Q': [1477, 1625, 1418],
    #               'type': 'Clot',   'marker': 'D', 'color': 'red'},
    # 'PA_wk':     {'Rm': [355, 471, 348],     'Q': [2659, 2030, 2697],
    #               'type': 'Tissue', 'marker': '^', 'color': 'blue'},
    # Monday 08.10.2026
    'T2':        {'Rm': [392],  'Q': [2321], 'type': 'Tissue', 'marker': '^', 'color': 'blue'},
    'T2P':       {'Rm': [417],  'Q': [2185], 'type': 'Tissue', 'marker': '^', 'color': 'blue'},
    'C1':        {'Rm': [891],  'Q': [1045], 'type': 'Clot',   'marker': 'D', 'color': 'red'},
    'C2':        {'Rm': [1031], 'Q': [994],  'type': 'Clot',   'marker': 'D', 'color': 'red'},
    'BL1':       {'Rm': [446],  'Q': [2094], 'type': 'Blood',  'marker': 's', 'color': 'green'},
}

# ============================================================
# Figure 1: Combined Rm vs Q scatter
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 6.5))

plotted = set()
for name, d in data.items():
    for r, q in zip(d['Rm'], d['Q']):
        label = d['type'] if d['type'] not in plotted else None
        ax.scatter(r, q, s=140, c=d['color'], marker=d['marker'],
                   edgecolors='k', zorder=5, label=label)
        plotted.add(d['type'])
        ax.annotate(name, (r, q), textcoords='offset points',
                    xytext=(6, 4), fontsize=8)

ax.axvline(550, color='gray', ls='--', lw=1.4, alpha=0.8)
ax.axhline(1800, color='gray', ls='--', lw=1.4, alpha=0.8)
ax.text(580, 1300, 'Clot region\n(Rm > 550 Ω)', fontsize=10,
        color='red', fontweight='bold')
ax.text(300, 2800, 'Blood / Tissue region\n(Rm < 500 Ω)', fontsize=10,
        color='blue', fontweight='bold')

ax.set_xlabel('Motional Resistance Rm (Ω)', fontsize=12)
ax.set_ylabel('Quality Factor Q', fontsize=12)
ax.set_title('Combined Results (Last Week + Today)\nTissue = T2/T2P only (with saline)',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, ls=':', alpha=0.6)
ax.set_xlim(250, 1200)
ax.set_ylim(800, 3200)

plt.tight_layout()
plt.savefig('Combined_Rm_vs_Q_T2only.png', dpi=150, bbox_inches='tight')
print('Saved Combined_Rm_vs_Q_T2only.png')
plt.close()

# ============================================================
# Figure 2: Grouped means ± std
# ============================================================
blood_rm  = [390, 382, 446]
blood_q   = [2300, 2350, 2094]
clot_rm   = [661, 621, 729, 891, 1031]
clot_q    = [1477, 1625, 1418, 1045, 994]
tissue_rm = [355, 471, 348, 392, 417]
tissue_q  = [2659, 2030, 2697, 2321, 2185]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

cats     = ['Blood', 'Tissue', 'Clot']
rm_means = [np.mean(blood_rm), np.mean(tissue_rm), np.mean(clot_rm)]
q_means  = [np.mean(blood_q),  np.mean(tissue_q),  np.mean(clot_q)]
rm_err   = [np.std(blood_rm),  np.std(tissue_rm),  np.std(clot_rm)]
q_err    = [np.std(blood_q),   np.std(tissue_q),   np.std(clot_q)]
cols     = ['green', 'blue', 'red']

axes[0].bar(cats, rm_means, yerr=rm_err, color=cols, edgecolor='k',
            width=0.55, capsize=5)
axes[0].set_ylabel('Rm (Ω)')
axes[0].set_title('Motional Resistance (mean ± std)')
axes[0].axhline(550, color='gray', ls='--', lw=1.2)
axes[0].grid(True, axis='y', ls=':', alpha=0.6)
axes[0].set_ylim(0, 1200)

axes[1].bar(cats, q_means, yerr=q_err, color=cols, edgecolor='k',
            width=0.55, capsize=5)
axes[1].set_ylabel('Q')
axes[1].set_title('Quality Factor (mean ± std)')
axes[1].axhline(1800, color='gray', ls='--', lw=1.2)
axes[1].grid(True, axis='y', ls=':', alpha=0.6)
axes[1].set_ylim(0, 3200)

fig.suptitle('Combined Loaded Samples (T2/T2P only for Tissue)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Combined_Grouped_T2only.png', dpi=150, bbox_inches='tight')
print('Saved Combined_Grouped_T2only.png')
plt.close()

print('Done.')