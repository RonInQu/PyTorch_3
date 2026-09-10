import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ============================================================
# Data (from ideal medial separation table)
# ============================================================
rm = {
    5:  np.array([150, 650, 250]),    # Blood, Clot, Wall
    10: np.array([212, 919, 353]),
    20: np.array([300, 1300, 500]),
}
q = {
    5:  np.array([3162, 728, 1903]),
    10: np.array([4542, 1027, 2710]),
    20: np.array([6406, 1460, 3836]),
}

colors  = ['green', 'red', 'blue']          # Blood, Clot, Wall
markers = {5: 'o', 10: 's', 20: '^'}        # circle, square, triangle
sizes   = {5: 140, 10: 140, 20: 160}

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 7))

for f in [5, 10, 20]:
    for i in range(3):
        ax.scatter(rm[f][i], q[f][i],
                   c=colors[i], marker=markers[f], s=sizes[f],
                   edgecolors='k', linewidths=0.8, zorder=5)

# Dashed lines connecting frequency progression for each media
for i, c in enumerate(colors):
    xs = [rm[5][i], rm[10][i], rm[20][i]]
    ys = [q[5][i],  q[10][i],  q[20][i]]
    ax.plot(xs, ys, color=c, ls='--', lw=1.2, alpha=0.55, zorder=2)

# Annotations
ax.annotate('Blood 5 MHz\n$R_m=150, Q=3162$', (150, 3162),
            xytext=(60, 2700), fontsize=8.5, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
ax.annotate('Clot 5 MHz\n$R_m=650, Q=728$', (650, 728),
            xytext=(480, 1050), fontsize=8.5, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
ax.annotate('Wall 5 MHz\n$R_m=250, Q=1903$', (250, 1903),
            xytext=(300, 1550), fontsize=8.5, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))

ax.annotate('20 MHz', (300, 6406), xytext=(340, 6100), fontsize=8, color='green')
ax.annotate('20 MHz', (1300, 1460), xytext=(1100, 1700), fontsize=8, color='red')
ax.annotate('20 MHz', (500, 3836), xytext=(540, 3600), fontsize=8, color='blue')

# Legends
legend_media = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
           markersize=11, markeredgecolor='k', label='Blood (Liquid)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=11, markeredgecolor='k', label='Blood Clot (Gel)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markersize=11, markeredgecolor='k', label='Vessel Wall (Contact)'),
]
legend_freq = [
    Line2D([0], [0], marker='o', color='gray', linestyle='None',
           markersize=11, markeredgecolor='k', label='5 MHz (circle)'),
    Line2D([0], [0], marker='s', color='gray', linestyle='None',
           markersize=11, markeredgecolor='k', label='10 MHz (square)'),
    Line2D([0], [0], marker='^', color='gray', linestyle='None',
           markersize=12, markeredgecolor='k', label='20 MHz (triangle)'),
]

leg1 = ax.legend(handles=legend_media, loc='upper right', title='Media', fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=legend_freq, loc='center right', title='Frequency', fontsize=9)

ax.set_xlabel(r'Motional Resistance $R_m$ (Ω)', fontsize=12)
ax.set_ylabel('Quality Factor Q', fontsize=12)
ax.set_title('Q vs $R_m$ – All Frequencies (5 / 10 / 20 MHz)\n'
             'Ideal Medial Separation Data',
             fontsize=13, fontweight='bold')
ax.grid(True, ls='--', alpha=0.45)
ax.set_xlim(50, 1450)
ax.set_ylim(400, 7000)

plt.tight_layout()
plt.savefig('Q_vs_Rm_all_frequencies.png', dpi=160, bbox_inches='tight')
plt.show()