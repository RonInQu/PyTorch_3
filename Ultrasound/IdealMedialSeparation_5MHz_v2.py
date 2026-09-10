import matplotlib.pyplot as plt
import numpy as np

# Data from QCM output table
states = ['Air (Baseline)', 'Blood (Liquid)', 'Blood Clot (Gel)', 'Vessel Wall (Contact)']
peak_g = np.array([99.12, 6.67, 1.54, 4.00])  # Peak Conductance G (mS)
q_factor = np.array([55527.8, 3162.7, 726.0, 3694.5])

# Motional resistance R_m (ohms) = 1 / G (siemens) = 1000 / G (mS)
r_m = 1000.0 / peak_g

colors = ['#7f7f7f', '#d62728', '#8c564b', '#1f77b4']

fig, ax = plt.subplots(figsize=(8, 5.5))

# Plot scatter points
for i, state in enumerate(states):
    ax.scatter(r_m[i], q_factor[i], color=colors[i], s=140, zorder=4)

# Annotation callouts with offsets to avoid overlapping text
offsets = [
    (12, 55528, 'Air (Baseline)\n$R_m = 10.1\ \Omega, Q = 55,528$', 'left', 'center'),
    (135, 2300, 'Blood (Liquid)\n$R_m = 149.9\ \Omega, Q = 3,163$', 'right', 'top'),
    (280, 4800, 'Vessel Wall (Contact)\n$R_m = 250.0\ \Omega, Q = 3,695$', 'left', 'bottom'),
    (580, 520, 'Blood Clot (Gel)\n$R_m = 649.4\ \Omega, Q = 726$', 'right', 'top')
]

for i, (x_ann, y_ann, txt, ha, va) in enumerate(offsets):
    ax.annotate(txt, (r_m[i], q_factor[i]), xytext=(x_ann, y_ann),
                textcoords='data',
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.7),
                fontsize=9.5, fontweight='bold', ha=ha, va=va)

ax.set_xlabel(r'Motional Resistance $R_m\ (\Omega)$ [Log Scale]', fontsize=11)
ax.set_ylabel('Quality Factor (Q) [Log Scale]', fontsize=11)
ax.set_title(r'Quality Factor vs. Motional Resistance ($R_m = 1 / G$)', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(5, 1500)
ax.set_ylim(300, 100000)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()