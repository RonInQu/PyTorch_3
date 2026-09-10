import matplotlib.pyplot as plt
import numpy as np

# Data excluding Air baseline
states = ['Blood (Liquid)', 'Blood Clot (Gel)', 'Vessel Wall (Contact)']
f_res = np.array([4999509, 4997508, 4990025])
f_air = 5000005.0
f_shift_kHz = (f_res - f_air) / 1000.0  # Resonance shift relative to Air in kHz
peak_g = np.array([6.67, 1.54, 4.00])    # Peak Conductance G (mS)
q_factor = np.array([3162.7, 728.0, 1903])

# Motional Resistance R_m (ohms) = 1000 / G (mS)
r_m = 1000.0 / peak_g

# colors = ['#d62728', '#8c564b', '#1f77b4']
colors = ['green', 'red', 'blue']

fig = plt.figure(figsize=(13, 5.5))

# Plot 1: Quality Factor vs. Motional Resistance R_m (Linear Axes)
ax1 = fig.add_subplot(1, 2, 1)
for i, state in enumerate(states):
    ax1.scatter(r_m[i], q_factor[i], color=colors[i], s=140, zorder=4)

ax1.annotate('Blood (Liquid)\n$R_m = 149.9\ \Omega, Q = 3,163$', (r_m[0], q_factor[0]), 
             xytext=(160, 2800), fontsize=9.5, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax1.annotate('Vessel Wall (Contact)\n$R_m = 250.0\ \Omega, Q = 3,695$', (r_m[2], q_factor[2]), 
             xytext=(280, 3500), fontsize=9.5, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax1.annotate('Blood Clot (Gel)\n$R_m = 649.4\ \Omega, Q = 726$', (r_m[1], q_factor[1]), 
             xytext=(480, 1100), fontsize=9.5, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax1.set_xlabel(r'Motional Resistance $R_m\ (\Omega)$', fontsize=11)
ax1.set_ylabel('Quality Factor (Q)', fontsize=11)
ax1.set_title(r'Quality Factor vs. Motional Resistance ($R_m = 1/G$)', fontsize=12, fontweight='bold')
ax1.set_xlim(100, 750)
ax1.set_ylim(500, 4200)
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot 2: 3D Feature Space without Air (Linear Axes)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for i, state in enumerate(states):
    ax2.scatter(f_shift_kHz[i], peak_g[i], q_factor[i], color=colors[i], s=120)

ax2.text(f_shift_kHz[0], peak_g[0], q_factor[0] + 150, ' Blood (Liquid)', fontsize=9, fontweight='bold')
ax2.text(f_shift_kHz[1], peak_g[1], q_factor[1] + 150, ' Blood Clot (Gel)', fontsize=9, fontweight='bold')
ax2.text(f_shift_kHz[2], peak_g[2], q_factor[2] + 150, ' Vessel Wall (Contact)', fontsize=9, fontweight='bold')

ax2.set_xlabel(r'$\Delta f$ (kHz)', fontsize=9, labelpad=10)
ax2.set_ylabel('Peak G (mS)', fontsize=9, labelpad=10)
ax2.set_zlabel('Q-Factor', fontsize=9, labelpad=10)
ax2.set_title(r'3D Feature Space ($\Delta f$, Peak G, Q)', fontsize=12, fontweight='bold')
ax2.view_init(elev=20, azim=130)

plt.tight_layout()
plt.show()