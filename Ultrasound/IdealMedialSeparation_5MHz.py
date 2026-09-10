import matplotlib.pyplot as plt
import numpy as np

# QCM Output Data
states = ['Air (Baseline)', 'Blood (Liquid)', 'Blood Clot (Gel)', 'Vessel Wall (Contact)']
f_res = np.array([5000005.0, 4999504.8, 4997503.8, 4990030.0])
f_shift_kHz = (f_res - f_res[0]) / 1000.0  # Frequency shift relative to Air (kHz)
peak_g = np.array([99.12, 6.67, 1.54, 4.00])
q_factor = np.array([55527.8, 3162.7, 726.0, 3694.5])

colors = ['#7f7f7f', '#d62728', '#8c564b', '#1f77b4']

fig = plt.figure(figsize=(13, 5.5))

# Graph 1: 2D Feature Space - Peak G vs. Q-Factor
ax1 = fig.add_subplot(1, 2, 1)
for i, state in enumerate(states):
    ax1.scatter(q_factor[i], peak_g[i], color=colors[i], s=130, zorder=4)
    ax1.annotate(f'  {state}', (q_factor[i], peak_g[i]), fontsize=10, fontweight='bold', va='center')

ax1.set_xlabel('Quality Factor (Q) [Log Scale]', fontsize=11)
ax1.set_ylabel('Peak Conductance G (mS) [Log Scale]', fontsize=11)
ax1.set_title('Dissipation vs. Damping: Peak G vs. Q-Factor', fontsize=12, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(400, 100000)
ax1.set_ylim(1, 150)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# Graph 2: 3D Feature Space - (Frequency Shift, Peak G, Q-Factor)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for i, state in enumerate(states):
    ax2.scatter(f_shift_kHz[i], np.log10(peak_g[i]), np.log10(q_factor[i]), color=colors[i], s=120)
    ax2.text(f_shift_kHz[i], np.log10(peak_g[i]), np.log10(q_factor[i]) + 0.08, f' {state}', fontsize=9, fontweight='bold')

ax2.set_xlabel(r'Resonance Shift $\Delta f$ (kHz)', fontsize=9, labelpad=10)
ax2.set_ylabel(r'$\log_{10}(\text{Peak G [mS]})$', fontsize=9, labelpad=10)
ax2.set_zlabel(r'$\log_{10}(\text{Q-Factor})$', fontsize=9, labelpad=10)
ax2.set_title(r'3D Feature Space ($\Delta f$, Peak G, Q)', fontsize=12, fontweight='bold')
ax2.view_init(elev=22, azim=130)

plt.tight_layout()
plt.show()