#!/usr/bin/env python3
"""
Analogous magnitude plot |Z| vs frequency (kHz) for series RC circuit.
Target: 50 kHz, C = 3.183 nF, R = 1 kΩ
"""

import numpy as np
import matplotlib.pyplot as plt

R = 1000.0          # 1 kΩ
f_target = 50e3     # 50 kHz
f_low = 5e3  # 5 kHz
C = 3.183e-9        # 3.183 nF

C_nF = C * 1e9
print(f"For R = {R/1000} kΩ and 45° at {f_target/1e3:.1f} kHz, use C = {C_nF:.3f} nF")

# Print |Z| at key frequencies
for freq in [5e3, 50e3, 100e3]:
    omega = 2 * np.pi * freq
    Z = R + 1 / (1j * omega * C)
    mag = np.abs(Z)
    phase = np.angle(Z, deg=True)
    print(f"At {freq/1e3:.0f} kHz: |Z| = {mag:.1f} Ω, Phase = {phase:.2f}°")

# Frequency sweep
f = np.logspace(3, 5, 1000)
omega = 2 * np.pi * f
Z = R + 1 / (1j * omega * C)
mag = np.abs(Z)
phase_deg = np.angle(Z, deg=True)

# Two-subplot figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Magnitude
ax1.semilogx(f / 1000, mag, 'b-', linewidth=2.5, label='|Z| (Magnitude)')
ax1.axhline(R * np.sqrt(2), color='orange', linestyle='--', linewidth=1.5, label=f'|Z| at 45° ≈ {R*np.sqrt(2):.0f} Ω')
ax1.axvline(f_low / 1000, color='k', linestyle='--', linewidth=1.5, label=f'{f_low/1e3:.0f} kHz target')
ax1.axvline(f_target / 1000, color='g', linestyle='--', linewidth=1.5, label=f'{f_target/1e3:.0f} kHz target')
ax1.grid(True, which='both', ls='--', alpha=0.7)
ax1.set_ylabel('|Z| (Ohms)', fontsize=12)
ax1.set_title(f'Series RC Impedance Magnitude |Z| vs Frequency\nR = 1 kΩ, C = {C_nF:.3f} nF (45° at 50 kHz)', fontsize=14)
ax1.legend(loc='upper right', fontsize=10)

# Phase (reference)
ax2.semilogx(f / 1000, phase_deg, 'purple', linewidth=2.5, label='Phase')
ax2.axhline(-45, color='r', linestyle='--', linewidth=1.5, label='45° target')
ax2.axvline(f_low / 1000, color='k', linestyle='--', linewidth=1.5)
ax2.axvline(f_target / 1000, color='g', linestyle='--', linewidth=1.5)
ax2.grid(True, which='both', ls='--', alpha=0.7)
ax2.set_xlabel('Frequency (kHz)', fontsize=12)
ax2.set_ylabel('Phase (degrees)', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)

# Annotation
fig.text(0.14, 0.18, 
         f'Key values:\n'
         f'At 5 kHz:   |Z| ≈ 10.05 kΩ, Phase ≈ -84.3°\n'
         f'At 50 kHz:  |Z| ≈ 1.41 kΩ, Phase = -45.0°\n'
         f'At 100 kHz: |Z| ≈ 1.12 kΩ, Phase ≈ -26.6°\n\n'
         f'Formula: Z = R + 1/(j 2π f C)\n'
         f'|Z| = sqrt(R² + (1/(2π f C))²)',
         fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.95))

plt.tight_layout()
# plt.savefig('/home/workdir/artifacts/lab_setup/rc_magnitude_plot_50kHz.png', dpi=300, bbox_inches='tight')
# plt.savefig('/home/workdir/artifacts/lab_setup/rc_magnitude_plot_50kHz.pdf', dpi=300, bbox_inches='tight')
# print("Plot saved")
plt.show()