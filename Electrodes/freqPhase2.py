#!/usr/bin/env python3
"""
RC phase plot for 50 kHz target (C = 3.183 nF) with phases at 5 & 100 kHz.
"""

import numpy as np
import matplotlib.pyplot as plt

R = 1000.0  # 1 kΩ
f_target = 50e3   # 50 kHz target
f_low = 5e3  # 5 kHz
C = 3.183e-9      # 3.183 nF (for exact 45° at 50 kHz)
C_nF = C * 1e9

print(f"For R = {R/1000} kΩ and 45° at {f_target/1e3:.1f} kHz, use C = {C_nF:.3f} nF")

# Phase at specific frequencies
for freq in [5e3, 100e3]:
    omega = 2 * np.pi * freq
    Z = R + 1 / (1j * omega * C)
    phase = np.angle(Z, deg=True)
    print(f"Phase at {freq/1e3:.0f} kHz: {phase:.2f}°")

# Bode plot
f = np.logspace(3, 5, 1000)
omega = 2 * np.pi * f
Z = R + 1 / (1j * omega * C)
phase_deg = np.angle(Z, deg=True)

plt.figure(figsize=(10, 6))
plt.semilogx(f / 1000, phase_deg, 'b-', linewidth=2.5, label='Phase Shift')
plt.axhline(-45, color='r', linestyle='--', linewidth=1.5, label='-45° target')
plt.axvline(f_low / 1000, color='k', linestyle='--', linewidth=1.5, label=f'{f_low/1e3:.0f} kHz target')
plt.axvline(f_target / 1000, color='g', linestyle='--', linewidth=1.5, label=f'{f_target/1e3:.0f} kHz target')

plt.grid(True, which='both', ls='--', alpha=0.7)
plt.xlabel('Frequency (kHz)', fontsize=12)
plt.ylabel('Phase (degrees)', fontsize=12)
plt.title(f'Series RC Phase Response (R in series with C)\nR = 1 kΩ, C = {C_nF:.3f} nF for 45° at 50 kHz', fontsize=14)

plt.text(0.02, 0.45, f'Required C for 45° at 50 kHz: {C_nF:.3f} nF\n\n'
         f'Phase at 5 kHz: -84.29°\nPhase at 100 kHz: -26.57°',
         transform=plt.gca().transAxes, fontsize=11,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.95, edgecolor="brown"))

plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()

# plt.savefig('/home/workdir/artifacts/lab_setup/rc_phase_plot_50kHz.png', dpi=300, bbox_inches='tight')
# plt.savefig('/home/workdir/artifacts/lab_setup/rc_phase_plot_50kHz.pdf', dpi=300, bbox_inches='tight')
# print("Plot saved as PNG and PDF")
plt.show()