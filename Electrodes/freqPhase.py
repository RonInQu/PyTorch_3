#!/usr/bin/env python3
"""
Updated RC phase plot: x-axis in kHz + capacitance annotation on figure.
"""

import numpy as np
import matplotlib.pyplot as plt

R = 1000.0  # ohms (1 kΩ)
f_target = 50e3  # <<< CHANGE THIS to your desired frequency in Hz (e.g. 50e3, 100e3, 150e3)
f_low = 5e3 # 5 kHz

C = 1 / (2 * np.pi * f_target * R)
C_nF = C * 1e9
print(f"For R = {R/1000} kΩ and 45° at {f_target/1e3:.1f} kHz, use C = {C_nF:.3f} nF")

# Bode plot
f = np.logspace(3, 5, 1000)  # 100 Hz to 1 MHz
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
plt.title(f'Series RC Phase Response\nR = 1 kΩ, C = {C_nF:.3f} nF for 45° at {f_target/1e3:.0f} kHz', fontsize=14)

# Capacitance annotation box
plt.text(0.02, 0.45, f'Required Capacitance for exactly 45° phase at {f_target/1e3:.0f} kHz:\n'
         f'C = {C_nF:.3f} nF\n\n'
         f'Formula: C = 1 / (2π × f × R)',
         transform=plt.gca().transAxes, fontsize=11,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.95, edgecolor="brown"))

plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()

# Save outputs
# plt.savefig('/home/workdir/artifacts/lab_setup/rc_phase_plot_kHz.png', dpi=300, bbox_inches='tight')
# plt.savefig('/home/workdir/artifacts/lab_setup/rc_phase_plot_kHz.pdf', dpi=300, bbox_inches='tight')
# print("Plot saved as PNG and PDF in /home/workdir/artifacts/lab_setup/")
plt.show()