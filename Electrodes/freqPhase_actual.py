#!/usr/bin/env python3
"""
Fixed version for your 1 kΩ + 3 nF series RC.
Accurate dynamic text box and cleaned legend.
"""

import numpy as np
import matplotlib.pyplot as plt

R = 1000.0          # 1 kΩ
C = 3.0e-9          # Your 3 nF capacitor
f_target = 50e3     # 50 kHz target
f_low = 5e3         # 5 kHz


C_nF = C * 1e9
print(f"Using R = {R/1000:.1f} kΩ and C = {C_nF:.3f} nF")

# Exact frequency for -45° (for reference)
f_45 = 1 / (2 * np.pi * R * C)
print(f"Exact frequency for -45° phase: {f_45/1e3:.2f} kHz")

# Print key values
key_freqs = [5e3, 50e3, 100e3]
print("\nCalculated values:")
for freq in key_freqs:
    omega = 2 * np.pi * freq
    Z = R + 1 / (1j * omega * C)
    mag = np.abs(Z)
    phase = np.angle(Z, deg=True)
    print(f"  {freq/1e3:5.0f} kHz → |Z| = {mag:8.1f} Ω   Phase = {phase:7.2f}°")

# Frequency sweep
f = np.logspace(3, 5, 1000)   # 1 kHz to 100 kHz
omega = 2 * np.pi * f
Z = R + 1 / (1j * omega * C)
mag = np.abs(Z)
phase_deg = np.angle(Z, deg=True)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# Magnitude
ax1.semilogx(f / 1000, mag, 'b-', linewidth=2.5, label='|Z| Magnitude')
ax1.axvline(f_target / 1000, color='g', linestyle='--', linewidth=1.5, label=f'{f_target/1e3:.0f} kHz target')
ax1.axvline(f_low / 1000, color='k', linestyle='--', linewidth=1.5, label=f'{f_low/1e3:.0f} kHz target')
ax1.grid(True, which='both', ls='--', alpha=0.7)
ax1.set_ylabel('|Z| (Ohms)', fontsize=13)
ax1.set_title(f'Series RC: Impedance Magnitude |Z| and Phase vs Frequency\n'
              f'R = 1 kΩ  |  C = {C_nF:.3f} nF', fontsize=14)
ax1.legend(loc='upper right', fontsize=10)

# Phase
ax2.semilogx(f / 1000, phase_deg, 'purple', linewidth=2.5, label='Phase')
ax2.axvline(f_target / 1000, color='g', linestyle='--', linewidth=1.5)
ax2.axvline(f_low / 1000, color='k', linestyle='--', linewidth=1.5)
ax2.axhline(-45, color='r', linestyle='--', linewidth=1.5, label='-45° reference')
ax2.grid(True, which='both', ls='--', alpha=0.7)
ax2.set_xlabel('Frequency (kHz)', fontsize=13)
ax2.set_ylabel('Phase (degrees)', fontsize=13)
ax2.legend(loc='upper left', fontsize=10)

# Dynamic, accurate annotation
key_text = "\n".join([
    f"At 5 kHz:   |Z| = {np.abs(R + 1/(1j*2*np.pi*5e3*C)):.0f} Ω,   Phase = {np.angle(R + 1/(1j*2*np.pi*5e3*C), deg=True):.1f}°",
    f"At 50 kHz:  |Z| = {np.abs(R + 1/(1j*2*np.pi*50e3*C)):.0f} Ω,   Phase = {np.angle(R + 1/(1j*2*np.pi*50e3*C), deg=True):.1f}°",
    f"At 100 kHz: |Z| = {np.abs(R + 1/(1j*2*np.pi*100e3*C)):.0f} Ω,   Phase = {np.angle(R + 1/(1j*2*np.pi*100e3*C), deg=True):.1f}°"
])

fig.text(0.13, 0.18,
         f'Key measured values with your C = {C_nF:.3f} nF:\n{key_text}\n\n'
         f'Formulas:\n'
         f'Z = R + 1/(j 2π f C)\n'
         f'|Z| = √(R² + (1/(2π f C))²)',
         fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.95, edgecolor="gray"))

plt.tight_layout()

# Uncomment to save
# plt.savefig('/home/workdir/artifacts/lab_setup/rc_bode_your_1k_3nF.png', dpi=300, bbox_inches='tight')
# plt.savefig('/home/workdir/artifacts/lab_setup/rc_bode_your_1k_3nF.pdf', dpi=300, bbox_inches='tight')

plt.show()