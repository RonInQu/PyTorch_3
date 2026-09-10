#!/usr/bin/env python3
"""
Adjusted theory for the measured series RC (RC2.TXT).
R and C fitted to the real impedance analyzer data.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----- Fitted values that match RC2.TXT -----
R = 1049.0          # Ω   (was 1000)
C = 2.872e-9        # F   (was 3.0 nF)
# --------------------------------------------

f_target = 50e3
f_low = 5e3
C_nF = C * 1e9

print(f"Using fitted R = {R:.1f} Ω  and  C = {C_nF:.3f} nF")
f_45 = 1 / (2 * np.pi * R * C)
print(f"Exact frequency for –45° phase: {f_45/1e3:.2f} kHz")

key_freqs = [5e3, 50e3, 100e3]
print("\nCalculated values:")
for freq in key_freqs:
    omega = 2 * np.pi * freq
    Z = R + 1 / (1j * omega * C)
    mag = np.abs(Z)
    phase = np.angle(Z, deg=True)
    print(f"  {freq/1e3:5.0f} kHz → |Z| = {mag:8.1f} Ω   Phase = {phase:7.2f}°")

# Frequency sweep
f = np.logspace(3, 5, 1000)
omega = 2 * np.pi * f
Z = R + 1 / (1j * omega * C)
mag = np.abs(Z)
phase_deg = np.angle(Z, deg=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax1.semilogx(f / 1000, mag, 'b-', linewidth=2.5, label='|Z| Magnitude')
ax1.axvline(f_target / 1000, color='g', linestyle='--', linewidth=1.5, label=f'{f_target/1e3:.0f} kHz')
ax1.axvline(f_low / 1000, color='k', linestyle='--', linewidth=1.5, label=f'{f_low/1e3:.0f} kHz')
ax1.grid(True, which='both', ls='--', alpha=0.7)
ax1.set_ylabel('|Z| (Ohms)', fontsize=13)
ax1.set_title(f'Series RC (fitted to measurement)\nR = {R:.0f} Ω  |  C = {C_nF:.3f} nF', fontsize=14)
ax1.legend(loc='upper right', fontsize=10)

ax2.semilogx(f / 1000, phase_deg, 'purple', linewidth=2.5, label='Phase')
ax2.axvline(f_target / 1000, color='g', linestyle='--', linewidth=1.5)
ax2.axvline(f_low / 1000, color='k', linestyle='--', linewidth=1.5)
ax2.axhline(-45, color='r', linestyle='--', linewidth=1.5, label='–45° reference')
ax2.grid(True, which='both', ls='--', alpha=0.7)
ax2.set_xlabel('Frequency (kHz)', fontsize=13)
ax2.set_ylabel('Phase (degrees)', fontsize=13)
ax2.legend(loc='upper left', fontsize=10)

key_text = "\n".join([
    f"At 5 kHz:   |Z| = {np.abs(R + 1/(1j*2*np.pi*5e3*C)):.0f} Ω,   Phase = {np.angle(R + 1/(1j*2*np.pi*5e3*C), deg=True):.1f}°",
    f"At 50 kHz:  |Z| = {np.abs(R + 1/(1j*2*np.pi*50e3*C)):.0f} Ω,   Phase = {np.angle(R + 1/(1j*2*np.pi*50e3*C), deg=True):.1f}°",
    f"At 100 kHz: |Z| = {np.abs(R + 1/(1j*2*np.pi*100e3*C)):.0f} Ω,   Phase = {np.angle(R + 1/(1j*2*np.pi*100e3*C), deg=True):.1f}°"
])

fig.text(0.13, 0.18,
         f'Key values with fitted C = {C_nF:.3f} nF, R = {R:.0f} Ω:\n{key_text}\n\n'
         f'Formulas:\n'
         f'Z = R + 1/(j 2π f C)\n'
         f'|Z| = √(R² + (1/(2π f C))²)',
         fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.95, edgecolor="gray"))

plt.tight_layout()
plt.show()