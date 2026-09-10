#!/usr/bin/env python3
"""
Series RC impedance: measured data (Agilent 4294A) vs theory.
Nominal: R = 1 kΩ, C = 3 nF
Best-fit: R ≈ 1100 Ω, C ≈ 2.95 nF
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
data_file = Path(r'C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Electrodes\RC_Circuit\RC2.TXT')
out_png   = Path(r'C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Electrodes\RC_Circuit\rc_measured_vs_theory.png')

# -------------------------------------------------------------------------
# Theoretical component values
# -------------------------------------------------------------------------
R_nom = 1000.0          # nominal 1 kΩ
C_nom = 3.0e-9          # nominal 3 nF

# R_fit = 1100.0          # least-squares fit to measured data
# C_fit = 2.95e-9         # ≈ 2.95 nF

R_fit = 1049.0          # least-squares fit to measured data
C_fit = 2.87e-9         # ≈ 2.95 nF

# -------------------------------------------------------------------------
# Parse 4294A dual-trace export (TRACE A = |Z|, TRACE B = Phase)
# -------------------------------------------------------------------------
def read_4294A_mag_phase(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\r\n') for ln in f]

    def extract(trace_label):
        start = None
        for i, line in enumerate(lines):
            if f'"TRACE: {trace_label}"' in line:
                start = i
                break
        if start is None:
            raise ValueError(f'TRACE {trace_label} not found')

        freqs, vals = [], []
        for line in lines[start + 1:]:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"TRACE:'):
                break
            if line.startswith('"'):
                continue
            parts = line.replace('\t', ' ').split()
            if len(parts) >= 2:
                try:
                    freqs.append(float(parts[0]))
                    vals.append(float(parts[1]))
                except ValueError:
                    break
            else:
                break
        return np.array(freqs), np.array(vals)

    f_mag, mag = extract('A')
    f_ph, phase = extract('B')

    if len(f_mag) != len(f_ph) or not np.allclose(f_mag, f_ph, atol=1.0):
        phase = np.interp(f_mag, f_ph, phase)

    return f_mag, mag, phase


freq, mag_meas, phase_meas = read_4294A_mag_phase(data_file)
print(f'Loaded {len(freq)} points  ({freq[0]/1e3:.1f} – {freq[-1]/1e3:.1f} kHz)')

# -------------------------------------------------------------------------
# Theory curves (dense log sweep for smooth lines)
# -------------------------------------------------------------------------
f_th = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), 2000)
omega_th = 2 * np.pi * f_th

def series_rc(R, C, omega):
    Z = R + 1.0 / (1j * omega * C)
    return np.abs(Z), np.angle(Z, deg=True)

mag_nom, phase_nom = series_rc(R_nom, C_nom, omega_th)
mag_fit, phase_fit = series_rc(R_fit, C_fit, omega_th)

# Key-point evaluation
targets = [5e3, 50e3, 100e3]
print('\nComparison at key frequencies:')
print(f'{"f (kHz)":>8} | {"|Z| meas":>10} {"|Z| nom":>10} {"|Z| fit":>10} | '
      f'{"φ meas":>8} {"φ nom":>8} {"φ fit":>8}')
for ft in targets:
    idx = np.argmin(np.abs(freq - ft))
    f_act = freq[idx]
    m_meas = mag_meas[idx]
    p_meas = phase_meas[idx]
    m_n, p_n = series_rc(R_nom, C_nom, 2*np.pi*f_act)
    m_f, p_f = series_rc(R_fit, C_fit, 2*np.pi*f_act)
    print(f'{f_act/1e3:8.2f} | {m_meas:10.1f} {m_n:10.1f} {m_f:10.1f} | '
          f'{p_meas:8.2f} {p_n:8.2f} {p_f:8.2f}')

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)

# Magnitude
ax1.semilogx(freq / 1e3, mag_meas, 'ko', markersize=4, alpha=0.7,
             label='Measured |Z| (4294A)')
ax1.semilogx(f_th / 1e3, mag_nom, 'b--', linewidth=2.2,
             label=f'Theory nominal  R={R_nom:.0f} Ω, C={C_nom*1e9:.1f} nF')
ax1.semilogx(f_th / 1e3, mag_fit, 'r-', linewidth=2.0,
             label=f'Theory fitted   R={R_fit:.0f} Ω, C={C_fit*1e9:.2f} nF')

ax1.axvline(5,  color='gray', ls=':', lw=1.2, alpha=0.8)
ax1.axvline(50, color='green', ls=':', lw=1.2, alpha=0.8)
ax1.set_ylabel('|Z| (Ω)', fontsize=13)
ax1.set_title('Series RC Impedance – Measured vs Theory\n'
              'Agilent 4294A  |  Nominal 1 kΩ + 3 nF',
              fontsize=14, fontweight='bold')
ax1.grid(True, which='both', ls='--', alpha=0.6)
ax1.legend(loc='upper right', fontsize=9.5)
ax1.set_ylim(bottom=800)

# Phase
ax2.semilogx(freq / 1e3, phase_meas, 'ko', markersize=4, alpha=0.7,
             label='Measured Phase (4294A)')
ax2.semilogx(f_th / 1e3, phase_nom, 'b--', linewidth=2.2,
             label=f'Theory nominal  R={R_nom:.0f} Ω, C={C_nom*1e9:.1f} nF')
ax2.semilogx(f_th / 1e3, phase_fit, 'r-', linewidth=2.0,
             label=f'Theory fitted   R={R_fit:.0f} Ω, C={C_fit*1e9:.2f} nF')

ax2.axhline(-45, color='purple', ls='--', lw=1.3, alpha=0.8, label='-45° reference')
ax2.axvline(5,  color='gray', ls=':', lw=1.2, alpha=0.8)
ax2.axvline(50, color='green', ls=':', lw=1.2, alpha=0.8)

ax2.set_xlabel('Frequency (kHz)', fontsize=13)
ax2.set_ylabel('Phase (degrees)', fontsize=13)
ax2.grid(True, which='both', ls='--', alpha=0.6)
ax2.legend(loc='upper left', fontsize=9.5)
ax2.set_ylim(-95, -15)

# Annotation box
def z_at(f0, R, C):
    Z = R + 1/(1j*2*np.pi*f0*C)
    return np.abs(Z), np.angle(Z, deg=True)

txt_lines = [
    'Key values (measured / nominal theory / fitted theory):',
    '',
]
for ft in [5e3, 50e3, 100e3]:
    idx = np.argmin(np.abs(freq - ft))
    m_m, p_m = mag_meas[idx], phase_meas[idx]
    m_n, p_n = z_at(freq[idx], R_nom, C_nom)
    m_f, p_f = z_at(freq[idx], R_fit, C_fit)
    txt_lines.append(
        f'{freq[idx]/1e3:5.1f} kHz:  |Z| {m_m:.0f} / {m_n:.0f} / {m_f:.0f} Ω   '
        f'φ {p_m:.1f} / {p_n:.1f} / {p_f:.1f} °'
    )

txt_lines += [
    '',
    'Formulas (series RC):',
    '  Z = R + 1/(j 2π f C)',
    '  |Z| = √(R² + (1/(2π f C))²)',
    '  φ = −atan(1/(2π f R C))',
]

fig.text(0.13, 0.02,
         '\n'.join(txt_lines),
         fontsize=9.5,
         family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                   alpha=0.95, edgecolor='gray'),
         verticalalignment='bottom')

plt.tight_layout(rect=[0, 0.22, 1, 1])
plt.savefig(out_png, dpi=180, bbox_inches='tight')
print(f'\nPlot saved to: {out_png}')
plt.show()