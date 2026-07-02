"""
Prepare 5 kHz sine wave for COMSOL import.
- Pure sine wave at 5 kHz, 560 mV peak-to-peak
- One period (200 µs), suitable for periodic interpolation
"""

import numpy as np
from pathlib import Path

# =====================================================================
#  PARAMETERS
# =====================================================================
freq = 5000.0        # Hz
ptp = 0.560          # V peak-to-peak
amplitude = ptp / 2  # V peak
period = 1.0 / freq  # 200 µs
n_points = 2000      # points per period

# =====================================================================
#  GENERATE SINE WAVE (one period)
# =====================================================================
t = np.linspace(0, period, n_points, endpoint=False)
v = amplitude * np.sin(2 * np.pi * freq * t)

print(f"5 kHz sine wave: {ptp*1e3:.0f} mV PTP, {amplitude*1e3:.0f} mV amplitude")
print(f"Period: {period*1e6:.0f} µs, {n_points} points")
print(f"PTP check: {v.max() - v.min():.4f} V")

# =====================================================================
#  EXPORT FOR COMSOL
#  Format: two-column text file (time [s], voltage [V])
#  COMSOL Interpolation function reads this directly.
# =====================================================================
out_path = Path(__file__).parent / "5kHz_waveform_comsol.txt"

header = (
    "% 5 kHz sine wave excitation for COMSOL interpolation\n"
    "% 560 mV peak-to-peak\n"
    "% One period (200 us), suitable for periodic interpolation\n"
    "% Column 1: time [s], Column 2: voltage [V]\n"
)

with open(out_path, 'w') as f:
    f.write(header)
    for ti, vi in zip(t, v):
        f.write(f"{ti:.10e} {vi:.10e}\n")

print(f"\nSaved: {out_path}")
print(f"  COMSOL: Definitions → Interpolation → File")
print(f"  Set 'Interpolation' to 'Linear', 'End condition' to 'Periodic'")
