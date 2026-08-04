import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# User settings
# ============================================================
data_dir = r'C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Ultrasound\ResonantFreq'
input_file = os.path.join(data_dir, 'Q7air_08.04.2026.TXT')   # <-- change as needed
output_plot = os.path.join(data_dir, 'crystal_GB_BvD.png')
# ============================================================

def read_4294A_GB(filename):
    """Read TRACE A (G) and TRACE B (B) from a 4294A dual-trace export."""
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\r\n') for ln in f]

    def extract_trace(trace_label):
        start = None
        for i, line in enumerate(lines):
            if f'"TRACE: {trace_label}"' in line:
                start = i
                break
        if start is None:
            raise ValueError(f'TRACE {trace_label} not found')

        freqs, values = [], []
        for line in lines[start+1:]:
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
                    values.append(float(parts[1]))
                except ValueError:
                    break
            else:
                break
        return np.array(freqs), np.array(values)

    freq_G, G = extract_trace('A')
    freq_B, B = extract_trace('B')

    print(f'Trace A (G) points: {len(freq_G)}')
    print(f'Trace B (B) points: {len(freq_B)}')

    if len(freq_G) == 0 or len(freq_B) == 0:
        raise RuntimeError('Failed to read one or both traces')

    if len(freq_G) != len(freq_B) or not np.allclose(freq_G, freq_B, atol=1.0):
        print('Interpolating B onto G frequency axis...')
        B = np.interp(freq_G, freq_B, B)

    return freq_G, G, B


# ------------------------------------------------------------------
# Read data
# ------------------------------------------------------------------
freq, G, B = read_4294A_GB(input_file)
omega = 2 * np.pi * freq

print(f'\nFreq range: {freq[0]/1e6:.6f} – {freq[-1]/1e6:.6f} MHz')

# ------------------------------------------------------------------
# Basic resonance parameters from G
# ------------------------------------------------------------------
imax = np.argmax(G)
fs = freq[imax]
Gmax = G[imax]
Rm = 1.0 / Gmax if Gmax > 0 else np.nan

# –3 dB bandwidth
half = Gmax / 2.0

left = np.where(G[:imax] <= half)[0]
if len(left) > 0:
    iL = left[-1]
    f_left = freq[iL] + (half - G[iL])*(freq[iL+1]-freq[iL])/(G[iL+1]-G[iL]+1e-30)
else:
    f_left = freq[0]

right = np.where(G[imax:] <= half)[0]
if len(right) > 0:
    iR = imax + right[0]
    f_right = freq[iR-1] + (half - G[iR-1])*(freq[iR]-freq[iR-1])/(G[iR]-G[iR-1]+1e-30)
else:
    f_right = freq[-1]

delta_f = f_right - f_left
Q = fs / delta_f if delta_f > 0 else np.nan

# ------------------------------------------------------------------
# Butterworth-van Dyke parameters
# ------------------------------------------------------------------
# Motional arm
Lm = Q * Rm / (2 * np.pi * fs)          # Henry
Cm = 1.0 / (Lm * (2 * np.pi * fs)**2)   # Farad

# Static capacitance C0 from susceptance far from resonance
# Use the average of B/(ω) on the lower-frequency side (away from the peak)
# (simple robust estimate)
n_baseline = max(10, len(freq)//10)          # first ~10 % of points
C0_est = np.mean(B[:n_baseline] / omega[:n_baseline])

# Alternative: median of the whole baseline regions
mask_low  = freq < (fs - 5*delta_f)
mask_high = freq > (fs + 5*delta_f)
if np.any(mask_low) or np.any(mask_high):
    C0_pts = np.concatenate([
        B[mask_low]  / omega[mask_low]  if np.any(mask_low)  else [],
        B[mask_high] / omega[mask_high] if np.any(mask_high) else []
    ])
    C0 = np.median(C0_pts) if len(C0_pts) > 0 else C0_est
else:
    C0 = C0_est

# ------------------------------------------------------------------
# Print results
# ------------------------------------------------------------------
print('\n========== Resonance Parameters ==========')
print(f'fs          = {fs:.3f} Hz  ({fs/1e6:.6f} MHz)')
print(f'Rm          = {Rm:.4f} Ω')
print(f'Q           = {Q:.1f}')
print(f'Δf (–3 dB)  = {delta_f:.3f} Hz')

print('\n========== Butterworth-van Dyke ==========')
print(f'Rm          = {Rm:.4f} Ω')
print(f'Lm          = {Lm*1e3:.4f} mH')
print(f'Cm          = {Cm*1e15:.4f} fF')
print(f'C0          = {C0*1e12:.4f} pF')

# Consistency check: fs from Lm, Cm
fs_check = 1.0 / (2 * np.pi * np.sqrt(Lm * Cm))
print(f'\nConsistency: 1/(2π√(Lm Cm)) = {fs_check/1e6:.6f} MHz')

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(freq/1e6, G*1000, 'b-', lw=1.2, label='G')
ax1.axhline(half*1000, color='r', ls='--', alpha=0.7, label='Half power')
ax1.axvline(fs/1e6, color='g', ls='--', alpha=0.7)
ax1.plot(fs/1e6, Gmax*1000, 'ro')
ax1.set_ylabel('Conductance G (mS)')
ax1.set_title(f'Crystal G–B  |  fs = {fs/1e6:.6f} MHz   Rm = {Rm:.2f} Ω   Q ≈ {Q:.0f}\n'
              f'Lm = {Lm*1e3:.2f} mH   Cm = {Cm*1e15:.2f} fF   C0 = {C0*1e12:.2f} pF')
ax1.legend(loc='best')
ax1.grid(True)

ax2.plot(freq/1e6, B*1000, 'm-', lw=1.2, label='B')
ax2.axhline(0, color='k', lw=0.6)
ax2.axvline(fs/1e6, color='g', ls='--', alpha=0.7)
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Susceptance B (mS)')
ax2.legend(loc='best')
ax2.grid(True)

plt.tight_layout()
plt.savefig(output_plot, dpi=150)
print(f'\nPlot saved to: {output_plot}')