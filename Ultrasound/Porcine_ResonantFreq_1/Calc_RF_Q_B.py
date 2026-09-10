import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# User settings
# ============================================================
data_dir = r'C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Ultrasound\ResonantFreq'
input_file = os.path.join(data_dir, 'Q7air_08.04.2026.TXT')
output_plot = os.path.join(data_dir, 'crystal_GB_air.png')
# ============================================================

def read_4294A_GB(filename):
    """
    Reads TRACE A (G) and TRACE B (B) from a 4294A dual-trace export.
    The instrument puts the trace value in the first data column (Real).
    """
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\r\n') for ln in f]

    def extract_trace(trace_label):
        """Find the named trace and return frequency + value arrays."""
        # Find the line that contains the trace header
        start = None
        for i, line in enumerate(lines):
            if f'"TRACE: {trace_label}"' in line:
                start = i
                break
        if start is None:
            raise ValueError(f'TRACE {trace_label} not found')

        freqs = []
        values = []

        # Start reading after the header block
        for line in lines[start+1:]:
            line = line.strip()
            if not line:
                continue
            # Stop if we hit another TRACE section
            if line.startswith('"TRACE:'):
                break
            # Skip remaining header lines
            if line.startswith('"'):
                continue

            # Data lines are tab- or space-separated
            parts = line.replace('\t', ' ').split()
            if len(parts) >= 2:
                try:
                    f = float(parts[0])
                    v = float(parts[1])
                    freqs.append(f)
                    values.append(v)
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
        raise RuntimeError('Failed to read one or both traces – check file format')

    # Make sure both traces share the same frequency axis
    if len(freq_G) != len(freq_B) or not np.allclose(freq_G, freq_B, atol=1.0):
        print('Interpolating B onto G frequency axis...')
        B = np.interp(freq_G, freq_B, B)

    return freq_G, G, B


# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------
freq, G, B = read_4294A_GB(input_file)

print(f'\nFreq range: {freq[0]/1e6:.6f} – {freq[-1]/1e6:.6f} MHz')
print(f'G  range: {G.min():.6f} to {G.max():.6f} S')
print(f'B  range: {B.min():.6f} to {B.max():.6f} S')

# Peak of conductance
imax = np.argmax(G)
f_peak = freq[imax]
G_peak = G[imax]
Rm = 1.0 / G_peak if G_peak > 0 else np.nan

print(f'\nPeak G = {G_peak:.6f} S at {f_peak:.3f} Hz ({f_peak/1e6:.6f} MHz)')
print(f'Rm ≈ {Rm:.3f} Ω')

# –3 dB bandwidth
half = G_peak / 2.0

left = np.where(G[:imax] <= half)[0]
if len(left) > 0:
    iL = left[-1]
    f_left = freq[iL] + (half - G[iL]) * (freq[iL+1] - freq[iL]) / (G[iL+1] - G[iL] + 1e-30)
else:
    f_left = freq[0]

right = np.where(G[imax:] <= half)[0]
if len(right) > 0:
    iR = imax + right[0]
    f_right = freq[iR-1] + (half - G[iR-1]) * (freq[iR] - freq[iR-1]) / (G[iR] - G[iR-1] + 1e-30)
else:
    f_right = freq[-1]

delta_f = f_right - f_left
Q = f_peak / delta_f if delta_f > 0 else np.nan

print(f'Δf (–3 dB) = {delta_f:.3f} Hz')
print(f'Q ≈ {Q:.1f}')

# Zero-crossing of B near the peak
search = slice(max(0, imax-40), min(len(B), imax+40))
crossings = np.where(np.diff(np.sign(B[search])))[0]
if len(crossings) > 0:
    iz = search.start + crossings[0]
    f_zero = freq[iz] - B[iz] * (freq[iz+1] - freq[iz]) / (B[iz+1] - B[iz] + 1e-30)
    print(f'B zero-crossing ≈ {f_zero:.3f} Hz ({f_zero/1e6:.6f} MHz)')

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(freq/1e6, G*1000, 'b-', lw=1.2, label='G')
ax1.axhline(half*1000, color='r', ls='--', alpha=0.7, label='Half power')
ax1.axvline(f_peak/1e6, color='g', ls='--', alpha=0.7)
ax1.plot(f_peak/1e6, G_peak*1000, 'ro')
ax1.set_ylabel('Conductance G (mS)')
ax1.set_title(f'Crystal G–B\nfs = {f_peak/1e6:.6f} MHz   Rm ≈ {Rm:.2f} Ω   Q ≈ {Q:.0f}')
ax1.legend(loc='best')
ax1.grid(True)

ax2.plot(freq/1e6, B*1000, 'm-', lw=1.2, label='B')
ax2.axhline(0, color='k', lw=0.6)
ax2.axvline(f_peak/1e6, color='g', ls='--', alpha=0.7)
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Susceptance B (mS)')
ax2.legend(loc='best')
ax2.grid(True)

plt.tight_layout()
plt.savefig(output_plot, dpi=150)
print(f'\nPlot saved to: {output_plot}')