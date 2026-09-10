import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# User settings – only change the input file name here
# ============================================================
data_dir = r'C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Ultrasound\Porcine_ResonantFreq_2'
input_file = os.path.join(data_dir, 'ACD3.TXT')   # <-- change this
# ============================================================

# Automatically build output names from the input file name
base_name = os.path.splitext(os.path.basename(input_file))[0]   # e.g. "Clot3"
output_plot    = os.path.join(data_dir, f'{base_name}_crystal_GB_BvD_Bfeatures.png')
output_results = os.path.join(data_dir, f'{base_name}_Results.txt')

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

# ------------------------------------------------------------------
# G-based parameters
# ------------------------------------------------------------------
imax = np.argmax(G)
fs = freq[imax]
Gmax = G[imax]
Rm = 1.0 / Gmax if Gmax > 0 else np.nan
B_at_fs = B[imax]

# –3 dB bandwidth from G
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
# B-curve features
# ------------------------------------------------------------------
i_Bmax = np.argmax(B)
f_Bmax = freq[i_Bmax]
Bmax = B[i_Bmax]

i_Bmin = np.argmin(B)
f_Bmin = freq[i_Bmin]
Bmin = B[i_Bmin]

delta_f_B = f_Bmin - f_Bmax
Q_from_B = fs / delta_f_B if delta_f_B > 0 else np.nan

# ------------------------------------------------------------------
# Butterworth-van Dyke parameters
# ------------------------------------------------------------------
Lm = Q * Rm / (2 * np.pi * fs)
Cm = 1.0 / (Lm * (2 * np.pi * fs)**2)

n_baseline = max(10, len(freq)//10)
C0_est = np.mean(B[:n_baseline] / omega[:n_baseline])
mask_low  = freq < (fs - 5*delta_f)
mask_high = freq > (fs + 5*delta_f)
C0_pts = []
if np.any(mask_low):
    C0_pts.append(B[mask_low] / omega[mask_low])
if np.any(mask_high):
    C0_pts.append(B[mask_high] / omega[mask_high])
C0 = np.median(np.concatenate(C0_pts)) if C0_pts else C0_est

# ------------------------------------------------------------------
# Collect results text
# ------------------------------------------------------------------
results = []
results.append(f'Trace A (G) points: {len(freq)}')
results.append(f'Trace B (B) points: {len(freq)}')
results.append('')
results.append('========== Resonance from G ==========')
results.append(f'fs              = {fs:.3f} Hz  ({fs/1e6:.6f} MHz)')
results.append(f'Rm              = {Rm:.4f} Ω')
results.append(f'Q (from G)      = {Q:.1f}')
results.append(f'Δf (–3 dB, G)   = {delta_f:.3f} Hz')
results.append(f'B at fs         = {B_at_fs*1000:.3f} mS')
results.append('')
results.append('========== Features from B curve ==========')
results.append(f'f_Bmax          = {f_Bmax:.3f} Hz  ({f_Bmax/1e6:.6f} MHz)   Bmax = {Bmax*1000:.3f} mS')
results.append(f'f_Bmin          = {f_Bmin:.3f} Hz  ({f_Bmin/1e6:.6f} MHz)   Bmin = {Bmin*1000:.3f} mS')
results.append(f'Δf (Bmax–Bmin)  = {delta_f_B:.3f} Hz')
results.append(f'Q (from B)      = {Q_from_B:.1f}')
results.append('')
results.append('========== Butterworth-van Dyke ==========')
results.append(f'Rm              = {Rm:.4f} Ω')
results.append(f'Lm              = {Lm*1e3:.4f} mH')
results.append(f'Cm              = {Cm*1e15:.4f} fF')
results.append(f'C0              = {C0*1e12:.4f} pF')

results_text = '\n'.join(results)
print(results_text)

# Save results to text file
with open(output_results, 'w', encoding='utf-8') as f:
    f.write(results_text + '\n')
print(f'\nResults saved to: {output_results}')

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(freq/1e6, G*1000, 'b-', lw=1.2, label='G')
ax1.axhline(half*1000, color='r', ls='--', alpha=0.7, label='Half power')
ax1.axvline(fs/1e6, color='g', ls='--', alpha=0.7)
ax1.plot(fs/1e6, Gmax*1000, 'ro')
ax1.set_ylabel('Conductance G (mS)')
ax1.set_title(f'fs = {fs/1e6:.6f} MHz   Rm = {Rm:.1f} Ω   Q ≈ {Q:.0f}\n'
              f'Lm = {Lm*1e3:.2f} mH   Cm = {Cm*1e15:.2f} fF   C0 = {C0*1e12:.1f} pF')
ax1.legend(loc='best')
ax1.grid(True)

ax2.plot(freq/1e6, B*1000, 'm-', lw=1.2, label='B')
ax2.axhline(0, color='k', lw=0.6)
ax2.axvline(fs/1e6, color='g', ls='--', alpha=0.7, label='fs (G peak)')
ax2.plot(f_Bmax/1e6, Bmax*1000, 'go', label='B max')
ax2.plot(f_Bmin/1e6, Bmin*1000, 'rs', label='B min')
ax2.plot(fs/1e6, B_at_fs*1000, 'k^', label='B at fs')
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Susceptance B (mS)')
ax2.legend(loc='best')
ax2.grid(True)

plt.tight_layout()
plt.savefig(output_plot, dpi=150)
print(f'Plot saved to: {output_plot}')