import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Directory containing the data and where the plot will be saved
data_dir = r'C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Ultrasound\ResonantFreq'

# Input data file
input_file = os.path.join(data_dir, 'Q6.TXT')

# Output plot file
output_plot = os.path.join(data_dir, 'crystal_G_air.png')

# Read the data
data = []
with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip().replace('\r', '')
        if not line or line.startswith('"') or line.startswith('4294') or 'Frequency' in line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                freq = float(parts[0])
                real = float(parts[1])
                imag = float(parts[2]) if len(parts) > 2 else 0.0
                data.append((freq, real, imag))
            except:
                pass

data = np.array(data)

# Remove duplicate frequencies (file contained 1602 lines for 801 points)
unique_f, idx = np.unique(data[:, 0], return_index=True)
freq = data[idx, 0]
G = data[idx, 1]          # Real part = Conductance

print('Unique points:', len(freq))
print('Freq range: {:.6f} - {:.6f} MHz'.format(freq[0]/1e6, freq[-1]/1e6))
print('G min: {:.6f} S, G max: {:.6f} S'.format(G.min(), G.max()))

# Find peak
imax = np.argmax(G)
f_peak = freq[imax]
G_peak = G[imax]
print('Peak G = {:.6f} S at f = {:.6f} Hz ({:.6f} MHz)'.format(G_peak, f_peak, f_peak/1e6))
print('Rm approx = {:.3f} Ohm'.format(1/G_peak if G_peak > 0 else float('nan')))

# Find -3 dB points (half-power = G_peak / 2)
half = G_peak / 2.0
print('Half power level = {:.6f} S'.format(half))

# Left side
left = np.where(G[:imax] <= half)[0]
if len(left) > 0:
    i_left = left[-1]
    f_left = freq[i_left] + (half - G[i_left]) * (freq[i_left+1] - freq[i_left]) / (G[i_left+1] - G[i_left] + 1e-30)
else:
    f_left = freq[0]

# Right side
right = np.where(G[imax:] <= half)[0]
if len(right) > 0:
    i_right = imax + right[0]
    f_right = freq[i_right-1] + (half - G[i_right-1]) * (freq[i_right] - freq[i_right-1]) / (G[i_right] - G[i_right-1] + 1e-30)
else:
    f_right = freq[-1]

delta_f = f_right - f_left
Q = f_peak / delta_f if delta_f > 0 else float('nan')

print('f_left  = {:.6f} Hz'.format(f_left))
print('f_right = {:.6f} Hz'.format(f_right))
print('Delta f = {:.6f} Hz'.format(delta_f))
print('Q = {:.1f}'.format(Q))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(freq/1e6, G*1000, 'b-', linewidth=1)
ax.axhline(half*1000, color='r', linestyle='--', label='Half power')
ax.axvline(f_peak/1e6, color='g', linestyle='--', label='Peak')
ax.plot(f_peak/1e6, G_peak*1000, 'ro')
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Conductance G (mS)')
ax.set_title('Crystal Conductance in Air\nfs = {:.6f} MHz, Q ≈ {:.0f}'.format(f_peak/1e6, Q))
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(output_plot, dpi=150)
print('Plot saved to:', output_plot)