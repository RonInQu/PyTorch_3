import pandas as pd
import matplotlib.pyplot as plt

# Load and filter initial timestamp
dfx1 = pd.read_parquet('Wall_solo.parquet')
if 'timestamp_ms' in dfx1.columns:
    dfx1 = dfx1[dfx1.timestamp_ms >= 1500]

# Copy and filter non-zero values for respective frequencies
dfx2 = dfx1.copy()
dfx3 = dfx1.copy()

col_0 = 'imp_mag_adj_0_ohm'
col_1 = 'imp_mag_adj_1_ohm' 
col_2 = 'imp_mag_adj_2_ohm'

if col_1 in dfx2.columns:
    dfx2 = dfx2[dfx2[col_1] > 0]

if col_2 in dfx3.columns:
    dfx3 = dfx3[dfx3[col_2] > 0]

# --- Plot 1: 50 kHz ---
plt.figure()
x1 = dfx1.timestamp_ms / 1000
y1 = dfx1[col_0]
mean_1 = y1.mean()

plt.plot(x1, y1, label='Impedance')
plt.axhline(y=mean_1, color='r', linestyle='--', label=f'Mean: {mean_1:.2f} Ω')
plt.text(x1.iloc[0], mean_1, f' Mean = {mean_1:.2f} Ω', color='r', va='bottom', fontweight='bold')
plt.title("Wall Impedance at 50 kHz")
plt.xlabel("Time (s)")
plt.ylabel("Impedance (Ω)")
plt.legend()
plt.show()

# --- Plot 2: 100 kHz ---
plt.figure()
x2 = dfx2.timestamp_ms / 1000
y2 = dfx2[col_1]
mean_2 = y2.mean()

plt.plot(x2, y2, label='Impedance')
plt.axhline(y=mean_2, color='r', linestyle='--', label=f'Mean: {mean_2:.2f} Ω')
plt.text(x2.iloc[0], mean_2, f' Mean = {mean_2:.2f} Ω', color='r', va='bottom', fontweight='bold')
plt.title("Wall Impedance at 100 kHz")
plt.xlabel("Time (s)")
plt.ylabel("Impedance (Ω)")
plt.legend()
plt.show()

# --- Plot 3: 12.5 kHz ---
plt.figure()
x3 = dfx3.timestamp_ms / 1000
y3 = dfx3[col_2]
mean_3 = y3.mean()

plt.plot(x3, y3, label='Impedance')
plt.axhline(y=mean_3, color='r', linestyle='--', label=f'Mean: {mean_3:.2f} Ω')
plt.text(x3.iloc[0], mean_3, f' Mean = {mean_3:.2f} Ω', color='r', va='bottom', fontweight='bold')
plt.title("Wall Impedance at 12.5 kHz")
plt.xlabel("Time (s)")
plt.ylabel("Impedance (Ω)")
plt.legend()
plt.show()