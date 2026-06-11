import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# Load the CSV file
name = "2026-05-12 206-104 Promedica_state.parquet"
# dfx1 = pd.read_csv(name+'.csv')

# Save as a Parquet file
# dfx1.to_parquet(name+".parquet", index=False)

col_names = ["timestamp_ms","light_style_i", "han_pressure_mmhg"]
dfx2 = pd.read_parquet(name, columns=col_names)
time = dfx2.timestamp_ms/1000
lbl_0 = dfx2.light_style_i
plt.figure()
plt.plot(time,lbl_0)
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout='tight')
# time = arr = np.arange(0, dfx1.shape[0]) / 1000

# 3. Plot on the first subplot (Row 0)
axs[0].plot(time, dfx2.light_style_i, color='blue')
axs[0].set_title('Label')
axs[0].set_xlabel('time')
axs[0].set_ylabel('label')

# 4. Plot on the second subplot (Row 1)
axs[1].plot(time, dfx2.han_pressure_mmhg, color='red')
axs[1].set_title('Pressure')
axs[1].set_xlabel('time')
axs[1].set_ylabel('Pressure, mmHg')

# axs[2].plot(time, dfx1.imp_mag_ohms, color='black')
# axs[2].set_title('Impedance')
# axs[2].set_xlabel('time')
# axs[2].set_ylabel('Impedance')

# 5. Display the plot
plt.show()