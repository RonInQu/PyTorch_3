# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:19:04 2026

@author: RonaldKurnik
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

name = "2026-05-12 206-104 Promedica_state.parquet"
dfx2 = pd.read_parquet(name)
time = dfx2.timestamp_ms/1000
# 1. Create the figure with shared X-axis (Updated to 3 rows)
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout='tight')

# 2. Plot on the first subplot (Row 0)
axs[0].plot(time, dfx2.Manual_GT, color='blue')
axs[0].set_title('Label')
axs[0].set_ylabel('label')

# 3. Prepare segments for the second subplot (Row 1)
points = np.array([time, dfx2.han_pressure_mmhg]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 4. Map labels to colors for each segment
color_map = {9: 'blue', 5: 'red', 4: 'green',
             'wall': 'blue', 'clot': 'red', 'blood': 'green'}
colors = [color_map.get(lbl, 'gray') for lbl in dfx2.Manual_GT[:-1]]

# 5. Create the LineCollection and add it to the second subplot
lc = LineCollection(segments, colors=colors, linewidths=1.5)
axs[1].add_collection(lc)

# 6. Adjust limits because collections do not autoscale the axes automatically
axs[1].set_xlim(time.min(), time.max())
axs[1].set_ylim(dfx2.han_pressure_mmhg.min() * 0.95, dfx2.han_pressure_mmhg.max() * 1.05)

# 7. Labels for the second subplot
axs[1].set_title('Pressure')
axs[1].set_ylabel('Pressure, mmHg')

# 8. Plot on the third subplot (Row 2)
axs[2].plot(time, dfx2.imp_mag_ohms, color='black')
axs[2].set_title('Impedance')
axs[2].set_xlabel('time, sec')
axs[2].set_ylabel('Impedance')

# Set global title
plt.suptitle("2026-05-12 206-104 Promedica_state")

# Display the plot with synchronized zoom
plt.show()