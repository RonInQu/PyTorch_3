# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:52:12 2026

@author: RonaldKurnik
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Load the CSV file
name = "2026-05-13 220-054 Centennial_state.parquet"
# df = pd.read_csv("LOG4_state.csv")

# Save as a Parquet file
# df.to_parquet("name", index=False)

col_names = ["timestamp_ms","light_style_i", "han_pressure_mmhg"]
# # 1. Open the file metadata stream
# parquet_file = pq.ParquetFile(name)
# # 2. Pull exactly 200,000 rows into memory
# batch_iter = parquet_file.iter_batches(batch_size=2000000)
# first_batch = next(batch_iter)
# # 3. Convert that single slice into your pandas DataFrame
# dfx2 = pa.Table.from_batches([first_batch]).to_pandas()

dfx2 = pd.read_parquet(name,columns=col_names)
dfx2 = pd.read_parquet(name)
time = dfx2.timestamp_ms/1000
lbl_0 = dfx2.cms_led_state_i
plt.figure()
plt.plot(time,lbl_0)
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout='tight')

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

# axs[2].plot(time, dfx2.imp_i, color='black')
# axs[2].set_title('Impedance')
# axs[2].set_xlabel('time')
# axs[2].set_ylabel('Impedance')

# 5. Display the plot
plt.show()



# 1. Create the figure with shared X-axis
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout='tight')

# 2. Plot on the first subplot (Row 0)
axs[0].plot(time, dfx2.light_style_i, color='blue')
axs[0].set_title('Label')
axs[0].set_ylabel('label')  # Note: sharex hides inner x-labels, so we only need it on the bottom

# 3. Prepare segments for the second subplot (Row 1)
# We reshape the time and pressure data into pairs of points: [[(x0, y0), (x1, y1)], [(x1, y1), (x2, y2)], ...]
points = np.array([time, dfx2.han_pressure_mmhg]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 4. Map labels to colors for each segment
# We define the mapping dictionary
color_map = {9: 'blue', 5: 'red', 4: 'green'}
# Default to a neutral color (e.g., 'gray') if a label doesn't match 4, 5, or 9
colors = [color_map.get(lbl, 'gray') for lbl in dfx2.light_style_i[:-1]]

# 5. Create the LineCollection and add it to the second subplot
lc = LineCollection(segments, colors=colors, linewidths=1.5)
axs[1].add_collection(lc)

# 6. Adjust limits because collections do not autoscale the axes automatically
axs[1].set_xlim(time.min(), time.max())
axs[1].set_ylim(dfx2.han_pressure_mmhg.min() * 0.95, dfx2.han_pressure_mmhg.max() * 1.05)

# 7. Labels for the second subplot
axs[1].set_title('Pressure')
axs[1].set_xlabel('time')
axs[1].set_ylabel('Pressure, mmHg')

plt.title("2026-05-12 206-104 Promedica_LOG4_state")


# Display the plot with synchronized zoom
plt.show()



fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout='tight')

# 2. Plot on the first subplot (Row 0)
axs[0].plot(time, dfx2.light_style_i, color='blue')
axs[0].set_title('Label')
axs[0].set_ylabel('label')

# --- OPTIMIZED LINE COLLECTION FOR ROW 1 ---

# Convert pandas series to numpy arrays immediately for speed
t_arr = time.to_numpy() if hasattr(time, 'to_numpy') else np.asarray(time)
p_arr = dfx2.han_pressure_mmhg.to_numpy()
lbl_arr = dfx2.light_style_i.to_numpy()

# Vectorized segment creation (No np.concatenate or reshaping loops)
points = np.column_stack((t_arr, p_arr))
segments = np.stack((points[:-1], points[1:]), axis=1)

# Vectorized color mapping using np.select (thousands of times faster than list loops)
condlist = [lbl_arr[:-1] == 9, lbl_arr[:-1] == 5, lbl_arr[:-1] == 4]
choicelist = ['blue', 'red', 'green']
colors = np.select(condlist, choicelist, default='gray')

# Create and add collection
lc = LineCollection(segments, colors=colors, linewidths=1.5)
axs[1].add_collection(lc)

# Explicitly set limits so it doesn't have to autoscale dynamically
axs[1].set_xlim(t_arr[0], t_arr[-1])
axs[1].set_ylim(p_arr.min() * 0.95, p_arr.max() * 1.05)

# --- END OPTIMIZATION ---

# 4. Labels for the second subplot
axs[1].set_title('Pressure')
axs[1].set_xlabel('time')
axs[1].set_ylabel('Pressure, mmHg')

axs[1].legend(loc='upper right')

plt.show()



# 1. Initialize Figure
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout='tight')

# 2. Plot Row 0 (Labels)
axs[0].plot(time, dfx2.light_style_i, color='blue')
axs[0].set_title('Label')
axs[0].set_ylabel('label')

# 3. Plot Row 1 (Pressure - plotted as a fast, continuous single line)
# We use a neutral dark color so the background colors stand out clearly
axs[1].plot(time, dfx2.han_pressure_mmhg, color='#333333', linewidth=1, label='Pressure')

# 4. FAST VECTORIZED BACKGROUND COLORING
t_arr = time.to_numpy() if hasattr(time, 'to_numpy') else np.asarray(time)
lbl_arr = dfx2.light_style_i.to_numpy()

# Find exactly where the state changes to avoid looping over every data point
change_indices = np.where(lbl_arr[:-1] != lbl_arr[1:])[0]
# Include the start and end of the dataset
block_edges = np.concatenate(([0], change_indices + 1, [len(lbl_arr) - 1]))

# Map your specific clinical states to colors
state_colors = {
    4: 'green',  # Blood
    9: 'blue',   # Wall
    5: 'red'     # Clot
}

# Track which labels we've added to the legend so we don't get duplicates
seen_labels = set()
label_names = {4: 'Blood', 9: 'Wall', 5: 'Clot'}

# Draw the colored background blocks (extremely fast because it's only a few dozen/hundred blocks)
for i in range(len(block_edges) - 1):
    idx_start = block_edges[i]
    idx_end = block_edges[i+1]
    
    state = lbl_arr[idx_start]
    color = state_colors.get(state, None)
    
    if color:
        t_start = t_arr[idx_start]
        t_end = t_arr[idx_end]
        
        # Only add a legend label the first time we encounter this state
        leg_label = None
        if state not in seen_labels:
            leg_label = label_names[state]
            seen_labels.add(state)
            
        axs[1].axvspan(t_start, t_end, color=color, alpha=0.25, label=leg_label)

# 5. Set limits and labels
axs[1].set_xlim(t_arr[0], t_arr[-1])
axs[1].set_title('Pressure')
axs[1].set_xlabel('time')
axs[1].set_ylabel('Pressure, mmHg')

# Add the legend to show Blood, Wall, and Clot
axs[1].legend(loc='upper right')

plt.show()