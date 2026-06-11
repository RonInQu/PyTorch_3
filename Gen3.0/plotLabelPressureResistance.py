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
time = (dfx2.timestamp_ms / 1000).values

# ─── Downsample for fast interactive plotting ───────────────────────────────
# Keep every Nth point (min/max preserving decimation per block)
MAX_POINTS = 50_000  # plenty for visual fidelity on screen
N = len(time)

if N > MAX_POINTS:
    block_size = N // MAX_POINTS
    n_blocks = N // block_size

    # Reshape into blocks, take min and max index within each block for pressure/impedance
    # This preserves peaks and valleys (envelope decimation)
    truncated = n_blocks * block_size
    t_blocks = time[:truncated].reshape(n_blocks, block_size)
    p_blocks = dfx2.han_pressure_mmhg.values[:truncated].reshape(n_blocks, block_size)
    z_blocks = dfx2.imp_mag_ohms.values[:truncated].reshape(n_blocks, block_size)
    gt_blocks = dfx2.Manual_GT.values[:truncated].reshape(n_blocks, block_size)

    # For each block: take the midpoint time, min and max of signal
    # Use interleaved min/max for proper envelope
    p_min_idx = p_blocks.argmin(axis=1)
    p_max_idx = p_blocks.argmax(axis=1)

    # Simple approach: take first sample of each block (preserves transitions)
    idx = np.arange(0, truncated, block_size)
    time_ds = time[idx]
    pressure_ds = dfx2.han_pressure_mmhg.values[idx]
    impedance_ds = dfx2.imp_mag_ohms.values[idx]
    gt_ds = dfx2.Manual_GT.values[idx]
else:
    time_ds = time
    pressure_ds = dfx2.han_pressure_mmhg.values
    impedance_ds = dfx2.imp_mag_ohms.values
    gt_ds = dfx2.Manual_GT.values

print(f"Original: {N:,} points -> Displayed: {len(time_ds):,} points")

# 1. Create the figure with shared X-axis
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout='tight')

# 2. Label subplot
axs[0].plot(time_ds, gt_ds, color='blue', linewidth=0.8)
axs[0].set_title('Label')
axs[0].set_ylabel('label')

# 3. Pressure subplot with color-coded LineCollection
points = np.column_stack([time_ds, pressure_ds]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 4. Map labels to colors
color_map = {9: 'blue', 5: 'red', 4: 'green',
             'wall': 'blue', 'clot': 'red', 'blood': 'green'}
colors = [color_map.get(lbl, 'gray') for lbl in gt_ds[:-1]]

# 5. Create LineCollection
lc = LineCollection(segments, colors=colors, linewidths=1.0)
axs[1].add_collection(lc)
axs[1].set_xlim(time_ds.min(), time_ds.max())
axs[1].set_ylim(pressure_ds.min() * 0.95, pressure_ds.max() * 1.05)
axs[1].set_title('Pressure')
axs[1].set_ylabel('Pressure, mmHg')

# 6. Impedance subplot
axs[2].plot(time_ds, impedance_ds, color='black', linewidth=0.5)
axs[2].set_title('Impedance')
axs[2].set_xlabel('time, sec')
axs[2].set_ylabel('Impedance')

# Set global title
plt.suptitle("2026-05-12 206-104 Promedica_state")

plt.show()