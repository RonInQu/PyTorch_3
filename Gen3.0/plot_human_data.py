"""Plot Gen3.0 human data: pressure time series, distributions, segment zoom."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

fname = sys.argv[1] if len(sys.argv) > 1 else '2026-05-12 206-104 Promedica_LOG4_state.parquet'
df = pd.read_parquet(fname)
df['time_sec'] = df['timestamp_ms'] / 1000.0

# Correct tissue mapping
tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}
df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')

colors = {'blood': 'green', 'clot': 'red', 'wall': 'blue', 'other': 'gray'}

# --- PLOT 1: Full time series (pressure + tissue labels) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle(f'Gen3.0 Human Data: {fname}', fontsize=12)

# Downsample for plotting
step = max(1, len(df) // 50000)
ds = df.iloc[::step]

ax1.plot(ds.time_sec / 60, ds.han_pressure_mmhg, 'k-', linewidth=0.3, alpha=0.5)
for tissue in ['blood', 'clot', 'wall']:
    mask = ds.tissue == tissue
    if mask.any():
        ax1.scatter(ds.time_sec[mask] / 60, ds.han_pressure_mmhg[mask],
                    c=colors[tissue], s=1, alpha=0.5, label=tissue)
ax1.set_ylabel('Pressure (mmHg)')
ax1.legend(loc='upper right', markerscale=5)
ax1.set_title('Pressure Time Series with Tissue Labels')
ax1.set_ylim(-50, 1500)

# State timeline
for tissue in ['blood', 'clot', 'wall']:
    mask = ds.tissue == tissue
    if mask.any():
        ax2.scatter(ds.time_sec[mask] / 60, [tissue] * mask.sum(),
                    c=colors[tissue], s=2, alpha=0.5)
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('State')
ax2.set_title('Tissue State Timeline')

plt.tight_layout()
plt.savefig('plot1_pressure_time_series.png', dpi=150)
print("Saved: plot1_pressure_time_series.png")
plt.close()

# --- PLOT 2: Zoom into wall segments ---
# Find longest wall segment
mask_w = df.tissue == 'wall'
seg_ids_w = (mask_w != mask_w.shift()).cumsum()[mask_w]
longest_wall = None
longest_dur = 0
for seg_id, grp in df.loc[seg_ids_w.index].groupby(seg_ids_w):
    dur = grp.time_sec.max() - grp.time_sec.min()
    if dur > longest_dur:
        longest_dur = dur
        longest_wall = grp

if longest_wall is not None:
    t_center = (longest_wall.time_sec.min() + longest_wall.time_sec.max()) / 2
    t_start = t_center - 30
    t_end = t_center + 30
    zoom = df[(df.time_sec >= t_start) & (df.time_sec <= t_end)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(zoom.time_sec, zoom.han_pressure_mmhg, 'k-', linewidth=0.5, alpha=0.4)
    for tissue in ['blood', 'clot', 'wall']:
        mask = zoom.tissue == tissue
        if mask.any():
            ax.scatter(zoom.time_sec[mask], zoom.han_pressure_mmhg[mask],
                       c=colors[tissue], s=3, alpha=0.7, label=tissue)
    ax.axhline(500, color='orange', linestyle='--', alpha=0.5, label='500 mmHg threshold')
    ax.axhline(200, color='purple', linestyle='--', alpha=0.5, label='200 mmHg threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (mmHg)')
    ax.set_title(f'Wall Segment Zoom (longest: {longest_dur:.1f}s)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plot2_wall_zoom.png', dpi=150)
    print("Saved: plot2_wall_zoom.png")
    plt.close()

# --- PLOT 3: Zoom into clot segments ---
mask_c = df.tissue == 'clot'
seg_ids_c = (mask_c != mask_c.shift()).cumsum()[mask_c]
longest_clot = None
longest_dur_c = 0
for seg_id, grp in df.loc[seg_ids_c.index].groupby(seg_ids_c):
    dur = grp.time_sec.max() - grp.time_sec.min()
    if dur > longest_dur_c:
        longest_dur_c = dur
        longest_clot = grp

if longest_clot is not None:
    t_center = (longest_clot.time_sec.min() + longest_clot.time_sec.max()) / 2
    t_start = t_center - 30
    t_end = t_center + 30
    zoom = df[(df.time_sec >= t_start) & (df.time_sec <= t_end)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(zoom.time_sec, zoom.han_pressure_mmhg, 'k-', linewidth=0.5, alpha=0.4)
    for tissue in ['blood', 'clot', 'wall']:
        mask = zoom.tissue == tissue
        if mask.any():
            ax.scatter(zoom.time_sec[mask], zoom.han_pressure_mmhg[mask],
                       c=colors[tissue], s=3, alpha=0.7, label=tissue)
    ax.axhline(500, color='orange', linestyle='--', alpha=0.5, label='500 mmHg threshold')
    ax.axhline(200, color='purple', linestyle='--', alpha=0.5, label='200 mmHg threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (mmHg)')
    ax.set_title(f'Clot Segment Zoom (longest: {longest_dur_c:.1f}s)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plot3_clot_zoom.png', dpi=150)
    print("Saved: plot3_clot_zoom.png")
    plt.close()

# --- PLOT 4: Pressure distributions by tissue ---
fig, ax = plt.subplots(figsize=(12, 6))
for tissue in ['blood', 'clot', 'wall']:
    sub = df[df.tissue == tissue]
    if len(sub) > 0:
        ax.hist(sub.han_pressure_mmhg, bins=100, alpha=0.5, density=True,
                color=colors[tissue], label=f"{tissue} (n={len(sub):,})")
ax.axvline(500, color='orange', linestyle='--', linewidth=2, label='500 mmHg')
ax.axvline(200, color='purple', linestyle='--', linewidth=2, label='200 mmHg')
ax.set_xlabel('Pressure (mmHg)')
ax.set_ylabel('Density')
ax.set_title('Pressure Distributions: Blood vs Clot vs Wall')
ax.legend()
ax.set_xlim(-50, 1200)
plt.tight_layout()
plt.savefig('plot4_pressure_distributions.png', dpi=150)
print("Saved: plot4_pressure_distributions.png")
plt.close()

# --- PLOT 5: Per-segment box plots ---
def get_segments_all(df, tissue_name):
    mask = df.tissue == tissue_name
    seg_ids = (mask != mask.shift()).cumsum()[mask]
    segments = []
    for seg_id, group in df.loc[seg_ids.index].groupby(seg_ids):
        dur = group.time_sec.max() - group.time_sec.min()
        if dur < 0.5:
            continue
        p = group.han_pressure_mmhg
        segments.append({
            'p_min': p.min(), 'p_mean': p.mean(),
            'frac_below_500': (p < 500).mean(),
            'frac_below_200': (p < 200).mean(),
            'p_range': p.max() - p.min(),
            'p_std': p.std() if len(p) > 1 else 0,
        })
    return pd.DataFrame(segments)

clot_df = get_segments_all(df, 'clot')
wall_df = get_segments_all(df, 'wall')
blood_df = get_segments_all(df, 'blood')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['p_min', 'frac_below_500', 'p_range']
titles = ['Segment Min Pressure', 'Fraction Below 500 mmHg', 'Pressure Range']

for ax, metric, title in zip(axes, metrics, titles):
    data = []
    labels = []
    for name, sdf in [('blood', blood_df), ('clot', clot_df), ('wall', wall_df)]:
        if len(sdf) > 0:
            data.append(sdf[metric].values)
            labels.append(f"{name}\n(n={len(sdf)})")
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['blue', 'red', 'green']):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot5_segment_boxplots.png', dpi=150)
print("Saved: plot5_segment_boxplots.png")
plt.close()

print("\nAll plots saved.")
