"""Quick pressure feature analysis for wall detection."""
import pandas as pd
import numpy as np

df = pd.read_parquet('LOG6_solo.parquet')
df['time_sec'] = df['timestamp_ms'] / 1000.0

tissue_map = {7:'blood', 10:'blood', 8:'clot', 11:'clot', 12:'clot', 13:'wall', 14:'wall'}
df['tissue'] = df['solo_state_i'].map(tissue_map).fillna('other')

# Compute derivatives (pressure rate of change)
dt = df['time_sec'].diff()
df['pressure_deriv'] = df['pressure_mmhg'].diff() / dt  # mmHg/sec
df['pressure_deriv_abs'] = df['pressure_deriv'].abs()

# Rolling stats (0.5s window ~ 56 samples, 2s ~ 224 samples)
win_short = 56   # 0.5s
win_long = 224   # 2s

df['pressure_std_05s'] = df['pressure_mmhg'].rolling(win_short, center=True).std()
df['pressure_std_2s'] = df['pressure_mmhg'].rolling(win_long, center=True).std()
df['pressure_range_2s'] = (df['pressure_mmhg'].rolling(win_long, center=True).max() -
                           df['pressure_mmhg'].rolling(win_long, center=True).min())
df['pressure_mean_2s'] = df['pressure_mmhg'].rolling(win_long, center=True).mean()

# Delta from blood baseline (global blood median)
blood_pressure_median = df[df.tissue == 'blood']['pressure_mmhg'].median()
df['pressure_delta'] = df['pressure_mmhg'] - blood_pressure_median

# Correlation: pressure vs impedance in rolling window
df['pressure_imp_corr_2s'] = (df['pressure_mmhg'].rolling(win_long, center=True)
                              .corr(df['imp_mag_adj_0_ohm']))

# Pressure min in window (captures dips during wall contact)
df['pressure_min_2s'] = df['pressure_mmhg'].rolling(win_long, center=True).min()
df['pressure_max_2s'] = df['pressure_mmhg'].rolling(win_long, center=True).max()

print('='*93)
print('PRESSURE FEATURE ANALYSIS: Blood vs Clot vs Wall')
print('='*93)

features = [
    ('pressure_mmhg',        'Raw pressure (mmHg)'),
    ('pressure_delta',       'Delta from blood baseline'),
    ('pressure_deriv',       'Pressure derivative (mmHg/s)'),
    ('pressure_deriv_abs',   '|Pressure derivative|'),
    ('pressure_std_05s',     'Pressure std (0.5s window)'),
    ('pressure_std_2s',      'Pressure std (2s window)'),
    ('pressure_range_2s',    'Pressure range (2s window)'),
    ('pressure_mean_2s',     'Pressure mean (2s window)'),
    ('pressure_min_2s',      'Pressure min (2s window)'),
    ('pressure_max_2s',      'Pressure max (2s window)'),
    ('pressure_imp_corr_2s', 'Pressure-impedance corr (2s)'),
]

header = f"{'Feature':<30} {'Blood':<12} {'Clot':<12} {'Wall':<12} {'C-B sep':<9} {'W-B sep':<9} {'C-W sep':<9}"
print(f"\n{header}")
print("-"*93)

for col, name in features:
    blood = df[df.tissue == 'blood'][col].dropna()
    clot = df[df.tissue == 'clot'][col].dropna()
    wall = df[df.tissue == 'wall'][col].dropna()

    b_med = blood.median()
    c_med = clot.median()
    w_med = wall.median()

    # Separation using pooled std
    bc_std = pd.concat([blood, clot]).std()
    bw_std = pd.concat([blood, wall]).std()
    cw_std = pd.concat([clot, wall]).std()

    cb_sep = abs(c_med - b_med) / bc_std if bc_std > 0 else 0
    wb_sep = abs(w_med - b_med) / bw_std if bw_std > 0 else 0
    cw_sep = abs(c_med - w_med) / cw_std if cw_std > 0 else 0

    print(f'{name:<30} {b_med:<12.2f} {c_med:<12.2f} {w_med:<12.2f} {cb_sep:<9.3f} {wb_sep:<9.3f} {cw_sep:<9.3f}')

# Percentile analysis
print("\n--- Pressure Distribution Details ---")
for tissue_name in ['blood', 'clot', 'wall']:
    sub = df[df.tissue == tissue_name]['pressure_mmhg']
    print(f"{tissue_name:>6}: p5={sub.quantile(0.05):.0f}, p25={sub.quantile(0.25):.0f}, "
          f"p50={sub.quantile(0.5):.0f}, p75={sub.quantile(0.75):.0f}, p95={sub.quantile(0.95):.0f}, "
          f"min={sub.min():.0f}, max={sub.max():.0f}")

# Derivative stats
print("\n--- Pressure Derivative Stats ---")
for tissue_name in ['blood', 'clot', 'wall']:
    sub = df[df.tissue == tissue_name]['pressure_deriv'].dropna()
    print(f"{tissue_name:>6}: mean={sub.mean():.2f}, std={sub.std():.2f}, "
          f"p5={sub.quantile(0.05):.1f}, p95={sub.quantile(0.95):.1f}")

# Answer the key question
print("\n--- KEY: Is wall pressure higher or lower than blood? ---")
b_p = df[df.tissue == 'blood']['pressure_mmhg']
w_p = df[df.tissue == 'wall']['pressure_mmhg']
c_p = df[df.tissue == 'clot']['pressure_mmhg']
print(f"Blood: median={b_p.median():.0f}, mean={b_p.mean():.1f}")
print(f"Clot:  median={c_p.median():.0f}, mean={c_p.mean():.1f}")
print(f"Wall:  median={w_p.median():.0f}, mean={w_p.mean():.1f}")
print(f"\nWall range: [{w_p.min():.0f}, {w_p.max():.0f}]")
print(f"% of wall samples with pressure < blood p25 ({b_p.quantile(0.25):.0f}): "
      f"{100*(w_p < b_p.quantile(0.25)).mean():.1f}%")
print(f"% of wall samples with pressure > blood p75 ({b_p.quantile(0.75):.0f}): "
      f"{100*(w_p > b_p.quantile(0.75)).mean():.1f}%")

# Pressure behavior AROUND transitions (before/during/after wall)
print("\n--- Pressure at Wall Transition Boundaries ---")
wall_mask = df.tissue == 'wall'
wall_start_indices = wall_mask.astype(int).diff().eq(1)
wall_end_indices = wall_mask.astype(int).diff().eq(-1)

print(f"Number of wall onset transitions: {wall_start_indices.sum()}")
# Look at pressure 1s before and during first 1s of wall
for idx in df.index[wall_start_indices]:
    loc = df.index.get_loc(idx)
    if loc > 112:  # need 1s before
        before = df.iloc[loc-112:loc]['pressure_mmhg'].mean()
        during = df.iloc[loc:min(loc+112, len(df))]['pressure_mmhg'].mean()
        t = df.iloc[loc]['time_sec']
        print(f"  t={t:.1f}s: before={before:.0f}, during={during:.0f}, delta={during-before:+.0f}")
