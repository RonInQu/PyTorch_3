"""Analyze pressure features in human data (Gen3.0 Promedica).
Correct mapping: label=light_style_i, pressure=han_pressure_mmhg
States: 4=SALINE_BLOOD, 5=CLOT, 9=WALL_LATCH
"""
import pandas as pd
import numpy as np
import sys

fname = sys.argv[1] if len(sys.argv) > 1 else '2026-05-12 206-104 Promedica_LOG4_state.parquet'
df = pd.read_parquet(fname)
df['time_sec'] = df['timestamp_ms'] / 1000.0

# Correct tissue mapping from light_style_i
# 4=IMP_STATE_2_SALINE_BLOOD, 5=IMP_STATE_3_CLOT, 9=WALL_LATCH
tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}
df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')

print(f"File: {fname}")
print(f"Duration: {(df.time_sec.max()-df.time_sec.min())/60:.1f} min, {len(df):,} samples, ~{len(df)/(df.time_sec.max()-df.time_sec.min()):.0f} Hz")
print(f"Label column: light_style_i")
print(f"Pressure column: han_pressure_mmhg")
print()
print("light_style_i value counts:")
print(df['light_style_i'].value_counts().sort_index().to_string())
print()
print(f"Tissue counts: blood={df[df.tissue=='blood'].shape[0]:,}, "
      f"clot={df[df.tissue=='clot'].shape[0]:,}, "
      f"wall={df[df.tissue=='wall'].shape[0]:,}, "
      f"other={df[df.tissue=='other'].shape[0]:,}")
print()

# Per-sample pressure stats
print('=' * 100)
print('PER-SAMPLE PRESSURE STATISTICS (han_pressure_mmhg)')
print('=' * 100)
for tissue in ['blood', 'clot', 'wall']:
    sub = df[df.tissue == tissue]
    if len(sub) == 0:
        print(f"  {tissue:<6}: NO SAMPLES")
        continue
    p = sub.han_pressure_mmhg
    print(f"  {tissue:<6}: n={len(sub):>9,}, p_med={p.median():.0f}, p_mean={p.mean():.0f}, "
          f"p_std={p.std():.0f}, p5={p.quantile(0.05):.0f}, p95={p.quantile(0.95):.0f}, "
          f"frac<500={(p < 500).mean():.3f}, frac<200={(p < 200).mean():.3f}")


def get_segments(df, tissue_name):
    mask = df.tissue == tissue_name
    seg_ids = (mask != mask.shift()).cumsum()
    seg_ids = seg_ids[mask]
    segments = []
    for seg_id, group in df.loc[seg_ids.index].groupby(seg_ids):
        dur = group.time_sec.max() - group.time_sec.min()
        if dur < 0.5:
            continue
        p = group.han_pressure_mmhg
        p_start = p.iloc[:min(20, len(p))].mean()
        segments.append({
            't_start': group.time_sec.min(), 'duration_s': dur,
            'n_samples': len(group),
            'p_min': p.min(), 'p_max': p.max(), 'p_mean': p.mean(),
            'p_range': p.max() - p.min(),
            'p_std': p.std() if len(p) > 1 else 0,
            'frac_below_500': (p < 500).mean(),
            'frac_below_200': (p < 200).mean(),
            'drop_depth': p_start - p.min(),
            'p_start': p_start,
        })
    return segments


clot_segs = get_segments(df, 'clot')
wall_segs = get_segments(df, 'wall')
blood_segs = get_segments(df, 'blood')

print(f"Blood segments (>0.5s): {len(blood_segs)}")
print(f"Clot segments (>0.5s): {len(clot_segs)}")
print(f"Wall segments (>0.5s): {len(wall_segs)}")

clot_df = pd.DataFrame(clot_segs) if clot_segs else pd.DataFrame()
wall_df = pd.DataFrame(wall_segs) if wall_segs else pd.DataFrame()

# Detailed segment listing
print()
print('=' * 100)
print('WALL SEGMENTS (state 9 = WALL_LATCH) — detailed')
print('=' * 100)
hdr = f"{'#':<3} {'t0(s)':<9} {'dur':<7} {'p_min':<7} {'p_mean':<8} {'p_range':<8} {'p_std':<7} {'drop':<7} {'%<500':<7} {'%<200':<7}"
print(hdr)
print('-' * len(hdr))
for i, s in enumerate(wall_segs):
    print(f"{i:<3} {s['t_start']:<9.1f} {s['duration_s']:<7.1f} "
          f"{s['p_min']:<7.0f} {s['p_mean']:<8.0f} {s['p_range']:<8.0f} "
          f"{s['p_std']:<7.1f} {s['drop_depth']:<7.0f} "
          f"{s['frac_below_500']:<7.2f} {s['frac_below_200']:<7.2f}")

print()
print('=' * 100)
print('CLOT SEGMENTS (state 5 = IMP_STATE_3_CLOT) — detailed')
print('=' * 100)
print(hdr)
print('-' * len(hdr))
for i, s in enumerate(clot_segs):
    print(f"{i:<3} {s['t_start']:<9.1f} {s['duration_s']:<7.1f} "
          f"{s['p_min']:<7.0f} {s['p_mean']:<8.0f} {s['p_range']:<8.0f} "
          f"{s['p_std']:<7.1f} {s['drop_depth']:<7.0f} "
          f"{s['frac_below_500']:<7.2f} {s['frac_below_200']:<7.2f}")

# Summary comparison
print()
print('=' * 100)
print('SUMMARY: Clot vs Wall per-segment medians')
print('=' * 100)
cols = ['duration_s', 'p_min', 'p_mean', 'p_range', 'p_std',
        'drop_depth', 'frac_below_500', 'frac_below_200']
print(f"\n{'Feature':<25} {'Clot median':<14} {'Wall median':<14} {'Ratio':<10}")
print('-' * 63)
for col in cols:
    c = clot_df[col].median() if len(clot_df) else 0
    w = wall_df[col].median() if len(wall_df) else 0
    ratio = w / (c + 1e-9)
    print(f"  {col:<23} {c:<14.2f} {w:<14.2f} {ratio:<10.2f}")


print()
print('=' * 100)
print('BLOOD SEGMENTS (state 4 = SALINE_BLOOD) — summary')
print('=' * 100)
if blood_segs:
    blood_df = pd.DataFrame(blood_segs)
    print(f"  n_segments: {len(blood_df)}")
    print(f"  p_min: median={blood_df.p_min.median():.0f}, range=[{blood_df.p_min.min():.0f}, {blood_df.p_min.max():.0f}]")
    print(f"  p_mean: median={blood_df.p_mean.median():.0f}")
    print(f"  frac<500: median={blood_df.frac_below_500.median():.2f}")
    print(f"  duration: median={blood_df.duration_s.median():.1f}s, max={blood_df.duration_s.max():.1f}s")
