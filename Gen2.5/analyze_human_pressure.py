"""Analyze pressure features in human data (Promedica LOG4)."""
import pandas as pd
import numpy as np
import sys

fname = sys.argv[1] if len(sys.argv) > 1 else '2026-05-12 206-104 Promedica_LOG4_state.parquet'
df = pd.read_parquet(fname)
df['time_sec'] = df['timestamp_ms'] / 1000.0

# Same tissue mapping
tissue_map = {7:'blood', 9:'blood', 10:'blood', 8:'clot', 11:'clot', 12:'clot', 13:'wall', 14:'wall'}
df['tissue'] = df['cms_led_state_i'].map(tissue_map).fillna('other')

print(f"File: {fname}")
print(f"Duration: {(df.time_sec.max()-df.time_sec.min())/60:.1f} min, {len(df):,} samples, ~{len(df)/(df.time_sec.max()-df.time_sec.min()):.0f} Hz")
print(f"Tissue counts: {df.tissue.value_counts().to_dict()}")
print(f"NOTE: imp_i is a counter (not instantaneous impedance). Only pressure available.")
print()


def get_segments(df, tissue_name):
    mask = df.tissue == tissue_name
    seg_ids = (mask != mask.shift()).cumsum()
    seg_ids = seg_ids[mask]
    segments = []
    for seg_id, group in df.loc[seg_ids.index].groupby(seg_ids):
        dur = group.time_sec.max() - group.time_sec.min()
        if dur < 0.5:
            continue
        p = group.cms_pressure_mmhg
        p_start = p.iloc[:min(20, len(p))].mean()
        segments.append({
            'state': int(group.cms_led_state_i.mode().iloc[0]),
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

# Per-state breakdown for clot
print()
print('=' * 100)
print('CLOT SEGMENTS by sub-state')
print('=' * 100)
clot_df = pd.DataFrame(clot_segs)
for st in sorted(clot_df.state.unique()):
    sub = clot_df[clot_df.state == st]
    print(f"  State {st} ({len(sub)} segments):")
    print(f"    p_min: median={sub.p_min.median():.0f}, range=[{sub.p_min.min():.0f}, {sub.p_min.max():.0f}]")
    print(f"    p_mean: median={sub.p_mean.median():.0f}")
    print(f"    frac<500: median={sub.frac_below_500.median():.2f}")
    print(f"    frac<200: median={sub.frac_below_200.median():.2f}")
    print(f"    duration: median={sub.duration_s.median():.1f}s")

# Per-state breakdown for wall
print()
print('=' * 100)
print('WALL SEGMENTS by sub-state')
print('=' * 100)
wall_df = pd.DataFrame(wall_segs)
for st in sorted(wall_df.state.unique()):
    sub = wall_df[wall_df.state == st]
    print(f"  State {st} ({len(sub)} segments):")
    print(f"    p_min: median={sub.p_min.median():.0f}, range=[{sub.p_min.min():.0f}, {sub.p_min.max():.0f}]")
    print(f"    p_mean: median={sub.p_mean.median():.0f}")
    print(f"    frac<500: median={sub.frac_below_500.median():.2f}")
    print(f"    frac<200: median={sub.frac_below_200.median():.2f}")
    print(f"    duration: median={sub.duration_s.median():.1f}s")

# Detailed segment listing
print()
print('=' * 100)
print('WALL SEGMENTS — detailed')
print('=' * 100)
hdr = f"{'#':<3} {'st':<4} {'t0(s)':<8} {'dur':<7} {'p_min':<7} {'p_mean':<8} {'p_range':<8} {'p_std':<7} {'%<500':<7} {'%<200':<7}"
print(hdr)
print('-' * len(hdr))
for i, s in enumerate(wall_segs):
    print(f"{i:<3} {s['state']:<4} {s['t_start']:<8.0f} {s['duration_s']:<7.1f} "
          f"{s['p_min']:<7.0f} {s['p_mean']:<8.0f} {s['p_range']:<8.0f} "
          f"{s['p_std']:<7.1f} {s['frac_below_500']:<7.2f} {s['frac_below_200']:<7.2f}")

print()
print('=' * 100)
print('CLOT SEGMENTS — detailed')
print('=' * 100)
print(hdr)
print('-' * len(hdr))
for i, s in enumerate(clot_segs):
    print(f"{i:<3} {s['state']:<4} {s['t_start']:<8.0f} {s['duration_s']:<7.1f} "
          f"{s['p_min']:<7.0f} {s['p_mean']:<8.0f} {s['p_range']:<8.0f} "
          f"{s['p_std']:<7.1f} {s['frac_below_500']:<7.2f} {s['frac_below_200']:<7.2f}")

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

# CRITICAL: Check what state 12 (show_clot) looks like — it has LOW pressure
# Compare state 11 (asp_in_clot) vs state 14 (wall) — the actual discrimination problem
print()
print('=' * 100)
print('THE REAL QUESTION: State 11 (asp_in_clot) vs State 14 (wall/lollipop)')
print('=' * 100)
s11 = clot_df[clot_df.state == 11]
s14 = wall_df[wall_df.state == 14]
if len(s11) > 0 and len(s14) > 0:
    print(f"\n{'Feature':<25} {'St11 (clot)':<14} {'St14 (wall)':<14}")
    print('-' * 53)
    for col in cols:
        c = s11[col].median()
        w = s14[col].median()
        print(f"  {col:<23} {c:<14.2f} {w:<14.2f}")

# Per-sample stats for blood/clot/wall
print()
print('=' * 100)
print('PER-SAMPLE STATISTICS (all samples, not segments)')
print('=' * 100)
for tissue in ['blood', 'clot', 'wall']:
    sub = df[df.tissue == tissue]
    p = sub.cms_pressure_mmhg
    print(f"  {tissue:<6}: n={len(sub):>8,}, p_med={p.median():.0f}, p_mean={p.mean():.0f}, "
          f"p_std={p.std():.0f}, p5={p.quantile(0.05):.0f}, p95={p.quantile(0.95):.0f}, "
          f"frac<500={( p < 500).mean():.3f}")
