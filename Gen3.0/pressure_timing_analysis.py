"""Segment-level pressure analysis: clot vs wall timing and depth."""
import pandas as pd
import numpy as np

df = pd.read_parquet('LOG3_solo.parquet')
df['time_sec'] = df['timestamp_ms'] / 1000.0
tissue_map = {7:'blood', 10:'blood', 8:'clot', 11:'clot', 12:'clot', 13:'wall', 14:'wall'}
df['tissue'] = df['solo_state_i'].map(tissue_map).fillna('other')


def get_segments(df, tissue_name):
    """Extract contiguous segments and compute per-segment pressure stats."""
    mask = df.tissue == tissue_name
    seg_ids = (mask != mask.shift()).cumsum()
    seg_ids = seg_ids[mask]
    segments = []
    for seg_id, group in df.loc[seg_ids.index].groupby(seg_ids):
        t0 = group.time_sec.min()
        t1 = group.time_sec.max()
        dur = t1 - t0
        p = group.pressure_mmhg
        p_start = p.iloc[:min(10, len(p))].mean()
        segments.append({
            'tissue': tissue_name,
            't_start': t0, 't_end': t1, 'duration_s': dur,
            'n_samples': len(group),
            'p_min': p.min(), 'p_max': p.max(), 'p_mean': p.mean(),
            'p_range': p.max() - p.min(), 'p_std': p.std() if len(p) > 1 else 0,
            'frac_below_500': (p < 500).mean(),
            'frac_below_200': (p < 200).mean(),
            'drop_depth': p_start - p.min(),
            'p_start': p_start,
            # Time to minimum (how fast does pressure drop?)
            'time_to_min_s': (p.idxmin() - group.index[0]) / 112.0 if len(p) > 1 else 0,
            # Time spent below 50% of start
            'frac_below_half_start': (p < p_start * 0.5).mean() if p_start > 0 else 0,
        })
    return segments


clot_segs = get_segments(df, 'clot')
wall_segs = get_segments(df, 'wall')

# Print clot segments
print('=' * 100)
print('CLOT SEGMENTS — Pressure Characteristics (dur > 0.5s)')
print('=' * 100)
hdr = f"{'#':<3} {'t0(s)':<8} {'dur':<6} {'p_min':<7} {'p_mean':<7} {'range':<7} {'std':<7} {'drop':<7} {'%<500':<7} {'%<200':<7} {'t_min':<7}"
print(hdr)
print('-' * len(hdr))
for i, s in enumerate(clot_segs):
    if s['duration_s'] > 0.5:
        print(f"{i:<3} {s['t_start']:<8.1f} {s['duration_s']:<6.1f} "
              f"{s['p_min']:<7.0f} {s['p_mean']:<7.0f} {s['p_range']:<7.0f} "
              f"{s['p_std']:<7.1f} {s['drop_depth']:<7.0f} "
              f"{s['frac_below_500']:<7.2f} {s['frac_below_200']:<7.2f} "
              f"{s['time_to_min_s']:<7.2f}")

# Print wall segments
print(f"\n{'=' * 100}")
print('WALL SEGMENTS — Pressure Characteristics (dur > 0.5s)')
print('=' * 100)
print(hdr)
print('-' * len(hdr))
for i, s in enumerate(wall_segs):
    if s['duration_s'] > 0.5:
        print(f"{i:<3} {s['t_start']:<8.1f} {s['duration_s']:<6.1f} "
              f"{s['p_min']:<7.0f} {s['p_mean']:<7.0f} {s['p_range']:<7.0f} "
              f"{s['p_std']:<7.1f} {s['drop_depth']:<7.0f} "
              f"{s['frac_below_500']:<7.2f} {s['frac_below_200']:<7.2f} "
              f"{s['time_to_min_s']:<7.2f}")

# Summary comparison
print(f"\n{'=' * 100}")
print('SUMMARY COMPARISON: Per-segment medians (segments > 0.5s)')
print('=' * 100)
clot_df = pd.DataFrame([s for s in clot_segs if s['duration_s'] > 0.5])
wall_df = pd.DataFrame([s for s in wall_segs if s['duration_s'] > 0.5])

cols = ['duration_s', 'p_min', 'p_mean', 'p_range', 'p_std',
        'drop_depth', 'frac_below_500', 'frac_below_200',
        'time_to_min_s', 'frac_below_half_start']
print(f"\n{'Feature':<25} {'Clot median':<14} {'Wall median':<14} {'Wall/Clot':<10} {'Useful?'}")
print('-' * 77)
for col in cols:
    c = clot_df[col].median() if len(clot_df) else 0
    w = wall_df[col].median() if len(wall_df) else 0
    ratio = w / (c + 1e-9)
    useful = '***' if abs(ratio - 1) > 0.5 or (col in ['frac_below_500', 'frac_below_200'] and w > 0.3) else ''
    print(f"  {col:<23} {c:<14.2f} {w:<14.2f} {ratio:<10.2f} {useful}")

# KEY: What about the TIMING of pressure relative to impedance change?
print(f"\n{'=' * 100}")
print('TIMING ANALYSIS: Does pressure drop BEFORE or AFTER impedance rises?')
print('=' * 100)

# For each wall segment, compare when pressure drops vs when impedance rises
# relative to segment start
for tissue_name, segs in [('clot', clot_segs), ('wall', wall_segs)]:
    timing_results = []
    for s in segs:
        if s['duration_s'] < 1.0:
            continue
        seg = df[(df.time_sec >= s['t_start']) & (df.time_sec <= s['t_end'])]
        if len(seg) < 20:
            continue
        p = seg.pressure_mmhg.values
        z = seg.imp_mag_adj_0_ohm.values
        # Find index of first significant pressure drop (>50 mmHg from start)
        p_start = p[:5].mean()
        p_drop_idx = np.where(p < p_start - 50)[0]
        # Find index of first significant impedance rise (>200 Ω from start)
        z_start = z[:5].mean()
        z_rise_idx = np.where(z > z_start + 200)[0]

        p_drop_time = p_drop_idx[0] / 112.0 if len(p_drop_idx) > 0 else None
        z_rise_time = z_rise_idx[0] / 112.0 if len(z_rise_idx) > 0 else None
        timing_results.append({
            't_start': s['t_start'],
            'p_drop_time': p_drop_time,
            'z_rise_time': z_rise_time,
            'p_leads_z': (p_drop_time < z_rise_time) if (p_drop_time is not None and z_rise_time is not None) else None,
            'lag_s': (z_rise_time - p_drop_time) if (p_drop_time is not None and z_rise_time is not None) else None,
        })

    print(f"\n  {tissue_name.upper()} segments with timing data:")
    valid = [t for t in timing_results if t['lag_s'] is not None]
    if valid:
        for t in valid[:10]:
            lead = 'P leads Z' if t['p_leads_z'] else 'Z leads P'
            print(f"    t={t['t_start']:.0f}s: pressure drops at +{t['p_drop_time']:.2f}s, "
                  f"impedance rises at +{t['z_rise_time']:.2f}s → {lead} by {abs(t['lag_s']):.2f}s")
        lags = [t['lag_s'] for t in valid]
        p_leads_count = sum(1 for t in valid if t['p_leads_z'])
        print(f"    → Pressure leads impedance in {p_leads_count}/{len(valid)} segments")
        print(f"    → Median lag: {np.median(lags):.3f}s (positive = P drops first)")
    else:
        print(f"    No segments with both pressure drop and impedance rise")
