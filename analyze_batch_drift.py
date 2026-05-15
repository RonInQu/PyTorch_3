"""
analyze_batch_drift.py — Compare old batch (85 studies) vs new batch (23 studies).
Identifies signal-level and labeling differences that cause distribution shift.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Directories ──
OLD_DIR = Path('vApril10_data_set') / 'training'
NEW_DIR = Path('vMay14_data_set') / 'training'

# ── Identify which studies are new ──
old_files = sorted(OLD_DIR.glob('*_labeled_segment.parquet'))
new_files = sorted(NEW_DIR.glob('*_labeled_segment.parquet'))
old_names = {f.stem.replace('_labeled_segment', '') for f in old_files}
new_names = {f.stem.replace('_labeled_segment', '') for f in new_files}
added_names = new_names - old_names

# Fixed test studies (excluded)
TEST_STUDIES = {'33CFB812', '819421BC', '847A1E3F', '8ECEADA6',
                'CENT0008', 'DD2DFAF4', 'F427536B', 'SUMM0127'}

old_train_names = old_names - TEST_STUDIES
new_only_names = added_names - TEST_STUDIES

print(f"Old batch training studies: {len(old_train_names)}")
print(f"New-only studies:           {len(new_only_names)}")
print(f"{'='*70}")


def analyze_study(filepath):
    """Extract per-study statistics from a labeled parquet."""
    df = pd.read_parquet(filepath)

    # Expect columns: 'R' (impedance) and 'label' (0=blood, 1=clot, 2=wall)
    r_col = 'R' if 'R' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else None

    r = df[r_col].values.astype(float)
    stats = {
        'n_samples': len(r),
        'r_mean': np.mean(r),
        'r_std': np.std(r),
        'r_median': np.median(r),
        'r_p10': np.percentile(r, 10),
        'r_p90': np.percentile(r, 90),
    }

    if label_col and label_col in df.columns:
        labels = df[label_col].values
        n = len(labels)
        stats['frac_blood'] = np.sum(labels == 0) / n
        stats['frac_clot'] = np.sum(labels == 1) / n
        stats['frac_wall'] = np.sum(labels == 2) / n

        # Event duration analysis (contiguous label runs)
        for cls, cls_name in [(1, 'clot'), (2, 'wall')]:
            mask = (labels == cls).astype(int)
            if mask.sum() == 0:
                stats[f'{cls_name}_n_events'] = 0
                stats[f'{cls_name}_mean_dur_s'] = 0
                stats[f'{cls_name}_max_dur_s'] = 0
            else:
                # Find contiguous runs
                d = np.diff(np.concatenate(([0], mask, [0])))
                starts = np.where(d == 1)[0]
                ends = np.where(d == -1)[0]
                durations = (ends - starts) / 150.0  # seconds at 150 Hz
                stats[f'{cls_name}_n_events'] = len(durations)
                stats[f'{cls_name}_mean_dur_s'] = np.mean(durations)
                stats[f'{cls_name}_max_dur_s'] = np.max(durations)

        # R-level per class
        for cls, cls_name in [(0, 'blood'), (1, 'clot'), (2, 'wall')]:
            cls_r = r[labels == cls]
            if len(cls_r) > 0:
                stats[f'r_mean_{cls_name}'] = np.mean(cls_r)
                stats[f'r_std_{cls_name}'] = np.std(cls_r)
            else:
                stats[f'r_mean_{cls_name}'] = np.nan
                stats[f'r_std_{cls_name}'] = np.nan

    return stats


# ── Analyze both batches ──
print("\nAnalyzing old batch...")
old_stats = {}
for f in sorted(OLD_DIR.glob('*_labeled_segment.parquet')):
    name = f.stem.replace('_labeled_segment', '')
    if name in old_train_names:
        old_stats[name] = analyze_study(f)

print(f"  Processed {len(old_stats)} studies")

print("Analyzing new-only batch...")
new_stats = {}
for f in sorted(NEW_DIR.glob('*_labeled_segment.parquet')):
    name = f.stem.replace('_labeled_segment', '')
    if name in new_only_names:
        new_stats[name] = analyze_study(f)

print(f"  Processed {len(new_stats)} studies")

# ── Compare distributions ──
print(f"\n{'='*70}")
print("BATCH COMPARISON: Old (85 train) vs New-only (23)")
print(f"{'='*70}")

metrics = [
    ('r_mean', 'Mean R (Ω)'),
    ('r_std', 'Std R (Ω)'),
    ('r_median', 'Median R (Ω)'),
    ('r_p10', 'P10 R (Ω)'),
    ('r_p90', 'P90 R (Ω)'),
    ('frac_blood', 'Blood fraction'),
    ('frac_clot', 'Clot fraction'),
    ('frac_wall', 'Wall fraction'),
    ('clot_n_events', 'Clot events/study'),
    ('clot_mean_dur_s', 'Clot mean duration (s)'),
    ('wall_n_events', 'Wall events/study'),
    ('wall_mean_dur_s', 'Wall mean duration (s)'),
    ('r_mean_blood', 'Blood R mean (Ω)'),
    ('r_mean_clot', 'Clot R mean (Ω)'),
    ('r_mean_wall', 'Wall R mean (Ω)'),
    ('r_std_clot', 'Clot R std (Ω)'),
    ('r_std_wall', 'Wall R std (Ω)'),
    ('n_samples', 'Samples/study'),
]

print(f"\n{'Metric':<25} {'Old Mean':>10} {'Old Std':>10} {'New Mean':>10} {'New Std':>10} {'Diff%':>8} {'Flag':>6}")
print("-" * 85)

for key, label in metrics:
    old_vals = [s[key] for s in old_stats.values() if key in s and not np.isnan(s.get(key, np.nan))]
    new_vals = [s[key] for s in new_stats.values() if key in s and not np.isnan(s.get(key, np.nan))]

    if not old_vals or not new_vals:
        continue

    old_mean = np.mean(old_vals)
    old_std = np.std(old_vals)
    new_mean = np.mean(new_vals)
    new_std = np.std(new_vals)

    if old_mean != 0:
        pct_diff = 100 * (new_mean - old_mean) / abs(old_mean)
    else:
        pct_diff = 0

    # Flag if >20% different or if new std is >2x old std
    flag = ""
    if abs(pct_diff) > 20:
        flag = "⚠️"
    if old_std > 0 and new_std / old_std > 2:
        flag = "⚠️⚠️"

    print(f"{label:<25} {old_mean:>10.1f} {old_std:>10.1f} {new_mean:>10.1f} {new_std:>10.1f} {pct_diff:>+7.1f}% {flag:>6}")

# ── Per-study detail for new batch (sorted by how "different" they are) ──
print(f"\n{'='*70}")
print("NEW STUDIES — DETAIL (sorted by blood R-level)")
print(f"{'='*70}")
print(f"{'Study':<12} {'R_mean':>7} {'Blood_R':>8} {'Clot_R':>7} {'Wall_R':>7} {'%Blood':>7} {'%Clot':>6} {'%Wall':>6} {'Clot_ev':>8} {'Wall_ev':>8}")
print("-" * 90)

# Sort by blood R mean
sorted_new = sorted(new_stats.items(), key=lambda x: x[1].get('r_mean_blood', 0))
for name, s in sorted_new:
    print(f"{name:<12} {s.get('r_mean',0):>7.0f} {s.get('r_mean_blood',0):>8.0f} "
          f"{s.get('r_mean_clot',0):>7.0f} {s.get('r_mean_wall',0):>7.0f} "
          f"{s.get('frac_blood',0):>7.1%} {s.get('frac_clot',0):>6.1%} {s.get('frac_wall',0):>6.1%} "
          f"{s.get('clot_n_events',0):>8} {s.get('wall_n_events',0):>8}")

# ── Studies with zero clot or wall events (potential labeling issue) ──
print(f"\n{'='*70}")
print("POTENTIAL ISSUES — Studies with unusual characteristics")
print(f"{'='*70}")

# No clot or wall at all
no_tissue = [n for n, s in new_stats.items()
             if s.get('frac_clot', 0) == 0 and s.get('frac_wall', 0) == 0]
if no_tissue:
    print(f"\n  Studies with NO clot/wall labels (100% blood):")
    for n in sorted(no_tissue):
        print(f"    {n}")

# Very short events only
short_events = [n for n, s in new_stats.items()
                if (s.get('clot_mean_dur_s', 999) < 5 and s.get('frac_clot', 0) > 0) or
                   (s.get('wall_mean_dur_s', 999) < 5 and s.get('frac_wall', 0) > 0)]
if short_events:
    print(f"\n  Studies with mean event duration < 5s (may be blanked by duration filter):")
    for n in sorted(short_events):
        s = new_stats[n]
        print(f"    {n}: clot={s.get('clot_mean_dur_s',0):.1f}s, wall={s.get('wall_mean_dur_s',0):.1f}s")

# Extreme R levels
old_blood_r_vals = [s['r_mean_blood'] for s in old_stats.values() if 'r_mean_blood' in s and not np.isnan(s['r_mean_blood'])]
if old_blood_r_vals:
    old_blood_mean = np.mean(old_blood_r_vals)
    old_blood_std = np.std(old_blood_r_vals)
    outliers = [(n, s['r_mean_blood']) for n, s in new_stats.items()
                if 'r_mean_blood' in s and not np.isnan(s['r_mean_blood'])
                and abs(s['r_mean_blood'] - old_blood_mean) > 2 * old_blood_std]
    if outliers:
        print(f"\n  Studies with blood R-level >2σ from old batch (old: {old_blood_mean:.0f} ± {old_blood_std:.0f} Ω):")
        for n, r in sorted(outliers, key=lambda x: x[1]):
            print(f"    {n}: blood R = {r:.0f} Ω  ({(r-old_blood_mean)/old_blood_std:+.1f}σ)")

# Very high wall fraction (possible mislabeling)
high_wall = [(n, s['frac_wall']) for n, s in new_stats.items() if s.get('frac_wall', 0) > 0.4]
if high_wall:
    print(f"\n  Studies with >40% wall (unusual — check labeling):")
    for n, frac in sorted(high_wall, key=lambda x: -x[1]):
        print(f"    {n}: {frac:.1%} wall")

print(f"\n{'='*70}")
print("DONE")
