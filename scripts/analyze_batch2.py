"""Analyze all v94 studies for clot/wall R characteristics."""
import pandas as pd, numpy as np, os

clot_events = [7, 11]
wall_events = [23]
blood_events = [6, 12]

results = []
for fname in sorted(os.listdir('v94_data_set/training')):
    path = os.path.join('v94_data_set/training', fname)
    df = pd.read_parquet(path)
    name = fname.replace('_labeled_segment.parquet', '')

    r = df['magRLoadAdjusted']
    lbl = df['label']

    clot_r = r[lbl == 1]
    wall_r = r[lbl == 2]
    blood_r = r[lbl == 0]

    if len(clot_r) == 0 or len(wall_r) == 0:
        results.append({
            'study': name, 'clot_n': len(clot_r), 'wall_n': len(wall_r),
            'clot_mean': None, 'wall_mean': None, 'clot_max': None, 'wall_max': None,
            'clot_p95': None, 'wall_p95': None, 'blood_mean': None,
            'mean_gap': None, 'max_gap': None, 'p95_gap': None, 'category': 'NO_TISSUE'
        })
        continue

    cm, wm = clot_r.mean(), wall_r.mean()
    cx, wx = clot_r.max(), wall_r.max()
    cp95 = clot_r.quantile(0.95)
    wp95 = wall_r.quantile(0.95)
    bm = blood_r.mean() if len(blood_r) > 0 else 800

    mean_gap = cm - wm
    max_gap = cx - wx
    p95_gap = cp95 - wp95

    if cm < wm:
        cat = 'NORMAL'
    else:
        cat = 'INVERTED'

    results.append({
        'study': name, 'clot_n': len(clot_r), 'wall_n': len(wall_r),
        'clot_mean': round(cm, 1), 'wall_mean': round(wm, 1),
        'clot_max': round(cx, 1), 'wall_max': round(wx, 1),
        'clot_p95': round(cp95, 1), 'wall_p95': round(wp95, 1),
        'blood_mean': round(bm, 1),
        'mean_gap': round(mean_gap, 1), 'max_gap': round(max_gap, 1),
        'p95_gap': round(p95_gap, 1), 'category': cat
    })

df_r = pd.DataFrame(results)
has_tissue = df_r[df_r['category'] != 'NO_TISSUE']

print(f"Total: {len(df_r)} studies")
n_norm = (df_r['category'] == 'NORMAL').sum()
n_inv = (df_r['category'] == 'INVERTED').sum()
n_no = (df_r['category'] == 'NO_TISSUE').sum()
print(f"NORMAL (clot_mean < wall_mean): {n_norm}")
print(f"INVERTED (clot_mean > wall_mean): {n_inv}")
print(f"NO_TISSUE: {n_no}")

# Among INVERTED, check spike differentiator
inv = df_r[df_r['category'] == 'INVERTED'].copy()
print(f"\nAmong INVERTED ({len(inv)}):")
print(f"  clot_max > wall_max: {(inv['max_gap'] > 0).sum()}")
print(f"  clot_max <= wall_max: {(inv['max_gap'] <= 0).sum()}")
print(f"  clot_p95 > wall_p95: {(inv['clot_p95'] > inv['wall_p95']).sum()}")
print(f"  clot_p95 <= wall_p95: {(inv['clot_p95'] <= inv['wall_p95']).sum()}")

norm = df_r[df_r['category'] == 'NORMAL'].copy()
print(f"\nAmong NORMAL ({len(norm)}):")
print(f"  clot_max > wall_max: {(norm['max_gap'] > 0).sum()}")
print(f"  clot_max <= wall_max: {(norm['max_gap'] <= 0).sum()}")

# Check which Batch 1 studies are in v94
batch1_train = ['15AC6217','16621B3E','1A8F0795','376DCB0D','3B90D74B','43140EA7',
    '48663E05','530618CC','6E7EB56C','73CB9CA1','81FC0C79','8860D580',
    'A225B105','AFF18ECE','BAPT0001','CENT0007','CENT0009','D25DD102',
    'D4793E80','F39B2DEA','HUNT0120','NASHUN01','STCLD002','SUMM0119']
batch1_test = ['33CFB812','819421BC','847A1E3F','8ECEADA6','CENT0008','DD2DFAF4','F427536B','SUMM0127']
batch1_excluded = ['0C7C8AB7','18BAA0D1','1C86FE05','2149DBD1','248FE3E6',
    '29CD8A13','3E146478','4633BDC0','65A19BFE','6CEE5D0B','7AFBE9EB',
    '7B42C83E','86FA6755','9C23F1D1','9D0CBDAA','A73ECEB3','B3C7E4A2',
    'BFF7FDB8','C9C8F4EB','CENT0006','SOMI0153','UHMAX001','FD47B6D5']
batch1_all = set(batch1_train + batch1_test + batch1_excluded)

new_studies = sorted([r['study'] for _, r in df_r.iterrows() if r['study'] not in batch1_all])
print(f"\nNew studies in v94 (not in Batch 1): {len(new_studies)}")
for s in new_studies:
    row = df_r[df_r['study'] == s].iloc[0]
    if row['category'] == 'NO_TISSUE':
        print(f"  {s}: NO CLOT/WALL DATA")
    else:
        print(f"  {s}: mean_gap={row['mean_gap']:+.1f}  max_gap={row['max_gap']:+.1f}  "
              f"p95_gap={row['p95_gap']:+.1f}  [{row['category']}]")

print("\n" + "=" * 120)
print(f"{'Study':12s}  {'mean_gap':>9s}  {'max_gap':>9s}  {'p95_gap':>9s}  "
      f"{'clot_max':>9s}  {'wall_max':>9s}  {'clot_p95':>9s}  {'wall_p95':>9s}  "
      f"{'clot_n':>7s}  {'wall_n':>7s}  {'cat':8s}  batch1")
print("=" * 120)
for _, row in df_r.sort_values('mean_gap').iterrows():
    if row['category'] == 'NO_TISSUE':
        b1 = 'excl' if row['study'] in batch1_excluded else ('train' if row['study'] in batch1_train else ('test' if row['study'] in batch1_test else 'NEW'))
        print(f"{row['study']:12s}  {'---':>9s}  {'---':>9s}  {'---':>9s}  "
              f"{'---':>9s}  {'---':>9s}  {'---':>9s}  {'---':>9s}  "
              f"{row['clot_n']:7d}  {row['wall_n']:7d}  NO_TISSUE  {b1}")
    else:
        b1 = 'excl' if row['study'] in batch1_excluded else ('train' if row['study'] in batch1_train else ('test' if row['study'] in batch1_test else 'NEW'))
        print(f"{row['study']:12s}  {row['mean_gap']:+9.1f}  {row['max_gap']:+9.1f}  {row['p95_gap']:+9.1f}  "
              f"{row['clot_max']:9.1f}  {row['wall_max']:9.1f}  {row['clot_p95']:9.1f}  {row['wall_p95']:9.1f}  "
              f"{row['clot_n']:7d}  {row['wall_n']:7d}  {row['category']:8s}  {b1}")
