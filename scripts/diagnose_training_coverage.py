"""Diagnostic: characterize training data class balance and resistance
distribution. Answers whether the training set contains enough high-R clot
examples to teach the model to distinguish clot from wall at high resistance."""

import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
train_dir = REPO / 'training_data'
test_dir = REPO / 'test_data'

files = sorted(train_dir.glob('*.parquet'))
print(f'Training files: {len(files)}')

total = {0: 0, 1: 0, 2: 0}
bins = [0, 800, 1000, 1200, 1400, 1600, 1800, 2000, 5000]
hist_by_label = {
    0: np.zeros(len(bins) - 1, dtype=np.int64),
    1: np.zeros(len(bins) - 1, dtype=np.int64),
    2: np.zeros(len(bins) - 1, dtype=np.int64),
}

for f in files:
    df = pd.read_parquet(f, columns=['magRLoadAdjusted', 'label'])
    for lbl in (0, 1, 2):
        mask = df['label'].values == lbl
        total[lbl] += int(mask.sum())
        r = df.loc[mask, 'magRLoadAdjusted'].values
        h, _ = np.histogram(r, bins=bins)
        hist_by_label[lbl] += h

grand = sum(total.values())
print()
print('CLASS TOTALS (training data):')
for lbl, name in [(0, 'blood'), (1, 'clot'), (2, 'wall')]:
    print(f'  {name:5s}: {total[lbl]:>12,d}  ({100*total[lbl]/grand:5.1f}%)')
print(f'  total: {grand:>12,d}')

print()
print('RESISTANCE DISTRIBUTION per class (counts):')
print(f'  {"bin":<15s} {"blood":>12s} {"clot":>12s} {"wall":>12s}')
for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
    label = f'{lo}-{hi}'
    print(f'  {label:<15s} {hist_by_label[0][i]:>12,d} {hist_by_label[1][i]:>12,d} {hist_by_label[2][i]:>12,d}')

print()
print('HIGH-R CLOT vs WALL COVERAGE (the key question):')
for thr in (1200, 1400, 1600, 1800):
    clot_hi = int(sum(hist_by_label[1][i] for i, lo in enumerate(bins[:-1]) if lo >= thr))
    wall_hi = int(sum(hist_by_label[2][i] for i, lo in enumerate(bins[:-1]) if lo >= thr))
    ratio = clot_hi / max(wall_hi, 1)
    print(f'  R >= {thr}:  clot={clot_hi:>10,d}  wall={wall_hi:>10,d}  clot/wall={ratio:.2f}')

print()
print('=' * 70)
print('COMPARISON: PALM0507 failure region')
print('=' * 70)
palm = pd.read_parquet(test_dir / 'PALM0507_labeled_segment.parquet',
                       columns=['timeInMS', 'magRLoadAdjusted', 'label'])
palm_ms = palm['timeInMS'].values
palm_r = palm['magRLoadAdjusted'].values
palm_lbl = palm['label'].values

# Failure region approximately 2900-3150s per the plot
lo_ms, hi_ms = 2900_000, 3150_000
region_mask = (palm_ms >= lo_ms) & (palm_ms <= hi_ms)
r_reg = palm_r[region_mask]
l_reg = palm_lbl[region_mask]

print(f'PALM0507 window 2900-3150s: {region_mask.sum():,} samples')
for lbl, name in [(0, 'blood'), (1, 'clot'), (2, 'wall')]:
    m = l_reg == lbl
    if m.sum() > 0:
        r = r_reg[m]
        print(f'  {name:5s}: {int(m.sum()):>7,d} samples, R min={r.min():.0f}, mean={r.mean():.0f}, max={r.max():.0f}')

# Where does the true-clot part of PALM live on the R scale?
clot_r_palm = r_reg[l_reg == 1]
if len(clot_r_palm) > 0:
    print()
    print(f'PALM true-clot in that window: R percentiles')
    for p in (5, 25, 50, 75, 95):
        print(f'    p{p:>2d}: {np.percentile(clot_r_palm, p):.0f}')

    # How many training clot samples fall in that same R range?
    p25, p75 = np.percentile(clot_r_palm, [25, 75])
    print()
    print(f'Training-set clot coverage in PALM failure R range (p25-p75 = {p25:.0f}-{p75:.0f}):')
    covered_clot = 0
    covered_wall = 0
    for f in files:
        df = pd.read_parquet(f, columns=['magRLoadAdjusted', 'label'])
        r = df['magRLoadAdjusted'].values
        l = df['label'].values
        in_range = (r >= p25) & (r <= p75)
        covered_clot += int(((l == 1) & in_range).sum())
        covered_wall += int(((l == 2) & in_range).sum())
    print(f'  Training clot samples with R in [{p25:.0f},{p75:.0f}]: {covered_clot:,}')
    print(f'  Training wall samples with R in [{p25:.0f},{p75:.0f}]: {covered_wall:,}')
    if covered_clot + covered_wall > 0:
        pct_clot = 100 * covered_clot / (covered_clot + covered_wall)
        print(f'  → In this R range, training data is {pct_clot:.1f}% clot, {100-pct_clot:.1f}% wall')
        print(f'  → Model prior at this R range biases toward: {"clot" if pct_clot > 50 else "wall"}')
