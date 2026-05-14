"""Compare meanR vs maxR polarity filters on training set."""
import pandas as pd
import numpy as np

df = pd.read_csv('analysis_data_drift/study_summaries.csv')
train = df[df['split'] == 'train'].copy()
train['max_ratio'] = train['clot_max_R'] / train['wall_max_R']

has_both = train.dropna(subset=['clot_wall_ratio', 'max_ratio'])
no_wall = train[train['n_wall_events'] == 0]
no_clot = train[train['n_clot_events'] == 0]

print(f"Total train: {len(train)}")
print(f"  Has both clot+wall: {len(has_both)}")
print(f"  No wall events: {len(no_wall)} -> {list(no_wall['study_id'])}")
print(f"  No clot events: {len(no_clot)} -> {list(no_clot['study_id'])}")

mean_normal = has_both[has_both['clot_wall_ratio'] <= 1.0]
max_normal = has_both[has_both['max_ratio'] > 1.0]
mean_inv = has_both[has_both['clot_wall_ratio'] > 1.0]
max_inv = has_both[has_both['max_ratio'] <= 1.0]

print(f"\nFilter comparison (studies with both clot+wall = {len(has_both)}):")
print(f"  meanR filter (clot_mean < wall_mean):  KEEP {len(mean_normal)} / DROP {len(mean_inv)}")
print(f"  maxR  filter (clot_max  > wall_max):   KEEP {len(max_normal)} / DROP {len(max_inv)}")

mean_keeps = set(mean_normal['study_id'])
max_keeps = set(max_normal['study_id'])
added = max_keeps - mean_keeps
lost = mean_keeps - max_keeps

print(f"\nmaxR filter vs meanR filter:")
print(f"  Added (mean-inverted but max-normal): {len(added)} studies")
for s in sorted(added):
    r = has_both[has_both['study_id'] == s].iloc[0]
    print(f"    {s:15s}  mean_ratio={r['clot_wall_ratio']:.3f}  max_ratio={r['max_ratio']:.3f}")

print(f"  Lost (mean-normal but max-inverted): {len(lost)} studies")
for s in sorted(lost):
    r = has_both[has_both['study_id'] == s].iloc[0]
    print(f"    {s:15s}  mean_ratio={r['clot_wall_ratio']:.3f}  max_ratio={r['max_ratio']:.3f}")

nowall = set(no_wall['study_id'])
noclot = set(no_clot['study_id'])
other = len(train) - len(has_both)  # studies without both
print(f"\nFinal training set sizes:")
print(f"  meanR filter: {len(mean_keeps)} (filtered) + {other} (no filter needed) = {len(mean_keeps) + other}")
print(f"  maxR  filter: {len(max_keeps)} (filtered) + {other} (no filter needed) = {len(max_keeps) + other}")
print(f"  No filter:    {len(train)}")

# Also check test set
test = df[df['split'] == 'test'].copy()
test['max_ratio'] = test['clot_max_R'] / test['wall_max_R']
test_both = test.dropna(subset=['clot_wall_ratio', 'max_ratio'])
print(f"\nTest set ({len(test)} total, {len(test_both)} with both):")
for _, r in test_both.iterrows():
    flag_mean = "INV" if r['clot_wall_ratio'] > 1.0 else "ok "
    flag_max = "INV" if r['max_ratio'] <= 1.0 else "ok "
    print(f"  {r['study_id']:15s}  mean_ratio={r['clot_wall_ratio']:.3f}({flag_mean})  "
          f"max_ratio={r['max_ratio']:.3f}({flag_max})")
