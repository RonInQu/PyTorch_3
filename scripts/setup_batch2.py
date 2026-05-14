"""
Batch 2 cherry-pick: split 59 studies (clot_max > wall_max) into train/test.
Copy from v94_data_set/training → training_data/, v94_data_set/testing → test_data/.
"""
import pandas as pd, numpy as np, os, shutil

clot_events = [7, 11]
wall_events = [23]
blood_events = [6, 12]

# Analyze all studies
results = []
for fname in sorted(os.listdir('v94_data_set/training')):
    path = os.path.join('v94_data_set/training', fname)
    df = pd.read_parquet(path)
    name = fname.replace('_labeled_segment.parquet', '')
    r = df['magRLoadAdjusted']
    lbl = df['label']
    clot_r = r[lbl == 1]
    wall_r = r[lbl == 2]
    
    if len(clot_r) == 0 or len(wall_r) == 0:
        results.append({'study': name, 'clot_max': None, 'wall_max': None,
                        'clot_n': len(clot_r), 'wall_n': len(wall_r),
                        'mean_gap': None, 'eligible': False, 'reason': 'no_tissue'})
        continue
    
    cx, wx = clot_r.max(), wall_r.max()
    cm, wm = clot_r.mean(), wall_r.mean()
    
    eligible = cx > wx
    reason = 'ok' if eligible else f'wall_max({wx:.0f}) >= clot_max({cx:.0f})'
    
    results.append({
        'study': name, 'clot_max': cx, 'wall_max': wx,
        'clot_n': len(clot_r), 'wall_n': len(wall_r),
        'mean_gap': round(cm - wm, 1),
        'clot_mean': round(cm, 1), 'wall_mean': round(wm, 1),
        'eligible': eligible, 'reason': reason
    })

df_r = pd.DataFrame(results)
eligible = df_r[df_r['eligible']].copy()
excluded = df_r[~df_r['eligible']].copy()

print(f"Eligible (clot_max > wall_max): {len(eligible)}")
print(f"Excluded: {len(excluded)}")
for _, row in excluded.iterrows():
    print(f"  {row['study']}: {row['reason']}")

# ─── Pick test set: ~12 studies, mix of normal/inverted, good tissue counts ───
# Prioritize: studies with both clot_n > 2000 and wall_n > 2000 for reliable metrics
eligible['is_inverted'] = eligible['mean_gap'] > 0
eligible['min_tissue'] = eligible[['clot_n', 'wall_n']].min(axis=1)

# Sort by min_tissue descending to pick studies with good data coverage
# Select ~6 normal + ~6 inverted for test
normal = eligible[~eligible['is_inverted']].sort_values('min_tissue', ascending=False)
inverted = eligible[eligible['is_inverted']].sort_values('min_tissue', ascending=False)

print(f"\nEligible NORMAL: {len(normal)}, INVERTED: {len(inverted)}")

# Pick 6 normal and 6 inverted for test
test_normal = normal.head(6)['study'].tolist()
test_inverted = inverted.head(6)['study'].tolist()
test_studies = sorted(test_normal + test_inverted)

train_studies = sorted(eligible[~eligible['study'].isin(test_studies)]['study'].tolist())

print(f"\n=== TEST ({len(test_studies)}) ===")
for s in test_studies:
    row = eligible[eligible['study'] == s].iloc[0]
    cat = 'INV' if row['is_inverted'] else 'NRM'
    print(f"  {s:12s}  [{cat}]  clot_n={row['clot_n']:6d}  wall_n={row['wall_n']:6d}  "
          f"mean_gap={row['mean_gap']:+7.1f}  clot_max={row['clot_max']:.0f}  wall_max={row['wall_max']:.0f}")

print(f"\n=== TRAIN ({len(train_studies)}) ===")
for s in train_studies:
    row = eligible[eligible['study'] == s].iloc[0]
    cat = 'INV' if row['is_inverted'] else 'NRM'
    print(f"  {s:12s}  [{cat}]  clot_n={row['clot_n']:6d}  wall_n={row['wall_n']:6d}  "
          f"mean_gap={row['mean_gap']:+7.1f}")

print(f"\n=== EXCLUDED ({len(excluded)}) ===")
for _, row in excluded.iterrows():
    print(f"  {row['study']:12s}  {row['reason']}")

# ─── Copy files ───
print("\n" + "=" * 60)
print("COPYING FILES")
print("=" * 60)

# Clear destination folders
for folder in ['training_data', 'test_data']:
    if os.path.exists(folder):
        old_files = os.listdir(folder)
        print(f"Clearing {folder}/ ({len(old_files)} old files)")
        for f in old_files:
            os.remove(os.path.join(folder, f))
    else:
        os.makedirs(folder)

# Copy train: v94_data_set/training → training_data
for s in train_studies:
    src = os.path.join('v94_data_set', 'training', f'{s}_labeled_segment.parquet')
    dst = os.path.join('training_data', f'{s}_labeled_segment.parquet')
    shutil.copy2(src, dst)

# Copy test: v94_data_set/testing → test_data
for s in test_studies:
    src = os.path.join('v94_data_set', 'testing', f'{s}_labeled_segment.parquet')
    dst = os.path.join('test_data', f'{s}_labeled_segment.parquet')
    shutil.copy2(src, dst)

print(f"Copied {len(train_studies)} training files (v94/training → training_data/)")
print(f"Copied {len(test_studies)} test files (v94/testing → test_data/)")

# Verify
print(f"\nVerification:")
print(f"  training_data/: {len(os.listdir('training_data'))} files")
print(f"  test_data/:     {len(os.listdir('test_data'))} files")
