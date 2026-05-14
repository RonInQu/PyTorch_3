"""
Batch 2 normal-only with SAME test set as Batch 1 (minus excluded).
Batch 1 test: 33CFB812, 819421BC, 847A1E3F, 8ECEADA6, CENT0008, DD2DFAF4, F427536B, SUMM0127
Excluded (wall_max >= clot_max): 33CFB812, 819421BC, F427536B
Remaining test: 847A1E3F, 8ECEADA6, CENT0008, DD2DFAF4, SUMM0127 (5 studies)

All other eligible normal studies go to training.
"""
import pandas as pd, numpy as np, os, shutil

clot_events = [7, 11]
wall_events = [23]

# Fixed test set (Batch 1 test minus excluded)
test_studies = sorted(['847A1E3F', '8ECEADA6', 'CENT0008', 'DD2DFAF4', 'SUMM0127'])

# Find all eligible normal studies (clot_mean < wall_mean, clot_max > wall_max)
eligible = []
for fname in sorted(os.listdir('v94_data_set/training')):
    path = os.path.join('v94_data_set/training', fname)
    df = pd.read_parquet(path)
    name = fname.replace('_labeled_segment.parquet', '')
    r = df['magRLoadAdjusted']
    lbl = df['label']
    clot_r = r[lbl == 1]
    wall_r = r[lbl == 2]

    if len(clot_r) == 0 or len(wall_r) == 0:
        continue

    cm, wm = clot_r.mean(), wall_r.mean()
    cx, wx = clot_r.max(), wall_r.max()

    if cm < wm and cx > wx:
        eligible.append({'study': name, 'clot_n': len(clot_r), 'wall_n': len(wall_r),
                         'mean_gap': round(cm - wm, 1)})

df_e = pd.DataFrame(eligible)
print(f"Eligible normal studies: {len(df_e)}")

# Train = all eligible minus the 5 fixed test studies
train_studies = sorted(df_e[~df_e['study'].isin(test_studies)]['study'].tolist())

# Verify test studies are in eligible set
for s in test_studies:
    if s in df_e['study'].values:
        print(f"  TEST: {s} ✓")
    else:
        print(f"  TEST: {s} ✗ NOT ELIGIBLE")

print(f"\n=== TEST ({len(test_studies)}) — same as Batch 1 ===")
for s in test_studies:
    row = df_e[df_e['study'] == s]
    if len(row) > 0:
        row = row.iloc[0]
        print(f"  {s:12s}  clot_n={row['clot_n']:6d}  wall_n={row['wall_n']:6d}  mean_gap={row['mean_gap']:+7.1f}")

print(f"\n=== TRAIN ({len(train_studies)}) ===")
for s in train_studies:
    row = df_e[df_e['study'] == s].iloc[0]
    print(f"  {s:12s}  clot_n={row['clot_n']:6d}  wall_n={row['wall_n']:6d}  mean_gap={row['mean_gap']:+7.1f}")

# Copy files
print("\n" + "=" * 60)
print("COPYING FILES")
print("=" * 60)

for folder in ['training_data', 'test_data']:
    if os.path.exists(folder):
        old_files = os.listdir(folder)
        print(f"Clearing {folder}/ ({len(old_files)} old files)")
        for f in old_files:
            os.remove(os.path.join(folder, f))
    else:
        os.makedirs(folder)

for s in train_studies:
    src = os.path.join('v94_data_set', 'training', f'{s}_labeled_segment.parquet')
    dst = os.path.join('training_data', f'{s}_labeled_segment.parquet')
    shutil.copy2(src, dst)

for s in test_studies:
    src = os.path.join('v94_data_set', 'testing', f'{s}_labeled_segment.parquet')
    dst = os.path.join('test_data', f'{s}_labeled_segment.parquet')
    shutil.copy2(src, dst)

print(f"Copied {len(train_studies)} training files (v94/training → training_data/)")
print(f"Copied {len(test_studies)} test files (v94/testing → test_data/)")

# Delete stale cache
cache_file = os.path.join('cache', 'features_w5.0s_s30_seq8_clot_wall_focused.npz')
if os.path.exists(cache_file):
    os.remove(cache_file)
    print(f"Deleted stale cache: {cache_file}")

print(f"\nVerification:")
print(f"  training_data/: {len(os.listdir('training_data'))} files")
print(f"  test_data/:     {len(os.listdir('test_data'))} files")
print(f"\nCompare to Batch 1: 24 train / 5 test (same test set)")
print(f"This run:          {len(train_studies)} train / {len(test_studies)} test")
