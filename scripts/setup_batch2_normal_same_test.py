"""
Batch 2 normal-only with EXACT same 8 test studies as Batch 1.
Test: 33CFB812, 819421BC, 847A1E3F, 8ECEADA6, CENT0008, DD2DFAF4, F427536B, SUMM0127
All other eligible normal studies (clot_mean < wall_mean) go to training.
Test studies kept regardless of clot_max vs wall_max — they are for evaluation only.
"""
import pandas as pd, numpy as np, os, shutil

clot_events = [7, 11]
wall_events = [23]

# Fixed test set — EXACT Batch 1 test
test_studies = sorted(['33CFB812', '819421BC', '847A1E3F', '8ECEADA6', 'CENT0008', 'DD2DFAF4', 'F427536B', 'SUMM0127'])

# Find all eligible normal studies for TRAINING (clot_mean < wall_mean, clot_max > wall_max)
eligible_train = []
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

    # Training: normal polarity with clot_max > wall_max (same criteria as before)
    if cm < wm and cx > wx and name not in test_studies:
        eligible_train.append({'study': name, 'clot_n': len(clot_r), 'wall_n': len(wall_r),
                               'mean_gap': round(cm - wm, 1)})

train_studies = sorted([r['study'] for r in eligible_train])

print(f"=== TEST ({len(test_studies)}) — EXACT Batch 1 test set ===")
for s in test_studies:
    print(f"  {s}")

print(f"\n=== TRAIN ({len(train_studies)}) ===")
for r in sorted(eligible_train, key=lambda x: x['study']):
    print(f"  {r['study']:12s}  clot_n={r['clot_n']:6d}  wall_n={r['wall_n']:6d}  mean_gap={r['mean_gap']:+7.1f}")

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
print(f"\nBatch 1: 24 train / 8 test")
print(f"This:    {len(train_studies)} train / 8 test (same test set, more training data)")
