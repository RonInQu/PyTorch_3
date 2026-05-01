"""
Batch 3 (April 6 data): clot_mean < wall_mean filter, same 8 test studies as Batch 1.
Source: vApril6_data_set/training → training_data/, vApril6_data_set/testing → test_data/
"""
import pandas as pd, numpy as np, os, shutil

clot_events = [7, 11]
wall_events = [23]

# Fixed test set — EXACT Batch 1 test
test_studies = sorted(['33CFB812', '819421BC', '847A1E3F', '8ECEADA6', 'CENT0008', 'DD2DFAF4', 'F427536B', 'SUMM0127'])

# Find all normal studies (clot_mean < wall_mean)
all_studies = []
eligible_train = []
for fname in sorted(os.listdir('vApril6_data_set/training')):
    path = os.path.join('vApril6_data_set/training', fname)
    df = pd.read_parquet(path)
    name = fname.replace('_labeled_segment.parquet', '')
    r = df['magRLoadAdjusted']
    lbl = df['label']
    clot_r = r[lbl == 1]
    wall_r = r[lbl == 2]

    if len(clot_r) == 0 or len(wall_r) == 0:
        all_studies.append({'study': name, 'clot_n': len(clot_r), 'wall_n': len(wall_r),
                           'mean_gap': None, 'category': 'NO_TISSUE'})
        continue

    cm, wm = clot_r.mean(), wall_r.mean()
    mean_gap = cm - wm
    cat = 'NORMAL' if cm < wm else 'INVERTED'
    all_studies.append({'study': name, 'clot_n': len(clot_r), 'wall_n': len(wall_r),
                       'mean_gap': round(mean_gap, 1), 'category': cat})

    if cm < wm and name not in test_studies:
        eligible_train.append({'study': name, 'clot_n': len(clot_r), 'wall_n': len(wall_r),
                               'mean_gap': round(mean_gap, 1)})

df_all = pd.DataFrame(all_studies)
n_total = len(df_all)
n_norm = (df_all['category'] == 'NORMAL').sum()
n_inv = (df_all['category'] == 'INVERTED').sum()
n_no = (df_all['category'] == 'NO_TISSUE').sum()

print(f"Total studies in vApril6_data_set: {n_total}")
print(f"  NORMAL (clot_mean < wall_mean): {n_norm}")
print(f"  INVERTED (clot_mean > wall_mean): {n_inv}")
print(f"  NO_TISSUE: {n_no}")

train_studies = sorted([r['study'] for r in eligible_train])

# Check test studies exist in this dataset
print(f"\n=== TEST ({len(test_studies)}) — EXACT Batch 1 test set ===")
for s in test_studies:
    exists = os.path.exists(os.path.join('vApril6_data_set', 'testing', f'{s}_labeled_segment.parquet'))
    row = df_all[df_all['study'] == s]
    if len(row) > 0:
        row = row.iloc[0]
        status = f"[{row['category']}]" if row['category'] != 'NO_TISSUE' else "[NO_TISSUE]"
    else:
        status = "[NOT IN DATASET]"
    print(f"  {s:12s}  testing_file={'YES' if exists else 'NO'}  {status}")

print(f"\n=== TRAIN ({len(train_studies)}) — filter: clot_mean < wall_mean ===")
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
    src = os.path.join('vApril6_data_set', 'training', f'{s}_labeled_segment.parquet')
    dst = os.path.join('training_data', f'{s}_labeled_segment.parquet')
    shutil.copy2(src, dst)

copied_test = 0
for s in test_studies:
    src = os.path.join('vApril6_data_set', 'testing', f'{s}_labeled_segment.parquet')
    dst = os.path.join('test_data', f'{s}_labeled_segment.parquet')
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied_test += 1
    else:
        print(f"  WARNING: {s} not found in vApril6_data_set/testing/")

print(f"Copied {len(train_studies)} training files (vApril6/training → training_data/)")
print(f"Copied {copied_test} test files (vApril6/testing → test_data/)")

# Delete stale cache
cache_file = os.path.join('cache', 'features_w5.0s_s30_seq8_clot_wall_focused.npz')
if os.path.exists(cache_file):
    os.remove(cache_file)
    print(f"Deleted stale cache: {cache_file}")

print(f"\nVerification:")
print(f"  training_data/: {len(os.listdir('training_data'))} files")
print(f"  test_data/:     {len(os.listdir('test_data'))} files")
print(f"\nBatch 1: 24 train / 8 test")
print(f"Batch 2: 31 train / 8 test")
print(f"This:    {len(train_studies)} train / {copied_test} test")
