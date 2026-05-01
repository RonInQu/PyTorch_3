"""
Batch 2 normal-only: 34 studies where clot_mean < wall_mean AND clot_max > wall_max.
Copy from v94_data_set/training → training_data/, v94_data_set/testing → test_data/.
"""
import pandas as pd, numpy as np, os, shutil

clot_events = [7, 11]
wall_events = [23]

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
        continue

    cm, wm = clot_r.mean(), wall_r.mean()
    cx, wx = clot_r.max(), wall_r.max()
    mean_gap = cm - wm

    # Normal = clot_mean < wall_mean AND clot_max > wall_max
    if mean_gap <= 0 and cx > wx:
        results.append({
            'study': name, 'clot_n': len(clot_r), 'wall_n': len(wall_r),
            'clot_mean': round(cm, 1), 'wall_mean': round(wm, 1),
            'clot_max': round(cx, 1), 'wall_max': round(wx, 1),
            'mean_gap': round(mean_gap, 1),
            'min_tissue': min(len(clot_r), len(wall_r))
        })

df_r = pd.DataFrame(results).sort_values('min_tissue', ascending=False)
print(f"Normal studies with clot_max > wall_max: {len(df_r)}")

# Pick ~8 for test (best tissue coverage), rest for train
test_studies = df_r.head(8)['study'].tolist()
train_studies = sorted(df_r[~df_r['study'].isin(test_studies)]['study'].tolist())
test_studies = sorted(test_studies)

print(f"\n=== TEST ({len(test_studies)}) ===")
for s in test_studies:
    row = df_r[df_r['study'] == s].iloc[0]
    print(f"  {s:12s}  clot_n={row['clot_n']:6d}  wall_n={row['wall_n']:6d}  "
          f"mean_gap={row['mean_gap']:+7.1f}  clot_max={row['clot_max']:.0f}  wall_max={row['wall_max']:.0f}")

print(f"\n=== TRAIN ({len(train_studies)}) ===")
for s in train_studies:
    row = df_r[df_r['study'] == s].iloc[0]
    print(f"  {s:12s}  clot_n={row['clot_n']:6d}  wall_n={row['wall_n']:6d}  "
          f"mean_gap={row['mean_gap']:+7.1f}  clot_max={row['clot_max']:.0f}  wall_max={row['wall_max']:.0f}")

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
