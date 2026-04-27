"""Quick check: compare training data characteristics."""
import pandas as pd
import numpy as np
import glob

train_files = sorted(glob.glob("training_data/*.parquet"))
print(f"=== TRAINING DATA: {len(train_files)} files ===\n")

all_blood_means = []
all_clot_means = []
all_wall_means = []

for f in train_files[:5]:
    df = pd.read_parquet(f)
    name = f.split("\\")[-1].split("/")[-1]
    r = df["magRLoadAdjusted"]
    blood = df[df["label"] == 0]["magRLoadAdjusted"]
    clot = df[df["label"] == 1]["magRLoadAdjusted"]
    wall = df[df["label"] == 2]["magRLoadAdjusted"]
    print(f"{name}")
    print(f"  R range: {r.min():.1f} - {r.max():.1f}, mean: {r.mean():.1f}")
    print(f"  Blood mean: {blood.mean():.1f} (n={len(blood)})")
    print(f"  Clot mean:  {clot.mean():.1f} (n={len(clot)})")
    print(f"  Wall mean:  {wall.mean():.1f} (n={len(wall)})")
    print(f"  Samples >5000: {(r > 5000).sum()}")
    print()

# Global stats across ALL files
print("=== GLOBAL STATS (all files) ===")
for f in train_files:
    df = pd.read_parquet(f)
    blood = df[df["label"] == 0]["magRLoadAdjusted"]
    clot = df[df["label"] == 1]["magRLoadAdjusted"]
    wall = df[df["label"] == 2]["magRLoadAdjusted"]
    all_blood_means.append(blood.mean())
    if len(clot) > 0: all_clot_means.append(clot.mean())
    if len(wall) > 0: all_wall_means.append(wall.mean())

print(f"Blood mean across studies: {np.mean(all_blood_means):.1f} +/- {np.std(all_blood_means):.1f}")
print(f"Clot mean across studies:  {np.mean(all_clot_means):.1f} +/- {np.std(all_clot_means):.1f}")
print(f"Wall mean across studies:  {np.mean(all_wall_means):.1f} +/- {np.std(all_wall_means):.1f}")
print(f"Total training files: {len(train_files)}")

# Check test data too
test_files = sorted(glob.glob("test_data/*.parquet"))
print(f"\n=== TEST DATA: {len(test_files)} files ===")
for f in test_files[:3]:
    df = pd.read_parquet(f)
    name = f.split("\\")[-1].split("/")[-1]
    r = df["magRLoadAdjusted"]
    print(f"{name}")
    print(f"  R range: {r.min():.1f} - {r.max():.1f}")
    print(f"  Samples >5000: {(r > 5000).sum()}")
    # Check if label -1 exists
    if (df["label"] == -1).any():
        print(f"  UNLABELED (label=-1): {(df['label']==-1).sum()}")
    print()
