"""Check how many training samples have da_label=clot/wall but label=blood,
and whether they have real R values (not noise-blanked ~800)."""
import pandas as pd
import numpy as np
import glob

train_files = sorted(glob.glob("training_data/*.parquet"))
print(f"Checking {len(train_files)} training files...\n")

total_da_clot_label_blood = 0
total_da_wall_label_blood = 0
total_real_r_da_nonblood = 0  # da!=0, label=0, R far from 800
total_samples = 0

studies_with_mismatch = []

for f in train_files:
    df = pd.read_parquet(f)
    name = f.split("\\")[-1].split("/")[-1].replace("_labeled_segment.parquet", "")
    total_samples += len(df)

    # da_label=1 (clot) but label=0 (blood)
    da1_lbl0 = (df["da_label"] == 1) & (df["label"] == 0)
    # da_label=2 (wall) but label=0 (blood)
    da2_lbl0 = (df["da_label"] == 2) & (df["label"] == 0)

    n1 = da1_lbl0.sum()
    n2 = da2_lbl0.sum()

    if n1 > 0 or n2 > 0:
        # Check if these have real R values (not noise-blanked to ~800)
        mismatch_mask = da1_lbl0 | da2_lbl0
        r_vals = df.loc[mismatch_mask, "magRLoadAdjusted"]
        # Noise-blanked samples cluster at ~800 +/- 5
        real_r = (r_vals < 780) | (r_vals > 820)
        n_real = real_r.sum()

        total_da_clot_label_blood += n1
        total_da_wall_label_blood += n2
        total_real_r_da_nonblood += n_real

        studies_with_mismatch.append({
            "study": name,
            "da_clot_lbl_blood": n1,
            "da_wall_lbl_blood": n2,
            "real_R_count": n_real,
            "R_mean": r_vals.mean(),
            "R_std": r_vals.std(),
            "R_min": r_vals.min(),
            "R_max": r_vals.max(),
        })

print(f"{'='*90}")
print(f"TRAINING DATA: da_label vs label MISMATCH ANALYSIS")
print(f"{'='*90}")
print(f"Total training samples: {total_samples:,}")
print(f"Samples with da_label=clot but label=blood: {total_da_clot_label_blood:,}")
print(f"Samples with da_label=wall but label=blood:  {total_da_wall_label_blood:,}")
print(f"Of those, with REAL R (not noise ~800±20):    {total_real_r_da_nonblood:,}")
print(f"\nStudies with mismatches: {len(studies_with_mismatch)}")

if studies_with_mismatch:
    print(f"\n{'Study':<20} {'da=clot':>10} {'da=wall':>10} {'real_R':>10} {'R_mean':>10} {'R_range':>20}")
    print("-" * 90)
    for s in sorted(studies_with_mismatch, key=lambda x: x["real_R_count"], reverse=True):
        print(f"{s['study']:<20} {s['da_clot_lbl_blood']:>10,} {s['da_wall_lbl_blood']:>10,} "
              f"{s['real_R_count']:>10,} {s['R_mean']:>10.1f} {s['R_min']:.0f}-{s['R_max']:.0f}")
