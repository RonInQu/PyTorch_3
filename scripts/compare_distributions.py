"""Compare clot/wall R distributions: original 62 studies vs 17 new studies."""
import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Original 62 training studies (from baseline_split_2026-04-07.txt)
original_62 = {
    "09419CF3", "0D9C36A0", "15A93526", "15AC6217", "16621B3E", "18A9A741",
    "1A8F0795", "24AFD80C", "26E955BA", "325A317A", "376DCB0D", "3B90D74B",
    "42CF0AE3", "453F37DC", "48663E05", "4B4BF4DB", "58F78079", "6E7EB56C",
    "73CB9CA1", "743CBF58", "7873BF1D", "81FC0C79", "847A1E3F", "8860D580",
    "8ECEADA6", "8EE40C79", "9C63125D", "B58B74D7", "B9E8EB7F", "BAPT0001",
    "CENT0006", "CENT0007", "CENT0009", "CENT0102", "CENT0161", "CENT0165",
    "D25DD102", "D4793E80", "DBEF90C4", "DD2DFAF4", "EA7C0500", "ELCA0179",
    "F39B2DEA", "F427536B", "F60DF902", "FE454F2D", "HUNT0130", "HUNT0134",
    "HUNT0136", "HUNT0177", "NASHUN01", "NASHUN02", "STCL0090", "STCLD002",
    "SUMM0119", "SUMM0149", "SUMM0152", "SUMM0154", "SUMM0163", "SUMM0183",
    "UH000008", "UNIH0148",
}

train_files = sorted(glob.glob("training_data/*.parquet"))

orig_clot_r, orig_wall_r, orig_blood_r = [], [], []
new_clot_r, new_wall_r, new_blood_r = [], [], []
orig_studies_clot_means, orig_studies_wall_means = [], []
new_studies_clot_means, new_studies_wall_means = [], []
new_study_names = []

for f in train_files:
    df = pd.read_parquet(f)
    name = f.split("\\")[-1].split("/")[-1].replace("_labeled_segment.parquet", "")
    
    blood = df[df["label"] == 0]["magRLoadAdjusted"].values
    clot = df[df["label"] == 1]["magRLoadAdjusted"].values
    wall = df[df["label"] == 2]["magRLoadAdjusted"].values
    
    if name in original_62:
        orig_blood_r.append(blood)
        orig_clot_r.append(clot)
        orig_wall_r.append(wall)
        if len(clot) > 0: orig_studies_clot_means.append((name, clot.mean(), clot.std(), len(clot)))
        if len(wall) > 0: orig_studies_wall_means.append((name, wall.mean(), wall.std(), len(wall)))
    else:
        new_study_names.append(name)
        new_blood_r.append(blood)
        new_clot_r.append(clot)
        new_wall_r.append(wall)
        if len(clot) > 0: new_studies_clot_means.append((name, clot.mean(), clot.std(), len(clot)))
        if len(wall) > 0: new_studies_wall_means.append((name, wall.mean(), wall.std(), len(wall)))

print(f"Original 62 studies matched: {62 - len(new_study_names) - (79 - 62 - len(new_study_names))}")
print(f"New studies ({len(new_study_names)}): {new_study_names}\n")

# Concatenate
orig_clot = np.concatenate(orig_clot_r) if orig_clot_r else np.array([])
orig_wall = np.concatenate(orig_wall_r) if orig_wall_r else np.array([])
new_clot = np.concatenate(new_clot_r) if new_clot_r else np.array([])
new_wall = np.concatenate(new_wall_r) if new_wall_r else np.array([])

print("=" * 90)
print("CLOT R DISTRIBUTION COMPARISON")
print("=" * 90)
print(f"{'':>20} {'Original 62':>20} {'New 17':>20} {'Delta':>15}")
print("-" * 75)
print(f"{'N samples':>20} {len(orig_clot):>20,} {len(new_clot):>20,}")
print(f"{'Mean':>20} {orig_clot.mean():>20.1f} {new_clot.mean():>20.1f} {new_clot.mean()-orig_clot.mean():>+15.1f}")
print(f"{'Std':>20} {orig_clot.std():>20.1f} {new_clot.std():>20.1f}")
print(f"{'Median':>20} {np.median(orig_clot):>20.1f} {np.median(new_clot):>20.1f}")
print(f"{'P5':>20} {np.percentile(orig_clot,5):>20.1f} {np.percentile(new_clot,5):>20.1f}")
print(f"{'P25':>20} {np.percentile(orig_clot,25):>20.1f} {np.percentile(new_clot,25):>20.1f}")
print(f"{'P75':>20} {np.percentile(orig_clot,75):>20.1f} {np.percentile(new_clot,75):>20.1f}")
print(f"{'P95':>20} {np.percentile(orig_clot,95):>20.1f} {np.percentile(new_clot,95):>20.1f}")
print(f"{'Min':>20} {orig_clot.min():>20.1f} {new_clot.min():>20.1f}")
print(f"{'Max':>20} {orig_clot.max():>20.1f} {new_clot.max():>20.1f}")

print(f"\n{'=' * 90}")
print("WALL R DISTRIBUTION COMPARISON")
print("=" * 90)
print(f"{'':>20} {'Original 62':>20} {'New 17':>20} {'Delta':>15}")
print("-" * 75)
print(f"{'N samples':>20} {len(orig_wall):>20,} {len(new_wall):>20,}")
print(f"{'Mean':>20} {orig_wall.mean():>20.1f} {new_wall.mean():>20.1f} {new_wall.mean()-orig_wall.mean():>+15.1f}")
print(f"{'Std':>20} {orig_wall.std():>20.1f} {new_wall.std():>20.1f}")
print(f"{'Median':>20} {np.median(orig_wall):>20.1f} {np.median(new_wall):>20.1f}")
print(f"{'P5':>20} {np.percentile(orig_wall,5):>20.1f} {np.percentile(new_wall,5):>20.1f}")
print(f"{'P25':>20} {np.percentile(orig_wall,25):>20.1f} {np.percentile(new_wall,25):>20.1f}")
print(f"{'P75':>20} {np.percentile(orig_wall,75):>20.1f} {np.percentile(new_wall,75):>20.1f}")
print(f"{'P95':>20} {np.percentile(orig_wall,95):>20.1f} {np.percentile(new_wall,95):>20.1f}")
print(f"{'Min':>20} {orig_wall.min():>20.1f} {new_wall.min():>20.1f}")
print(f"{'Max':>20} {orig_wall.max():>20.1f} {new_wall.max():>20.1f}")

# Clot-Wall overlap
print(f"\n{'=' * 90}")
print("CLOT vs WALL OVERLAP")
print("=" * 90)
for label, clot, wall in [("Original 62", orig_clot, orig_wall), ("New 17", new_clot, new_wall)]:
    # Overlap = fraction of clot samples that fall within wall's P5-P95 range
    w_lo, w_hi = np.percentile(wall, 5), np.percentile(wall, 95)
    clot_in_wall_range = ((clot >= w_lo) & (clot <= w_hi)).sum() / len(clot) * 100
    c_lo, c_hi = np.percentile(clot, 5), np.percentile(clot, 95)
    wall_in_clot_range = ((wall >= c_lo) & (wall <= c_hi)).sum() / len(wall) * 100
    print(f"\n{label}:")
    print(f"  Clot R in wall's P5-P95 range [{w_lo:.0f}-{w_hi:.0f}]: {clot_in_wall_range:.1f}%")
    print(f"  Wall R in clot's P5-P95 range [{c_lo:.0f}-{c_hi:.0f}]: {wall_in_clot_range:.1f}%")

# Per-study breakdown for new 17
print(f"\n{'=' * 90}")
print("NEW STUDIES — PER-STUDY CLOT/WALL R MEANS")
print("=" * 90)
print(f"{'Study':<15} {'Clot mean':>10} {'Clot std':>10} {'Clot N':>8} {'Wall mean':>10} {'Wall std':>10} {'Wall N':>8} {'Gap':>8}")
print("-" * 90)
new_clot_dict = {s[0]: s for s in new_studies_clot_means}
new_wall_dict = {s[0]: s for s in new_studies_wall_means}
all_new = set(s[0] for s in new_studies_clot_means) | set(s[0] for s in new_studies_wall_means)
for name in sorted(all_new):
    c = new_clot_dict.get(name, (name, 0, 0, 0))
    w = new_wall_dict.get(name, (name, 0, 0, 0))
    gap = w[1] - c[1] if c[3] > 0 and w[3] > 0 else float('nan')
    print(f"{name:<15} {c[1]:>10.1f} {c[2]:>10.1f} {c[3]:>8,} {w[1]:>10.1f} {w[2]:>10.1f} {w[3]:>8,} {gap:>+8.1f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, data_orig, data_new, title in [
    (axes[0, 0], orig_clot, new_clot, "Clot R"),
    (axes[0, 1], orig_wall, new_wall, "Wall R"),
]:
    p1, p99 = np.percentile(np.concatenate([data_orig, data_new]), [1, 99])
    ax.hist(data_orig[(data_orig >= p1) & (data_orig <= p99)], bins=100, alpha=0.6, 
            density=True, label=f"Original 62 (n={len(data_orig):,})", color="blue")
    ax.hist(data_new[(data_new >= p1) & (data_new <= p99)], bins=100, alpha=0.6, 
            density=True, label=f"New 17 (n={len(data_new):,})", color="red")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlabel("R (Ω)")

# Clot vs Wall overlay for each group
for ax, clot, wall, title in [
    (axes[1, 0], orig_clot, orig_wall, "Original 62: Clot vs Wall"),
    (axes[1, 1], new_clot, new_wall, "New 17: Clot vs Wall"),
]:
    all_data = np.concatenate([clot, wall])
    p1, p99 = np.percentile(all_data, [1, 99])
    ax.hist(clot[(clot >= p1) & (clot <= p99)], bins=100, alpha=0.6, 
            density=True, label=f"Clot (n={len(clot):,})", color="red")
    ax.hist(wall[(wall >= p1) & (wall <= p99)], bins=100, alpha=0.6, 
            density=True, label=f"Wall (n={len(wall):,})", color="blue")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlabel("R (Ω)")

plt.suptitle("Clot/Wall R Distribution: Original 62 vs New 17 Studies", fontsize=14)
plt.tight_layout()
plt.savefig("clot_wall_R_distribution_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved plot: clot_wall_R_distribution_comparison.png")
