"""Plot clot/wall R distributions for original 62 and new 17 studies — per-study view."""
import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Original 62 training studies
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

# Collect per-study stats
orig_stats = []  # (name, clot_mean, wall_mean, clot_n, wall_n)
new_stats = []

for f in train_files:
    df = pd.read_parquet(f)
    name = f.split("\\")[-1].split("/")[-1].replace("_labeled_segment.parquet", "")
    clot = df[df["label"] == 1]["magRLoadAdjusted"].values
    wall = df[df["label"] == 2]["magRLoadAdjusted"].values
    
    entry = (name, 
             clot.mean() if len(clot) > 0 else np.nan,
             wall.mean() if len(wall) > 0 else np.nan,
             len(clot), len(wall))
    
    if name in original_62:
        orig_stats.append(entry)
    else:
        new_stats.append(entry)

# ── Figure 1: Per-study clot mean vs wall mean scatter ──
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left: scatter plot clot_mean vs wall_mean
ax = axes[0]
for stats, color, label in [(orig_stats, "blue", "Original 62"), (new_stats, "red", "New 17")]:
    clot_means = [s[1] for s in stats if not np.isnan(s[1]) and not np.isnan(s[2])]
    wall_means = [s[2] for s in stats if not np.isnan(s[1]) and not np.isnan(s[2])]
    names = [s[0] for s in stats if not np.isnan(s[1]) and not np.isnan(s[2])]
    ax.scatter(clot_means, wall_means, c=color, s=60, alpha=0.7, label=label, edgecolors='k', linewidth=0.5)
    # Label new studies
    if label == "New 17":
        for n, cx, wy in zip(names, clot_means, wall_means):
            ax.annotate(n, (cx, wy), fontsize=6, alpha=0.8, xytext=(3, 3), textcoords='offset points')

# Diagonal line: clot_mean == wall_mean
lims = [700, 2500]
ax.plot(lims, lims, 'k--', alpha=0.4, label="clot=wall line")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Clot R mean (Ω)", fontsize=12)
ax.set_ylabel("Wall R mean (Ω)", fontsize=12)
ax.set_title("Per-Study: Clot Mean vs Wall Mean R", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Right: gap (wall_mean - clot_mean) histogram
ax = axes[1]
orig_gaps = [s[2] - s[1] for s in orig_stats if not np.isnan(s[1]) and not np.isnan(s[2])]
new_gaps = [s[2] - s[1] for s in new_stats if not np.isnan(s[1]) and not np.isnan(s[2])]
bins = np.arange(-1500, 1500, 50)
ax.hist(orig_gaps, bins=bins, alpha=0.6, color="blue", label=f"Original 62 (n={len(orig_gaps)})")
ax.hist(new_gaps, bins=bins, alpha=0.6, color="red", label=f"New 17 (n={len(new_gaps)})")
ax.axvline(0, color='k', linestyle='--', alpha=0.5)
ax.axvline(np.mean(orig_gaps), color='blue', linestyle='-', alpha=0.8, linewidth=2, label=f"Orig mean gap: {np.mean(orig_gaps):+.0f}")
ax.axvline(np.mean(new_gaps), color='red', linestyle='-', alpha=0.8, linewidth=2, label=f"New mean gap: {np.mean(new_gaps):+.0f}")
ax.set_xlabel("Wall mean − Clot mean (Ω)", fontsize=12)
ax.set_ylabel("Number of studies", fontsize=12)
ax.set_title("Per-Study Gap: Wall − Clot R Mean", fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle("Original 62 vs New 17 Studies — Clot/Wall R Distributions", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("clot_wall_per_study_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: clot_wall_per_study_comparison.png")

# ── Figure 2: Pooled histograms (clot and wall separately) with per-group overlay ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Collect pooled data
orig_clot_all, orig_wall_all = [], []
new_clot_all, new_wall_all = [], []
for f in train_files:
    df = pd.read_parquet(f)
    name = f.split("\\")[-1].split("/")[-1].replace("_labeled_segment.parquet", "")
    clot = df[df["label"] == 1]["magRLoadAdjusted"].values
    wall = df[df["label"] == 2]["magRLoadAdjusted"].values
    if name in original_62:
        orig_clot_all.append(clot)
        orig_wall_all.append(wall)
    else:
        new_clot_all.append(clot)
        new_wall_all.append(wall)

orig_clot = np.concatenate(orig_clot_all)
orig_wall = np.concatenate(orig_wall_all)
new_clot = np.concatenate(new_clot_all)
new_wall = np.concatenate(new_wall_all)

bins_r = np.arange(600, 3500, 20)

# Top-left: Original 62 clot vs wall
ax = axes[0, 0]
ax.hist(orig_clot, bins=bins_r, alpha=0.6, density=True, color="red", label=f"Clot (n={len(orig_clot):,})")
ax.hist(orig_wall, bins=bins_r, alpha=0.6, density=True, color="blue", label=f"Wall (n={len(orig_wall):,})")
ax.set_title("Original 62: Clot vs Wall R", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlabel("R (Ω)")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.3)

# Top-right: New 17 clot vs wall
ax = axes[0, 1]
ax.hist(new_clot, bins=bins_r, alpha=0.6, density=True, color="red", label=f"Clot (n={len(new_clot):,})")
ax.hist(new_wall, bins=bins_r, alpha=0.6, density=True, color="blue", label=f"Wall (n={len(new_wall):,})")
ax.set_title("New 17: Clot vs Wall R", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlabel("R (Ω)")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.3)

# Bottom-left: Clot comparison (orig vs new)
ax = axes[1, 0]
ax.hist(orig_clot, bins=bins_r, alpha=0.6, density=True, color="blue", label=f"Original 62 (n={len(orig_clot):,})")
ax.hist(new_clot, bins=bins_r, alpha=0.6, density=True, color="red", label=f"New 17 (n={len(new_clot):,})")
ax.set_title("Clot R: Original 62 vs New 17", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlabel("R (Ω)")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.3)

# Bottom-right: Wall comparison (orig vs new)
ax = axes[1, 1]
ax.hist(orig_wall, bins=bins_r, alpha=0.6, density=True, color="blue", label=f"Original 62 (n={len(orig_wall):,})")
ax.hist(new_wall, bins=bins_r, alpha=0.6, density=True, color="red", label=f"New 17 (n={len(new_wall):,})")
ax.set_title("Wall R: Original 62 vs New 17", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlabel("R (Ω)")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.3)

plt.suptitle("Pooled R Distributions — Original 62 vs New 17", fontsize=15)
plt.tight_layout()
plt.savefig("clot_wall_pooled_histograms.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: clot_wall_pooled_histograms.png")
