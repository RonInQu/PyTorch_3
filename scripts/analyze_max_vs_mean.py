"""Quick analysis: max R ratio vs mean R ratio for clot/wall polarity."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("analysis_data_drift")

# ── Load event-level data for deeper analysis ──
df_events = pd.read_csv(OUT_DIR / "all_events.csv")
df_studies = pd.read_csv(OUT_DIR / "study_summaries.csv")

# Recompute ratio using MAX R
df_studies['max_ratio'] = df_studies['clot_max_R'] / df_studies['wall_max_R']
both = df_studies.dropna(subset=['max_ratio', 'clot_wall_ratio']).copy()

print("=" * 70)
print("MAX R ratio (clot_max / wall_max) vs MEAN R ratio")
print("=" * 70)

print(f"\n  MAX ratio:  mean={both['max_ratio'].mean():.3f}  std={both['max_ratio'].std():.3f}")
print(f"    clot_max > wall_max: {(both['max_ratio'] > 1.0).sum()} / {len(both)}"
      f" ({(both['max_ratio'] > 1.0).mean()*100:.0f}%)")
print(f"    clot_max < wall_max: {(both['max_ratio'] < 1.0).sum()} / {len(both)}"
      f" ({(both['max_ratio'] < 1.0).mean()*100:.0f}%)")

print(f"\n  MEAN ratio: mean={both['clot_wall_ratio'].mean():.3f}  std={both['clot_wall_ratio'].std():.3f}")
print(f"    clot_mean > wall_mean: {(both['clot_wall_ratio'] > 1.0).sum()} / {len(both)}"
      f" ({(both['clot_wall_ratio'] > 1.0).mean()*100:.0f}%)")
print(f"    clot_mean < wall_mean: {(both['clot_wall_ratio'] < 1.0).sum()} / {len(both)}"
      f" ({(both['clot_wall_ratio'] < 1.0).mean()*100:.0f}%)")

# Studies where mean-ratio says "inverted" but max-ratio says "normal"
confused = both[(both['clot_wall_ratio'] > 1.0) & (both['max_ratio'] <= 1.0)]
print(f"\n  Mean-inverted BUT max-normal (rise/fall effect): {len(confused)}")
for _, r in confused.iterrows():
    print(f"    {r['study_id']:15s}  mean_ratio={r['clot_wall_ratio']:.3f}"
          f"  max_ratio={r['max_ratio']:.3f}")

# Truly hard: max clot < max wall
hard = both[both['max_ratio'] < 1.0].sort_values('max_ratio')
print(f"\n  Truly hard (clot_max < wall_max): {len(hard)} / {len(both)}")
for _, r in hard.iterrows():
    print(f"    {r['study_id']:15s}  max_ratio={r['max_ratio']:.3f}"
          f"  clot_max={r['clot_max_R']:.0f}  wall_max={r['wall_max_R']:.0f}"
          f"  mean_ratio={r['clot_wall_ratio']:.3f}  split={r['split']}")

# ── Per-event: compare clot vs wall by max R ──
clot_ev = df_events[df_events['label'] == 1]
wall_ev = df_events[df_events['label'] == 2]

print(f"\n{'=' * 70}")
print("PER-EVENT statistics")
print(f"{'=' * 70}")
for name, evs in [("Clot", clot_ev), ("Wall", wall_ev)]:
    print(f"\n  {name} events ({len(evs)}):")
    for col in ['max_R', 'mean_R', 'median_R', 'std_R', 'range_R', 'duration_sec']:
        v = evs[col]
        print(f"    {col:15s}  mean={v.mean():8.1f}  median={v.median():8.1f}"
              f"  std={v.std():8.1f}  min={v.min():8.1f}  max={v.max():8.1f}")

# ── Class balance analysis ──
print(f"\n{'=' * 70}")
print("CLASS BALANCE (sample counts)")
print(f"{'=' * 70}")

for split in ['train', 'test']:
    sub = df_studies[df_studies['split'] == split]
    # Sum up total samples by label from the study summaries
    # We need event-level data to compute total samples
    ev_split = df_events[df_events['split'] == split]
    clot_samples = ev_split[ev_split['label'] == 1]['n_samples'].sum()
    wall_samples = ev_split[ev_split['label'] == 2]['n_samples'].sum()
    # Blood = total - clot - wall (approximate from study totals)
    total = sub['total_samples'].sum()
    blood_samples = total - clot_samples - wall_samples
    print(f"\n  {split.upper()}: {total:,} total samples")
    print(f"    Blood: {blood_samples:>10,} ({blood_samples/total*100:.1f}%)")
    print(f"    Clot:  {clot_samples:>10,} ({clot_samples/total*100:.1f}%)")
    print(f"    Wall:  {wall_samples:>10,} ({wall_samples/total*100:.1f}%)")
    print(f"    Clot+Wall: {clot_samples+wall_samples:>10,} ({(clot_samples+wall_samples)/total*100:.1f}%)")

# ── Plot: max ratio vs mean ratio ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Clot/Wall Analysis: Max R vs Mean R", fontsize=13)

ax = axes[0]
colors = ['red' if s == 'test' else 'blue' for s in both['split']]
ax.scatter(both['clot_wall_ratio'], both['max_ratio'], c=colors, alpha=0.6, s=50)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Mean R ratio (clot_mean / wall_mean)")
ax.set_ylabel("Max R ratio (clot_max / wall_max)")
ax.set_title("Max vs Mean Ratio (blue=train, red=test)")
# Annotate quadrants
ax.text(0.6, 2.5, "Normal max\nInverted mean", fontsize=8, ha='center', style='italic')
ax.text(1.4, 2.5, "Both say\nclot > wall", fontsize=8, ha='center', style='italic')
ax.text(0.6, 0.7, "Both say\nwall > clot", fontsize=8, ha='center', style='italic')
ax.text(1.4, 0.7, "Normal mean\nInverted max", fontsize=8, ha='center', style='italic')

ax = axes[1]
ax.hist(both['max_ratio'], bins=25, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='ratio=1')
ax.set_xlabel("Max R ratio (clot_max / wall_max)")
ax.set_title(f"Max Ratio Distribution (n={len(both)})")
ax.legend()

ax = axes[2]
# Boxplot: clot max R vs wall max R (per-event)
ax.boxplot([clot_ev['max_R'].values, wall_ev['max_R'].values],
           tick_labels=['Clot', 'Wall'], showfliers=False)
ax.set_ylabel("Max R (ohms)")
ax.set_title(f"Per-Event Max R: Clot ({len(clot_ev)}) vs Wall ({len(wall_ev)})")

plt.tight_layout()
fig.savefig(OUT_DIR / "max_vs_mean_ratio.png", dpi=150)
print(f"\nSaved: {OUT_DIR / 'max_vs_mean_ratio.png'}")
plt.close()
