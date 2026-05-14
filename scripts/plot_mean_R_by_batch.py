"""
Plot mean R distribution for clot vs wall for each of the 4 batches.
Shows per-study mean R as scatter/strip plots with distributions.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# ═══════════════════════════════════════════
# BATCH DEFINITIONS
# ═══════════════════════════════════════════
# Batch 1: v93, cherry-picked 24 training studies
batch1_train = ['15AC6217', '16621B3E', '1A8F0795', '376DCB0D', '3B90D74B', '43140EA7',
                '48663E05', '530618CC', '6E7EB56C', '73CB9CA1', '81FC0C79', '8860D580',
                'A225B105', 'AFF18ECE', 'BAPT0001', 'CENT0007', 'CENT0009', 'D25DD102',
                'D4793E80', 'F39B2DEA', 'HUNT0120', 'NASHUN01', 'STCLD002', 'SUMM0119']

test_studies = ['33CFB812', '819421BC', '847A1E3F', '8ECEADA6', 'CENT0008', 'DD2DFAF4', 'F427536B', 'SUMM0127']

batches = {
    'Batch 1\n(v93, 24 train)': ('v93_data_set/training', batch1_train),
    'Batch 2\n(v94, 31 train)': ('v94_data_set/training', None),  # None = auto-detect
    'Batch 3\n(Apr6, 34 train)': ('vApril6_data_set/training', None),
    'Batch 4\n(Apr10, 41 train)': ('vApril10_data_set/training', None),
}

def get_mean_R(data_dir, study_list=None):
    """Get per-study mean R for clot and wall. Only normal-polarity (clot_mean < wall_mean)."""
    results = []
    
    if study_list:
        # Batch 1: specific studies from flat directory
        files = [os.path.join(data_dir, f'{s}_labeled_segment.parquet') for s in study_list]
    else:
        # Batches 2-4: all files in the training directory
        files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) 
                 if f.endswith('.parquet')]
    
    for fpath in files:
        if not os.path.exists(fpath):
            continue
        name = os.path.basename(fpath).replace('_labeled_segment.parquet', '')
        if name in test_studies:
            continue
            
        df = pd.read_parquet(fpath, columns=['magRLoadAdjusted', 'label'])
        r = df['magRLoadAdjusted']
        lbl = df['label']
        
        clot_r = r[lbl == 1]
        wall_r = r[lbl == 2]
        
        if len(clot_r) == 0 or len(wall_r) == 0:
            continue
        
        cm = clot_r.mean()
        wm = wall_r.mean()
        
        # Only normal polarity
        if cm < wm:
            results.append({'study': name, 'clot_mean': cm, 'wall_mean': wm})
    
    return results

# ═══════════════════════════════════════════
# COLLECT DATA
# ═══════════════════════════════════════════
print("Collecting mean R for each batch...")
batch_data = {}
for label, (data_dir, study_list) in batches.items():
    print(f"  {label.split(chr(10))[0]}: {data_dir}")
    results = get_mean_R(data_dir, study_list)
    batch_data[label] = results
    print(f"    → {len(results)} normal-polarity studies")

# ═══════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True)
fig.suptitle('Mean R Distribution: Clot vs Wall (Training Studies, Normal Polarity Only)', 
             fontsize=14, fontweight='bold', y=0.98)

for ax, (batch_label, results) in zip(axes, batch_data.items()):
    if not results:
        ax.set_title(batch_label)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        continue
    
    clot_means = [r['clot_mean'] for r in results]
    wall_means = [r['wall_mean'] for r in results]
    
    # Strip plot with jitter
    np.random.seed(42)
    jitter_c = np.random.uniform(-0.15, 0.15, len(clot_means))
    jitter_w = np.random.uniform(-0.15, 0.15, len(wall_means))
    
    ax.scatter(0 + jitter_c, clot_means, color='red', alpha=0.6, s=40, label='Clot', zorder=3)
    ax.scatter(1 + jitter_w, wall_means, color='blue', alpha=0.6, s=40, label='Wall', zorder=3)
    
    # Box plots
    bp = ax.boxplot([clot_means, wall_means], positions=[0, 1], widths=0.5,
                    patch_artist=True, zorder=2,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    bp['boxes'][0].set_facecolor((1, 0.6, 0.6, 0.3))
    bp['boxes'][1].set_facecolor((0.6, 0.6, 1, 0.3))
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Clot', 'Wall'])
    ax.set_title(batch_label, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate medians
    med_c = np.median(clot_means)
    med_w = np.median(wall_means)
    ax.axhline(med_c, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(med_w, color='blue', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(0.02, 0.95, f'n={len(results)}', transform=ax.transAxes, fontsize=9,
            va='top', ha='left', color='gray')
    ax.text(0.02, 0.88, f'Med clot: {med_c:.0f} Ω', transform=ax.transAxes, fontsize=9,
            va='top', ha='left', color='red')
    ax.text(0.02, 0.81, f'Med wall: {med_w:.0f} Ω', transform=ax.transAxes, fontsize=9,
            va='top', ha='left', color='blue')
    ax.text(0.02, 0.74, f'Gap: {med_w - med_c:.0f} Ω', transform=ax.transAxes, fontsize=9,
            va='top', ha='left', color='black')

axes[0].set_ylabel('Mean R (Ω)', fontsize=12)
axes[0].legend(loc='lower right', fontsize=9)

plt.tight_layout()
out_path = 'mean_R_clot_wall_by_batch.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
