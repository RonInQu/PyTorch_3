"""
Analyze event durations across training data.
Quantifies spike vs sustained events for clot and wall.
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "vApril6_data_set" / "training"  # Batch 3
SAMPLE_RATE = 150

test_studies = ['33CFB812','819421BC','847A1E3F','8ECEADA6','CENT0008','DD2DFAF4','F427536B','SUMM0127']

events = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.parquet'): continue
    nm = fname.replace('_labeled_segment.parquet','')
    if nm in test_studies: continue
    df = pd.read_parquet(DATA_DIR / fname, columns=['magRLoadAdjusted','label'])
    r = df['magRLoadAdjusted'].values
    labels = df['label'].values
    clot_r, wall_r = r[labels==1], r[labels==2]
    if len(clot_r)==0 or len(wall_r)==0: continue
    if clot_r.mean() >= wall_r.mean(): continue

    for cls, cls_name in [(1,'clot'),(2,'wall')]:
        mask = (labels==cls).astype(int)
        diff = np.diff(np.concatenate([[0], mask, [0]]))
        starts = np.where(diff==1)[0]
        ends = np.where(diff==-1)[0]
        for s, e in zip(starts, ends):
            seg_r = r[s:e]
            events.append({
                'study': nm, 'class': cls_name,
                'duration_sec': (e-s)/SAMPLE_RATE, 'n_samples': e-s,
                'peak_R': seg_r.max(), 'mean_R': seg_r.mean(), 'std_R': seg_r.std(),
            })

df_events = pd.DataFrame(events)
print(f"Total events: {len(df_events)}")

# Duration stats
for cls in ['clot','wall']:
    d = df_events[df_events['class']==cls]['duration_sec'].values
    print(f"\n{cls.upper()} (n={len(d)}):")
    print(f"  Median: {np.median(d):.2f}s  Mean: {d.mean():.2f}s")
    for lo, hi, label in [(0,1,'<1s (spikes)'),(1,3,'1-3s'),(3,5,'3-5s'),(5,10,'5-10s'),(10,30,'10-30s'),(30,9999,'>30s')]:
        n = ((d>=lo)&(d<hi)).sum()
        samp = df_events[(df_events['class']==cls)&(df_events['duration_sec']>=lo)&(df_events['duration_sec']<hi)]['n_samples'].sum()
        total_samp = df_events[df_events['class']==cls]['n_samples'].sum()
        print(f"    {label:15s}: {n:4d} events ({100*n/len(d):5.1f}%)  {samp:8d} samples ({100*samp/total_samp:5.1f}%)")

# Height threshold analysis (can DA call spikes by height?)
print("\n\nSPIKE HEIGHT ANALYSIS (events < 3s):")
for thresh in [1200, 1500, 1800, 2000]:
    for cls in ['clot','wall']:
        short = df_events[(df_events['class']==cls)&(df_events['duration_sec']<3)]
        above = (short['peak_R']>thresh).sum()
        print(f"  {cls} peak>{thresh}: {above}/{len(short)} ({100*above/max(1,len(short)):.0f}%)")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Event Duration Analysis — Batch 3', fontsize=14, fontweight='bold')

ax = axes[0,0]
for cls, color in [('clot','red'),('wall','blue')]:
    d = df_events[df_events['class']==cls]['duration_sec'].values
    ax.hist(d, bins=50, alpha=0.5, color=color, label=cls)
ax.set_xlabel('Duration (s)'); ax.set_ylabel('Count'); ax.set_title('Duration Distribution')
ax.axvline(3, color='gray', linestyle='--'); ax.legend(); ax.set_xlim(0, 60)

ax = axes[0,1]
for cls, color in [('clot','red'),('wall','blue')]:
    d = df_events[df_events['class']==cls]['duration_sec'].values
    ax.hist(d, bins=np.logspace(-1, 2.5, 50), alpha=0.5, color=color, label=cls)
ax.set_xscale('log'); ax.set_xlabel('Duration (s)'); ax.set_title('Duration (Log Scale)'); ax.legend()
ax.axvline(3, color='gray', linestyle='--')

ax = axes[1,0]
for cls, color in [('clot','red'),('wall','blue')]:
    sub = df_events[df_events['class']==cls]
    ax.scatter(sub['duration_sec'], sub['peak_R'], color=color, alpha=0.3, s=15, label=cls)
ax.set_xlabel('Duration (s)'); ax.set_ylabel('Peak R (Ω)'); ax.set_title('Peak R vs Duration')
ax.axvline(3, color='gray', linestyle='--'); ax.axhline(1800, color='green', linestyle=':')
ax.legend(); ax.set_xlim(0, 60); ax.set_ylim(500, 5000)

ax = axes[1,1]
for cls, color in [('clot','red'),('wall','blue')]:
    sub = df_events[df_events['class']==cls].sort_values('duration_sec')
    cum = sub['n_samples'].cumsum() / sub['n_samples'].sum() * 100
    ax.plot(sub['duration_sec'].values, cum.values, color=color, label=cls, linewidth=2)
ax.set_xlabel('Duration (s)'); ax.set_ylabel('Cumulative % of Samples')
ax.set_title('Cumulative Sample Coverage'); ax.legend(); ax.grid(alpha=0.3)
ax.axvline(3, color='gray', linestyle='--'); ax.axvline(5, color='green', linestyle=':')

plt.tight_layout()
plt.savefig('event_duration_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: event_duration_analysis.png")