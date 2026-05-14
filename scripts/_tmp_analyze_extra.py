"""Analyze extra studies in Batch 4 not in Batch 3."""
import os, pandas as pd, numpy as np
os.environ['PYARROW_IGNORE_TIMEZONE']='1'

b3_dir = 'vApril6_data_set/training'
b4_dir = 'vApril10_data_set/training'
test_studies = ['33CFB812','819421BC','847A1E3F','8ECEADA6','CENT0008','DD2DFAF4','F427536B','SUMM0127']

b3_studies = set()
for f in os.listdir(b3_dir):
    if f.endswith('.parquet'):
        nm = f.replace('_labeled_segment.parquet','')
        if nm not in test_studies:
            df = pd.read_parquet(os.path.join(b3_dir, f), columns=['magRLoadAdjusted','label'])
            cr = df.loc[df['label']==1, 'magRLoadAdjusted']
            wr = df.loc[df['label']==2, 'magRLoadAdjusted']
            if len(cr)>0 and len(wr)>0 and cr.mean() < wr.mean():
                b3_studies.add(nm)

print(f'Batch 3 studies: {len(b3_studies)}')

# Find extra studies in batch 4
extra_studies = []
for f in sorted(os.listdir(b4_dir)):
    if f.endswith('.parquet'):
        nm = f.replace('_labeled_segment.parquet','')
        if nm in test_studies: continue
        df = pd.read_parquet(os.path.join(b4_dir, f), columns=['magRLoadAdjusted','label'])
        cr = df.loc[df['label']==1, 'magRLoadAdjusted']
        wr = df.loc[df['label']==2, 'magRLoadAdjusted']
        if len(cr)==0 or len(wr)==0: continue
        cm, wm = cr.mean(), wr.mean()
        if cm >= wm: continue
        if nm not in b3_studies:
            gap = wm - cm
            extra_studies.append({'study': nm, 'clot_mean': cm, 'wall_mean': wm, 'gap': gap,
                                  'clot_n': len(cr), 'wall_n': len(wr)})

print(f'Extra studies in Batch 4 (not in Batch 3): {len(extra_studies)}')
print(f"{'Study':<12} {'Clot Mean':>10} {'Wall Mean':>10} {'Gap':>8} {'Clot_n':>8} {'Wall_n':>8}")
for s in sorted(extra_studies, key=lambda x: x['gap']):
    print(f"{s['study']:<12} {s['clot_mean']:>10.0f} {s['wall_mean']:>10.0f} {s['gap']:>8.0f} {s['clot_n']:>8} {s['wall_n']:>8}")

gaps = [s['gap'] for s in extra_studies]
if gaps:
    print(f'\nExtra studies median gap: {np.median(gaps):.0f} vs Batch 3 median gap: 201')
    print(f'Extra studies with gap < 100: {sum(1 for g in gaps if g < 100)}/{len(gaps)}')
