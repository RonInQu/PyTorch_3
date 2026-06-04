"""Redo analysis with Z <= 3500 ohm filter (excluding disconnects/spikes)."""
import pandas as pd
import numpy as np

tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}
files = {
    'Promedica': '2026-05-12 206-104 Promedica_state.parquet',
    'Centennial 054': '2026-05-13 220-054 Centennial_state.parquet',
    'Centennial 055': '2026-05-13 220-055 Centennial_state.parquet',
}

Z_MAX = 3500

print(f'ANALYSIS WITH Z <= {Z_MAX} ohm FILTER (Manual_GT)')
print('=' * 90)

# ─── Table 1: Characteristics ──────────────────────────────────────────────
case_data = {}
for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['Manual_GT'].map(tissue_map).fillna('other')
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms <= Z_MAX)]

    print(f'\n  {name}:')
    print(f'    {"Tissue":>6} | {"n":>9} | {"Z median":>8}  {"Z IQR":>18} | '
          f'{"P median":>8}  {"P IQR":>14}')
    print(f'    {"-" * 78}')
    for t in ['blood', 'clot', 'wall']:
        sub = labeled[labeled.tissue == t]
        if len(sub) == 0:
            print(f'    {t:>6} | {"n/a":>9} |')
            continue
        z = sub['imp_mag_ohms']
        p = sub['han_pressure_mmhg'].astype(np.int32)
        print(f'    {t:>6} | {len(sub):>9,} | {z.median():>8.0f}  [{z.quantile(0.25):.0f}, {z.quantile(0.75):.0f}]'
              f'       | {p.median():>8.0f}  [{p.quantile(0.25):.0f}, {p.quantile(0.75):.0f}]')

    # Fraction excluded
    all_labeled = df[df.tissue.isin(['blood', 'clot', 'wall'])]
    excluded_pct = (all_labeled.imp_mag_ohms > Z_MAX).mean() * 100
    print(f'    (Excluded Z>{Z_MAX}: {excluded_pct:.1f}% of labeled data)')

    case_data[name] = labeled

# ─── Wall vs Clot: Impedance with Z<=3500 ─────────────────────────────────
print('\n\n')
print(f'WALL vs CLOT DISCRIMINATION (Z <= {Z_MAX}, Manual_GT)')
print('=' * 90)

for name, labeled in case_data.items():
    wc = labeled[labeled.tissue.isin(['clot', 'wall'])].copy()
    n_clot = (wc.tissue == 'clot').sum()
    n_wall = (wc.tissue == 'wall').sum()

    if n_clot < 10:
        print(f'\n  {name}: SKIPPED (clot={n_clot} - insufficient)')
        continue

    z = wc['imp_mag_ohms']
    p = wc['han_pressure_mmhg'].astype(np.int32)
    t_wc = wc['tissue']

    print(f'\n  {name} (n_clot={n_clot:,}, n_wall={n_wall:,}):')

    # Impedance: Z > X -> clot
    print(f'    Impedance (Z > X -> clot):')
    best_f1 = 0
    best_t = 0
    for thr in range(800, 3500, 25):
        pred_clot = z > thr
        ctp = (pred_clot & (t_wc == 'clot')).sum()
        wfp = (pred_clot & (t_wc == 'wall')).sum()
        c_rec = ctp / n_clot
        c_prec = ctp / (ctp + wfp) if (ctp + wfp) > 0 else 0
        c_f1 = 2 * c_rec * c_prec / (c_rec + c_prec) if (c_rec + c_prec) > 0 else 0
        if c_f1 > best_f1:
            best_f1 = c_f1
            best_t = thr

    for thr in [best_t - 100, best_t - 50, best_t, best_t + 50, best_t + 100]:
        if thr < 500:
            continue
        pred_clot = z > thr
        ctp = (pred_clot & (t_wc == 'clot')).sum()
        wfp = (pred_clot & (t_wc == 'wall')).sum()
        c_rec = ctp / n_clot
        c_prec = ctp / (ctp + wfp) if (ctp + wfp) > 0 else 0
        c_f1 = 2 * c_rec * c_prec / (c_rec + c_prec) if (c_rec + c_prec) > 0 else 0
        marker = ' <-- best' if thr == best_t else ''
        print(f'      Z>{thr:>4}: clot_rec={c_rec:.3f}  clot_prec={c_prec:.3f}  F1={c_f1:.3f}{marker}')

    # Pressure: P < X -> wall
    print(f'    Pressure (P < X -> wall):')
    for thr in [200, 300, 400, 500]:
        pred_wall = p < thr
        wtp = (pred_wall & (t_wc == 'wall')).sum()
        cfp = (pred_wall & (t_wc == 'clot')).sum()
        rec = wtp / n_wall
        prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(f'      P<{thr:>3}: wall_rec={rec:.3f}  wall_prec={prec:.3f}  F1={f1:.3f}')

    # Combined: P<300 AND Z<=best -> wall (or equivalently Z>best -> clot)
    print(f'    Combined (P<300 -> wall, Z>{best_t} -> clot):')
    pred = pd.Series('blood', index=wc.index)
    pred[p < 300] = 'wall'
    pred[z > best_t] = 'clot'  # clot overrides wall if Z is high
    # Wall metrics
    wtp = ((pred == 'wall') & (t_wc == 'wall')).sum()
    wfp = ((pred == 'wall') & (t_wc == 'clot')).sum()
    wfn = ((pred != 'wall') & (t_wc == 'wall')).sum()
    w_rec = wtp / n_wall
    w_prec = wtp / (wtp + wfp) if (wtp + wfp) > 0 else 0
    w_f1 = 2 * w_rec * w_prec / (w_rec + w_prec) if (w_rec + w_prec) > 0 else 0
    # Clot metrics
    ctp = ((pred == 'clot') & (t_wc == 'clot')).sum()
    cfp = ((pred == 'clot') & (t_wc == 'wall')).sum()
    cfn = ((pred != 'clot') & (t_wc == 'clot')).sum()
    c_rec = ctp / n_clot
    c_prec = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0
    c_f1 = 2 * c_rec * c_prec / (c_rec + c_prec) if (c_rec + c_prec) > 0 else 0
    print(f'      Wall: rec={w_rec:.3f}  prec={w_prec:.3f}  F1={w_f1:.3f}')
    print(f'      Clot: rec={c_rec:.3f}  prec={c_prec:.3f}  F1={c_f1:.3f}')

# ─── Full 3-class with Z<=3500 ─────────────────────────────────────────────
print('\n\n')
print(f'3-CLASS PERFORMANCE (Z <= {Z_MAX}, Manual_GT, pooled)')
print('=' * 90)

all_true = []
all_pred = []

for name, labeled in case_data.items():
    z = labeled['imp_mag_ohms']
    p = labeled['han_pressure_mmhg'].astype(np.int32)
    true = labeled['tissue']

    # Strategy: P<300 -> wall, Z>1500 -> clot, else blood
    # (try a universal clot threshold)
    pred = pd.Series('blood', index=labeled.index)
    pred[p < 300] = 'wall'
    pred[z > 1500] = 'clot'

    all_true.append(true)
    all_pred.append(pred)

true = pd.concat(all_true)
pred = pd.concat(all_pred)

print(f'\n  Rule: P<300 -> wall, Z>1500 -> clot, else blood')
print(f'  Total samples: {len(true):,}')
print(f'\n  Predicted ->')
print(f'  {"True":>8} {"blood":>10} {"clot":>10} {"wall":>10} | {"Recall":>8}')
print(f'  {"-" * 58}')
for t in ['blood', 'clot', 'wall']:
    mask = true == t
    n = mask.sum()
    print(f'  {t:>8} {(pred[mask]=="blood").sum():>10,} {(pred[mask]=="clot").sum():>10,} '
          f'{(pred[mask]=="wall").sum():>10,} | {(pred[mask]==t).sum()/n:>7.1%}')

print(f'\n  Per-class:')
f1s = []
for t in ['blood', 'clot', 'wall']:
    tp = ((true == t) & (pred == t)).sum()
    fp = ((true != t) & (pred == t)).sum()
    fn = ((true == t) & (pred != t)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    f1s.append(f1)
    print(f'    {t}: prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}')
print(f'\n  F1-macro = {np.mean(f1s):.3f}')

# Try different clot thresholds
print(f'\n  Sweep clot Z threshold (with P<300 -> wall fixed):')
for z_thresh in [1200, 1300, 1400, 1500, 1600, 1800, 2000]:
    pred2 = pd.Series('blood', index=pd.RangeIndex(len(true)))
    p_all = pd.concat([case_data[n]['han_pressure_mmhg'].astype(np.int32) for n in case_data])
    z_all = pd.concat([case_data[n]['imp_mag_ohms'] for n in case_data])
    pred2[p_all.values < 300] = 'wall'
    pred2[z_all.values > z_thresh] = 'clot'
    f1s2 = []
    for t in ['blood', 'clot', 'wall']:
        tp = ((true.values == t) & (pred2.values == t)).sum()
        fp = ((true.values != t) & (pred2.values == t)).sum()
        fn = ((true.values == t) & (pred2.values != t)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s2.append(f1)
    print(f'    Z>{z_thresh:>4}: blood_F1={f1s2[0]:.3f}  clot_F1={f1s2[1]:.3f}  wall_F1={f1s2[2]:.3f}  macro={np.mean(f1s2):.3f}')
