"""Redo pressure/impedance classification analysis using Manual_GT labels."""
import pandas as pd
import numpy as np

tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}

files = {
    'Promedica': '2026-05-12 206-104 Promedica_state.parquet',
    'Centennial 054': '2026-05-13 220-054 Centennial_state.parquet',
    'Centennial 055': '2026-05-13 220-055 Centennial_state.parquet',
}

print('=' * 90)
print('CLASSIFICATION ANALYSIS USING MANUAL GROUND TRUTH (Manual_GT)')
print('=' * 90)

# ─── Per-case characteristics ───────────────────────────────────────────────
print('\n\nTABLE 1: Per-Case Characteristics (Manual_GT)')
print('-' * 90)

all_results = []
for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['Manual_GT'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]

    # Exclude open-circuit impedance
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    print(f'\n{name} (excluding open-circuit Z):')
    print(f'  {"Tissue":>6} | {"n":>9} | {"Z med":>7} {"Z IQR":>18} | '
          f'{"P_han med":>9} {"P_han IQR":>16}')
    print(f'  {"-" * 80}')
    for t in ['blood', 'clot', 'wall']:
        sub = labeled[labeled.tissue == t]
        if len(sub) == 0:
            print(f'  {t:>6} | {"--":>9} |')
            continue
        z = sub['imp_mag_ohms']
        p = sub['han_pressure_mmhg'].astype(np.int32)
        print(f'  {t:>6} | {len(sub):>9,} | {z.median():>7.0f} [{z.quantile(0.25):.0f}, {z.quantile(0.75):.0f}]'
              f'       | {p.median():>9.0f} [{p.quantile(0.25):.0f}, {p.quantile(0.75):.0f}]')

    all_results.append({
        'name': name,
        'df': labeled,
    })

# ─── Wall vs Clot discrimination ───────────────────────────────────────────
print('\n\n')
print('=' * 90)
print('WALL vs CLOT DISCRIMINATION (Manual_GT)')
print('=' * 90)

for res in all_results:
    name = res['name']
    labeled = res['df']

    wc = labeled[labeled.tissue.isin(['clot', 'wall'])].copy()
    t_wc = wc['tissue']
    n_clot = (t_wc == 'clot').sum()
    n_wall = (t_wc == 'wall').sum()

    if n_clot < 10 or n_wall < 10:
        print(f'\n{name}: SKIPPED (clot={n_clot}, wall={n_wall} - insufficient samples)')
        continue

    han_p = wc['han_pressure_mmhg'].astype(np.int32)
    z = wc['imp_mag_ohms']

    print(f'\n{name} (n_clot={n_clot:,}, n_wall={n_wall:,}):')

    # 1) HAN pressure sweep
    print(f'  HAN pressure threshold (han_P < X -> wall):')
    best_f1_p = 0
    best_thresh_p = 0
    for thresh in [150, 200, 250, 300, 400, 500]:
        pred = han_p < thresh
        wtp = (pred & (t_wc == 'wall')).sum()
        wfn = (~pred & (t_wc == 'wall')).sum()
        cfp = (pred & (t_wc == 'clot')).sum()
        rec = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
        prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        marker = ' <-- best' if f1 > best_f1_p else ''
        if f1 > best_f1_p:
            best_f1_p = f1
            best_thresh_p = thresh
        print(f'    P<{thresh:>3}: recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}{marker}')

    # 2) Impedance sweep
    print(f'  Impedance threshold (Z > X -> clot, i.e. Z < X -> wall):')
    best_f1_z = 0
    best_thresh_z = 0
    for thresh in range(1000, 5000, 100):
        pred = z < thresh
        wtp = (pred & (t_wc == 'wall')).sum()
        cfp = (pred & (t_wc == 'clot')).sum()
        rec = wtp / n_wall if n_wall > 0 else 0
        prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        if f1 > best_f1_z:
            best_f1_z = f1
            best_thresh_z = thresh
    # Print around best
    for thresh in [best_thresh_z - 200, best_thresh_z - 100, best_thresh_z, best_thresh_z + 100, best_thresh_z + 200]:
        if thresh < 500:
            continue
        pred = z < thresh
        wtp = (pred & (t_wc == 'wall')).sum()
        cfp = (pred & (t_wc == 'clot')).sum()
        rec = wtp / n_wall if n_wall > 0 else 0
        prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        marker = ' <-- best' if thresh == best_thresh_z else ''
        print(f'    Z<{thresh:>4}: recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}{marker}')

    # 3) Combined AND: P<best AND Z<best
    print(f'  Combined rules:')
    pred_and = (han_p < best_thresh_p) & (z < best_thresh_z)
    wtp = (pred_and & (t_wc == 'wall')).sum()
    cfp = (pred_and & (t_wc == 'clot')).sum()
    rec = wtp / n_wall
    prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
    print(f'    P<{best_thresh_p} AND Z<{best_thresh_z}: recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}')

    pred_or = (han_p < best_thresh_p) | (z < best_thresh_z)
    wtp = (pred_or & (t_wc == 'wall')).sum()
    cfp = (pred_or & (t_wc == 'clot')).sum()
    rec = wtp / n_wall
    prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
    print(f'    P<{best_thresh_p} OR Z<{best_thresh_z}:  recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}')

# ─── Pooled 3-class confusion matrix ───────────────────────────────────────
print('\n\n')
print('=' * 90)
print('3-CLASS CONFUSION MATRIX (Manual_GT, pooled)')
print('=' * 90)

# Use per-case best pressure threshold, but a single combined approach
# Strategy: P<300 -> wall candidate, Z > per-case-blood-median*2 -> clot, else blood
all_true = []
all_pred = []

for res in all_results:
    labeled = res['df']
    name = res['name']
    z = labeled['imp_mag_ohms']
    p = labeled['han_pressure_mmhg'].astype(np.int32)
    true = labeled['tissue']

    blood_z_med = labeled[labeled.tissue == 'blood']['imp_mag_ohms'].median()
    clot_thresh = blood_z_med * 2.0

    pred = pd.Series('blood', index=labeled.index)
    pred[(p < 300) & (z < clot_thresh)] = 'wall'
    pred[z > clot_thresh] = 'clot'

    all_true.append(true)
    all_pred.append(pred)
    print(f'\n  {name}: blood_Z_med={blood_z_med:.0f}, clot_thresh(Z>{clot_thresh:.0f})')

true = pd.concat(all_true)
pred = pd.concat(all_pred)

print(f'\n  Pooled samples: {len(true):,}')
print(f'\n  Predicted ->')
print(f'  {"True v":<8} {"blood":>10} {"clot":>10} {"wall":>10} | {"Recall":>8}')
print(f'  {"-" * 58}')
for t in ['blood', 'clot', 'wall']:
    mask = true == t
    n = mask.sum()
    as_blood = (pred[mask] == 'blood').sum()
    as_clot = (pred[mask] == 'clot').sum()
    as_wall = (pred[mask] == 'wall').sum()
    recall = (pred[mask] == t).sum() / n if n > 0 else 0
    print(f'  {t:<8} {as_blood:>10,} {as_clot:>10,} {as_wall:>10,} | {recall:>7.1%}')

print(f'\n  Per-class metrics:')
f1s = []
for t in ['blood', 'clot', 'wall']:
    tp = ((true == t) & (pred == t)).sum()
    fp = ((true != t) & (pred == t)).sum()
    fn = ((true == t) & (pred != t)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    f1s.append(f1)
    print(f'    {t}: precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}')
print(f'\n  F1-macro = {np.mean(f1s):.3f}')

# ─── Compare with old state-machine labels ─────────────────────────────────
print('\n\n')
print('=' * 90)
print('COMPARISON: Manual_GT vs State-Machine (light_style_i) as ground truth')
print('=' * 90)
print('\n  Using same combined rule: P<300 AND Z<2*blood_Z -> wall, Z>2*blood_Z -> clot')
print('\n  (Previous analysis used light_style_i labels)')
print()

# Redo with state machine labels for comparison
all_true_sm = []
all_pred_sm = []
sm_tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}

for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue_sm'] = df['light_style_i'].map(sm_tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]
    labeled = df[(df.tissue_sm.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    z = labeled['imp_mag_ohms']
    p = labeled['han_pressure_mmhg'].astype(np.int32)
    true_sm = labeled['tissue_sm']
    blood_z_med = labeled[labeled.tissue_sm == 'blood']['imp_mag_ohms'].median()
    clot_thresh = blood_z_med * 2.0

    pred_sm = pd.Series('blood', index=labeled.index)
    pred_sm[(p < 300) & (z < clot_thresh)] = 'wall'
    pred_sm[z > clot_thresh] = 'clot'

    all_true_sm.append(true_sm)
    all_pred_sm.append(pred_sm)

true_sm = pd.concat(all_true_sm)
pred_sm = pd.concat(all_pred_sm)

print(f'  State-machine labels (light_style_i):')
f1s_sm = []
for t in ['blood', 'clot', 'wall']:
    tp = ((true_sm == t) & (pred_sm == t)).sum()
    fp = ((true_sm != t) & (pred_sm == t)).sum()
    fn = ((true_sm == t) & (pred_sm != t)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    f1s_sm.append(f1)
    print(f'    {t}: precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}')
print(f'  F1-macro (state-machine GT) = {np.mean(f1s_sm):.3f}')

print(f'\n  Manual_GT labels:')
for i, t in enumerate(['blood', 'clot', 'wall']):
    print(f'    {t}: F1={f1s[i]:.3f}')
print(f'  F1-macro (Manual GT) = {np.mean(f1s):.3f}')

print(f'\n  Delta: F1-macro improved by {np.mean(f1s) - np.mean(f1s_sm):.3f} '
      f'({(np.mean(f1s) - np.mean(f1s_sm))/np.mean(f1s_sm)*100:.1f}%)')
