"""Analyze impedance from new Gen3.0 parquets with imp_mag_ohms."""
import pandas as pd
import numpy as np

tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}

files = {
    'Promedica': '2026-05-12 206-104 Promedica_state.parquet',
    'Centennial 055': '2026-05-13 220-055 Centennial_state.parquet',
}

print('=' * 90)
print('IMPEDANCE CLASSIFICATION ANALYSIS (excluding open-circuit samples)')
print('=' * 90)

all_results = []
for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]

    # Filter: labeled tissue only, exclude open-circuit
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    blood = labeled[labeled.tissue == 'blood']['imp_mag_ohms']
    clot = labeled[labeled.tissue == 'clot']['imp_mag_ohms']
    wall = labeled[labeled.tissue == 'wall']['imp_mag_ohms']

    print(f'\n{name}:')
    print(f'  Blood: median={blood.median():.0f}, IQR=[{blood.quantile(0.25):.0f}, {blood.quantile(0.75):.0f}]')
    print(f'  Clot:  median={clot.median():.0f}, IQR=[{clot.quantile(0.25):.0f}, {clot.quantile(0.75):.0f}]')
    print(f'  Wall:  median={wall.median():.0f}, IQR=[{wall.quantile(0.25):.0f}, {wall.quantile(0.75):.0f}]')

    # Key observation: wall is BETWEEN blood and clot
    print(f'\n  Separation:')
    print(f'    Blood-Wall gap: wall_p25 - blood_p75 = {wall.quantile(0.25) - blood.quantile(0.75):.0f} ohms')
    print(f'    Wall-Clot gap:  clot_p25 - wall_p75 = {clot.quantile(0.25) - wall.quantile(0.75):.0f} ohms')

    # Threshold analysis: clot vs not-clot
    print(f'\n  Threshold sweep: Z > X -> predict CLOT')
    non_clot = pd.concat([blood, wall])
    for thresh in [1500, 1800, 2000, 2200, 2500]:
        clot_tp = (clot > thresh).sum()
        clot_fn = (clot <= thresh).sum()
        non_clot_fp = (non_clot > thresh).sum()
        recall = clot_tp / len(clot) if len(clot) > 0 else 0
        prec = clot_tp / (clot_tp + non_clot_fp) if (clot_tp + non_clot_fp) > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        print(f'    Z > {thresh:>4}: recall={recall:.3f}  prec={prec:.3f}  F1={f1:.3f}  (clot_tp={clot_tp}, fp={non_clot_fp})')

    all_results.append({'name': name, 'blood': blood, 'clot': clot, 'wall': wall})

# ─── Combined pressure + impedance analysis ────────────────────────────────
print('\n')
print('=' * 90)
print('COMBINED PRESSURE + IMPEDANCE ANALYSIS')
print('=' * 90)

for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    print(f'\n{name}:')
    print(f'  Combined rule: Wall = (P < 300) AND (Z < 1800)')
    print(f'                 Clot = (Z > 2000)')
    print(f'                 Blood = everything else')

    for t in ['blood', 'clot', 'wall']:
        sub = labeled[labeled.tissue == t]
        z = sub['imp_mag_ohms']
        p = sub['han_pressure_mmhg']

        wall_pred = (p < 300) & (z < 1800)
        clot_pred = z > 2000
        blood_pred = ~wall_pred & ~clot_pred

        print(f'  {t:>5}: n={len(sub):>7,}  pred_wall={wall_pred.mean()*100:>5.1f}%  '
              f'pred_clot={clot_pred.mean()*100:>5.1f}%  pred_blood={blood_pred.mean()*100:>5.1f}%')

# ─── 3-class confusion matrix ──────────────────────────────────────────────
print('\n')
print('=' * 90)
print('3-CLASS CONFUSION MATRIX (combined rule, pooled Promedica + Centennial 055)')
print('=' * 90)

all_true = []
all_pred = []

for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    z = labeled['imp_mag_ohms']
    p = labeled['han_pressure_mmhg']

    pred = pd.Series('blood', index=labeled.index)
    pred[(p < 300) & (z < 1800)] = 'wall'
    pred[z > 2000] = 'clot'

    all_true.append(labeled['tissue'])
    all_pred.append(pred)

true = pd.concat(all_true)
pred = pd.concat(all_pred)

print('\n  Predicted ->')
print(f'  {"True v":<8} {"blood":>10} {"clot":>10} {"wall":>10} | {"Recall":>8}')
print(f'  {"-"*55}')
for t in ['blood', 'clot', 'wall']:
    mask = true == t
    n = mask.sum()
    as_blood = ((pred[mask] == 'blood').sum())
    as_clot = ((pred[mask] == 'clot').sum())
    as_wall = ((pred[mask] == 'wall').sum())
    recall = (pred[mask] == t).sum() / n if n > 0 else 0
    print(f'  {t:<8} {as_blood:>10,} {as_clot:>10,} {as_wall:>10,} | {recall:>7.1%}')

print(f'\n  Precision:')
for t in ['blood', 'clot', 'wall']:
    tp = ((true == t) & (pred == t)).sum()
    pred_pos = (pred == t).sum()
    prec = tp / pred_pos if pred_pos > 0 else 0
    print(f'    {t}: {prec:.1%}  ({tp:,} / {pred_pos:,})')

# F1 per class
print(f'\n  F1 per class:')
for t in ['blood', 'clot', 'wall']:
    tp = ((true == t) & (pred == t)).sum()
    fp = ((true != t) & (pred == t)).sum()
    fn = ((true == t) & (pred != t)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f'    {t}: F1={f1:.3f}  (prec={prec:.3f}, recall={rec:.3f})')

f1s = []
for t in ['blood', 'clot', 'wall']:
    tp = ((true == t) & (pred == t)).sum()
    fp = ((true != t) & (pred == t)).sum()
    fn = ((true == t) & (pred != t)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    f1s.append(f1)
print(f'\n  F1-macro = {np.mean(f1s):.3f}')
