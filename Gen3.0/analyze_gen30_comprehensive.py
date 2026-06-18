"""Comprehensive Gen3.0 analysis: clot vs wall with impedance + pressure (Manual_GT).
Compares all 3 human cases and benchmarks against Gen2.5 porcine results.
"""
import pandas as pd
import numpy as np

tissue_num = {4: 'blood', 5: 'clot', 9: 'wall'}
Z_MAX = 3500  # exclude disconnects/spikes

files = {
    'Promedica': '2026-05-12 206-104 Promedica_state.parquet',
    'Centennial 054': '2026-05-13 220-054 Centennial_state.parquet',
    'Centennial 055': '2026-05-13 220-055 Centennial_state.parquet',
}

# ═══════════════════════════════════════════════════════════════════════════
# 1. LABEL CHECK
# ═══════════════════════════════════════════════════════════════════════════
print('=' * 90)
print('1. LABEL SUMMARY (Manual_GT)')
print('=' * 90)

case_data = {}
for name, f in files.items():
    df = pd.read_parquet(f)
    gt = df['Manual_GT']
    print(f'\n  {name}:  ({len(df):,} rows, Manual_GT dtype={gt.dtype})')

    # Normalize to string tissue labels
    if gt.dtype == object:
        df['tissue'] = gt.copy()
        df.loc[~df.tissue.isin(['blood', 'clot', 'wall']), 'tissue'] = 'other'
    else:
        df['tissue'] = gt.map(tissue_num).fillna('other')

    vc = df['tissue'].value_counts()
    for t in ['blood', 'clot', 'wall', 'other']:
        if t in vc.index:
            print(f'    {t:>6}: {vc[t]:>10,}')

    # Filter to labeled tissue with valid impedance
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms <= Z_MAX)].copy()
    labeled['han_p'] = labeled['han_pressure_mmhg'].astype(np.int32)
    case_data[name] = labeled

# ═══════════════════════════════════════════════════════════════════════════
# 2. SIGNAL CHARACTERISTICS BY TISSUE
# ═══════════════════════════════════════════════════════════════════════════
print('\n\n')
print('=' * 90)
print('2. SIGNAL CHARACTERISTICS BY TISSUE (Z <= 3500, Manual_GT)')
print('=' * 90)

for name, labeled in case_data.items():
    print(f'\n  {name} ({len(labeled):,} samples after Z filter):')
    print(f'    {"":>6} | {"n":>9} | {"Z med":>6} {"Z p25":>6} {"Z p75":>6} | '
          f'{"P med":>6} {"P p25":>6} {"P p75":>6}')
    print(f'    {"-" * 70}')
    for t in ['blood', 'clot', 'wall']:
        sub = labeled[labeled.tissue == t]
        if len(sub) == 0:
            print(f'    {t:>6} | {"n/a":>9} |')
            continue
        z = sub['imp_mag_ohms']
        p = sub['han_p']
        print(f'    {t:>6} | {len(sub):>9,} | {z.median():>6.0f} {z.quantile(0.25):>6.0f} '
              f'{z.quantile(0.75):>6.0f} | {p.median():>6.0f} {p.quantile(0.25):>6.0f} {p.quantile(0.75):>6.0f}')

# ═══════════════════════════════════════════════════════════════════════════
# 3. WALL vs CLOT DISCRIMINATION
# ═══════════════════════════════════════════════════════════════════════════
print('\n\n')
print('=' * 90)
print('3. WALL vs CLOT DISCRIMINATION (Manual_GT)')
print('=' * 90)

for name, labeled in case_data.items():
    wc = labeled[labeled.tissue.isin(['clot', 'wall'])].copy()
    n_clot = (wc.tissue == 'clot').sum()
    n_wall = (wc.tissue == 'wall').sum()

    if n_clot < 10 or n_wall < 10:
        print(f'\n  {name}: SKIPPED (clot={n_clot:,}, wall={n_wall:,})')
        continue

    z = wc['imp_mag_ohms']
    p = wc['han_p']
    t_wc = wc['tissue']

    print(f'\n  {name} (clot={n_clot:,}, wall={n_wall:,}):')

    # a) Pressure threshold sweep
    print(f'    a) HAN pressure (P < X -> wall):')
    for thresh in [200, 300, 400, 500]:
        pred = p < thresh
        wtp = (pred & (t_wc == 'wall')).sum()
        cfp = (pred & (t_wc == 'clot')).sum()
        rec = wtp / n_wall
        prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(f'       P<{thresh:>3}: wall_rec={rec:.3f}  wall_prec={prec:.3f}  F1={f1:.3f}')

    # b) Impedance threshold sweep (Z > X -> clot)
    print(f'    b) Impedance (Z > X -> clot):')
    best_f1_z = 0
    best_t_z = 0
    for thr in range(800, 3500, 25):
        pred_clot = z > thr
        ctp = (pred_clot & (t_wc == 'clot')).sum()
        wfp = (pred_clot & (t_wc == 'wall')).sum()
        c_rec = ctp / n_clot
        c_prec = ctp / (ctp + wfp) if (ctp + wfp) > 0 else 0
        c_f1 = 2 * c_rec * c_prec / (c_rec + c_prec) if (c_rec + c_prec) > 0 else 0
        if c_f1 > best_f1_z:
            best_f1_z = c_f1
            best_t_z = thr

    for thr in [best_t_z - 100, best_t_z, best_t_z + 100]:
        if thr < 500:
            continue
        pred_clot = z > thr
        ctp = (pred_clot & (t_wc == 'clot')).sum()
        wfp = (pred_clot & (t_wc == 'wall')).sum()
        c_rec = ctp / n_clot
        c_prec = ctp / (ctp + wfp) if (ctp + wfp) > 0 else 0
        c_f1 = 2 * c_rec * c_prec / (c_rec + c_prec) if (c_rec + c_prec) > 0 else 0
        marker = ' <-- best' if thr == best_t_z else ''
        print(f'       Z>{thr:>4}: clot_rec={c_rec:.3f}  clot_prec={c_prec:.3f}  F1={c_f1:.3f}{marker}')

    # c) Combined: P<300 -> wall, Z>best -> clot (clot overrides)
    print(f'    c) Combined (P<300 -> wall, Z>{best_t_z} -> clot overrides):')
    pred = pd.Series('neither', index=wc.index)
    pred[p < 300] = 'wall'
    pred[z > best_t_z] = 'clot'
    # Wall metrics
    wtp = ((pred == 'wall') & (t_wc == 'wall')).sum()
    wfp = ((pred == 'wall') & (t_wc == 'clot')).sum()
    w_rec = wtp / n_wall
    w_prec = wtp / (wtp + wfp) if (wtp + wfp) > 0 else 0
    w_f1 = 2 * w_rec * w_prec / (w_rec + w_prec) if (w_rec + w_prec) > 0 else 0
    # Clot metrics
    ctp = ((pred == 'clot') & (t_wc == 'clot')).sum()
    cfp = ((pred == 'clot') & (t_wc == 'wall')).sum()
    c_rec = ctp / n_clot
    c_prec = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0
    c_f1 = 2 * c_rec * c_prec / (c_rec + c_prec) if (c_rec + c_prec) > 0 else 0
    print(f'       Wall: rec={w_rec:.3f}  prec={w_prec:.3f}  F1={w_f1:.3f}')
    print(f'       Clot: rec={c_rec:.3f}  prec={c_prec:.3f}  F1={c_f1:.3f}')

# ═══════════════════════════════════════════════════════════════════════════
# 4. POOLED WALL DETECTION (binary: wall vs not-wall)
# ═══════════════════════════════════════════════════════════════════════════
print('\n\n')
print('=' * 90)
print('4. POOLED WALL DETECTION (wall vs everything else, Manual_GT)')
print('=' * 90)

all_true = []
all_pred = {}
for thresh in [200, 300, 400, 500]:
    all_pred[thresh] = []

for name, labeled in case_data.items():
    true_wall = (labeled['tissue'] == 'wall')
    all_true.append(true_wall)
    p = labeled['han_p']
    for thresh in [200, 300, 400, 500]:
        all_pred[thresh].append(p < thresh)

true_all = pd.concat(all_true)
n_wall_total = true_all.sum()
n_other_total = (~true_all).sum()
print(f'\n  Total: {n_wall_total:,} wall, {n_other_total:,} not-wall')

print(f'\n  {"Thresh":>8} | {"Recall":>8} {"Precision":>10} {"F1":>8} | {"TP":>8} {"FP":>8} {"FN":>8}')
print(f'  {"-" * 65}')
for thresh in [200, 300, 400, 500]:
    pred_all = pd.concat(all_pred[thresh])
    tp = (pred_all & true_all).sum()
    fp = (pred_all & ~true_all).sum()
    fn = (~pred_all & true_all).sum()
    rec = tp / (tp + fn)
    prec = tp / (tp + fp)
    f1 = 2 * rec * prec / (rec + prec)
    print(f'  P<{thresh:>3}   | {rec:>8.3f} {prec:>10.3f} {f1:>8.3f} | {tp:>8,} {fp:>8,} {fn:>8,}')

# ═══════════════════════════════════════════════════════════════════════════
# 5. COMPARISON WITH GEN2.5 PORCINE DATA
# ═══════════════════════════════════════════════════════════════════════════
print('\n\n')
print('=' * 90)
print('5. GEN3.0 vs GEN2.5 COMPARISON')
print('=' * 90)

print("""
  Gen2.5 (porcine cadaver, 4 cases, single-frequency impedance):
    - Electrode: catheter-body (standard placement)
    - Blood Z median: ~380-480 ohms
    - Clot Z median:  ~700-1500 ohms (elevated, 2-3x blood)
    - Wall Z median:  ~450-600 ohms (near blood, slight elevation)
    - Wall detection:  pressure drop is the primary indicator
    - Clot detection:  impedance rise is the primary indicator
    - Blood baseline:  760-790 mmHg (atmospheric + hydrostatic)

  Gen3.0 (human clinical, 3 cases, different electrode placement):""")

# Compute pooled Gen3.0 stats
for t in ['blood', 'clot', 'wall']:
    all_z = pd.concat([d[d.tissue == t]['imp_mag_ohms'] for d in case_data.values()])
    all_p = pd.concat([d[d.tissue == t]['han_p'] for d in case_data.values()])
    if len(all_z) == 0:
        continue
    print(f'    - {t.capitalize()} Z median: {all_z.median():.0f} ohms '
          f'(IQR [{all_z.quantile(0.25):.0f}, {all_z.quantile(0.75):.0f}])')
    print(f'      {t.capitalize()} P median: {all_p.median():.0f} mmHg '
          f'(IQR [{all_p.quantile(0.25):.0f}, {all_p.quantile(0.75):.0f}])')

print("""
  Key differences:
    Gen2.5: Clot has ELEVATED Z (2-3x blood), Wall Z ~ Blood Z
    Gen3.0: ??? (see results above for updated Manual_GT)

    Gen2.5: Wall pressure drops are clear (P < 200 mmHg during wall contact)
    Gen3.0: Wall pressure drops are clear (consistent with Gen2.5)

    Gen2.5: Impedance is the PRIMARY clot discriminator
    Gen3.0: Pressure is the PRIMARY wall discriminator

  Consistency check (pressure-based wall detection):
    Gen2.5 P<300 wall F1: ~0.75 (pooled across 4 porcine cases)
    Gen3.0 P<300 wall F1: see Table 4 above
""")
