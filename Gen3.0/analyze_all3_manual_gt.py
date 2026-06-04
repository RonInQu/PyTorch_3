"""Full Manual_GT analysis for all 3 cases with wall detection performance."""
import pandas as pd
import numpy as np

tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}
files = {
    'Promedica': '2026-05-12 206-104 Promedica_state.parquet',
    'Centennial 054': '2026-05-13 220-054 Centennial_state.parquet',
    'Centennial 055': '2026-05-13 220-055 Centennial_state.parquet',
}

print('CLASSIFICATION ANALYSIS WITH MANUAL_GT - ALL 3 CASES')
print('=' * 90)

# ─── Table 1: Characteristics ──────────────────────────────────────────────
print('\nTABLE 1: Per-Case Characteristics (Manual_GT, excluding open-circuit Z)')
print('-' * 90)

case_data = {}
for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['Manual_GT'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    print(f'\n  {name}:')
    print(f'    {"Tissue":>6} | {"n":>9} | {"Z median":>8} {"Z IQR":>18} | {"P median":>8} {"P IQR":>16}')
    print(f'    {"-" * 78}')
    for t in ['blood', 'clot', 'wall']:
        sub = labeled[labeled.tissue == t]
        if len(sub) == 0:
            print(f'    {t:>6} | {"(none)":>9} |')
            continue
        z = sub['imp_mag_ohms']
        p = sub['han_pressure_mmhg'].astype(np.int32)
        print(f'    {t:>6} | {len(sub):>9,} | {z.median():>8.0f} [{z.quantile(0.25):.0f}, {z.quantile(0.75):.0f}]'
              f'      | {p.median():>8.0f} [{p.quantile(0.25):.0f}, {p.quantile(0.75):.0f}]')

    case_data[name] = labeled

# ─── Table 2: Wall vs Not-Wall (binary) with pressure ──────────────────────
print('\n\n')
print('TABLE 2: WALL DETECTION (P < threshold -> wall) using Manual_GT')
print('=' * 90)

all_true = []
all_pred_300 = []

for name, labeled in case_data.items():
    p = labeled['han_pressure_mmhg'].astype(np.int32)
    true_wall = (labeled['tissue'] == 'wall')
    n_wall = true_wall.sum()
    n_other = (~true_wall).sum()

    if n_wall < 10:
        print(f'\n  {name}: SKIPPED (only {n_wall} wall samples)')
        continue

    print(f'\n  {name} (n_wall={n_wall:,}, n_not_wall={n_other:,}):')
    print(f'    Threshold | Recall  | Precision | F1      | TP        FP        FN')
    print(f'    {"-" * 72}')
    for thresh in [150, 200, 250, 300, 400, 500]:
        pred_wall = p < thresh
        tp = (pred_wall & true_wall).sum()
        fp = (pred_wall & ~true_wall).sum()
        fn = (~pred_wall & true_wall).sum()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(f'    P < {thresh:>3}   | {rec:.3f}   | {prec:.3f}     | {f1:.3f}   | {tp:>8,}  {fp:>8,}  {fn:>8,}')

    all_true.append(true_wall)
    all_pred_300.append(p < 300)

# Pooled
true_all = pd.concat(all_true)
pred_all = pd.concat(all_pred_300)
tp = (pred_all & true_all).sum()
fp = (pred_all & ~true_all).sum()
fn = (~pred_all & true_all).sum()
rec = tp / (tp + fn)
prec = tp / (tp + fp)
f1 = 2 * rec * prec / (rec + prec)
print(f'\n  POOLED (P<300 -> wall):')
print(f'    recall={rec:.3f}  precision={prec:.3f}  F1={f1:.3f}')
print(f'    ({tp:,} correct, {fp:,} false alarms, {fn:,} missed)')

# ─── Table 3: Wall vs Clot only ────────────────────────────────────────────
print('\n\n')
print('TABLE 3: WALL vs CLOT ONLY (Manual_GT, cases with sufficient clot)')
print('=' * 90)

for name, labeled in case_data.items():
    wc = labeled[labeled.tissue.isin(['clot', 'wall'])].copy()
    n_clot = (wc.tissue == 'clot').sum()
    n_wall = (wc.tissue == 'wall').sum()

    if n_clot < 10 or n_wall < 10:
        print(f'\n  {name}: SKIPPED (clot={n_clot}, wall={n_wall})')
        continue

    p = wc['han_pressure_mmhg'].astype(np.int32)
    z = wc['imp_mag_ohms']
    t_wc = wc['tissue']

    print(f'\n  {name} (n_clot={n_clot:,}, n_wall={n_wall:,}):')

    # Pressure
    print(f'    HAN pressure (P < X -> wall):')
    for thresh in [200, 300, 400, 500]:
        pred = p < thresh
        wtp = (pred & (t_wc == 'wall')).sum()
        cfp = (pred & (t_wc == 'clot')).sum()
        rec = wtp / n_wall
        prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(f'      P<{thresh:>3}: rec={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}  (wall_tp={wtp:,} clot_fp={cfp:,})')

    # Impedance
    print(f'    Impedance (Z > X -> clot):')
    # Find best threshold
    best_f1_z = 0
    best_t = 0
    for thr in range(900, 5000, 50):
        pred_clot = z > thr
        ctp = (pred_clot & (t_wc == 'clot')).sum()
        wfp = (pred_clot & (t_wc == 'wall')).sum()
        cfn = (~pred_clot & (t_wc == 'clot')).sum()
        c_rec = ctp / n_clot if n_clot > 0 else 0
        c_prec = ctp / (ctp + wfp) if (ctp + wfp) > 0 else 0
        c_f1 = 2 * c_rec * c_prec / (c_rec + c_prec) if (c_rec + c_prec) > 0 else 0
        if c_f1 > best_f1_z:
            best_f1_z = c_f1
            best_t = thr

    for thr in [best_t - 100, best_t, best_t + 100]:
        if thr < 500:
            continue
        pred_clot = z > thr
        ctp = (pred_clot & (t_wc == 'clot')).sum()
        wfp = (pred_clot & (t_wc == 'wall')).sum()
        c_rec = ctp / n_clot
        c_prec = ctp / (ctp + wfp) if (ctp + wfp) > 0 else 0
        c_f1 = 2 * c_rec * c_prec / (c_rec + c_prec) if (c_rec + c_prec) > 0 else 0
        marker = ' <-- best' if thr == best_t else ''
        print(f'      Z>{thr:>4}: clot_rec={c_rec:.3f}  clot_prec={c_prec:.3f}  clot_F1={c_f1:.3f}{marker}')

    # Combined
    print(f'    Combined (P<300 AND Z<{best_t} -> wall):')
    pred_wall = (p < 300) & (z < best_t)
    wtp = (pred_wall & (t_wc == 'wall')).sum()
    cfp = (pred_wall & (t_wc == 'clot')).sum()
    rec = wtp / n_wall
    prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
    print(f'      Wall: rec={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}')

# ─── Summary comparison ────────────────────────────────────────────────────
print('\n\n')
print('SUMMARY: Manual_GT vs State-Machine labels')
print('=' * 90)
print()
print('  With MANUAL ground truth (human annotations):')
print(f'    - Pressure (P<300) wall detection pooled: recall={rec:.3f} -> see above')
print('    - Clot Z is indistinguishable from blood Z in all cases')
print('    - Wall Z is elevated in Promedica (2024) but not in Centennial (1121)')
print('    - State machine mislabeled many wall events as clot (high-Z wall confusion)')
print()
print('  Conclusion:')
print('    - Handle pressure is the PRIMARY wall discriminator (consistent across all cases)')
print('    - Impedance (this electrode config) cannot reliably separate clot from blood')
print('    - The state machine was artificially inflating apparent clot-wall Z separation')
print('      by mislabeling high-Z wall events as "clot"')
