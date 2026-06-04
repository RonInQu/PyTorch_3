"""Compare pressure-only vs impedance-only vs combined for wall/clot discrimination."""
import pandas as pd
import numpy as np

tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}
files = {
    'Promedica': '2026-05-12 206-104 Promedica_state.parquet',
    'Centennial 054': '2026-05-13 220-054 Centennial_state.parquet',
    'Centennial 055': '2026-05-13 220-055 Centennial_state.parquet',
}

print('CRITICAL OBSERVATION: Impedance ordering differs between cases!')
print('=' * 80)
print()
for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    print(f'{name}:')
    for t in ['blood', 'clot', 'wall']:
        sub = labeled[labeled.tissue == t]
        print(f'  {t:>5}: Z median = {sub.imp_mag_ohms.median():>6.0f} ohms,  '
              f'P median = {sub.han_pressure_mmhg.median():>5.0f} mmHg')
    print()

print()
print('WALL vs CLOT DISCRIMINATION: Pressure-only vs Impedance-only vs Combined')
print('=' * 80)

for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    # Wall vs Clot only
    wc = labeled[labeled.tissue.isin(['clot', 'wall'])].copy()
    z_wc = wc['imp_mag_ohms']
    p_wc = wc['han_pressure_mmhg']
    t_wc = wc['tissue']
    n_clot = (t_wc == 'clot').sum()
    n_wall = (t_wc == 'wall').sum()

    print(f'\n{name} (n_clot={n_clot:,}, n_wall={n_wall:,}):')

    # 1) Pressure only: P < 300 -> wall
    pred = (p_wc < 300)
    wall_tp = (pred & (t_wc == 'wall')).sum()
    wall_fn = (~pred & (t_wc == 'wall')).sum()
    clot_fp = (pred & (t_wc == 'clot')).sum()
    rec = wall_tp / (wall_tp + wall_fn)
    prec = wall_tp / (wall_tp + clot_fp) if (wall_tp + clot_fp) > 0 else 0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
    print(f'  Pressure only (P<300 -> wall):')
    print(f'    Wall recall={rec:.3f}  precision={prec:.3f}  F1={f1:.3f}')

    # 2) Impedance only: use optimal threshold (brute force)
    best_f1_z = 0
    best_thresh_z = 0
    best_dir = 'below'
    # Try Z < thresh -> wall (Promedica pattern) and Z > thresh -> wall (if inverted)
    for thresh in range(900, 3000, 50):
        # Z < thresh -> wall
        pred_below = z_wc < thresh
        wtp = (pred_below & (t_wc == 'wall')).sum()
        wfn = (~pred_below & (t_wc == 'wall')).sum()
        cfp = (pred_below & (t_wc == 'clot')).sum()
        r = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
        p = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f = 2 * r * p / (r + p) if (r + p) > 0 else 0
        if f > best_f1_z:
            best_f1_z = f
            best_thresh_z = thresh
            best_dir = 'below'
            best_rec_z, best_prec_z = r, p

    print(f'  Impedance only (Z<{best_thresh_z} -> wall):')
    print(f'    Wall recall={best_rec_z:.3f}  precision={best_prec_z:.3f}  F1={best_f1_z:.3f}')

    # 3) Combined: P<300 AND Z<optimal_z -> wall (strict)
    pred_strict = (p_wc < 300) & (z_wc < best_thresh_z)
    wtp = (pred_strict & (t_wc == 'wall')).sum()
    wfn = (~pred_strict & (t_wc == 'wall')).sum()
    cfp = (pred_strict & (t_wc == 'clot')).sum()
    r = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
    p = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
    f = 2 * r * p / (r + p) if (r + p) > 0 else 0
    print(f'  Combined AND (P<300 AND Z<{best_thresh_z} -> wall):')
    print(f'    Wall recall={r:.3f}  precision={p:.3f}  F1={f:.3f}')

    # 4) Combined OR: P<300 OR Z<(blood_median*1.1) -> wall
    blood_z_med = labeled[labeled.tissue == 'blood']['imp_mag_ohms'].median()
    pred_or = (p_wc < 300) | (z_wc < blood_z_med * 1.05)
    wtp = (pred_or & (t_wc == 'wall')).sum()
    wfn = (~pred_or & (t_wc == 'wall')).sum()
    cfp = (pred_or & (t_wc == 'clot')).sum()
    r = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
    p = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
    f = 2 * r * p / (r + p) if (r + p) > 0 else 0
    print(f'  Combined OR (P<300 OR Z<{blood_z_med*1.05:.0f} -> wall):')
    print(f'    Wall recall={r:.3f}  precision={p:.3f}  F1={f:.3f}')

    # 5) Pressure sweep for best threshold
    print(f'\n  Pressure threshold sweep (wall vs clot only):')
    for pt in [150, 200, 250, 300, 400, 500]:
        pred_p = p_wc < pt
        wtp = (pred_p & (t_wc == 'wall')).sum()
        wfn = (~pred_p & (t_wc == 'wall')).sum()
        cfp = (pred_p & (t_wc == 'clot')).sum()
        r = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
        p_val = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f = 2 * r * p_val / (r + p_val) if (r + p_val) > 0 else 0
        print(f'    P<{pt:>3}: recall={r:.3f}  prec={p_val:.3f}  F1={f:.3f}')
