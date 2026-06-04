"""Analyze CMS pressure vs HAN pressure for clot/wall discrimination."""
import pandas as pd
import numpy as np

tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}
files = {
    'Promedica': '2026-05-12 206-104 Promedica_state.parquet',
    'Centennial 054': '2026-05-13 220-054 Centennial_state.parquet',
    'Centennial 055': '2026-05-13 220-055 Centennial_state.parquet',
}

print('CMS vs HAN PRESSURE BY TISSUE')
print('=' * 90)

for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    print(f'\n{name}:')
    print(f'  {"Tissue":>6} | {"han_P med":>10} {"han_P IQR":>16} | '
          f'{"cms_P med":>10} {"cms_P IQR":>16} | {"han-cms":>8}')
    print(f'  {"-"*88}')
    for t in ['blood', 'clot', 'wall']:
        sub = labeled[labeled.tissue == t]
        hp = sub['han_pressure_mmhg']
        cp = sub['cms_pressure_mmhg']
        diff = hp - cp
        print(f'  {t:>6} | {hp.median():>10.0f} [{hp.quantile(0.25):.0f}, {hp.quantile(0.75):.0f}]'
              f'       | {cp.median():>10.0f} [{cp.quantile(0.25):.0f}, {cp.quantile(0.75):.0f}]'
              f'       | {diff.median():>8.0f}')

# ─── CMS pressure discrimination ───────────────────────────────────────────
print('\n\n')
print('CMS PRESSURE: WALL vs CLOT DISCRIMINATION')
print('=' * 90)

for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    wc = labeled[labeled.tissue.isin(['clot', 'wall'])].copy()
    t_wc = wc['tissue']
    n_clot = (t_wc == 'clot').sum()
    n_wall = (t_wc == 'wall').sum()

    print(f'\n{name} (n_clot={n_clot:,}, n_wall={n_wall:,}):')

    # CMS pressure sweep
    print(f'  CMS pressure threshold (cms_P < X -> wall):')
    for thresh in [200, 300, 400, 500, 600]:
        pred = wc['cms_pressure_mmhg'] < thresh
        wtp = (pred & (t_wc == 'wall')).sum()
        wfn = (~pred & (t_wc == 'wall')).sum()
        cfp = (pred & (t_wc == 'clot')).sum()
        rec = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
        prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(f'    cms_P < {thresh:>3}: recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}')

    # HAN pressure sweep (for comparison)
    print(f'  HAN pressure threshold (han_P < X -> wall):')
    for thresh in [200, 300, 400, 500, 600]:
        pred = wc['han_pressure_mmhg'] < thresh
        wtp = (pred & (t_wc == 'wall')).sum()
        wfn = (~pred & (t_wc == 'wall')).sum()
        cfp = (pred & (t_wc == 'clot')).sum()
        rec = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
        prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(f'    han_P < {thresh:>3}: recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}')

    # Pressure difference (han - cms)
    wc_diff = wc['han_pressure_mmhg'] - wc['cms_pressure_mmhg']
    print(f'  Pressure DIFFERENCE (han_P - cms_P) by tissue:')
    for t in ['clot', 'wall']:
        sub_diff = wc_diff[t_wc == t]
        print(f'    {t}: median={sub_diff.median():.0f}  '
              f'IQR=[{sub_diff.quantile(0.25):.0f}, {sub_diff.quantile(0.75):.0f}]  '
              f'p10={sub_diff.quantile(0.10):.0f}  p90={sub_diff.quantile(0.90):.0f}')

    # Difference threshold
    print(f'  Pressure DIFFERENCE threshold (han-cms < X -> wall):')
    for thresh in [-200, -100, -50, 0, 50, 100]:
        pred = wc_diff < thresh
        wtp = (pred & (t_wc == 'wall')).sum()
        wfn = (~pred & (t_wc == 'wall')).sum()
        cfp = (pred & (t_wc == 'clot')).sum()
        rec = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
        prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(f'    diff < {thresh:>4}: recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}')

# ─── Combined: han + cms + impedance ───────────────────────────────────────
print('\n\n')
print('COMBINED FEATURES: han_P + cms_P + Z')
print('=' * 90)

for name, f in files.items():
    df = pd.read_parquet(f)
    df['tissue'] = df['light_style_i'].map(tissue_map).fillna('other')
    mode_val = df['imp_mag_ohms'].mode().iloc[0]
    labeled = df[(df.tissue.isin(['blood', 'clot', 'wall'])) & (df.imp_mag_ohms < mode_val * 0.9)]

    wc = labeled[labeled.tissue.isin(['clot', 'wall'])].copy()
    t_wc = wc['tissue']

    han_p = wc['han_pressure_mmhg']
    cms_p = wc['cms_pressure_mmhg']
    z = wc['imp_mag_ohms']
    diff_p = han_p - cms_p

    print(f'\n{name}:')

    # Combined: han_P<300 AND cms_P<500 -> wall
    pred = (han_p < 300) & (cms_p < 500)
    wtp = (pred & (t_wc == 'wall')).sum()
    wfn = (~pred & (t_wc == 'wall')).sum()
    cfp = (pred & (t_wc == 'clot')).sum()
    rec = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
    prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
    print(f'  han_P<300 AND cms_P<500:  recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}')

    # Combined: han_P<300 AND cms_P<300 -> wall
    pred = (han_p < 300) & (cms_p < 300)
    wtp = (pred & (t_wc == 'wall')).sum()
    wfn = (~pred & (t_wc == 'wall')).sum()
    cfp = (pred & (t_wc == 'clot')).sum()
    rec = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
    prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
    print(f'  han_P<300 AND cms_P<300:  recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}')

    # All three: han_P<300 AND cms_P<500 AND Z < adaptive threshold
    blood_z = labeled[labeled.tissue == 'blood']['imp_mag_ohms'].median()
    z_thresh = blood_z * 1.8
    pred = (han_p < 300) & (cms_p < 500) & (z < z_thresh)
    wtp = (pred & (t_wc == 'wall')).sum()
    wfn = (~pred & (t_wc == 'wall')).sum()
    cfp = (pred & (t_wc == 'clot')).sum()
    rec = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0
    prec = wtp / (wtp + cfp) if (wtp + cfp) > 0 else 0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
    print(f'  han_P<300 AND cms_P<500 AND Z<{z_thresh:.0f}:  recall={rec:.3f}  prec={prec:.3f}  F1={f1:.3f}')
