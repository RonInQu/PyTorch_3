"""analyze_bunch1.py — Characterize all 55 Bunch 1 studies for train/test selection."""
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("v93_data_set")
files = sorted(data_dir.glob("*_labeled_segment.parquet"))

print(f"{'Study':<12} {'Blood':>7} {'Clot':>7} {'Wall':>7} {'Unlbl':>7} "
      f"{'%Tissue':>8} {'Clot_R':>8} {'Wall_R':>8} {'Gap':>8} {'DA_err%':>8} {'Notes'}")
print("─" * 110)

results = []
for f in files:
    name = f.stem.replace("_labeled_segment", "")
    df = pd.read_parquet(f)
    r = df["magRLoadAdjusted"].values
    gt = df["label"].values if "label" in df.columns else None
    da = df["da_label"].values if "da_label" in df.columns else None
    
    if gt is None:
        print(f"{name:<12} NO LABELS")
        continue
    
    n_blood = (gt == 0).sum()
    n_clot = (gt == 1).sum()
    n_wall = (gt == 2).sum()
    n_unlabeled = (gt == -1).sum()
    n_total = len(gt)
    pct_tissue = 100.0 * (n_clot + n_wall) / (n_total - n_unlabeled + 1e-8)
    
    clot_r = r[gt == 1].mean() if n_clot > 0 else 0
    wall_r = r[gt == 2].mean() if n_wall > 0 else 0
    gap = wall_r - clot_r if (n_clot > 0 and n_wall > 0) else np.nan
    
    # DA error rate on tissue events (clot + wall only)
    tissue_mask = (gt == 1) | (gt == 2)
    da_err = 0.0
    if da is not None and tissue_mask.sum() > 0:
        da_err = 100.0 * (da[tissue_mask] != gt[tissue_mask]).sum() / tissue_mask.sum()
    
    notes = []
    if n_clot == 0 and n_wall == 0:
        notes.append("BLOOD_ONLY")
    elif n_clot == 0:
        notes.append("NO_CLOT")
    elif n_wall == 0:
        notes.append("NO_WALL")
    if gap is not None and not np.isnan(gap) and gap < 0:
        notes.append("INVERTED")
    if pct_tissue < 2:
        notes.append("LOW_TISSUE")
    
    print(f"{name:<12} {n_blood:>7,} {n_clot:>7,} {n_wall:>7,} {n_unlabeled:>7,} "
          f"{pct_tissue:>7.1f}% {clot_r:>8.0f} {wall_r:>8.0f} {gap:>+8.0f} {da_err:>7.1f}% "
          f"{'  '.join(notes)}")
    
    results.append({
        "name": name, "n_blood": n_blood, "n_clot": n_clot, "n_wall": n_wall,
        "n_unlabeled": n_unlabeled, "pct_tissue": pct_tissue,
        "clot_r": clot_r, "wall_r": wall_r, "gap": gap, "da_err_pct": da_err,
        "notes": notes
    })

# ── Summary recommendations ──
print(f"\n{'═' * 110}")
print("SUMMARY")
print(f"{'═' * 110}")
print(f"Total studies: {len(results)}")
print(f"Studies with both clot AND wall: {sum(1 for r in results if r['n_clot']>0 and r['n_wall']>0)}")
print(f"Blood-only studies: {sum(1 for r in results if 'BLOOD_ONLY' in r['notes'])}")
print(f"Inverted (clot R > wall R): {sum(1 for r in results if 'INVERTED' in r['notes'])}")
print(f"Low tissue (<2%): {sum(1 for r in results if 'LOW_TISSUE' in r['notes'])}")

# Best test candidates: have both clot+wall, high DA error, normal gap
print(f"\n── BEST TEST CANDIDATES (both clot+wall, DA errors >20%, normal gap) ──")
test_cands = [r for r in results 
              if r['n_clot'] > 0 and r['n_wall'] > 0 
              and r['da_err_pct'] > 20 
              and (np.isnan(r['gap']) or r['gap'] > 0)]
test_cands.sort(key=lambda x: -x['da_err_pct'])
for r in test_cands:
    print(f"  {r['name']:<12} tissue={r['pct_tissue']:.1f}%  DA_err={r['da_err_pct']:.1f}%  "
          f"gap={r['gap']:+.0f}  clot_n={r['n_clot']:,}  wall_n={r['n_wall']:,}")

# Problematic studies
print(f"\n── PROBLEMATIC (exclude or use cautiously) ──")
for r in results:
    if r['notes']:
        print(f"  {r['name']:<12} {', '.join(r['notes'])}")