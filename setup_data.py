"""
Batch 7: Full duration data (no 7s filter), no maxR filter.
Source: vApril10_data_set/training → training_data/
        vApril10_data_set/testing  → test_data/

Both dirs contain all 93 studies with full-duration labels (no short-event removal).
All non-test studies go to training. Max-inverted studies also copied to excluded_data/.

NOTE: This script will OVERWRITE training_data/, test_data/, and excluded_data/.
      Do NOT run while another training is in progress!
"""
import pandas as pd
import numpy as np
import os
import shutil

# ── Configuration ──
# Source directories (both have all 93 studies)
SRC_TRAIN_DIR = os.path.join('vApril10_data_set', 'training')  # outliers removed → training_data/
SRC_TEST_DIR  = os.path.join('vApril10_data_set', 'testing')   # outliers included → test_data/

# Fixed test set (original 8)
FIXED_TEST = sorted([
    '33CFB812', '819421BC', '847A1E3F', '8ECEADA6',
    'CENT0008', 'DD2DFAF4', 'F427536B', 'SUMM0127'
])

# Expanded test (14 from vApril10 test dir)
EXPANDED_TEST = sorted([
    '00F628C9', '33CFB812', '43140EA7', '4E3747A0', '530618CC',
    '819421BC', 'A225B105', 'AFF18ECE', 'CENT0008', 'HACK0140',
    'HUNT0120', 'STCLD001', 'SUMM0127', 'UHMAX001'
])

# ALL test = the 14 from vApril10 test dir (same as before)
# 4 fixed-8 studies (847A1E3F, 8ECEADA6, DD2DFAF4, F427536B) are NOT in the
# test dir — they go to training instead (if they pass the maxR filter).
ALL_TEST = sorted(set(EXPANDED_TEST))

clot_events = [7, 11]  # event types for clot
wall_events = [23]      # event types for wall


def compute_study_stats(filepath):
    """Compute clot/wall max R and mean R for a study."""
    df = pd.read_parquet(filepath, engine='pyarrow')
    r = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
    lbl = df['label'].to_numpy(dtype=np.int64)

    clot_mask = lbl == 1
    wall_mask = lbl == 2

    stats = {
        'clot_n': int(clot_mask.sum()),
        'wall_n': int(wall_mask.sum()),
    }

    if stats['clot_n'] > 0:
        clot_r = r[clot_mask]
        stats['clot_max'] = float(clot_r.max())
        stats['clot_mean'] = float(clot_r.mean())
    else:
        stats['clot_max'] = None
        stats['clot_mean'] = None

    if stats['wall_n'] > 0:
        wall_r = r[wall_mask]
        stats['wall_max'] = float(wall_r.max())
        stats['wall_mean'] = float(wall_r.mean())
    else:
        stats['wall_max'] = None
        stats['wall_mean'] = None

    return stats


def main():
    print("=" * 70)
    print("  BATCH 7: ALL studies, full duration (no 7s filter)")
    print("=" * 70)

    # ── Analyze all candidate training studies ──
    all_studies = []

    candidates = sorted(os.listdir(SRC_TRAIN_DIR))
    for fname in candidates:
        if not fname.endswith('.parquet'):
            continue
        name = fname.replace('_labeled_segment.parquet', '')
        filepath = os.path.join(SRC_TRAIN_DIR, fname)
        stats = compute_study_stats(filepath)

        has_clot = stats['clot_n'] > 0
        has_wall = stats['wall_n'] > 0

        if has_clot and has_wall:
            max_ratio = stats['clot_max'] / stats['wall_max']
            mean_ratio = stats['clot_mean'] / stats['wall_mean']
            if max_ratio > 1.0:
                category = 'NORMAL_MAX'
            else:
                category = 'INVERTED_MAX'
        elif not has_clot and not has_wall:
            category = 'NO_TISSUE'
            max_ratio = None
            mean_ratio = None
        else:
            category = 'PARTIAL'  # has one but not both
            max_ratio = None
            mean_ratio = None

        all_studies.append({
            'study': name, 'category': category,
            'max_ratio': max_ratio, 'mean_ratio': mean_ratio,
            **stats,
        })

    # Summary
    df_all = pd.DataFrame(all_studies)
    n_total = len(df_all)
    n_norm = (df_all['category'] == 'NORMAL_MAX').sum()
    n_inv = (df_all['category'] == 'INVERTED_MAX').sum()
    n_part = (df_all['category'] == 'PARTIAL').sum()
    n_no = (df_all['category'] == 'NO_TISSUE').sum()

    print(f"\nSource pool: {n_total} studies")
    print(f"  NORMAL_MAX  (clot_max > wall_max): {n_norm}")
    print(f"  INVERTED_MAX (clot_max < wall_max): {n_inv}")
    print(f"  PARTIAL (clot or wall only):        {n_part}")
    print(f"  NO_TISSUE:                          {n_no}")

    # Identify max-inverted for excluded_data/ (informational + evaluation)
    test_set_all = set(ALL_TEST)
    inverted = [s for s in all_studies if s['category'] == 'INVERTED_MAX']
    if inverted:
        print(f"\n  MAX-INVERTED ({len(inverted)} studies — clot_max < wall_max):")
        print(f"    These are INCLUDED in training but also copied to excluded_data/ for evaluation")
        for s in sorted(inverted, key=lambda x: x['max_ratio'] or 0):
            tag = " [TEST]" if s['study'] in test_set_all else " [TRAIN]"
            print(f"    {s['study']:15s}  max_ratio={s['max_ratio']:.3f}"
                  f"  clot_max={s['clot_max']:.0f}  wall_max={s['wall_max']:.0f}"
                  f"  mean_ratio={s['mean_ratio']:.3f}{tag}")

    # All non-test studies go to training (NO FILTER)
    train_studies = sorted([s['study'] for s in all_studies if s['study'] not in test_set_all])

    print(f"\n  Accounting: {len(train_studies)} train + {len(ALL_TEST)} test = {len(train_studies) + len(ALL_TEST)}"
          f" / {n_total} total")

    print(f"\n{'=' * 70}")
    print(f"  TRAINING: {len(train_studies)} studies (ALL, no filter)")
    print(f"{'=' * 70}")
    for s in all_studies:
        if s['study'] in train_studies:
            mr = f"max_ratio={s['max_ratio']:.3f}" if s['max_ratio'] else "no_ratio"
            cat_tag = " *INV*" if s['category'] == 'INVERTED_MAX' else ""
            print(f"  {s['study']:15s}  {s['category']:12s}  {mr}"
                  f"  clot_n={s['clot_n']:6d}  wall_n={s['wall_n']:6d}{cat_tag}")

    # Test studies
    print(f"\n{'=' * 70}")
    print(f"  TEST: {len(ALL_TEST)} studies")
    print(f"{'=' * 70}")
    for name in ALL_TEST:
        in_fixed = name in FIXED_TEST
        tag = "[FIXED-8]" if in_fixed else "[EXPANDED]"
        filepath = os.path.join(SRC_TEST_DIR, f'{name}_labeled_segment.parquet')
        if os.path.exists(filepath):
            stats = compute_study_stats(filepath)
            if stats['clot_max'] and stats['wall_max']:
                mr = stats['clot_max'] / stats['wall_max']
                mm = stats['clot_mean'] / stats['wall_mean']
                print(f"  {name:15s}  {tag:10s}  max_ratio={mr:.3f}  mean_ratio={mm:.3f}")
            else:
                print(f"  {name:15s}  {tag:10s}  partial/no tissue")
        else:
            print(f"  {name:15s}  {tag:10s}  FILE NOT FOUND")

    # ── COPY FILES ──
    print(f"\n{'=' * 70}")
    print(f"  READY TO COPY")
    print(f"{'=' * 70}")
    print(f"  Training:     {len(train_studies)} files → training_data/")
    print(f"  Test:         {len(ALL_TEST)} files → test_data/")
    inverted_non_test = [s for s in inverted if s['study'] not in test_set_all]
    print(f"  Excluded eval: {len(inverted_non_test)} files → excluded_data/ (max-inverted, for evaluation)")

    response = input("\n  Proceed with copying? (yes/no): ").strip().lower()
    if response != 'yes':
        print("  Aborted. No files changed.")
        return

    # Clear and copy
    for folder in ['training_data', 'test_data', 'excluded_data']:
        if os.path.exists(folder):
            old_files = [f for f in os.listdir(folder) if f.endswith('.parquet')]
            print(f"  Clearing {folder}/ ({len(old_files)} old files)")
            for f in old_files:
                os.remove(os.path.join(folder, f))
        else:
            os.makedirs(folder)

    for s in train_studies:
        src = os.path.join(SRC_TRAIN_DIR, f'{s}_labeled_segment.parquet')
        dst = os.path.join('training_data', f'{s}_labeled_segment.parquet')
        shutil.copy2(src, dst)

    copied_test = 0
    for s in ALL_TEST:
        src = os.path.join(SRC_TEST_DIR, f'{s}_labeled_segment.parquet')
        dst = os.path.join('test_data', f'{s}_labeled_segment.parquet')
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied_test += 1
        else:
            print(f"  WARNING: {s} not found in test source dir")

    # Copy max-inverted non-test studies to excluded_data/ (from test source, with outliers)
    for s in inverted_non_test:
        src = os.path.join(SRC_TEST_DIR, f"{s['study']}_labeled_segment.parquet")
        dst = os.path.join('excluded_data', f"{s['study']}_labeled_segment.parquet")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: {s['study']} not found in test source dir for excluded_data/")

    print(f"\n  Copied {len(train_studies)} training files")
    print(f"  Copied {copied_test} test files")
    print(f"  Copied {len(inverted_non_test)} excluded eval files")

    # Delete stale caches
    for cache_dir in ['cache', os.path.join('experiments', 'scattering', 'cache')]:
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if f.endswith('.npz'):
                    os.remove(os.path.join(cache_dir, f))
                    print(f"  Deleted stale cache: {os.path.join(cache_dir, f)}")

    print(f"\n  Verification:")
    print(f"    training_data/:  {len(os.listdir('training_data'))} files")
    print(f"    test_data/:      {len(os.listdir('test_data'))} files")
    print(f"    excluded_data/:  {len(os.listdir('excluded_data'))} files")
    print(f"\n  Historical comparison:")
    print(f"    Batch 1 (meanR, v93):     24 train / 8 test")
    print(f"    Batch 2 (meanR, v94):     31 train / 8 test")
    print(f"    Batch 3 (meanR, Apr6):    34 train / 8 test")
    print(f"    Batch 4 (meanR, Apr10):   41 train / 8 test")
    print(f"    ALL (no filter, old):     79 train / 14 test")
    print(f"    Batch 5 (maxR, Apr10):    69 train / 14 test")
    print(f"    Batch 6 (ALL, Apr10):     79 train / 14 test")
    print(f"    Batch 7 (ALL, full dur):  {len(train_studies)} train / {copied_test} test  ← THIS")


if __name__ == '__main__':
    main()
