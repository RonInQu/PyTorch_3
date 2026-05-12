"""
Analyze how clot/wall characteristics change across studies.

Groups studies by site prefix (CENT, HUNT, SUMM, hex-ID, etc.)
and computes per-study statistics for clot and wall events:
  - Duration (seconds)
  - Height (mean R, max R, median R)
  - R ratio (clot_mean / wall_mean)
  - Blood baseline R
  - Number of events per study

Outputs:
  1. Console summary table
  2. CSV with all per-event stats
  3. Plots showing feature distributions by site group
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "training_data"
TEST_DIR  = PROJECT_ROOT / "test_data"
OUT_DIR   = PROJECT_ROOT / "analysis_data_drift"
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 150  # Hz

def get_site_group(study_id):
    """Classify study into site group by naming pattern."""
    prefixes = ['CENT', 'HUNT', 'SUMM', 'BAPT', 'ELCA', 'STCL', 'NASH', 'SOMI', 'UHMA', 'UH00', 'UNIH', 'HACK']
    for p in prefixes:
        if study_id.startswith(p):
            return p
    return 'HEX'  # anonymized hex IDs


def extract_events(resistance, labels, sample_rate=SAMPLE_RATE):
    """Extract contiguous clot/wall events and compute stats for each."""
    events = []
    n = len(labels)
    i = 0
    while i < n:
        lbl = labels[i]
        if lbl in (1, 2):  # clot or wall
            j = i
            while j < n and labels[j] == lbl:
                j += 1
            seg = resistance[i:j]
            dur_sec = (j - i) / sample_rate
            events.append({
                'label': int(lbl),
                'label_name': 'clot' if lbl == 1 else 'wall',
                'start_idx': i,
                'end_idx': j,
                'duration_sec': dur_sec,
                'n_samples': j - i,
                'mean_R': float(np.mean(seg)),
                'median_R': float(np.median(seg)),
                'max_R': float(np.max(seg)),
                'min_R': float(np.min(seg)),
                'std_R': float(np.std(seg)),
                'range_R': float(np.max(seg) - np.min(seg)),
            })
            i = j
        else:
            i += 1
    return events


def compute_blood_stats(resistance, labels):
    """Stats for blood (label=0) segments."""
    blood_mask = labels == 0
    if not blood_mask.any():
        return {'blood_mean_R': np.nan, 'blood_median_R': np.nan, 'blood_std_R': np.nan}
    blood_r = resistance[blood_mask]
    return {
        'blood_mean_R': float(np.mean(blood_r)),
        'blood_median_R': float(np.median(blood_r)),
        'blood_std_R': float(np.std(blood_r)),
    }


def analyze_all_studies():
    """Main analysis: loop through all parquets, extract event-level stats."""

    all_events = []
    study_summaries = []

    for data_dir, split in [(TRAIN_DIR, 'train'), (TEST_DIR, 'test')]:
        parquets = sorted(data_dir.glob("*_labeled_segment.parquet"))
        for fp in parquets:
            study_id = fp.stem.replace('_labeled_segment', '')
            df = pd.read_parquet(fp, engine='pyarrow')

            if 'magRLoadAdjusted' not in df.columns or 'label' not in df.columns:
                continue

            resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
            labels = df['label'].to_numpy(dtype=np.int64)

            site = get_site_group(study_id)
            events = extract_events(resistance, labels)
            blood = compute_blood_stats(resistance, labels)

            # Per-event records
            for ev in events:
                ev['study_id'] = study_id
                ev['site'] = site
                ev['split'] = split
                ev['study_duration_sec'] = len(resistance) / SAMPLE_RATE
                ev.update(blood)
                all_events.append(ev)

            # Per-study summary
            clot_events = [e for e in events if e['label'] == 1]
            wall_events = [e for e in events if e['label'] == 2]

            summary = {
                'study_id': study_id,
                'site': site,
                'split': split,
                'total_samples': len(resistance),
                'study_duration_sec': len(resistance) / SAMPLE_RATE,
                'n_clot_events': len(clot_events),
                'n_wall_events': len(wall_events),
                'clot_total_sec': sum(e['duration_sec'] for e in clot_events),
                'wall_total_sec': sum(e['duration_sec'] for e in wall_events),
                'clot_mean_dur': np.mean([e['duration_sec'] for e in clot_events]) if clot_events else np.nan,
                'wall_mean_dur': np.mean([e['duration_sec'] for e in wall_events]) if wall_events else np.nan,
                'clot_mean_R': np.mean([e['mean_R'] for e in clot_events]) if clot_events else np.nan,
                'wall_mean_R': np.mean([e['mean_R'] for e in wall_events]) if wall_events else np.nan,
                'clot_max_R': np.max([e['max_R'] for e in clot_events]) if clot_events else np.nan,
                'wall_max_R': np.max([e['max_R'] for e in wall_events]) if wall_events else np.nan,
            }
            summary.update(blood)
            # Ratio: key difficulty indicator
            if clot_events and wall_events:
                summary['clot_wall_ratio'] = summary['clot_mean_R'] / summary['wall_mean_R'] if summary['wall_mean_R'] > 0 else np.nan
            else:
                summary['clot_wall_ratio'] = np.nan

            study_summaries.append(summary)

    return pd.DataFrame(all_events), pd.DataFrame(study_summaries)


def print_summary_table(df_studies):
    """Print grouped summary by site."""
    print("\n" + "=" * 100)
    print("PER-SITE GROUP SUMMARY")
    print("=" * 100)

    for site in sorted(df_studies['site'].unique()):
        group = df_studies[df_studies['site'] == site]
        n_train = (group['split'] == 'train').sum()
        n_test = (group['split'] == 'test').sum()
        print(f"\n--- {site} ({len(group)} studies: {n_train} train, {n_test} test) ---")
        cols = ['clot_mean_R', 'wall_mean_R', 'blood_mean_R', 'clot_wall_ratio',
                'clot_mean_dur', 'wall_mean_dur', 'n_clot_events', 'n_wall_events']
        for c in cols:
            vals = group[c].dropna()
            if len(vals) > 0:
                print(f"  {c:25s}  mean={vals.mean():8.1f}  std={vals.std():8.1f}  "
                      f"min={vals.min():8.1f}  max={vals.max():8.1f}  n={len(vals)}")


def plot_distributions(df_events, df_studies):
    """Generate diagnostic plots."""

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Clot/Wall Event Characteristics by Site Group", fontsize=14, y=0.98)

    # 1. Clot mean R by site
    ax = axes[0, 0]
    clot_events = df_events[df_events['label'] == 1]
    sites = sorted(clot_events['site'].unique())
    data = [clot_events[clot_events['site'] == s]['mean_R'].values for s in sites]
    ax.boxplot(data, labels=sites, showfliers=False)
    ax.set_title("Clot Mean R by Site")
    ax.set_ylabel("R (ohms)")
    ax.tick_params(axis='x', rotation=45)

    # 2. Wall mean R by site
    ax = axes[0, 1]
    wall_events = df_events[df_events['label'] == 2]
    data = [wall_events[wall_events['site'] == s]['mean_R'].values for s in sites if s in wall_events['site'].values]
    wall_sites = [s for s in sites if s in wall_events['site'].values]
    ax.boxplot(data, labels=wall_sites, showfliers=False)
    ax.set_title("Wall Mean R by Site")
    ax.set_ylabel("R (ohms)")
    ax.tick_params(axis='x', rotation=45)

    # 3. Blood mean R by site
    ax = axes[0, 2]
    data = [df_studies[df_studies['site'] == s]['blood_mean_R'].dropna().values for s in sites]
    ax.boxplot(data, labels=sites, showfliers=False)
    ax.set_title("Blood Baseline R by Site")
    ax.set_ylabel("R (ohms)")
    ax.tick_params(axis='x', rotation=45)

    # 4. Clot duration by site
    ax = axes[1, 0]
    data = [clot_events[clot_events['site'] == s]['duration_sec'].values for s in sites]
    ax.boxplot(data, labels=sites, showfliers=False)
    ax.set_title("Clot Duration by Site")
    ax.set_ylabel("Duration (sec)")
    ax.tick_params(axis='x', rotation=45)

    # 5. Wall duration by site
    ax = axes[1, 1]
    data = [wall_events[wall_events['site'] == s]['duration_sec'].values for s in sites if s in wall_events['site'].values]
    ax.boxplot(data, labels=wall_sites, showfliers=False)
    ax.set_title("Wall Duration by Site")
    ax.set_ylabel("Duration (sec)")
    ax.tick_params(axis='x', rotation=45)

    # 6. Clot/Wall R ratio by site (study-level)
    ax = axes[1, 2]
    data = [df_studies[df_studies['site'] == s]['clot_wall_ratio'].dropna().values for s in sites]
    data_filtered = [(d, s) for d, s in zip(data, sites) if len(d) > 0]
    if data_filtered:
        ax.boxplot([d for d, _ in data_filtered], labels=[s for _, s in data_filtered], showfliers=False)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='ratio=1 (equal)')
    ax.set_title("Clot/Wall R Ratio by Site")
    ax.set_ylabel("Ratio (clot_mean / wall_mean)")
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)

    # 7. Scatter: clot_mean_R vs wall_mean_R (per study)
    ax = axes[2, 0]
    for site in sites:
        sub = df_studies[df_studies['site'] == site]
        marker = 'x' if sub.iloc[0]['split'] == 'test' else 'o'
        ax.scatter(sub['clot_mean_R'], sub['wall_mean_R'], label=site, alpha=0.7, s=40)
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'r--', alpha=0.3, label='clot=wall')
    ax.set_xlabel("Clot Mean R")
    ax.set_ylabel("Wall Mean R")
    ax.set_title("Clot vs Wall Mean R (per study)")
    ax.legend(fontsize=7, ncol=2)

    # 8. Histogram: clot_wall_ratio
    ax = axes[2, 1]
    ratios = df_studies['clot_wall_ratio'].dropna()
    ax.hist(ratios, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=1.0, color='red', linestyle='--', label='ratio=1')
    ax.set_xlabel("Clot/Wall R Ratio")
    ax.set_ylabel("Count (studies)")
    ax.set_title(f"Clot/Wall Ratio Distribution (n={len(ratios)})")
    ax.legend()

    # 9. Per-study: number of events
    ax = axes[2, 2]
    ax.scatter(df_studies['n_clot_events'], df_studies['n_wall_events'], alpha=0.6,
               c=[{'train': 'blue', 'test': 'red'}[s] for s in df_studies['split']])
    ax.set_xlabel("# Clot Events")
    ax.set_ylabel("# Wall Events")
    ax.set_title("Event Counts per Study (blue=train, red=test)")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "event_distributions_by_site.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'event_distributions_by_site.png'}")
    plt.close()

    # ── Second figure: "hard" vs "easy" studies ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle("Study Difficulty Analysis", fontsize=13)

    # Ratio histogram colored by split
    ax = axes2[0]
    train_ratios = df_studies[df_studies['split'] == 'train']['clot_wall_ratio'].dropna()
    test_ratios = df_studies[df_studies['split'] == 'test']['clot_wall_ratio'].dropna()
    ax.hist(train_ratios, bins=20, alpha=0.6, label=f'Train (n={len(train_ratios)})', color='blue')
    ax.hist(test_ratios, bins=20, alpha=0.6, label=f'Test (n={len(test_ratios)})', color='red')
    ax.axvline(x=1.0, color='black', linestyle='--')
    ax.set_xlabel("Clot/Wall R Ratio")
    ax.set_title("Ratio Distribution: Train vs Test")
    ax.legend()

    # Blood baseline scatter
    ax = axes2[1]
    ax.scatter(df_studies[df_studies['split']=='train']['blood_mean_R'],
               df_studies[df_studies['split']=='train']['clot_wall_ratio'],
               alpha=0.6, label='Train', color='blue')
    ax.scatter(df_studies[df_studies['split']=='test']['blood_mean_R'],
               df_studies[df_studies['split']=='test']['clot_wall_ratio'],
               alpha=0.6, label='Test', color='red', marker='x', s=60)
    ax.set_xlabel("Blood Mean R (ohms)")
    ax.set_ylabel("Clot/Wall R Ratio")
    ax.set_title("Blood Baseline vs Difficulty")
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax.legend()

    # Duration vs R for clot events
    ax = axes2[2]
    ax.scatter(clot_events['duration_sec'], clot_events['mean_R'], alpha=0.3, s=10)
    ax.set_xlabel("Clot Duration (sec)")
    ax.set_ylabel("Clot Mean R (ohms)")
    ax.set_title(f"Clot: Duration vs Height (n={len(clot_events)} events)")

    plt.tight_layout()
    fig2.savefig(OUT_DIR / "study_difficulty_analysis.png", dpi=150)
    print(f"Saved: {OUT_DIR / 'study_difficulty_analysis.png'}")
    plt.close()

    # ── Third figure: inverted vs normal studies ──
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    df_sorted = df_studies.dropna(subset=['clot_wall_ratio']).sort_values('clot_wall_ratio')
    colors = ['red' if r > 1.0 else 'green' for r in df_sorted['clot_wall_ratio']]
    edge = ['black' if s == 'test' else 'none' for s in df_sorted['split']]
    ax3.barh(range(len(df_sorted)), df_sorted['clot_wall_ratio'], color=colors,
             edgecolor=edge, linewidth=1.5)
    ax3.axvline(x=1.0, color='black', linestyle='--', linewidth=2)
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels(df_sorted['study_id'], fontsize=5)
    ax3.set_xlabel("Clot/Wall R Ratio")
    ax3.set_title("All Studies: Clot/Wall R Ratio (green=normal, red=inverted, black border=test)")
    plt.tight_layout()
    fig3.savefig(OUT_DIR / "all_studies_ratio_ranked.png", dpi=150)
    print(f"Saved: {OUT_DIR / 'all_studies_ratio_ranked.png'}")
    plt.close()


def print_inverted_studies(df_studies):
    """Flag studies where clot_mean_R > wall_mean_R (inverted polarity)."""
    inv = df_studies[df_studies['clot_wall_ratio'] > 1.0].sort_values('clot_wall_ratio', ascending=False)
    print(f"\n{'=' * 80}")
    print(f"INVERTED STUDIES (clot_mean_R > wall_mean_R): {len(inv)} / {len(df_studies)}")
    print(f"{'=' * 80}")
    if len(inv) > 0:
        for _, row in inv.iterrows():
            print(f"  {row['study_id']:15s}  site={row['site']:6s}  split={row['split']:5s}  "
                  f"ratio={row['clot_wall_ratio']:.3f}  "
                  f"clot_R={row['clot_mean_R']:.0f}  wall_R={row['wall_mean_R']:.0f}  "
                  f"blood_R={row['blood_mean_R']:.0f}")


def print_train_test_comparison(df_studies):
    """Compare distributions between train and test sets."""
    print(f"\n{'=' * 80}")
    print("TRAIN vs TEST DISTRIBUTION COMPARISON")
    print(f"{'=' * 80}")
    metrics = ['blood_mean_R', 'clot_mean_R', 'wall_mean_R', 'clot_wall_ratio',
               'clot_mean_dur', 'wall_mean_dur', 'n_clot_events', 'n_wall_events']
    train = df_studies[df_studies['split'] == 'train']
    test = df_studies[df_studies['split'] == 'test']
    print(f"  {'Metric':25s}  {'Train mean':>12s}  {'Train std':>10s}  {'Test mean':>12s}  {'Test std':>10s}  {'Gap%':>8s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*8}")
    for m in metrics:
        tv = train[m].dropna()
        ev = test[m].dropna()
        if len(tv) > 0 and len(ev) > 0:
            gap = (ev.mean() - tv.mean()) / tv.mean() * 100 if tv.mean() != 0 else 0
            print(f"  {m:25s}  {tv.mean():12.1f}  {tv.std():10.1f}  "
                  f"{ev.mean():12.1f}  {ev.std():10.1f}  {gap:+7.1f}%")


if __name__ == "__main__":
    print("Analyzing clot/wall characteristics across all studies...")
    df_events, df_studies = analyze_all_studies()

    # Save raw data
    df_events.to_csv(OUT_DIR / "all_events.csv", index=False)
    df_studies.to_csv(OUT_DIR / "study_summaries.csv", index=False)
    print(f"Saved: {OUT_DIR / 'all_events.csv'} ({len(df_events)} events)")
    print(f"Saved: {OUT_DIR / 'study_summaries.csv'} ({len(df_studies)} studies)")

    # Print summaries
    print_summary_table(df_studies)
    print_inverted_studies(df_studies)
    print_train_test_comparison(df_studies)

    # Generate plots
    plot_distributions(df_events, df_studies)

    # Final count summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total studies:  {len(df_studies)} ({(df_studies['split']=='train').sum()} train, {(df_studies['split']=='test').sum()} test)")
    print(f"  Total events:   {len(df_events)} ({(df_events['label']==1).sum()} clot, {(df_events['label']==2).sum()} wall)")
    n_inv = (df_studies['clot_wall_ratio'] > 1.0).sum()
    print(f"  Inverted:       {n_inv} / {len(df_studies)} studies have clot_mean > wall_mean")
    clot_durs = df_events[df_events['label'] == 1]['duration_sec']
    wall_durs = df_events[df_events['label'] == 2]['duration_sec']
    print(f"  Clot durations: mean={clot_durs.mean():.1f}s  median={clot_durs.median():.1f}s  "
          f"min={clot_durs.min():.1f}s  max={clot_durs.max():.1f}s")
    print(f"  Wall durations: mean={wall_durs.mean():.1f}s  median={wall_durs.median():.1f}s  "
          f"min={wall_durs.min():.1f}s  max={wall_durs.max():.1f}s")
    print(f"\nDone. Results in: {OUT_DIR}")
