"""
data_inventory_183.py

Walks the full 183-file labeled dataset at
    C:\\Users\\RonaldKurnik\\Inquis Medical\\DataScience - Documents\\Working\\
    Ronald Kurnik\\19August2026\\processedResults\\training

and produces a per-file inventory:
  - filename, file size (MB)
  - is currently used as training data in the workspace
  - is currently used as test data in the workspace
  - total samples, duration (s)
  - label counts and fractions (blood / clot / wall / unlabeled)
  - mean/std/min/max of resistance overall and per class
  - number of contiguous clot events, number of contiguous wall events
  - mean event duration for clot and wall (s)

Output: analysis_data_drift/data_inventory_183.csv
Prints a summary to stdout, and writes a short summary to
    analysis_data_drift/data_inventory_summary.txt

Read-only w.r.t. the source data. Does not modify any parquet.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working"
    r"\Ronald Kurnik\19August2026\processedResults\training"
)
CUR_TRAIN_DIR = PROJECT_ROOT / "training_data"
CUR_TEST_DIR = PROJECT_ROOT / "test_data"

OUT_DIR = PROJECT_ROOT / "analysis_data_drift"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "data_inventory_183.csv"
SUMMARY_PATH = OUT_DIR / "data_inventory_summary.txt"

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ = 150
LABEL_NAMES = {0: "blood", 1: "clot", 2: "wall"}


def _count_events(labels: np.ndarray, target_label: int) -> tuple[int, float]:
    """Count contiguous runs of `target_label` and return (n_events, mean_duration_s).

    A run of length L samples has duration L / SAMPLE_RATE_HZ seconds.
    Returns (0, 0.0) if there are no runs.
    """
    if labels.size == 0:
        return 0, 0.0
    is_target = labels == target_label
    if not is_target.any():
        return 0, 0.0
    # Boundaries: positions where is_target changes value
    diff = np.diff(is_target.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    n = int(lengths.size)
    mean_dur = float(lengths.mean() / SAMPLE_RATE_HZ)
    return n, mean_dur


def _class_stat(resistance: np.ndarray, labels: np.ndarray, target: int) -> tuple[float, float]:
    """Return (mean, std) of resistance where labels == target. NaN if empty."""
    mask = labels == target
    if not mask.any():
        return float("nan"), float("nan")
    r = resistance[mask]
    return float(r.mean()), float(r.std())


def analyze_one_file(path: Path,
                     current_train_set: set[str],
                     current_test_set: set[str]) -> Optional[dict]:
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        print(f"  FAILED read {path.name}: {e}")
        return None

    required = {"timeInMS", "magRLoadAdjusted", "label"}
    missing = required - set(df.columns)
    if missing:
        print(f"  {path.name}: missing columns {missing}, skipping")
        return None

    resistance = df["magRLoadAdjusted"].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)
    n = int(len(labels))
    if n == 0:
        return None

    file_size_mb = path.stat().st_size / (1024 * 1024)
    duration_s = n / SAMPLE_RATE_HZ

    # Label counts (0=blood, 1=clot, 2=wall, other = unlabeled)
    n_blood = int((labels == 0).sum())
    n_clot = int((labels == 1).sum())
    n_wall = int((labels == 2).sum())
    n_other = n - (n_blood + n_clot + n_wall)

    # Class-conditional resistance stats
    r_mean_blood, r_std_blood = _class_stat(resistance, labels, 0)
    r_mean_clot, r_std_clot = _class_stat(resistance, labels, 1)
    r_mean_wall, r_std_wall = _class_stat(resistance, labels, 2)

    # Overall resistance stats
    r_mean = float(resistance.mean())
    r_std = float(resistance.std())
    r_min = float(resistance.min())
    r_max = float(resistance.max())

    # Event counts
    n_clot_events, mean_clot_dur_s = _count_events(labels, 1)
    n_wall_events, mean_wall_dur_s = _count_events(labels, 2)

    # DA presence
    has_da = "da_label" in df.columns

    stem = path.stem  # e.g. "00F628C9_labeled_segment"
    in_cur_train = stem in current_train_set
    in_cur_test = stem in current_test_set

    return {
        "filename": path.name,
        "stem": stem,
        "file_size_mb": round(file_size_mb, 3),
        "in_cur_train": in_cur_train,
        "in_cur_test": in_cur_test,
        "in_cur_unused": (not in_cur_train) and (not in_cur_test),
        "has_da_label": has_da,
        "n_samples": n,
        "duration_s": round(duration_s, 1),
        # counts
        "n_blood": n_blood,
        "n_clot": n_clot,
        "n_wall": n_wall,
        "n_unlabeled": n_other,
        # fractions
        "frac_blood": round(n_blood / n, 4),
        "frac_clot": round(n_clot / n, 4),
        "frac_wall": round(n_wall / n, 4),
        "frac_unlabeled": round(n_other / n, 4),
        # events
        "n_clot_events": n_clot_events,
        "n_wall_events": n_wall_events,
        "mean_clot_event_s": round(mean_clot_dur_s, 3),
        "mean_wall_event_s": round(mean_wall_dur_s, 3),
        # resistance summaries
        "R_mean_overall": round(r_mean, 1),
        "R_std_overall": round(r_std, 1),
        "R_min_overall": round(r_min, 1),
        "R_max_overall": round(r_max, 1),
        "R_mean_blood": round(r_mean_blood, 1) if r_mean_blood == r_mean_blood else np.nan,
        "R_std_blood": round(r_std_blood, 1) if r_std_blood == r_std_blood else np.nan,
        "R_mean_clot": round(r_mean_clot, 1) if r_mean_clot == r_mean_clot else np.nan,
        "R_std_clot": round(r_std_clot, 1) if r_std_clot == r_std_clot else np.nan,
        "R_mean_wall": round(r_mean_wall, 1) if r_mean_wall == r_mean_wall else np.nan,
        "R_std_wall": round(r_std_wall, 1) if r_std_wall == r_std_wall else np.nan,
    }


def build_inventory() -> pd.DataFrame:
    print(f"Source dir : {SOURCE_DIR}")
    print(f"Output CSV : {CSV_PATH}")
    print()

    if not SOURCE_DIR.exists():
        print(f"ERROR: source directory does not exist: {SOURCE_DIR}")
        sys.exit(1)

    files = sorted(SOURCE_DIR.glob("*_labeled_segment.parquet"))
    print(f"Found {len(files)} labeled files in source.")

    # Current workspace membership (stem-based lookup)
    cur_train_set = {p.stem for p in CUR_TRAIN_DIR.glob("*_labeled_segment.parquet")}
    cur_test_set = {p.stem for p in CUR_TEST_DIR.glob("*_labeled_segment.parquet")}
    print(f"Currently in {CUR_TRAIN_DIR.name}/: {len(cur_train_set)}")
    print(f"Currently in {CUR_TEST_DIR.name}/ : {len(cur_test_set)}")

    rows = []
    t0 = time.perf_counter()
    for i, p in enumerate(files, 1):
        rec = analyze_one_file(p, cur_train_set, cur_test_set)
        if rec is not None:
            rows.append(rec)
        if i % 20 == 0 or i == len(files):
            elapsed = time.perf_counter() - t0
            print(f"  [{i:3d}/{len(files)}] elapsed {elapsed:5.1f}s")

    df = pd.DataFrame(rows)
    df = df.sort_values("filename").reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nWrote {CSV_PATH}")
    return df


def _fmt_frac(x: float) -> str:
    return f"{100 * x:5.1f}%"


def print_and_save_summary(df: pd.DataFrame) -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 78)
    out(f"DATA INVENTORY SUMMARY  ({len(df)} files)")
    out("=" * 78)

    # ── Membership breakdown ──
    n_train = int(df["in_cur_train"].sum())
    n_test = int(df["in_cur_test"].sum())
    n_unused = int(df["in_cur_unused"].sum())
    out(f"\nCurrent workspace membership:")
    out(f"  In training_data/ : {n_train:3d}")
    out(f"  In test_data/     : {n_test:3d}")
    out(f"  Not used anywhere : {n_unused:3d}   <-- available to grow train/test")

    # ── Aggregate class balance ──
    total_samples = int(df["n_samples"].sum())
    total_blood = int(df["n_blood"].sum())
    total_clot = int(df["n_clot"].sum())
    total_wall = int(df["n_wall"].sum())
    total_other = int(df["n_unlabeled"].sum())
    out(f"\nAggregate class balance across all {len(df)} files:")
    out(f"  Total samples : {total_samples:,}")
    out(f"  Blood         : {total_blood:>12,}  ({_fmt_frac(total_blood/total_samples)})")
    out(f"  Clot          : {total_clot:>12,}  ({_fmt_frac(total_clot/total_samples)})")
    out(f"  Wall          : {total_wall:>12,}  ({_fmt_frac(total_wall/total_samples)})")
    out(f"  Unlabeled     : {total_other:>12,}  ({_fmt_frac(total_other/total_samples)})")

    total_2class = total_clot + total_wall
    out(f"\n  Clot vs wall only (V9 training target):")
    out(f"    Clot : {total_clot:>12,}  ({_fmt_frac(total_clot/max(total_2class,1))})")
    out(f"    Wall : {total_wall:>12,}  ({_fmt_frac(total_wall/max(total_2class,1))})")

    # ── Files with zero of a class ──
    n_no_clot = int((df["n_clot"] == 0).sum())
    n_no_wall = int((df["n_wall"] == 0).sum())
    n_no_blood = int((df["n_blood"] == 0).sum())
    n_no_events = int(((df["n_clot"] == 0) & (df["n_wall"] == 0)).sum())
    out(f"\nFiles with zero samples of a given class:")
    out(f"  No blood : {n_no_blood:3d}")
    out(f"  No clot  : {n_no_clot:3d}")
    out(f"  No wall  : {n_no_wall:3d}")
    out(f"  No clot AND no wall : {n_no_events:3d}   (pure blood streams)")

    # ── Duration distribution ──
    d = df["duration_s"]
    out(f"\nDuration (seconds):")
    out(f"  min {d.min():7.1f}   p25 {d.quantile(0.25):7.1f}   median {d.median():7.1f}"
        f"   p75 {d.quantile(0.75):7.1f}   max {d.max():7.1f}")
    out(f"  total recorded time : {int(d.sum())} s  ({d.sum()/3600:.1f} hours)")

    # ── Resistance distribution ──
    out(f"\nResistance (Ohms) mean-per-file summary:")
    for col, lbl in [("R_mean_blood", "blood"),
                     ("R_mean_clot", "clot"),
                     ("R_mean_wall", "wall")]:
        v = df[col].dropna()
        if len(v):
            out(f"  {lbl:>5} : n_files_with_class={len(v):3d}   "
                f"mean_of_means={v.mean():7.1f}   "
                f"p25={v.quantile(0.25):7.1f}   p75={v.quantile(0.75):7.1f}   "
                f"max={v.max():7.1f}")

    # ── Class-fraction distribution ──
    out(f"\nPer-file blood fraction distribution:")
    fb = df["frac_blood"]
    out(f"  p25={_fmt_frac(fb.quantile(0.25))}   median={_fmt_frac(fb.median())}"
        f"   p75={_fmt_frac(fb.quantile(0.75))}   max={_fmt_frac(fb.max())}")
    out(f"Per-file clot fraction distribution:")
    fc = df["frac_clot"]
    out(f"  p25={_fmt_frac(fc.quantile(0.25))}   median={_fmt_frac(fc.median())}"
        f"   p75={_fmt_frac(fc.quantile(0.75))}   max={_fmt_frac(fc.max())}")
    out(f"Per-file wall fraction distribution:")
    fw = df["frac_wall"]
    out(f"  p25={_fmt_frac(fw.quantile(0.25))}   median={_fmt_frac(fw.median())}"
        f"   p75={_fmt_frac(fw.quantile(0.75))}   max={_fmt_frac(fw.max())}")

    # ── Files with highest clot / wall content ──
    out(f"\nTop 10 files by clot fraction:")
    top_clot = df.nlargest(10, "frac_clot")[
        ["stem", "duration_s", "frac_clot", "frac_wall", "frac_blood",
         "in_cur_train", "in_cur_test"]
    ]
    for _, r in top_clot.iterrows():
        out(f"  {r['stem']:<40} dur={r['duration_s']:>6.0f}s "
            f"clot={_fmt_frac(r['frac_clot'])} wall={_fmt_frac(r['frac_wall'])} "
            f"blood={_fmt_frac(r['frac_blood'])} "
            f"train={r['in_cur_train']} test={r['in_cur_test']}")

    out(f"\nTop 10 files by wall fraction:")
    top_wall = df.nlargest(10, "frac_wall")[
        ["stem", "duration_s", "frac_clot", "frac_wall", "frac_blood",
         "in_cur_train", "in_cur_test"]
    ]
    for _, r in top_wall.iterrows():
        out(f"  {r['stem']:<40} dur={r['duration_s']:>6.0f}s "
            f"clot={_fmt_frac(r['frac_clot'])} wall={_fmt_frac(r['frac_wall'])} "
            f"blood={_fmt_frac(r['frac_blood'])} "
            f"train={r['in_cur_train']} test={r['in_cur_test']}")

    # ── Current test set profile ──
    out(f"\nCurrent test_data/ profile ({int(df['in_cur_test'].sum())} files):")
    cur_test_df = df[df["in_cur_test"]].sort_values("stem")
    for _, r in cur_test_df.iterrows():
        out(f"  {r['stem']:<40} dur={r['duration_s']:>6.0f}s "
            f"clot={_fmt_frac(r['frac_clot'])} wall={_fmt_frac(r['frac_wall'])} "
            f"blood={_fmt_frac(r['frac_blood'])} "
            f"n_clot_events={r['n_clot_events']} n_wall_events={r['n_wall_events']}")

    # ── Where does mass live? ──
    out(f"\nMass distribution — top 10 files by sample count:")
    top_size = df.nlargest(10, "n_samples")[
        ["stem", "n_samples", "duration_s", "frac_blood", "frac_clot", "frac_wall",
         "in_cur_train", "in_cur_test"]
    ]
    for _, r in top_size.iterrows():
        out(f"  {r['stem']:<40} n={r['n_samples']:>8,}  dur={r['duration_s']:>6.0f}s  "
            f"blood={_fmt_frac(r['frac_blood'])} clot={_fmt_frac(r['frac_clot'])} "
            f"wall={_fmt_frac(r['frac_wall'])} train={r['in_cur_train']} test={r['in_cur_test']}")

    out("=" * 78)

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {SUMMARY_PATH}")


def main():
    df = build_inventory()
    print_and_save_summary(df)


if __name__ == "__main__":
    main()
