"""
build_train_test_split_183.py

Creates a stratified train/test split of the 183-file labeled dataset and
copies files into the workspace training_data/ and test_data/ folders so
V9 can run end-to-end on the full data.

Source (read-only):
    C:\\Users\\RonaldKurnik\\Inquis Medical\\DataScience - Documents\\Working\\
    Ronald Kurnik\\19August2026\\processedResults\\training

Workspace destination (overwritten, but backed up first):
    <PROJECT_ROOT>/training_data/
    <PROJECT_ROOT>/test_data/

Split rules:
  - TARGET_TEST_N = 33 (about 18%% of 183)
  - Stratified across a 3x3 grid of (clot fraction bucket) x (wall fraction bucket).
  - The current 10 workspace test files are PINNED to the new test set so all
    prior V9 inference metrics remain directly comparable. Additional test files
    are drawn from the remaining 173 to fill the strata that are under-represented.
  - Everything else goes to training.

Safety:
  - The current training_data/ and test_data/ folders are copied to
    training_data.backup_<timestamp>/ and test_data.backup_<timestamp>/ before
    being cleared and repopulated.
  - Source files are only READ. No writes happen in the source directory.

Outputs:
  - training_data/ and test_data/ populated with the chosen files
  - analysis_data_drift/train_test_manifest.csv (which file went where + why)
  - analysis_data_drift/train_test_split_summary.txt
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working"
    r"\Ronald Kurnik\19August2026\processedResults\training"
)
INVENTORY_CSV = PROJECT_ROOT / "analysis_data_drift" / "data_inventory_183.csv"

TRAIN_DIR = PROJECT_ROOT / "training_data"
TEST_DIR = PROJECT_ROOT / "test_data"
OUT_DIR = PROJECT_ROOT / "analysis_data_drift"

TARGET_TEST_N = 33
SEED = 42

# Bucket edges for stratification. Chosen from inventory quartiles so each
# bucket has a reasonable number of files.
CLOT_EDGES = [0.05, 0.15]   # buckets: <5%, 5-15%, >=15%
WALL_EDGES = [0.03, 0.10]   # buckets: <3%, 3-10%, >=10%

# ── Helpers ──────────────────────────────────────────────────────────────────

def _bucket(x: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def _stratum_key(row: pd.Series) -> str:
    cb = _bucket(row["frac_clot"], CLOT_EDGES)
    wb = _bucket(row["frac_wall"], WALL_EDGES)
    return f"c{cb}_w{wb}"


def backup_dir(src: Path, tag: str) -> Path | None:
    """Copy src to <src>.backup_<tag>_<timestamp>/ if it exists. Return the
    backup path, or None if src was empty/missing."""
    if not src.exists():
        return None
    files = list(src.glob("*"))
    if not files:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = src.parent / f"{src.name}.backup_{tag}_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for f in files:
        if f.is_file():
            shutil.copy2(f, backup / f.name)
    return backup


def clear_dir(d: Path) -> int:
    """Remove all *_labeled_segment.parquet from d. Return count removed."""
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        return 0
    n = 0
    for p in d.glob("*_labeled_segment.parquet"):
        p.unlink()
        n += 1
    return n


def copy_files(stems: list[str], dst: Path) -> tuple[int, int]:
    """Copy each stem from SOURCE_DIR to dst. Returns (n_copied, n_missing)."""
    dst.mkdir(parents=True, exist_ok=True)
    n_ok, n_miss = 0, 0
    for stem in stems:
        src_file = SOURCE_DIR / f"{stem}.parquet"
        if not src_file.exists():
            print(f"  WARNING: source missing: {src_file.name}")
            n_miss += 1
            continue
        shutil.copy2(src_file, dst / src_file.name)
        n_ok += 1
    return n_ok, n_miss


# ── Split algorithm ──────────────────────────────────────────────────────────

def stratified_test_selection(df: pd.DataFrame,
                              pinned_test_stems: set[str],
                              target_n: int,
                              rng: np.random.Generator) -> list[str]:
    """Choose `target_n` test files.

    Pinned files (currently in workspace test_data/) are always included.
    Remaining slots are filled to keep each stratum's train/test ratio close
    to the global ratio.
    """
    df = df.copy()
    df["stratum"] = df.apply(_stratum_key, axis=1)

    # Start with pinned stems (they may span multiple strata).
    chosen: list[str] = [s for s in df["stem"] if s in pinned_test_stems]
    print(f"  Pinned to test:   {len(chosen)}   (existing workspace test files)")

    remaining_needed = target_n - len(chosen)
    if remaining_needed <= 0:
        return chosen[:target_n]

    # Count files per stratum, count pinned per stratum, and compute how many
    # more we want from each stratum proportional to its population.
    stratum_counts = df["stratum"].value_counts().to_dict()
    pinned_by_stratum: dict[str, int] = {}
    for s in df.itertuples():
        if s.stem in chosen:
            pinned_by_stratum[s.stratum] = pinned_by_stratum.get(s.stratum, 0) + 1

    print(f"  Files per stratum (population / pinned):")
    for k in sorted(stratum_counts):
        print(f"    {k:>10}  {stratum_counts[k]:3d} / {pinned_by_stratum.get(k, 0):3d}")

    # Ideal test count per stratum = pop_share * target_n
    total = len(df)
    ideal = {k: v * target_n / total for k, v in stratum_counts.items()}

    # We must add extras only from strata whose pinned count is below their
    # ideal share. Round using largest-remainder to hit exactly remaining_needed.
    deficit = {k: max(0.0, ideal[k] - pinned_by_stratum.get(k, 0))
               for k in stratum_counts}
    total_deficit = sum(deficit.values())
    if total_deficit <= 0:
        # Fallback: fill from any non-pinned files randomly.
        candidates = df[~df["stem"].isin(chosen)]["stem"].tolist()
        rng.shuffle(candidates)
        chosen.extend(candidates[:remaining_needed])
        return chosen[:target_n]

    raw_alloc = {k: remaining_needed * deficit[k] / total_deficit
                 for k in deficit}
    # Largest-remainder rounding
    int_alloc = {k: int(np.floor(v)) for k, v in raw_alloc.items()}
    assigned = sum(int_alloc.values())
    remainders = sorted(
        raw_alloc.items(),
        key=lambda kv: (kv[1] - int_alloc[kv[0]]),
        reverse=True,
    )
    ri = 0
    while assigned < remaining_needed and ri < len(remainders):
        k = remainders[ri][0]
        int_alloc[k] += 1
        assigned += 1
        ri += 1

    print(f"  Adding to test per stratum (beyond pinned):")
    for k in sorted(int_alloc):
        if int_alloc[k] > 0:
            print(f"    {k:>10}  +{int_alloc[k]}")

    # Draw the extras. Prefer files with clot/wall content close to the middle
    # of each stratum so the test set doesn't accidentally pick only the extreme
    # examples. Randomize with seed for reproducibility.
    for stratum, n_extra in int_alloc.items():
        if n_extra <= 0:
            continue
        pool = df[(df["stratum"] == stratum) & (~df["stem"].isin(chosen))].copy()
        if len(pool) == 0:
            continue
        # Random draw
        take = rng.choice(pool["stem"].to_numpy(), size=min(n_extra, len(pool)),
                          replace=False)
        chosen.extend(take.tolist())

    return chosen[:target_n]


def main():
    if not INVENTORY_CSV.exists():
        print(f"ERROR: inventory CSV not found: {INVENTORY_CSV}")
        print("Run scripts/data_inventory_183.py first.")
        sys.exit(1)

    df = pd.read_csv(INVENTORY_CSV)
    print(f"Loaded inventory: {len(df)} files.")

    # Pinned files = whatever is currently in workspace test_data/
    pinned_test_stems = {p.stem for p in TEST_DIR.glob("*_labeled_segment.parquet")}
    print(f"Currently pinned (existing test set): {len(pinned_test_stems)}")

    rng = np.random.default_rng(SEED)

    test_stems = stratified_test_selection(df, pinned_test_stems, TARGET_TEST_N, rng)
    test_set = set(test_stems)
    train_stems = [s for s in df["stem"] if s not in test_set]

    print(f"\nDecided split: train={len(train_stems)}  test={len(test_stems)}")

    # ── Backup existing dirs ──
    print("\nBacking up existing training_data/ and test_data/ ...")
    train_backup = backup_dir(TRAIN_DIR, "presplit")
    test_backup = backup_dir(TEST_DIR, "presplit")
    print(f"  training_data backup: {train_backup}")
    print(f"  test_data backup    : {test_backup}")

    # ── Clear and repopulate ──
    n_removed_train = clear_dir(TRAIN_DIR)
    n_removed_test = clear_dir(TEST_DIR)
    print(f"  cleared {n_removed_train} old files from training_data/")
    print(f"  cleared {n_removed_test} old files from test_data/")

    print("\nCopying files into training_data/ ...")
    n_train_ok, n_train_miss = copy_files(train_stems, TRAIN_DIR)
    print(f"  train: {n_train_ok} copied, {n_train_miss} missing")

    print("Copying files into test_data/ ...")
    n_test_ok, n_test_miss = copy_files(test_stems, TEST_DIR)
    print(f"  test:  {n_test_ok} copied, {n_test_miss} missing")

    # ── Write manifest ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _row(stem: str, split: str) -> dict:
        r = df[df["stem"] == stem]
        if len(r) == 0:
            return {"stem": stem, "split": split, "reason": "unknown"}
        r = r.iloc[0]
        pinned = stem in pinned_test_stems
        reason = "pinned_existing_test" if pinned else "stratified_draw"
        if split == "train":
            reason = "not_in_test"
        return {
            "stem": stem,
            "split": split,
            "reason": reason,
            "frac_clot": r["frac_clot"],
            "frac_wall": r["frac_wall"],
            "frac_blood": r["frac_blood"],
            "duration_s": r["duration_s"],
            "stratum": _stratum_key(r),
        }

    manifest_rows = [_row(s, "test") for s in test_stems] + \
                    [_row(s, "train") for s in train_stems]
    manifest = pd.DataFrame(manifest_rows).sort_values(["split", "stem"])
    manifest_path = OUT_DIR / "train_test_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nWrote manifest: {manifest_path}")

    # ── Summary ──
    summary_lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        summary_lines.append(s)

    out("\n" + "=" * 78)
    out(f"TRAIN/TEST SPLIT SUMMARY  (seed={SEED})")
    out("=" * 78)
    out(f"Source directory : {SOURCE_DIR}")
    out(f"Target test count: {TARGET_TEST_N}")
    out(f"Pinned test files: {len(pinned_test_stems)}  (existing workspace test set)")
    out(f"Actual train / test: {len(train_stems)} / {len(test_stems)}")
    if train_backup:
        out(f"Backup (train)   : {train_backup.name}")
    if test_backup:
        out(f"Backup (test)    : {test_backup.name}")

    # Stratum balance table
    df["stratum"] = df.apply(_stratum_key, axis=1)
    df["split"] = df["stem"].apply(lambda s: "test" if s in test_set else "train")
    counts = df.groupby(["stratum", "split"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["test_pct"] = (counts.get("test", 0) / counts["total"] * 100).round(1)
    out("\nStratum balance:")
    out(f"  {'stratum':<12}{'train':>8}{'test':>8}{'total':>8}{'test %':>10}")
    for k, row in counts.iterrows():
        out(f"  {k:<12}{int(row.get('train', 0)):>8d}"
            f"{int(row.get('test', 0)):>8d}{int(row['total']):>8d}"
            f"{row['test_pct']:>9.1f}%")

    # Aggregate class balance in each split
    def _class_totals(sub: pd.DataFrame) -> tuple[int, int, int, int]:
        b = int((sub["frac_blood"] * sub["n_samples"]).sum())
        c = int((sub["frac_clot"] * sub["n_samples"]).sum())
        w = int((sub["frac_wall"] * sub["n_samples"]).sum())
        n = int(sub["n_samples"].sum())
        return b, c, w, n

    tr_sub = df[df["split"] == "train"]
    te_sub = df[df["split"] == "test"]
    for name, sub in (("train", tr_sub), ("test", te_sub)):
        b, c, w, n = _class_totals(sub)
        out(f"\n{name.upper()} class balance across {len(sub)} files "
            f"({n:,} samples):")
        out(f"  Blood: {b:>12,}  ({100*b/n:5.1f}%)")
        out(f"  Clot : {c:>12,}  ({100*c/n:5.1f}%)")
        out(f"  Wall : {w:>12,}  ({100*w/n:5.1f}%)")
        n_2c = c + w
        if n_2c > 0:
            out(f"  Clot vs wall only : clot={100*c/n_2c:5.1f}%  wall={100*w/n_2c:5.1f}%")

    out("\nTest set file list:")
    te_view = df[df["split"] == "test"].sort_values("stem")
    for _, r in te_view.iterrows():
        pin = "  (pinned)" if r["stem"] in pinned_test_stems else ""
        out(f"  {r['stem']:<40} dur={r['duration_s']:>6.0f}s  "
            f"clot={100*r['frac_clot']:5.1f}%  wall={100*r['frac_wall']:5.1f}%  "
            f"stratum={r['stratum']}{pin}")

    summary_path = OUT_DIR / "train_test_split_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")
    print("\nNext steps:")
    print("  1. python src/data/fit_scaler_V9.py")
    print("  2. python src/training/train_gru_V9.py --force-extract")
    print("  3. python src/models/gru_torch_V9.py")


if __name__ == "__main__":
    main()
