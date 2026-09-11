"""
Audit label integrity across every *_labeled_segment.parquet in training_data/
and test_data/.

For each file, report:
  - baseline_R : median R where label == 0 (blood)
  - baseline_p95 : upper edge of the blood distribution
  - For BOTH `label` (GT) and `da_label` (DA):
      * how many clot samples have R <= baseline_p95
      * how many wall samples have R <= baseline_p95
      * fraction relative to total clot/wall samples in that file
      * "at-baseline fraction" of the label — how much of what the file
        calls "clot" is actually sitting at the blood level

Writes:
  analysis_data_drift/label_audit_summary.csv    (per-file numbers)
  analysis_data_drift/label_audit_flagged.txt    (files with suspicious levels)

Two thresholds worth flagging:
  * GT flag: any clot/wall count at-baseline > 500 samples AND > 5% of that class
      → indicates the training label file itself is corrupted
  * DA flag: any da_label=1/2 at-baseline fraction > 20%
      → indicates the DA-benchmark for that file is unreliable

Usage:
  python scripts/audit_label_integrity.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [ROOT / "training_data", ROOT / "test_data"]
OUT_DIR = ROOT / "analysis_data_drift"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUT_DIR / "label_audit_summary.csv"
TXT_OUT = OUT_DIR / "label_audit_flagged.txt"

# Thresholds
GT_MIN_COUNT = 500          # samples
GT_MIN_FRAC = 0.05          # 5% of that class
DA_MIN_FRAC = 0.20          # 20% of that class


def audit_file(pq_path: Path) -> dict:
    df = pd.read_parquet(pq_path)
    R = df["magRLoadAdjusted"].to_numpy()
    gt = df["label"].to_numpy().astype(int)
    da = (
        df["da_label"].to_numpy().astype(int)
        if "da_label" in df.columns
        else np.full(len(df), -99)
    )

    n = len(df)
    blood_R = R[gt == 0]
    baseline_med = float(np.median(blood_R)) if len(blood_R) else float("nan")
    baseline_p95 = float(np.percentile(blood_R, 95)) if len(blood_R) else float("nan")

    def _at_baseline(mask_lbl: np.ndarray) -> int:
        return int(((R <= baseline_p95) & mask_lbl).sum())

    gt_clot_mask = gt == 1
    gt_wall_mask = gt == 2
    n_gt_clot = int(gt_clot_mask.sum())
    n_gt_wall = int(gt_wall_mask.sum())
    gt_clot_at_bl = _at_baseline(gt_clot_mask)
    gt_wall_at_bl = _at_baseline(gt_wall_mask)

    da_clot_mask = da == 1
    da_wall_mask = da == 2
    n_da_clot = int(da_clot_mask.sum())
    n_da_wall = int(da_wall_mask.sum())
    da_clot_at_bl = _at_baseline(da_clot_mask)
    da_wall_at_bl = _at_baseline(da_wall_mask)

    def frac(num: int, den: int) -> float:
        return float(num) / float(den) if den > 0 else 0.0

    return dict(
        file=pq_path.stem.replace("_labeled_segment", ""),
        n_samples=n,
        baseline_med=round(baseline_med, 1),
        baseline_p95=round(baseline_p95, 1),
        n_gt_clot=n_gt_clot,
        gt_clot_at_baseline=gt_clot_at_bl,
        gt_clot_at_baseline_frac=round(frac(gt_clot_at_bl, n_gt_clot), 4),
        n_gt_wall=n_gt_wall,
        gt_wall_at_baseline=gt_wall_at_bl,
        gt_wall_at_baseline_frac=round(frac(gt_wall_at_bl, n_gt_wall), 4),
        n_da_clot=n_da_clot,
        da_clot_at_baseline=da_clot_at_bl,
        da_clot_at_baseline_frac=round(frac(da_clot_at_bl, n_da_clot), 4),
        n_da_wall=n_da_wall,
        da_wall_at_baseline=da_wall_at_bl,
        da_wall_at_baseline_frac=round(frac(da_wall_at_bl, n_da_wall), 4),
    )


def main():
    rows = []
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for pq in sorted(d.glob("*_labeled_segment.parquet")):
            try:
                row = audit_file(pq)
                row["source_dir"] = d.name
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR on {pq.name}: {e}")

    if not rows:
        print("No files found.")
        return

    df = pd.DataFrame(rows)
    df = df[
        [
            "file",
            "source_dir",
            "n_samples",
            "baseline_med",
            "baseline_p95",
            "n_gt_clot",
            "gt_clot_at_baseline",
            "gt_clot_at_baseline_frac",
            "n_gt_wall",
            "gt_wall_at_baseline",
            "gt_wall_at_baseline_frac",
            "n_da_clot",
            "da_clot_at_baseline",
            "da_clot_at_baseline_frac",
            "n_da_wall",
            "da_wall_at_baseline",
            "da_wall_at_baseline_frac",
        ]
    ]
    df.to_csv(CSV_OUT, index=False)
    print(f"\nWrote {CSV_OUT}")
    print(f"Audited {len(df)} files.")

    # ----- Flag GT corruption -----
    gt_clot_bad = df[
        (df["gt_clot_at_baseline"] >= GT_MIN_COUNT)
        & (df["gt_clot_at_baseline_frac"] >= GT_MIN_FRAC)
    ]
    gt_wall_bad = df[
        (df["gt_wall_at_baseline"] >= GT_MIN_COUNT)
        & (df["gt_wall_at_baseline_frac"] >= GT_MIN_FRAC)
    ]

    # ----- Flag DA corruption -----
    da_clot_bad = df[df["da_clot_at_baseline_frac"] >= DA_MIN_FRAC]
    da_wall_bad = df[df["da_wall_at_baseline_frac"] >= DA_MIN_FRAC]

    lines = []
    lines.append("=" * 72)
    lines.append("LABEL INTEGRITY AUDIT")
    lines.append("=" * 72)
    lines.append(f"Files audited: {len(df)}")
    lines.append(
        f"Baseline_p95 across files: median={df['baseline_p95'].median():.1f}, "
        f"p10={df['baseline_p95'].quantile(0.1):.1f}, "
        f"p90={df['baseline_p95'].quantile(0.9):.1f}"
    )
    lines.append("")
    lines.append(
        f"GT-corruption threshold: at-baseline count >= {GT_MIN_COUNT} AND frac >= {GT_MIN_FRAC:.0%}"
    )
    lines.append(f"DA-corruption threshold: at-baseline frac >= {DA_MIN_FRAC:.0%}")
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"GT CLOT-ON-BASELINE — {len(gt_clot_bad)} files flagged")
    lines.append("-" * 72)
    for _, r in gt_clot_bad.sort_values("gt_clot_at_baseline", ascending=False).iterrows():
        lines.append(
            f"  {r['file']:>32s}  [{r['source_dir']:>13s}]  "
            f"gt_clot={r['n_gt_clot']:>6d}  "
            f"at_bl={r['gt_clot_at_baseline']:>6d} ({r['gt_clot_at_baseline_frac']:.1%})"
        )
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"GT WALL-ON-BASELINE — {len(gt_wall_bad)} files flagged")
    lines.append("-" * 72)
    for _, r in gt_wall_bad.sort_values("gt_wall_at_baseline", ascending=False).iterrows():
        lines.append(
            f"  {r['file']:>32s}  [{r['source_dir']:>13s}]  "
            f"gt_wall={r['n_gt_wall']:>6d}  "
            f"at_bl={r['gt_wall_at_baseline']:>6d} ({r['gt_wall_at_baseline_frac']:.1%})"
        )
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"DA CLOT-ON-BASELINE — {len(da_clot_bad)} files flagged")
    lines.append("-" * 72)
    for _, r in da_clot_bad.sort_values("da_clot_at_baseline_frac", ascending=False).iterrows():
        lines.append(
            f"  {r['file']:>32s}  [{r['source_dir']:>13s}]  "
            f"da_clot={r['n_da_clot']:>6d}  "
            f"at_bl={r['da_clot_at_baseline']:>6d} ({r['da_clot_at_baseline_frac']:.1%})"
        )
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"DA WALL-ON-BASELINE — {len(da_wall_bad)} files flagged")
    lines.append("-" * 72)
    for _, r in da_wall_bad.sort_values("da_wall_at_baseline_frac", ascending=False).iterrows():
        lines.append(
            f"  {r['file']:>32s}  [{r['source_dir']:>13s}]  "
            f"da_wall={r['n_da_wall']:>6d}  "
            f"at_bl={r['da_wall_at_baseline']:>6d} ({r['da_wall_at_baseline_frac']:.1%})"
        )
    lines.append("")

    txt = "\n".join(lines)
    TXT_OUT.write_text(txt, encoding="utf-8")
    print(f"Wrote {TXT_OUT}")
    print()
    print(txt)


if __name__ == "__main__":
    main()
