"""
Create DA-vs-GT clot/wall disagreement visuals over raw testing data.

User request coverage:
1) For each file: plot magRLoadAdjusted and highlight where GT != DA
   (clot/wall only; blood ignored).
2) Compile all error segments into an easy-to-review set.

Outputs under analysis_data_drift/da_gt_disagreement_review/:
- per_file_plots/<study>_da_gt_disagreement.png
- segment_clips/<study>_segXXXX_<type>_<start>-<end>s.png
- da_gt_error_segments.csv
- da_gt_file_summary.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SOURCE_DIR = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\19August2026\processedResults\testing"
)
DEFAULT_OUT_DIR = Path("analysis_data_drift") / "da_gt_disagreement_review"


@dataclass
class Segment:
    file_stem: str
    seg_id: int
    start_idx: int
    end_idx: int
    start_s: float
    end_s: float
    duration_s: float
    n_samples: int
    kind: str
    gt_clot_frac: float
    da_clot_frac: float


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    if mask.size == 0:
        return segments

    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start, i - 1))
            start = None

    if start is not None:
        segments.append((start, mask.size - 1))

    return segments


def classify_segment(gt_slice: np.ndarray, da_slice: np.ndarray) -> tuple[str, float, float]:
    gt_is_clot = gt_slice == 1
    da_is_clot = da_slice == 1

    gt_clot_frac = float(gt_is_clot.mean())
    da_clot_frac = float(da_is_clot.mean())

    gt_all_clot = bool(np.all(gt_slice == 1))
    gt_all_wall = bool(np.all(gt_slice == 2))
    da_all_clot = bool(np.all(da_slice == 1))
    da_all_wall = bool(np.all(da_slice == 2))

    if gt_all_clot and da_all_wall:
        return "GT_clot_DA_wall", gt_clot_frac, da_clot_frac
    if gt_all_wall and da_all_clot:
        return "GT_wall_DA_clot", gt_clot_frac, da_clot_frac
    return "mixed_clot_wall_disagreement", gt_clot_frac, da_clot_frac


def plot_per_file(
    times_s: np.ndarray,
    r: np.ndarray,
    gt: np.ndarray,
    da: np.ndarray,
    err_segments: list[Segment],
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(times_s, r, color="#1f2937", linewidth=0.8, alpha=0.9, label="magRLoadAdjusted")

    added_red = False
    added_blue = False
    added_gray = False

    for seg in err_segments:
        if seg.kind == "GT_clot_DA_wall":
            color = "#dc2626"
            label = "DA error: called wall, GT clot"
            if added_red:
                label = None
            added_red = True
        elif seg.kind == "GT_wall_DA_clot":
            color = "#2563eb"
            label = "DA error: called clot, GT wall"
            if added_blue:
                label = None
            added_blue = True
        else:
            color = "#6b7280"
            label = "DA error: mixed clot/wall disagreement"
            if added_gray:
                label = None
            added_gray = True

        ax1.axvspan(seg.start_s, seg.end_s, color=color, alpha=0.22, label=label)

    ax1.set_title(f"{out_path.stem.replace('_da_gt_disagreement', '')} - DA vs GT disagreement (clot/wall only)")
    ax1.set_ylabel("magRLoadAdjusted")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)

    gt_cw = np.where(np.isin(gt, [1, 2]), gt, np.nan)
    da_cw = np.where(np.isin(da, [1, 2]), da, np.nan)

    ax2.plot(times_s, gt_cw, color="#16a34a", linewidth=1.0, drawstyle="steps-mid", label="GT")
    ax2.plot(times_s, da_cw, color="#f59e0b", linewidth=1.0, drawstyle="steps-mid", alpha=0.9, label="DA")
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(["clot", "wall"])
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("label")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_segment_clip(
    times_s: np.ndarray,
    r: np.ndarray,
    gt: np.ndarray,
    da: np.ndarray,
    seg: Segment,
    out_path: Path,
    context_s: float,
) -> None:
    lo_t = seg.start_s - context_s
    hi_t = seg.end_s + context_s
    clip_mask = (times_s >= lo_t) & (times_s <= hi_t)

    if clip_mask.sum() < 3:
        return

    t = times_s[clip_mask]
    rr = r[clip_mask]
    gg = gt[clip_mask]
    dd = da[clip_mask]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(t, rr, color="#111827", linewidth=1.0)

    color = "#dc2626" if seg.kind == "GT_clot_DA_wall" else "#2563eb" if seg.kind == "GT_wall_DA_clot" else "#6b7280"
    ax1.axvspan(seg.start_s, seg.end_s, color=color, alpha=0.25)
    ax1.set_ylabel("magR")
    ax1.set_title(
        f"{seg.file_stem} seg {seg.seg_id:04d} | {seg.kind} | "
        f"{seg.start_s:.1f}-{seg.end_s:.1f}s ({seg.duration_s:.2f}s)"
    )
    ax1.grid(True, alpha=0.25)

    gg_cw = np.where(np.isin(gg, [1, 2]), gg, np.nan)
    dd_cw = np.where(np.isin(dd, [1, 2]), dd, np.nan)
    ax2.plot(t, gg_cw, color="#16a34a", linewidth=1.0, drawstyle="steps-mid", label="GT")
    ax2.plot(t, dd_cw, color="#f59e0b", linewidth=1.0, drawstyle="steps-mid", label="DA")
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(["clot", "wall"])
    ax2.set_xlabel("time (s)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def process_file(fp: Path, per_file_dir: Path, clips_dir: Path, context_s: float) -> tuple[list[Segment], dict]:
    df = pd.read_parquet(fp)
    required = ["timeInMS", "magRLoadAdjusted", "label", "da_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {fp.name}")

    t = df["timeInMS"].to_numpy(dtype=np.float64) / 1000.0
    r = df["magRLoadAdjusted"].to_numpy(dtype=np.float32)
    gt = df["label"].to_numpy(dtype=np.int16)
    da = df["da_label"].to_numpy(dtype=np.int16)

    cw_mask = np.isin(gt, [1, 2]) & np.isin(da, [1, 2])
    mismatch = cw_mask & (gt != da)

    segments_idx = contiguous_segments(mismatch)

    segments: list[Segment] = []
    stem = fp.stem.replace("_labeled_segment", "")

    for i, (s, e) in enumerate(segments_idx, start=1):
        gt_slice = gt[s : e + 1]
        da_slice = da[s : e + 1]
        kind, gt_clot_frac, da_clot_frac = classify_segment(gt_slice, da_slice)
        seg = Segment(
            file_stem=stem,
            seg_id=i,
            start_idx=s,
            end_idx=e,
            start_s=float(t[s]),
            end_s=float(t[e]),
            duration_s=float(t[e] - t[s]),
            n_samples=int(e - s + 1),
            kind=kind,
            gt_clot_frac=gt_clot_frac,
            da_clot_frac=da_clot_frac,
        )
        segments.append(seg)

    plot_per_file(
        times_s=t,
        r=r,
        gt=gt,
        da=da,
        err_segments=segments,
        out_path=per_file_dir / f"{stem}_da_gt_disagreement.png",
    )

    for seg in segments:
        clip_name = (
            f"{seg.file_stem}_seg{seg.seg_id:04d}_{seg.kind}_"
            f"{seg.start_s:.1f}-{seg.end_s:.1f}s.png"
        )
        plot_segment_clip(
            times_s=t,
            r=r,
            gt=gt,
            da=da,
            seg=seg,
            out_path=clips_dir / clip_name,
            context_s=context_s,
        )

    summary = {
        "file": stem,
        "n_rows": len(df),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "n_clot_wall_samples": int(cw_mask.sum()),
        "n_mismatch_samples": int(mismatch.sum()),
        "mismatch_frac_in_clot_wall": float(mismatch.sum() / cw_mask.sum()) if cw_mask.sum() > 0 else np.nan,
        "n_error_segments": len(segments),
    }

    return segments, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DA-vs-GT clot/wall disagreements and compile error segments.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Folder with *_labeled_segment.parquet files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output folder")
    parser.add_argument("--context-sec", type=float, default=2.0, help="Context around each error segment clip")
    args = parser.parse_args()

    source_dir = args.source_dir
    out_dir = args.output_dir
    per_file_dir = out_dir / "per_file_plots"
    clips_dir = out_dir / "segment_clips"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = sorted(source_dir.glob("*_labeled_segment.parquet"))
    if not files:
        raise FileNotFoundError(f"No *_labeled_segment.parquet files found in {source_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    all_segments: list[dict] = []
    all_summaries: list[dict] = []

    print(f"Found {len(files)} files in {source_dir}")

    for idx, fp in enumerate(files, start=1):
        print(f"[{idx:3d}/{len(files)}] Processing {fp.name} ...")
        segments, summary = process_file(fp, per_file_dir, clips_dir, args.context_sec)
        all_summaries.append(summary)

        for seg in segments:
            all_segments.append(
                {
                    "file": seg.file_stem,
                    "segment_id": seg.seg_id,
                    "kind": seg.kind,
                    "start_idx": seg.start_idx,
                    "end_idx": seg.end_idx,
                    "start_s": seg.start_s,
                    "end_s": seg.end_s,
                    "duration_s": seg.duration_s,
                    "n_samples": seg.n_samples,
                    "gt_clot_frac": seg.gt_clot_frac,
                    "da_clot_frac": seg.da_clot_frac,
                    "clip_file": (
                        f"segment_clips/{seg.file_stem}_seg{seg.seg_id:04d}_{seg.kind}_"
                        f"{seg.start_s:.1f}-{seg.end_s:.1f}s.png"
                    ),
                    "per_file_plot": f"per_file_plots/{seg.file_stem}_da_gt_disagreement.png",
                }
            )

    seg_df = pd.DataFrame(all_segments).sort_values(["file", "start_s"]).reset_index(drop=True) if all_segments else pd.DataFrame()
    sum_df = pd.DataFrame(all_summaries).sort_values("file").reset_index(drop=True)

    seg_csv = out_dir / "da_gt_error_segments.csv"
    sum_csv = out_dir / "da_gt_file_summary.csv"
    seg_df.to_csv(seg_csv, index=False)
    sum_df.to_csv(sum_csv, index=False)

    n_segments = 0 if seg_df.empty else len(seg_df)
    total_err_samples = int(sum_df["n_mismatch_samples"].sum()) if not sum_df.empty else 0
    print("\nDone.")
    print(f"Per-file plots: {per_file_dir}")
    print(f"Segment clips: {clips_dir}")
    print(f"Segment catalog: {seg_csv}")
    print(f"File summary: {sum_csv}")
    print(f"Total error segments: {n_segments:,}")
    print(f"Total clot/wall mismatch samples: {total_err_samples:,}")


if __name__ == "__main__":
    main()
