"""
Build slide-style visuals to show clot/wall similarity and DA disagreement.

Outputs under:
analysis_data_drift/da_gt_disagreement_review/slide_style_similarity/

1) Per-file slide-style plots (top mismatch files):
   top_files/<file>_slide_style.png

2) All-segment combined views:
   segment_similarity_scatter.png
   segment_similarity_hexbin.png

Design intent:
- Similar visual language to the shared slide (black context + red/blue classes).
- Explicitly mark DA-vs-GT disagreement points so similarity is easy to inspect.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import plotly.graph_objects as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SOURCE_DIR = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\19August2026\processedResults\testing"
)
BASE_DIR = Path("analysis_data_drift") / "da_gt_disagreement_review"
SUMMARY_CSV = BASE_DIR / "da_gt_file_summary.csv"
SEGMENTS_CSV = BASE_DIR / "da_gt_error_segments.csv"
OUT_DIR = BASE_DIR / "slide_style_similarity"

HTML_MAX_CONTEXT_POINTS = 12000
HTML_MAX_CLASS_POINTS = 16000
HTML_MAX_CLOUD_POINTS_PER_KIND = 18000


def _sanitize_filename(name: str, max_len: int = 100) -> str:
    s = str(name).strip()
    # Remove Windows-invalid filename chars and control chars.
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip(" ._")
    if not s:
        s = "unnamed"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _path_for_save(path: Path) -> str:
    """Return a platform-compatible save path string.

    On Windows, prepend the long-path prefix to avoid occasional path handling
    issues in downstream libraries.
    """
    p = path.resolve()
    s = str(p)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def _savefig_robust(fig, out_path: Path, dpi: int = 200) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(_path_for_save(out_path), dpi=dpi, bbox_inches="tight")
        return out_path
    except OSError:
        # Fallback to a guaranteed-safe short filename in the same directory.
        h = hashlib.sha1(str(out_path).encode("utf-8", errors="ignore")).hexdigest()[:8]
        safe_stem = _sanitize_filename(out_path.stem, max_len=60)
        fallback = out_path.parent / f"{safe_stem}_{h}{out_path.suffix}"
        fig.savefig(_path_for_save(fallback), dpi=dpi, bbox_inches="tight")
        print(f"  WARNING: failed to save original filename; used fallback: {fallback.name}")
        return fallback


def _write_html_robust(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_html(_path_for_save(out_path), include_plotlyjs="cdn")
        return out_path
    except OSError:
        h = hashlib.sha1(str(out_path).encode("utf-8", errors="ignore")).hexdigest()[:8]
        safe_stem = _sanitize_filename(out_path.stem, max_len=60)
        fallback = out_path.parent / f"{safe_stem}_{h}{out_path.suffix}"
        fig.write_html(_path_for_save(fallback), include_plotlyjs="cdn")
        print(f"  WARNING: failed to save original HTML filename; used fallback: {fallback.name}")
        return fallback


def _downsample_idx(mask: np.ndarray, max_points: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= max_points:
        return idx
    keep = np.linspace(0, len(idx) - 1, max_points, dtype=int)
    return idx[keep]


def choose_top_files(summary_csv: Path, top_n: int) -> list[str]:
    df = pd.read_csv(summary_csv)
    df = df.sort_values(["n_mismatch_samples", "mismatch_frac_in_clot_wall"], ascending=False)
    return df.head(top_n)["file"].astype(str).tolist()


def plot_file_slide_style(file_stem: str, source_dir: Path, out_path: Path) -> dict:
    fp = source_dir / f"{file_stem}_labeled_segment.parquet"
    if not fp.exists():
        return {"file": file_stem, "ok": False, "reason": "missing parquet"}

    df = pd.read_parquet(fp, columns=["timeInMS", "magRLoadAdjusted", "label", "da_label"])
    t = df["timeInMS"].to_numpy(dtype=np.float64) / 1000.0
    r = df["magRLoadAdjusted"].to_numpy(dtype=np.float32)
    gt = df["label"].to_numpy(dtype=np.int16)
    da = df["da_label"].to_numpy(dtype=np.int16)

    cw_gt = np.isin(gt, [1, 2])
    cw_da = np.isin(da, [1, 2])
    cw_both = cw_gt & cw_da
    mismatch = cw_both & (gt != da)

    non_cw_gt = ~np.isin(gt, [1, 2])
    gt_clot = cw_gt & (gt == 1)
    gt_wall = cw_gt & (gt == 2)

    err_gt_clot_da_wall = mismatch & (gt == 1) & (da == 2)
    err_gt_wall_da_clot = mismatch & (gt == 2) & (da == 1)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Render all non-clot/wall regions as a skinny black line only.
    # This keeps blood / unknown context visible without making it look like a
    # separate dense blood point cloud.
    r_non_cw = np.where(non_cw_gt, r, np.nan)
    ax.plot(t, r_non_cw, color="black", linewidth=0.8, alpha=0.55, label="non-clot/wall context")

    # Add a faint full-run trace underneath so temporal continuity remains easy to follow.
    ax.plot(t, r, color="black", linewidth=0.5, alpha=0.10)

    # GT clot/wall overlays
    ax.scatter(t[gt_clot], r[gt_clot], c="#dc2626", s=8, alpha=0.70, label="GT clot")
    ax.scatter(t[gt_wall], r[gt_wall], c="#2563eb", s=8, alpha=0.70, label="GT wall")

    # Disagreement markers
    ax.scatter(
        t[err_gt_clot_da_wall],
        r[err_gt_clot_da_wall],
        facecolors="none",
        edgecolors="#f59e0b",
        s=28,
        linewidths=0.8,
        label="DA error: called wall, GT clot",
    )
    ax.scatter(
        t[err_gt_wall_da_clot],
        r[err_gt_wall_da_clot],
        facecolors="none",
        edgecolors="#10b981",
        s=28,
        linewidths=0.8,
        label="DA error: called clot, GT wall",
    )

    n_cw = int(cw_both.sum())
    n_mis = int(mismatch.sum())
    mis_frac = (n_mis / n_cw) if n_cw > 0 else np.nan

    ax.set_title(
        f"{file_stem} - Slide Style View (GT clot/wall over full signal)\n"
        f"DA disagreement in clot/wall only: {n_mis:,}/{n_cw:,} ({mis_frac:.1%})",
        fontsize=14,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Resistance (ohms)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    _savefig_robust(fig, out_path, dpi=200)
    plt.close(fig)

    html_path = out_path.with_suffix(".html")
    plot_file_slide_style_html(
        file_stem=file_stem,
        t=t,
        r=r,
        non_cw_gt=non_cw_gt,
        gt_clot=gt_clot,
        gt_wall=gt_wall,
        err_gt_clot_da_wall=err_gt_clot_da_wall,
        err_gt_wall_da_clot=err_gt_wall_da_clot,
        n_mis=n_mis,
        n_cw=n_cw,
        mis_frac=mis_frac,
        out_path=html_path,
    )

    return {
        "file": file_stem,
        "ok": True,
        "n_cw_samples": n_cw,
        "n_mismatch_samples": n_mis,
        "mismatch_frac": mis_frac,
        "n_err_gt_clot_da_wall": int(err_gt_clot_da_wall.sum()),
        "n_err_gt_wall_da_clot": int(err_gt_wall_da_clot.sum()),
    }


def plot_file_slide_style_html(
    file_stem: str,
    t: np.ndarray,
    r: np.ndarray,
    non_cw_gt: np.ndarray,
    gt_clot: np.ndarray,
    gt_wall: np.ndarray,
    err_gt_clot_da_wall: np.ndarray,
    err_gt_wall_da_clot: np.ndarray,
    n_mis: int,
    n_cw: int,
    mis_frac: float,
    out_path: Path,
) -> None:
    fig = go.Figure()

    context_idx = _downsample_idx(non_cw_gt, HTML_MAX_CONTEXT_POINTS)
    full_idx = _downsample_idx(np.ones(len(t), dtype=bool), HTML_MAX_CONTEXT_POINTS)
    clot_idx = _downsample_idx(gt_clot, HTML_MAX_CLASS_POINTS)
    wall_idx = _downsample_idx(gt_wall, HTML_MAX_CLASS_POINTS)
    err_clot_idx = np.flatnonzero(err_gt_clot_da_wall)
    err_wall_idx = np.flatnonzero(err_gt_wall_da_clot)

    fig.add_trace(
        go.Scattergl(
            x=t[context_idx],
            y=r[context_idx],
            mode="lines",
            name="non-clot/wall context",
            line=dict(color="rgba(0,0,0,0.55)", width=1),
            hovertemplate="t=%{x:.2f}s<br>R=%{y:.1f} ohms<extra>context</extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=t[full_idx],
            y=r[full_idx],
            mode="lines",
            name="full trace",
            line=dict(color="rgba(0,0,0,0.12)", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=t[clot_idx],
            y=r[clot_idx],
            mode="markers",
            name="GT clot (downsampled)",
            marker=dict(color="#dc2626", size=4, opacity=0.65),
            hovertemplate="t=%{x:.2f}s<br>R=%{y:.1f} ohms<extra>GT clot</extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=t[wall_idx],
            y=r[wall_idx],
            mode="markers",
            name="GT wall (downsampled)",
            marker=dict(color="#2563eb", size=4, opacity=0.65),
            hovertemplate="t=%{x:.2f}s<br>R=%{y:.1f} ohms<extra>GT wall</extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=t[err_clot_idx],
            y=r[err_clot_idx],
            mode="markers",
            name="DA error: called wall, GT clot",
            marker=dict(color="rgba(0,0,0,0)", size=7, line=dict(color="#f59e0b", width=1)),
            hovertemplate="t=%{x:.2f}s<br>R=%{y:.1f} ohms<extra>DA wall, GT clot</extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=t[err_wall_idx],
            y=r[err_wall_idx],
            mode="markers",
            name="DA error: called clot, GT wall",
            marker=dict(color="rgba(0,0,0,0)", size=7, line=dict(color="#10b981", width=1)),
            hovertemplate="t=%{x:.2f}s<br>R=%{y:.1f} ohms<extra>DA clot, GT wall</extra>",
        )
    )

    fig.update_layout(
        title=(
            f"{file_stem} - Slide Style View (GT clot/wall over full signal)<br>"
            f"DA disagreement in clot/wall only: {n_mis:,}/{n_cw:,} ({mis_frac:.1%})"
        ),
        xaxis_title="Time (seconds)",
        yaxis_title="Resistance (ohms)",
        template="plotly_white",
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
        hovermode="closest",
        annotations=[
            dict(
                x=0.0,
                y=1.10,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                text=(
                    f"Interactive view is downsampled for speed: context <= {HTML_MAX_CONTEXT_POINTS:,} pts, "
                    f"GT clot/wall <= {HTML_MAX_CLASS_POINTS:,} pts each; disagreement markers are full resolution."
                ),
                font=dict(size=11, color="#444"),
            )
        ],
    )

    _write_html_robust(fig, out_path)


def _resample(x: np.ndarray, n: int = 140) -> np.ndarray:
    if len(x) < 2:
        return np.zeros(n, dtype=np.float32)
    xp = np.linspace(0, 1, len(x), dtype=np.float64)
    xnew = np.linspace(0, 1, n, dtype=np.float64)
    return np.interp(xnew, xp, x.astype(np.float64)).astype(np.float32)


def build_segment_point_cloud(source_dir: Path, segments_df: pd.DataFrame, max_segments_per_kind: int = 220):
    points = []
    for kind in ["GT_wall_DA_clot", "GT_clot_DA_wall"]:
        sub = segments_df[segments_df["kind"] == kind].copy()
        sub = sub.sort_values("duration_s", ascending=False).head(max_segments_per_kind)

        for _, row in sub.iterrows():
            file_stem = str(row["file"])
            fp = source_dir / f"{file_stem}_labeled_segment.parquet"
            if not fp.exists():
                continue

            arr = pd.read_parquet(fp, columns=["magRLoadAdjusted"])["magRLoadAdjusted"].to_numpy(dtype=np.float32)
            s = max(0, int(row["start_idx"]))
            e = min(len(arr) - 1, int(row["end_idx"]))
            if e <= s + 3:
                continue

            seg = arr[s : e + 1]
            # time-normalized representation keeps shape while allowing overlay.
            seg_r = _resample(seg, n=140)
            x = np.linspace(0.0, 100.0, len(seg_r), dtype=np.float32)

            for xi, yi in zip(x, seg_r):
                points.append((kind, float(xi), float(yi)))

    cloud = pd.DataFrame(points, columns=["kind", "x_pct", "ohms"])
    return cloud


def plot_segment_similarity_scatter(cloud: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))

    m_a = cloud["kind"] == "GT_wall_DA_clot"
    m_b = cloud["kind"] == "GT_clot_DA_wall"

    ax.scatter(cloud.loc[m_a, "x_pct"], cloud.loc[m_a, "ohms"], s=2.0, alpha=0.08, c="#2563eb", label="GT wall / DA clot")
    ax.scatter(cloud.loc[m_b, "x_pct"], cloud.loc[m_b, "ohms"], s=2.0, alpha=0.08, c="#dc2626", label="GT clot / DA wall")

    # Median trend per type
    for kind, color in [("GT_wall_DA_clot", "#1d4ed8"), ("GT_clot_DA_wall", "#b91c1c")]:
        sub = cloud[cloud["kind"] == kind]
        med = sub.groupby("x_pct", as_index=False)["ohms"].median()
        ax.plot(med["x_pct"], med["ohms"], color=color, linewidth=2.2)

    ax.set_title("All Error Segments Overlaid in Impedance Space (time-normalized)")
    ax.set_xlabel("Normalized segment time (%)")
    ax.set_ylabel("magRLoadAdjusted (ohms)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    plt.tight_layout()
    _savefig_robust(fig, out_path, dpi=200)
    plt.close(fig)

    plot_segment_similarity_scatter_html(cloud, out_path.with_suffix(".html"))


def plot_segment_similarity_scatter_html(cloud: pd.DataFrame, out_path: Path) -> None:
    fig = go.Figure()

    sub_a = cloud[cloud["kind"] == "GT_wall_DA_clot"]
    sub_b = cloud[cloud["kind"] == "GT_clot_DA_wall"]

    if len(sub_a) > HTML_MAX_CLOUD_POINTS_PER_KIND:
        sub_a = sub_a.iloc[np.linspace(0, len(sub_a) - 1, HTML_MAX_CLOUD_POINTS_PER_KIND, dtype=int)]
    if len(sub_b) > HTML_MAX_CLOUD_POINTS_PER_KIND:
        sub_b = sub_b.iloc[np.linspace(0, len(sub_b) - 1, HTML_MAX_CLOUD_POINTS_PER_KIND, dtype=int)]

    fig.add_trace(
        go.Scattergl(
            x=sub_a["x_pct"],
            y=sub_a["ohms"],
            mode="markers",
            name="GT wall / DA clot",
            marker=dict(color="rgba(37,99,235,0.10)", size=3),
            hovertemplate="x=%{x:.1f}%<br>R=%{y:.1f} ohms<extra>GT wall / DA clot</extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=sub_b["x_pct"],
            y=sub_b["ohms"],
            mode="markers",
            name="GT clot / DA wall",
            marker=dict(color="rgba(220,38,38,0.10)", size=3),
            hovertemplate="x=%{x:.1f}%<br>R=%{y:.1f} ohms<extra>GT clot / DA wall</extra>",
        )
    )

    for kind, color, name in [
        ("GT_wall_DA_clot", "#1d4ed8", "median GT wall / DA clot"),
        ("GT_clot_DA_wall", "#b91c1c", "median GT clot / DA wall"),
    ]:
        sub = cloud[cloud["kind"] == kind]
        med = sub.groupby("x_pct", as_index=False)["ohms"].median()
        fig.add_trace(
            go.Scatter(
                x=med["x_pct"],
                y=med["ohms"],
                mode="lines",
                name=name,
                line=dict(color=color, width=3),
                hovertemplate="x=%{x:.1f}%<br>median=%{y:.1f} ohms<extra></extra>",
            )
        )

    fig.update_layout(
        title="All Error Segments Overlaid in Impedance Space (time-normalized)",
        xaxis_title="Normalized segment time (%)",
        yaxis_title="magRLoadAdjusted (ohms)",
        template="plotly_white",
        hovermode="closest",
        annotations=[
            dict(
                x=0.0,
                y=1.08,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                text=f"Interactive cloud downsampled to <= {HTML_MAX_CLOUD_POINTS_PER_KIND:,} points per class for responsiveness.",
                font=dict(size=11, color="#444"),
            )
        ],
    )

    _write_html_robust(fig, out_path)


def plot_segment_similarity_hexbin(cloud: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    sub_a = cloud[cloud["kind"] == "GT_wall_DA_clot"]
    sub_b = cloud[cloud["kind"] == "GT_clot_DA_wall"]

    hb1 = ax1.hexbin(sub_a["x_pct"], sub_a["ohms"], gridsize=65, bins="log", cmap="Blues")
    hb2 = ax2.hexbin(sub_b["x_pct"], sub_b["ohms"], gridsize=65, bins="log", cmap="Reds")

    ax1.set_title(f"GT wall / DA clot\npoints={len(sub_a):,}")
    ax2.set_title(f"GT clot / DA wall\npoints={len(sub_b):,}")

    for ax in (ax1, ax2):
        ax.set_xlabel("Normalized segment time (%)")
        ax.grid(True, alpha=0.15)
    ax1.set_ylabel("magRLoadAdjusted (ohms)")

    cb1 = fig.colorbar(hb1, ax=ax1)
    cb1.set_label("log10 bin count")
    cb2 = fig.colorbar(hb2, ax=ax2)
    cb2.set_label("log10 bin count")

    fig.suptitle("Segment Similarity Density in Impedance Space", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig_robust(fig, out_path, dpi=200)
    plt.close(fig)

    plot_segment_similarity_hexbin_html(cloud, out_path.with_suffix(".html"))


def plot_segment_similarity_hexbin_html(cloud: pd.DataFrame, out_path: Path) -> None:
    sub_a = cloud[cloud["kind"] == "GT_wall_DA_clot"]
    sub_b = cloud[cloud["kind"] == "GT_clot_DA_wall"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram2d(
            x=sub_a["x_pct"],
            y=sub_a["ohms"],
            colorscale="Blues",
            nbinsx=65,
            nbinsy=65,
            colorbar=dict(title="count"),
            visible=True,
            name="GT wall / DA clot",
            hovertemplate="x=%{x:.1f}%<br>R=%{y:.1f} ohms<br>count=%{z}<extra>GT wall / DA clot</extra>",
        )
    )
    fig.add_trace(
        go.Histogram2d(
            x=sub_b["x_pct"],
            y=sub_b["ohms"],
            colorscale="Reds",
            nbinsx=65,
            nbinsy=65,
            colorbar=dict(title="count"),
            visible=False,
            name="GT clot / DA wall",
            hovertemplate="x=%{x:.1f}%<br>R=%{y:.1f} ohms<br>count=%{z}<extra>GT clot / DA wall</extra>",
        )
    )

    fig.update_layout(
        title="Segment Similarity Density in Impedance Space",
        xaxis_title="Normalized segment time (%)",
        yaxis_title="magRLoadAdjusted (ohms)",
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(label="GT wall / DA clot", method="update", args=[{"visible": [True, False]}]),
                    dict(label="GT clot / DA wall", method="update", args=[{"visible": [False, True]}]),
                ],
                x=0.0,
                y=1.12,
            )
        ],
    )

    _write_html_robust(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--top-files", type=int, default=24)
    args = parser.parse_args()

    if not SUMMARY_CSV.exists() or not SEGMENTS_CSV.exists():
        raise FileNotFoundError(
            "Expected da_gt_file_summary.csv and da_gt_error_segments.csv under analysis_data_drift/da_gt_disagreement_review"
        )

    top_files = choose_top_files(SUMMARY_CSV, top_n=args.top_files)

    out_files_dir = OUT_DIR / "top_files"
    out_files_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for i, stem in enumerate(top_files, start=1):
        print(f"[{i:2d}/{len(top_files)}] {stem}")
        safe_stem = _sanitize_filename(stem)
        s = plot_file_slide_style(
            file_stem=stem,
            source_dir=args.source_dir,
            out_path=out_files_dir / f"{safe_stem}_slide_style.png",
        )
        stats.append(s)

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT_DIR / "top_file_slide_style_stats.csv", index=False)

    seg = pd.read_csv(SEGMENTS_CSV)
    seg = seg[seg["kind"].isin(["GT_wall_DA_clot", "GT_clot_DA_wall"])].copy()

    cloud = build_segment_point_cloud(args.source_dir, seg, max_segments_per_kind=220)
    cloud.to_csv(OUT_DIR / "segment_similarity_cloud_points.csv", index=False)

    plot_segment_similarity_scatter(cloud, OUT_DIR / "segment_similarity_scatter.png")
    plot_segment_similarity_hexbin(cloud, OUT_DIR / "segment_similarity_hexbin.png")

    print("\nSaved outputs:")
    print(f"  {OUT_DIR}")
    print(f"  per-file plots: {out_files_dir}")
    print(f"  segment scatter: {OUT_DIR / 'segment_similarity_scatter.png'}")
    print(f"  segment hexbin : {OUT_DIR / 'segment_similarity_hexbin.png'}")


if __name__ == "__main__":
    main()
