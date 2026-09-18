from __future__ import annotations

"""
AllEvents internal helper variant for slide-style clot/wall similarity views.

This variant consumes dataframes whose label column may come from the
LabelingWithDuration_V8 all-events truth resolution logic.

It does not re-resolve event_type_1/2/3 itself; it visualizes the resulting
label and da_label columns at full output resolution.
"""

import hashlib
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HTML_MAX_CONTEXT_POINTS = 12000
HTML_MAX_CLASS_POINTS = 16000


def _path_for_save(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return "\\\\?\\" + text
    return text


def _fallback_path(out_path: Path) -> Path:
    digest = hashlib.sha1(str(out_path).encode("utf-8", errors="ignore")).hexdigest()[:8]
    safe_stem = out_path.stem.replace(" ", "_")[:60] or "slide_style"
    return out_path.parent / f"{safe_stem}_{digest}{out_path.suffix}"


def _savefig_robust(fig, out_path: Path, dpi: int = 200) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(_path_for_save(out_path), dpi=dpi, bbox_inches="tight")
        return out_path
    except OSError:
        fallback = _fallback_path(out_path)
        fig.savefig(_path_for_save(fallback), dpi=dpi, bbox_inches="tight")
        print(f"  WARNING: failed to save original filename; used fallback: {fallback.name}")
        return fallback


def _write_html_robust(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_html(_path_for_save(out_path), include_plotlyjs="cdn")
        return out_path
    except OSError:
        fallback = _fallback_path(out_path)
        fig.write_html(_path_for_save(fallback), include_plotlyjs="cdn")
        print(f"  WARNING: failed to save original HTML filename; used fallback: {fallback.name}")
        return fallback


def _downsample_idx(mask: np.ndarray, max_points: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= max_points:
        return idx
    keep = np.linspace(0, len(idx) - 1, max_points, dtype=int)
    return idx[keep]


def _plot_file_slide_style_html(
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
    )

    _write_html_robust(fig, out_path)


def plot_file_slide_style_from_df(file_stem: str, df: pd.DataFrame, out_path: Path) -> dict[str, object]:
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

    r_non_cw = np.where(non_cw_gt, r, np.nan)
    ax.plot(t, r_non_cw, color="black", linewidth=0.8, alpha=0.55, label="non-clot/wall context")
    ax.plot(t, r, color="black", linewidth=0.5, alpha=0.10)
    ax.scatter(t[gt_clot], r[gt_clot], c="#dc2626", s=8, alpha=0.70, label="GT clot")
    ax.scatter(t[gt_wall], r[gt_wall], c="#2563eb", s=8, alpha=0.70, label="GT wall")
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

    _plot_file_slide_style_html(
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
        out_path=out_path.with_suffix(".html"),
    )

    return {
        "file": file_stem,
        "ok": True,
        "n_cw_samples": n_cw,
        "n_mismatch_samples": n_mis,
        "mismatch_frac": mis_frac,
    }