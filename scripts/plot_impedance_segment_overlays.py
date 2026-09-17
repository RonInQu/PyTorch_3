"""
Create impedance-space overlays and stacked views for DA-vs-GT error segments.

Categories:
- GT_wall_DA_clot  (GT wall, DA called clot)
- GT_clot_DA_wall  (GT clot, DA called wall)

Reads:
- analysis_data_drift/da_gt_disagreement_review/da_gt_error_segments.csv
- raw testing parquets (processedResults/testing)

Writes:
- analysis_data_drift/da_gt_disagreement_review/error_atlas/impedance_views/
    overlay_absolute_ohms.png
    overlay_delta_from_start.png
    stacked_heatmap_absolute_ohms.png
    stacked_heatmap_delta_from_start.png
    impedance_overlay_stats.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEG_CSV = Path("analysis_data_drift/da_gt_disagreement_review/da_gt_error_segments.csv")
SOURCE_DIR = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\19August2026\processedResults\testing"
)
OUT_DIR = Path("analysis_data_drift/da_gt_disagreement_review/error_atlas/impedance_views")

KIND_A = "GT_wall_DA_clot"
KIND_B = "GT_clot_DA_wall"

RESAMPLE_N = 240
MAX_OVERLAY_LINES = 180


def resample(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) == 1:
        return np.full(n, float(x[0]), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
    xnew = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.interp(xnew, xp, x.astype(np.float64)).astype(np.float32)


def read_segment_signal(file_stem: str, start_idx: int, end_idx: int) -> np.ndarray:
    fp = SOURCE_DIR / f"{file_stem}_labeled_segment.parquet"
    if not fp.exists():
        return np.zeros(0, dtype=np.float32)
    df = pd.read_parquet(fp, columns=["magRLoadAdjusted"])
    r = df["magRLoadAdjusted"].to_numpy(dtype=np.float32)
    s = max(0, int(start_idx))
    e = min(len(r) - 1, int(end_idx))
    if e < s:
        return np.zeros(0, dtype=np.float32)
    return r[s : e + 1]


def build_matrices(seg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_abs, B_abs, A_dlt, B_dlt = [], [], [], []

    for _, row in seg.iterrows():
        kind = str(row["kind"])
        if kind not in (KIND_A, KIND_B):
            continue
        x = read_segment_signal(str(row["file"]), int(row["start_idx"]), int(row["end_idx"]))
        if len(x) < 4 or not np.all(np.isfinite(x)):
            continue

        xa = resample(x, RESAMPLE_N)
        xd = xa - float(xa[0])

        if kind == KIND_A:
            A_abs.append(xa)
            A_dlt.append(xd)
        else:
            B_abs.append(xa)
            B_dlt.append(xd)

    A_abs = np.vstack(A_abs) if A_abs else np.zeros((0, RESAMPLE_N), dtype=np.float32)
    B_abs = np.vstack(B_abs) if B_abs else np.zeros((0, RESAMPLE_N), dtype=np.float32)
    A_dlt = np.vstack(A_dlt) if A_dlt else np.zeros((0, RESAMPLE_N), dtype=np.float32)
    B_dlt = np.vstack(B_dlt) if B_dlt else np.zeros((0, RESAMPLE_N), dtype=np.float32)
    return A_abs, B_abs, A_dlt, B_dlt


def downsample_rows(M: np.ndarray, max_rows: int) -> np.ndarray:
    if len(M) <= max_rows:
        return M
    step = int(np.ceil(len(M) / max_rows))
    return M[::step]


def plot_overlay_two_panel(A: np.ndarray, B: np.ndarray, title: str, ylab: str, out_path: Path) -> None:
    x = np.linspace(0.0, 1.0, RESAMPLE_N)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    def draw(ax, M, color, label):
        S = downsample_rows(M, MAX_OVERLAY_LINES)
        for r in S:
            ax.plot(x, r, color=color, alpha=0.06, linewidth=0.8)
        if len(M) > 0:
            med = np.median(M, axis=0)
            q25 = np.percentile(M, 25, axis=0)
            q75 = np.percentile(M, 75, axis=0)
            ax.plot(x, med, color=color, linewidth=2.2, label=f"median ({label})")
            ax.fill_between(x, q25, q75, color=color, alpha=0.20, label="IQR")

    draw(ax1, A, "#2563eb", "GT wall / DA clot")
    ax1.set_title(f"GT wall / DA clot (n={len(A)})")
    ax1.set_xlabel("normalized segment time")
    ax1.set_ylabel(ylab)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=8)

    draw(ax2, B, "#dc2626", "GT clot / DA wall")
    ax2.set_title(f"GT clot / DA wall (n={len(B)})")
    ax2.set_xlabel("normalized segment time")
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=8)

    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_two_panel(A: np.ndarray, B: np.ndarray, title: str, cbar: str, out_path: Path, robust: bool = True) -> None:
    # Sort by row median to reveal structured bands.
    if len(A) > 0:
        A = A[np.argsort(np.median(A, axis=1))]
    if len(B) > 0:
        B = B[np.argsort(np.median(B, axis=1))]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    if robust:
        all_vals = np.concatenate([A.ravel() if len(A) else np.array([]), B.ravel() if len(B) else np.array([])])
        if len(all_vals) > 0:
            vmin = float(np.percentile(all_vals, 2))
            vmax = float(np.percentile(all_vals, 98))
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin = vmax = None

    im1 = ax1.imshow(A, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_title(f"GT wall / DA clot (n={len(A)})")
    ax1.set_xlabel("normalized time bin")
    ax1.set_ylabel("segment index (sorted)")

    im2 = ax2.imshow(B, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_title(f"GT clot / DA wall (n={len(B)})")
    ax2.set_xlabel("normalized time bin")

    cb = fig.colorbar(im2, ax=[ax1, ax2], location="right", fraction=0.05, pad=0.03)
    cb.set_label(cbar)

    fig.suptitle(title, fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_stats(A_abs: np.ndarray, B_abs: np.ndarray, A_dlt: np.ndarray, B_dlt: np.ndarray, out_csv: Path) -> None:
    def summarize(name: str, M: np.ndarray) -> dict:
        if len(M) == 0:
            return {
                "category": name,
                "n_segments": 0,
                "median_start_ohm": np.nan,
                "median_end_ohm": np.nan,
                "median_delta_end_ohm": np.nan,
                "median_range_ohm": np.nan,
                "median_std_ohm": np.nan,
            }
        ranges = np.ptp(M, axis=1)
        stds = np.std(M, axis=1)
        return {
            "category": name,
            "n_segments": int(len(M)),
            "median_start_ohm": float(np.median(M[:, 0])),
            "median_end_ohm": float(np.median(M[:, -1])),
            "median_delta_end_ohm": float(np.median(M[:, -1] - M[:, 0])),
            "median_range_ohm": float(np.median(ranges)),
            "median_std_ohm": float(np.median(stds)),
        }

    rows = [
        summarize("GT_wall_DA_clot_abs", A_abs),
        summarize("GT_clot_DA_wall_abs", B_abs),
        summarize("GT_wall_DA_clot_delta", A_dlt),
        summarize("GT_clot_DA_wall_delta", B_dlt),
    ]
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    if not SEG_CSV.exists():
        raise FileNotFoundError(f"Missing segment CSV: {SEG_CSV}")
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing source directory: {SOURCE_DIR}")

    seg = pd.read_csv(SEG_CSV)
    seg = seg[seg["kind"].isin([KIND_A, KIND_B])].copy()

    A_abs, B_abs, A_dlt, B_dlt = build_matrices(seg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_overlay_two_panel(
        A_abs,
        B_abs,
        title="Error Segments Overlaid in Impedance Space (Absolute Ohms)",
        ylab="magRLoadAdjusted (ohms)",
        out_path=OUT_DIR / "overlay_absolute_ohms.png",
    )

    plot_overlay_two_panel(
        A_dlt,
        B_dlt,
        title="Error Segments Overlaid (Delta from Segment Start)",
        ylab="delta magR from segment start (ohms)",
        out_path=OUT_DIR / "overlay_delta_from_start.png",
    )

    plot_heatmap_two_panel(
        A_abs,
        B_abs,
        title="Stacked Error Segments (Absolute Ohms)",
        cbar="magRLoadAdjusted (ohms)",
        out_path=OUT_DIR / "stacked_heatmap_absolute_ohms.png",
        robust=True,
    )

    plot_heatmap_two_panel(
        A_dlt,
        B_dlt,
        title="Stacked Error Segments (Delta from Start)",
        cbar="delta ohms",
        out_path=OUT_DIR / "stacked_heatmap_delta_from_start.png",
        robust=True,
    )

    save_stats(A_abs, B_abs, A_dlt, B_dlt, OUT_DIR / "impedance_overlay_stats.csv")

    print(f"Saved impedance views to: {OUT_DIR}")
    print(f"A segments (GT wall / DA clot): {len(A_abs)}")
    print(f"B segments (GT clot / DA wall): {len(B_abs)}")


if __name__ == "__main__":
    main()
