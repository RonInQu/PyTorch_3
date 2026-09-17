"""
Build a compact error-atlas view from DA-vs-GT disagreement segments.

Inputs:
- analysis_data_drift/da_gt_disagreement_review/da_gt_error_segments.csv
- raw files under processedResults/testing

Outputs:
- analysis_data_drift/da_gt_disagreement_review/error_atlas/
    error_type_counts.png
    duration_hist_by_type.png
    shape_overlay_resampled_z.png
    shape_mean_band.png
    pca_projection_by_type.png
    segment_feature_table.csv
    feature_group_summary.csv
    logistic_feature_importance.csv
    error_math_summary.txt

Purpose:
- Visualize both error types together at once.
- Quantify which segment-level math features separate the two error types.
- Provide an evidence-based direction for ML improvements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SEGMENTS_CSV = Path("analysis_data_drift/da_gt_disagreement_review/da_gt_error_segments.csv")
DEFAULT_SOURCE_DIR = Path(
    r"C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\19August2026\processedResults\testing"
)
OUT_DIR = Path("analysis_data_drift/da_gt_disagreement_review/error_atlas")

KIND_A = "GT_wall_DA_clot"  # DA called clot, GT says wall
KIND_B = "GT_clot_DA_wall"  # DA called wall, GT says clot

RESAMPLE_N = 200


@dataclass
class SegmentSlice:
    kind: str
    duration_s: float
    n_samples: int
    signal: np.ndarray


def robust_z(x: np.ndarray) -> np.ndarray:
    med = float(np.median(x))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    scale = iqr if iqr > 1e-6 else (float(np.std(x)) + 1e-6)
    return (x - med) / scale


def resample_to_n(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) == 1:
        return np.full(n, float(x[0]), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, len(x))
    fp = x.astype(np.float32)
    xnew = np.linspace(0.0, 1.0, n)
    return np.interp(xnew, xp, fp).astype(np.float32)


def load_segments() -> pd.DataFrame:
    if not SEGMENTS_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {SEGMENTS_CSV}")
    seg = pd.read_csv(SEGMENTS_CSV)
    need = {"file", "kind", "start_idx", "end_idx", "start_s", "end_s", "duration_s", "n_samples"}
    miss = sorted(list(need - set(seg.columns)))
    if miss:
        raise ValueError(f"Missing columns in segments CSV: {miss}")
    return seg


def load_signal_slice(source_dir: Path, file_stem: str, start_idx: int, end_idx: int) -> np.ndarray:
    fp = source_dir / f"{file_stem}_labeled_segment.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing raw source file: {fp}")
    df = pd.read_parquet(fp, columns=["magRLoadAdjusted"])
    r = df["magRLoadAdjusted"].to_numpy(dtype=np.float32)
    s = max(0, int(start_idx))
    e = min(len(r) - 1, int(end_idx))
    if e < s:
        return np.zeros(0, dtype=np.float32)
    return r[s : e + 1]


def compute_segment_features(x: np.ndarray, duration_s: float, n_samples: int) -> dict:
    if len(x) < 2:
        return {
            "duration_s": duration_s,
            "n_samples": n_samples,
            "r_mean": np.nan,
            "r_std": np.nan,
            "r_range": np.nan,
            "r_iqr": np.nan,
            "r_p10": np.nan,
            "r_p90": np.nan,
            "slope_total": np.nan,
            "diff_abs_mean": np.nan,
            "diff_std": np.nan,
            "curv_abs_mean": np.nan,
            "z_peak": np.nan,
            "z_valley": np.nan,
        }

    d1 = np.diff(x)
    d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0], dtype=np.float32)
    z = robust_z(x)

    slope_total = float((x[-1] - x[0]) / max(duration_s, 1e-6))

    return {
        "duration_s": float(duration_s),
        "n_samples": int(n_samples),
        "r_mean": float(np.mean(x)),
        "r_std": float(np.std(x)),
        "r_range": float(np.ptp(x)),
        "r_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "r_p10": float(np.percentile(x, 10)),
        "r_p90": float(np.percentile(x, 90)),
        "slope_total": slope_total,
        "diff_abs_mean": float(np.mean(np.abs(d1))),
        "diff_std": float(np.std(d1)),
        "curv_abs_mean": float(np.mean(np.abs(d2))),
        "z_peak": float(np.max(z)),
        "z_valley": float(np.min(z)),
    }


def build_dataset(seg_df: pd.DataFrame, source_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    mat_a = []
    mat_b = []

    for _, r in seg_df.iterrows():
        kind = str(r["kind"])
        if kind not in (KIND_A, KIND_B):
            continue

        file_stem = str(r["file"])
        x = load_signal_slice(source_dir, file_stem, int(r["start_idx"]), int(r["end_idx"]))
        if len(x) < 4:
            continue

        feat = compute_segment_features(x, float(r["duration_s"]), int(r["n_samples"]))
        feat["file"] = file_stem
        feat["kind"] = kind
        feat["kind_bin"] = 0 if kind == KIND_A else 1
        rows.append(feat)

        z_res = resample_to_n(robust_z(x), RESAMPLE_N)
        if kind == KIND_A:
            mat_a.append(z_res)
        else:
            mat_b.append(z_res)

    feat_df = pd.DataFrame(rows)
    A = np.vstack(mat_a) if mat_a else np.zeros((0, RESAMPLE_N), dtype=np.float32)
    B = np.vstack(mat_b) if mat_b else np.zeros((0, RESAMPLE_N), dtype=np.float32)
    return feat_df, A, B


def save_counts_plot(seg_df: pd.DataFrame, out_dir: Path) -> None:
    counts = seg_df["kind"].value_counts().reindex([KIND_A, KIND_B]).fillna(0).astype(int)
    labels = ["GT wall / DA clot", "GT clot / DA wall"]
    vals = counts.values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=["#2563eb", "#dc2626"])
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, str(int(v)), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("segment count")
    ax.set_title("DA-vs-GT error segments by type")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "error_type_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_duration_hist(seg_df: pd.DataFrame, out_dir: Path) -> None:
    a = seg_df.loc[seg_df["kind"] == KIND_A, "duration_s"].to_numpy()
    b = seg_df.loc[seg_df["kind"] == KIND_B, "duration_s"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, min(360, max(np.nanmax(a) if len(a) else 1, np.nanmax(b) if len(b) else 1)), 50)
    ax.hist(a, bins=bins, alpha=0.55, color="#2563eb", label="GT wall / DA clot")
    ax.hist(b, bins=bins, alpha=0.55, color="#dc2626", label="GT clot / DA wall")
    ax.set_xlabel("segment duration (s)")
    ax.set_ylabel("count")
    ax.set_title("Error segment duration distribution")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "duration_hist_by_type.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_shape_overlay(A: np.ndarray, B: np.ndarray, out_dir: Path) -> None:
    x = np.linspace(0, 1, RESAMPLE_N)

    fig, ax = plt.subplots(figsize=(10, 5))

    max_lines = 120
    if len(A) > 0:
        step = max(1, len(A) // max_lines)
        for row in A[::step]:
            ax.plot(x, row, color="#2563eb", alpha=0.05, linewidth=0.8)
        ax.plot(x, A.mean(axis=0), color="#1d4ed8", linewidth=2.0, label="GT wall / DA clot mean")
    if len(B) > 0:
        step = max(1, len(B) // max_lines)
        for row in B[::step]:
            ax.plot(x, row, color="#dc2626", alpha=0.05, linewidth=0.8)
        ax.plot(x, B.mean(axis=0), color="#b91c1c", linewidth=2.0, label="GT clot / DA wall mean")

    ax.set_xlabel("normalized segment time")
    ax.set_ylabel("robust z-score of magR")
    ax.set_title("All error segment shapes overlaid (time-normalized)")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "shape_overlay_resampled_z.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_shape_band(A: np.ndarray, B: np.ndarray, out_dir: Path) -> None:
    x = np.linspace(0, 1, RESAMPLE_N)

    fig, ax = plt.subplots(figsize=(10, 5))

    if len(A) > 0:
        ma = A.mean(axis=0)
        sa = A.std(axis=0)
        ax.plot(x, ma, color="#1d4ed8", linewidth=2.0, label="GT wall / DA clot mean")
        ax.fill_between(x, ma - sa, ma + sa, color="#93c5fd", alpha=0.35)

    if len(B) > 0:
        mb = B.mean(axis=0)
        sb = B.std(axis=0)
        ax.plot(x, mb, color="#b91c1c", linewidth=2.0, label="GT clot / DA wall mean")
        ax.fill_between(x, mb - sb, mb + sb, color="#fca5a5", alpha=0.35)

    ax.set_xlabel("normalized segment time")
    ax.set_ylabel("robust z-score of magR")
    ax.set_title("Mean +/- 1 std shape by error type")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "shape_mean_band.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pca_plot(feat_df: pd.DataFrame, feat_cols: list[str], out_dir: Path) -> None:
    clean = feat_df.dropna(subset=feat_cols + ["kind_bin"]).copy()
    if len(clean) < 10:
        return

    X = clean[feat_cols].to_numpy(dtype=np.float64)
    y = clean["kind_bin"].to_numpy(dtype=int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(7, 6))
    m0 = y == 0
    m1 = y == 1
    ax.scatter(Z[m0, 0], Z[m0, 1], s=18, alpha=0.55, color="#2563eb", label="GT wall / DA clot")
    ax.scatter(Z[m1, 0], Z[m1, 1], s=18, alpha=0.55, color="#dc2626", label="GT clot / DA wall")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("Segment-feature PCA projection")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca_projection_by_type.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def logistic_math(feat_df: pd.DataFrame, feat_cols: list[str], out_dir: Path) -> dict:
    clean = feat_df.dropna(subset=feat_cols + ["kind_bin"]).copy()
    if len(clean) < 20:
        return {
            "ok": False,
            "msg": "Too few clean rows for logistic analysis",
        }

    X = clean[feat_cols].to_numpy(dtype=np.float64)
    y = clean["kind_bin"].to_numpy(dtype=int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    accs = []
    f1s = []

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        clf = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)
        clf.fit(Xtr, y[tr])

        p = clf.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], yhat))
        f1s.append(f1_score(y[te], yhat))

    scaler_full = StandardScaler()
    Xs = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)
    clf_full.fit(Xs, y)

    coef = clf_full.coef_[0]
    imp = pd.DataFrame({
        "feature": feat_cols,
        "coef": coef,
        "abs_coef": np.abs(coef),
        "direction": np.where(coef > 0, KIND_B, KIND_A),
    }).sort_values("abs_coef", ascending=False)
    imp.to_csv(out_dir / "logistic_feature_importance.csv", index=False)

    return {
        "ok": True,
        "n": int(len(clean)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "class_counts": dict(pd.Series(y).value_counts().sort_index()),
        "top_features": imp.head(8).to_dict(orient="records"),
    }


def group_summary(feat_df: pd.DataFrame, feat_cols: list[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    for kind, sub in feat_df.groupby("kind"):
        row = {"kind": kind, "n": int(len(sub))}
        for c in feat_cols:
            row[f"{c}_mean"] = float(np.nanmean(sub[c]))
            row[f"{c}_median"] = float(np.nanmedian(sub[c]))
            row[f"{c}_std"] = float(np.nanstd(sub[c]))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "feature_group_summary.csv", index=False)
    return out


def write_math_summary(seg_df: pd.DataFrame, feat_df: pd.DataFrame, log_res: dict, out_dir: Path) -> None:
    c = seg_df["kind"].value_counts().to_dict()
    n_a = int(c.get(KIND_A, 0))
    n_b = int(c.get(KIND_B, 0))

    lines = []
    lines.append("ERROR ATLAS - MATHEMATICAL SUMMARY")
    lines.append("=" * 44)
    lines.append(f"Segments total (two types): {n_a + n_b}")
    lines.append(f"  {KIND_A}: {n_a}")
    lines.append(f"  {KIND_B}: {n_b}")
    lines.append("")

    if log_res.get("ok", False):
        lines.append("Logistic separation of error type from segment-level math features")
        lines.append("(5-fold CV, balanced classes)")
        lines.append(f"  n samples used: {log_res['n']}")
        lines.append(f"  ROC-AUC: {log_res['auc_mean']:.3f} +/- {log_res['auc_std']:.3f}")
        lines.append(f"  Accuracy: {log_res['acc_mean']:.3f} +/- {log_res['acc_std']:.3f}")
        lines.append(f"  F1: {log_res['f1_mean']:.3f} +/- {log_res['f1_std']:.3f}")
        lines.append("")
        lines.append("Top discriminative features (absolute logistic coefficient)")
        for i, rec in enumerate(log_res["top_features"], start=1):
            lines.append(
                f"  {i:2d}. {rec['feature']}: coef={rec['coef']:+.3f}, favors={rec['direction']}"
            )
    else:
        lines.append("Logistic analysis not run: " + str(log_res.get("msg", "unknown")))

    out_path = out_dir / "error_math_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    source_dir = DEFAULT_SOURCE_DIR
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_df = load_segments()
    seg_df = seg_df[seg_df["kind"].isin([KIND_A, KIND_B])].copy()

    save_counts_plot(seg_df, out_dir)
    save_duration_hist(seg_df, out_dir)

    feat_df, A, B = build_dataset(seg_df, source_dir)
    feat_df.to_csv(out_dir / "segment_feature_table.csv", index=False)

    save_shape_overlay(A, B, out_dir)
    save_shape_band(A, B, out_dir)

    feature_cols = [
        "duration_s",
        "n_samples",
        "r_mean",
        "r_std",
        "r_range",
        "r_iqr",
        "r_p10",
        "r_p90",
        "slope_total",
        "diff_abs_mean",
        "diff_std",
        "curv_abs_mean",
        "z_peak",
        "z_valley",
    ]

    group_summary(feat_df, feature_cols, out_dir)
    save_pca_plot(feat_df, feature_cols, out_dir)
    log_res = logistic_math(feat_df, feature_cols, out_dir)
    write_math_summary(seg_df, feat_df, log_res, out_dir)

    print(f"Saved error atlas to: {out_dir}")
    print(f"Segments used in feature table: {len(feat_df)}")
    if log_res.get("ok", False):
        print(
            "Logistic CV metrics: "
            f"AUC={log_res['auc_mean']:.3f}+/-{log_res['auc_std']:.3f}, "
            f"ACC={log_res['acc_mean']:.3f}+/-{log_res['acc_std']:.3f}, "
            f"F1={log_res['f1_mean']:.3f}+/-{log_res['f1_std']:.3f}"
        )


if __name__ == "__main__":
    main()
