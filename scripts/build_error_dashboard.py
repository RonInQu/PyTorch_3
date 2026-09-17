"""
Build a single 4-panel management dashboard for DA-vs-GT clot/wall errors.

Inputs (from prior scripts):
- analysis_data_drift/da_gt_disagreement_review/da_gt_error_segments.csv
- analysis_data_drift/da_gt_disagreement_review/error_atlas/segment_feature_table.csv
- analysis_data_drift/da_gt_disagreement_review/error_atlas/error_math_summary.txt

Output:
- analysis_data_drift/da_gt_disagreement_review/error_atlas/management_error_dashboard.png
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path("analysis_data_drift") / "da_gt_disagreement_review"
ATLAS_DIR = BASE_DIR / "error_atlas"
SEG_CSV = BASE_DIR / "da_gt_error_segments.csv"
FEAT_CSV = ATLAS_DIR / "segment_feature_table.csv"
MATH_TXT = ATLAS_DIR / "error_math_summary.txt"
OUT_PNG = ATLAS_DIR / "management_error_dashboard.png"

KIND_A = "GT_wall_DA_clot"
KIND_B = "GT_clot_DA_wall"


def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    def extract(pattern: str) -> str:
        m = re.search(pattern, text)
        return m.group(1) if m else "n/a"

    return {
        "auc": extract(r"ROC-AUC:\s*([0-9.]+\s*\+/-\s*[0-9.]+)"),
        "acc": extract(r"Accuracy:\s*([0-9.]+\s*\+/-\s*[0-9.]+)"),
        "f1": extract(r"F1:\s*([0-9.]+\s*\+/-\s*[0-9.]+)"),
    }


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / max(1, (len(a) + len(b) - 2))
    if pooled <= 1e-12:
        return np.nan
    return (mb - ma) / np.sqrt(pooled)


def _bootstrap_ci_d(a: np.ndarray, b: np.ndarray, n_boot: int = 1200, seed: int = 42) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    dvals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ai = a[rng.integers(0, len(a), len(a))]
        bi = b[rng.integers(0, len(b), len(b))]
        dvals[i] = _cohen_d(ai, bi)
    lo = float(np.nanpercentile(dvals, 2.5))
    hi = float(np.nanpercentile(dvals, 97.5))
    return lo, hi


def _cv_logreg_1d(z: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, accs = [], []
    for tr, te in skf.split(z, y):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
        clf.fit(z[tr].reshape(-1, 1), y[tr])
        p = clf.predict_proba(z[te].reshape(-1, 1))[:, 1]
        yh = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], yh))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)), float(np.std(accs))


def build_dashboard() -> None:
    seg = pd.read_csv(SEG_CSV)
    seg = seg[seg["kind"].isin([KIND_A, KIND_B])].copy()

    feat = pd.read_csv(FEAT_CSV)
    feat = feat[feat["kind"].isin([KIND_A, KIND_B])].copy()

    metrics = parse_metrics(MATH_TXT)

    color_a = "#2563eb"
    color_b = "#dc2626"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_counts, ax_dur, ax_shape, ax_lda = axes.ravel()

    # Panel 1: counts
    c = seg["kind"].value_counts().reindex([KIND_A, KIND_B]).fillna(0).astype(int)
    labels = ["GT wall / DA clot", "GT clot / DA wall"]
    vals = c.values
    bars = ax_counts.bar(labels, vals, color=[color_a, color_b], alpha=0.9)
    for b in bars:
        h = b.get_height()
        ax_counts.text(b.get_x() + b.get_width() / 2, h + max(vals) * 0.02, f"{int(h)}", ha="center", va="bottom", fontsize=11)
    ax_counts.set_title("Error Segment Counts")
    ax_counts.set_ylabel("Count")
    ax_counts.grid(axis="y", alpha=0.2)

    # Panel 2: duration hist
    a = seg.loc[seg["kind"] == KIND_A, "duration_s"].to_numpy()
    b = seg.loc[seg["kind"] == KIND_B, "duration_s"].to_numpy()
    max_d = min(360.0, float(np.nanmax(seg["duration_s"])) if len(seg) else 1.0)
    bins = np.linspace(0.0, max(1.0, max_d), 45)
    ax_dur.hist(a, bins=bins, color=color_a, alpha=0.55, label="GT wall / DA clot")
    ax_dur.hist(b, bins=bins, color=color_b, alpha=0.55, label="GT clot / DA wall")
    ax_dur.set_title("Segment Duration Distribution")
    ax_dur.set_xlabel("Duration (s)")
    ax_dur.set_ylabel("Count")
    ax_dur.grid(alpha=0.2)
    ax_dur.legend(fontsize=9)

    # Panel 3: standardized effect sizes (Cohen's d) with 95% bootstrap CI.
    # Positive d => larger in KIND_B (GT clot / DA wall), negative => KIND_A.
    plot_features = [
        ("diff_abs_mean", "mean |dR|"),
        ("r_std", "std(R)"),
        ("curv_abs_mean", "mean |d2R|"),
        ("r_range", "range(R)"),
        ("z_peak", "z_peak"),
        ("z_valley", "z_valley"),
    ]
    a_sub = feat[feat["kind"] == KIND_A]
    b_sub = feat[feat["kind"] == KIND_B]

    rows = []
    for col, label in plot_features:
        da = a_sub[col].to_numpy(dtype=np.float64)
        db = b_sub[col].to_numpy(dtype=np.float64)
        d = _cohen_d(da, db)
        lo, hi = _bootstrap_ci_d(da, db)
        rows.append((label, d, lo, hi))

    edf = pd.DataFrame(rows, columns=["label", "d", "lo", "hi"]).dropna().sort_values("d")
    y_pos = np.arange(len(edf))
    colors = ["#2563eb" if v < 0 else "#dc2626" for v in edf["d"].to_numpy()]
    ax_shape.barh(y_pos, edf["d"], color=colors, alpha=0.9)
    xerr = np.vstack((edf["d"] - edf["lo"], edf["hi"] - edf["d"]))
    ax_shape.errorbar(edf["d"], y_pos, xerr=xerr, fmt="none", ecolor="#111827", elinewidth=1.2, capsize=3)
    ax_shape.axvline(0.0, color="#374151", linestyle="--", linewidth=1.1)
    ax_shape.set_yticks(y_pos)
    ax_shape.set_yticklabels(edf["label"])
    ax_shape.set_xlabel("Standardized effect size (Cohen's d)")
    ax_shape.set_title("Error-Type Fingerprints (Standardized, 95% CI)")
    ax_shape.grid(axis="x", alpha=0.2)
    ax_shape.text(
        0.01,
        0.02,
        "d > 0 favors GT clot / DA wall, d < 0 favors GT wall / DA clot",
        transform=ax_shape.transAxes,
        fontsize=8,
        color="#374151",
    )

    # Panel 4: supervised LDA projection (replaces PCA)
    feat_cols = [
        "duration_s", "n_samples", "r_mean", "r_std", "r_range", "r_iqr",
        "r_p10", "r_p90", "slope_total", "diff_abs_mean", "diff_std",
        "curv_abs_mean", "z_peak", "z_valley"
    ]
    clean = feat.dropna(subset=feat_cols).copy()
    clean["kind_bin"] = (clean["kind"] == KIND_B).astype(int)

    X = clean[feat_cols].to_numpy(dtype=np.float64)
    y = clean["kind_bin"].to_numpy(dtype=int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lda = LDA(n_components=1)
    z = lda.fit_transform(Xs, y).ravel()
    auc_lda, auc_sd, acc_lda, acc_sd = _cv_logreg_1d(z, y)

    z0 = z[y == 0]
    z1 = z[y == 1]
    ax_lda.hist(z0, bins=42, alpha=0.55, color=color_a, density=True, label="GT wall / DA clot")
    ax_lda.hist(z1, bins=42, alpha=0.55, color=color_b, density=True, label="GT clot / DA wall")
    ax_lda.axvline(float(np.mean(z0)), color="#1d4ed8", linestyle="--", linewidth=1.2)
    ax_lda.axvline(float(np.mean(z1)), color="#b91c1c", linestyle="--", linewidth=1.2)
    ax_lda.set_title("LDA Separation of Error Types (Supervised)")
    ax_lda.set_xlabel("LDA axis")
    ax_lda.set_ylabel("Density")
    ax_lda.grid(alpha=0.2)
    ax_lda.legend(fontsize=8)
    ax_lda.text(
        0.01,
        0.98,
        f"CV on LDA axis: AUC {auc_lda:.3f} +/- {auc_sd:.3f}, ACC {acc_lda:.3f} +/- {acc_sd:.3f}",
        transform=ax_lda.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#d1d5db"),
    )

    # Global title + KPI text
    total = int(len(seg))
    fig.suptitle("DA vs GT Clot/Wall Error Dashboard", fontsize=18, y=0.98)
    kpi = (
        f"Total segments: {total} | GT wall/DA clot: {int(c.get(KIND_A, 0))} | GT clot/DA wall: {int(c.get(KIND_B, 0))}\n"
        f"Logistic separability (5-fold CV): AUC {metrics['auc']} | Accuracy {metrics['acc']} | F1 {metrics['f1']}"
    )
    fig.text(0.5, 0.93, kpi, ha="center", va="top", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_dashboard()
    print(f"Saved: {OUT_PNG}")
