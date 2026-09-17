"""
Create a supervised 1D LDA view for DA-vs-GT error segment types.

This complements PCA: PCA is unsupervised variance compression, while LDA
optimizes class separation directly.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

FEAT_CSV = Path("analysis_data_drift/da_gt_disagreement_review/error_atlas/segment_feature_table.csv")
OUT_PNG = Path("analysis_data_drift/da_gt_disagreement_review/error_atlas/lda_projection_by_type.png")

K0 = "GT_wall_DA_clot"
K1 = "GT_clot_DA_wall"

cols = [
    "duration_s", "n_samples", "r_mean", "r_std", "r_range", "r_iqr",
    "r_p10", "r_p90", "slope_total", "diff_abs_mean", "diff_std",
    "curv_abs_mean", "z_peak", "z_valley"
]


def cv_logreg(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, accs = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        yh = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], yh))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    df = pd.read_csv(FEAT_CSV)
    df = df[df["kind"].isin([K0, K1])].dropna(subset=cols).copy()

    y = (df["kind"] == K1).astype(int).to_numpy()
    X = df[cols].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lda = LDA(n_components=1)
    z = lda.fit_transform(Xs, y).ravel()

    auc, auc_sd, acc, acc_sd = cv_logreg(z.reshape(-1, 1), y)

    z0 = z[y == 0]
    z1 = z[y == 1]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bins = 45
    ax.hist(z0, bins=bins, alpha=0.55, color="#2563eb", density=True, label="GT wall / DA clot")
    ax.hist(z1, bins=bins, alpha=0.55, color="#dc2626", density=True, label="GT clot / DA wall")

    ax.axvline(np.mean(z0), color="#1d4ed8", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(z1), color="#b91c1c", linestyle="--", linewidth=1.5)

    ax.set_title("Supervised LDA Projection of Error Types")
    ax.set_xlabel("LDA axis (maximizes class separation)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()

    txt = (
        f"N={len(y)}  |  CV Logistic on LDA axis: "
        f"AUC={auc:.3f} +/- {auc_sd:.3f}, ACC={acc:.3f} +/- {acc_sd:.3f}"
    )
    ax.text(0.01, 0.98, txt, transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#d1d5db"))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print(txt)


if __name__ == "__main__":
    main()
