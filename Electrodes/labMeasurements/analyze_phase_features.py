from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_label(raw: str) -> str:
    s = str(raw).strip().lower()
    alias = {
        "blood": "blood",
        "clot": "clot",
        "tissue #1": "tissue_1_with_blood",
        "with blood": "tissue_1_with_blood",
        "tissue #2": "tissue_2_less_blood",
        "less blood": "tissue_2_less_blood",
        "tissue #3": "tissue_3_no_blood",
        "no blood": "tissue_3_no_blood",
    }
    return alias.get(s, s.replace(" ", "_"))


def parse_csv(input_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_csv)
    label_col = raw.columns[0]

    df = raw.copy()
    df["kHz"] = pd.to_numeric(df["kHz"], errors="coerce")
    df["ohm"] = pd.to_numeric(df["ohm"], errors="coerce")
    df["phase"] = pd.to_numeric(df["phase"], errors="coerce")

    # Keep rows that contain actual measurements.
    df = df.dropna(subset=["kHz", "ohm", "phase"]).copy()

    # Map labels directly from first column; when empty, forward-fill.
    labels = df[label_col].replace("", np.nan).ffill().map(normalize_label)
    df["sample"] = labels

    return df[["sample", "kHz", "ohm", "phase"]]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    pz = df.pivot_table(index="sample", columns="kHz", values="ohm", aggfunc="mean")
    pp = df.pivot_table(index="sample", columns="kHz", values="phase", aggfunc="mean")

    needed = [5.0, 50.0, 100.0]
    missing = [k for k in needed if k not in pz.columns or k not in pp.columns]
    if missing:
        raise ValueError(f"Missing required frequencies in data: {missing}")

    f = pd.DataFrame(index=pz.index)
    f["Z_5"] = pz[5.0]
    f["Z_50"] = pz[50.0]
    f["Z_100"] = pz[100.0]
    f["phase_5"] = pp[5.0]
    f["phase_50"] = pp[50.0]
    f["phase_100"] = pp[100.0]

    f["dphase_50_5"] = f["phase_50"] - f["phase_5"]
    f["dphase_100_5"] = f["phase_100"] - f["phase_5"]
    f["dZ_100_5"] = f["Z_100"] - f["Z_5"]
    f["Zratio_100_5"] = f["Z_100"] / f["Z_5"]
    f["drop_pct_100_vs_5"] = 100.0 * (f["Z_5"] - f["Z_100"]) / f["Z_5"]

    # Phase slope in deg/decade over [5, 50, 100] kHz
    x = np.log10(np.array([5.0, 50.0, 100.0]))
    slopes = []
    for s in f.index:
        y = np.array([f.loc[s, "phase_5"], f.loc[s, "phase_50"], f.loc[s, "phase_100"]])
        slopes.append(np.polyfit(x, y, 1)[0])
    f["phase_slope_deg_per_dec"] = slopes

    return f


def classify_from_phase(row: pd.Series) -> str:
    p100 = row["phase_100"]
    dph = row["dphase_100_5"]

    # Heuristic rules from current dataset.
    if p100 > -8 and dph > -5:
        return "blood-dominant"
    if p100 < -22 and dph < -12:
        return "wall-rich"
    if p100 < -14 and dph < -10:
        return "clot-or-wall-rich"
    return "mixed/transition"


def save_threshold_report(features: pd.DataFrame, out_txt: Path) -> None:
    lines = []
    lines.append("Phase-based quick classification report")
    lines.append("")
    lines.append("Rules:")
    lines.append("1) blood-dominant: phase_100 > -8 and dphase_100_5 > -5")
    lines.append("2) wall-rich:      phase_100 < -22 and dphase_100_5 < -12")
    lines.append("3) clot/wall-rich: phase_100 < -14 and dphase_100_5 < -10")
    lines.append("4) else: mixed/transition")
    lines.append("")
    lines.append("Per-sample results:")

    for sample, row in features.iterrows():
        cls = classify_from_phase(row)
        lines.append(
            f"- {sample}: class={cls}, phase_5={row['phase_5']:.3f}, "
            f"phase_50={row['phase_50']:.3f}, phase_100={row['phase_100']:.3f}, "
            f"dphase_100_5={row['dphase_100_5']:.3f}"
        )

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def make_plots(df: pd.DataFrame, features: pd.DataFrame, out_dir: Path) -> None:
    # Plot 1: phase vs frequency for each sample
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for sample, g in df.groupby("sample"):
        g = g.sort_values("kHz")
        ax1.plot(g["kHz"], g["phase"], marker="o", linewidth=2, label=sample)
    ax1.set_xscale("log")
    ax1.set_xticks([5, 50, 100])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("Frequency [kHz]")
    ax1.set_ylabel("Phase [deg]")
    ax1.set_title("Phase vs Frequency")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig(out_dir / "phase_vs_frequency.png", dpi=160)
    plt.close(fig1)

    # Plot 2: phase_100 vs phase drop
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    x = features["phase_100"].values
    y = features["dphase_100_5"].values
    ax2.scatter(x, y, s=90)
    for sample, row in features.iterrows():
        ax2.annotate(sample, (row["phase_100"], row["dphase_100_5"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax2.axvline(-8, color="gray", linestyle="--", linewidth=1)
    ax2.axvline(-14, color="gray", linestyle=":", linewidth=1)
    ax2.axvline(-22, color="gray", linestyle="--", linewidth=1)
    ax2.axhline(-5, color="gray", linestyle="--", linewidth=1)
    ax2.axhline(-10, color="gray", linestyle=":", linewidth=1)
    ax2.axhline(-12, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("phase_100 [deg]")
    ax2.set_ylabel("dphase_100_5 [deg]")
    ax2.set_title("Phase Features for Tissue Separation")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "phase_feature_scatter.png", dpi=160)
    plt.close(fig2)

    # Plot 2b: publication-style decision scatter with shaded regions.
    fig2b, ax2b = plt.subplots(figsize=(8.6, 5.6))

    # Region boundaries in (phase_100, dphase_100_5) space.
    x_left, x_right = -28.0, -2.0
    y_low, y_high = -20.0, 1.0

    # Blood-dominant region: p100 > -8 and dphase > -5.
    ax2b.fill_betweenx(
        [ -5.0, y_high],
        [-8.0, -8.0],
        [x_right, x_right],
        color="#d9f2d9",
        alpha=0.65,
        zorder=0,
        label="Blood-dominant zone",
    )

    # Clot/wall-rich region: p100 < -14 and dphase < -10.
    ax2b.fill_betweenx(
        [y_low, -10.0],
        [x_left, x_left],
        [-14.0, -14.0],
        color="#fde0dd",
        alpha=0.65,
        zorder=0,
        label="Clot/wall-rich zone",
    )

    # Wall-rich subregion: p100 < -22 and dphase < -12.
    ax2b.fill_betweenx(
        [y_low, -12.0],
        [x_left, x_left],
        [-22.0, -22.0],
        color="#d6eaf8",
        alpha=0.8,
        zorder=0,
        label="Wall-rich zone",
    )

    # Mixed/transition region as a broad central band.
    ax2b.fill_betweenx(
        [y_low, y_high],
        [-14.0, -14.0],
        [-8.0, -8.0],
        color="#fdf3cf",
        alpha=0.55,
        zorder=0,
        label="Mixed/transition band",
    )

    class_colors = {
        "blood-dominant": "#2ca02c",
        "mixed/transition": "#ff7f0e",
        "clot-or-wall-rich": "#d62728",
        "wall-rich": "#1f77b4",
    }

    for sample, row in features.iterrows():
        cls = classify_from_phase(row)
        color = class_colors.get(cls, "#444444")
        ax2b.scatter(
            row["phase_100"],
            row["dphase_100_5"],
            s=120,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax2b.annotate(
            sample,
            (row["phase_100"], row["dphase_100_5"]),
            fontsize=9,
            xytext=(6, 6),
            textcoords="offset points",
            zorder=4,
        )

    # Decision boundaries.
    ax2b.axvline(-8, color="gray", linestyle="--", linewidth=1.2)
    ax2b.axvline(-14, color="gray", linestyle=":", linewidth=1.2)
    ax2b.axvline(-22, color="gray", linestyle="--", linewidth=1.2)
    ax2b.axhline(-5, color="gray", linestyle="--", linewidth=1.2)
    ax2b.axhline(-10, color="gray", linestyle=":", linewidth=1.2)
    ax2b.axhline(-12, color="gray", linestyle="--", linewidth=1.2)

    ax2b.set_xlim(x_left, x_right)
    ax2b.set_ylim(y_low, y_high)
    ax2b.set_xlabel("phase_100 [deg]")
    ax2b.set_ylabel("dphase_100_5 [deg]")
    ax2b.set_title("Phase Decision Map: Blood vs Mixed vs Clot/Wall")
    ax2b.grid(True, alpha=0.25)
    ax2b.legend(loc="upper left", fontsize=8)
    fig2b.tight_layout()
    fig2b.savefig(out_dir / "phase_feature_scatter_decision_zones.png", dpi=180)
    plt.close(fig2b)

    # Plot 3: grouped bars for the three tissue conditions.
    tissue_order = [
        "tissue_1_with_blood",
        "tissue_2_less_blood",
        "tissue_3_no_blood",
    ]
    tissue_labels = ["With blood", "Less blood", "No blood"]
    tissue_colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]

    available = [s for s in tissue_order if s in features.index]
    if available:
        freq_cols = ["phase_5", "phase_50", "phase_100"]
        freq_names = ["5 kHz", "50 kHz", "100 kHz"]
        x = np.arange(len(freq_cols))
        width = 0.22

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        for i, sample in enumerate(available):
            y = np.abs(features.loc[sample, freq_cols].values.astype(float))
            label = tissue_labels[tissue_order.index(sample)]
            color = tissue_colors[tissue_order.index(sample)]
            ax3.bar(x + (i - (len(available) - 1) / 2) * width, y, width=width, label=label, color=color, edgecolor="black")

        ax3.set_xticks(x)
        ax3.set_xticklabels(freq_names)
        ax3.set_ylabel("|Phase| [deg]")
        ax3.set_title("Three Tissue Conditions: |Phase| by Frequency")
        ax3.grid(axis="y", alpha=0.3)
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(out_dir / "tissue_three_condition_phase_bars.png", dpi=160)
        plt.close(fig3)

        # Plot 4: grouped bars of |delta phase| relative to 50 kHz, including clot.
        delta_order = [
            "clot",
            "tissue_1_with_blood",
            "tissue_2_less_blood",
            "tissue_3_no_blood",
        ]
        delta_labels = ["Clot", "With blood", "Less blood", "No blood"]
        delta_colors = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]
        delta_available = [s for s in delta_order if s in features.index]

        fig4, ax4 = plt.subplots(figsize=(8, 5))
        for i, sample in enumerate(delta_available):
            phase_vals = np.abs(features.loc[sample, freq_cols].values.astype(float))
            delta_vals = np.abs(phase_vals - phase_vals[1])
            label = delta_labels[delta_order.index(sample)]
            color = delta_colors[delta_order.index(sample)]
            ax4.bar(x + (i - (len(delta_available) - 1) / 2) * width, delta_vals, width=width, label=label, color=color, edgecolor="black")

        ax4.set_xticks(x)
        ax4.set_xticklabels(freq_names)
        ax4.set_ylabel("|Δ|Phase|| from 50 kHz [deg]")
        ax4.set_title("Clot + Tissue Conditions: |Phase Delta| Relative to 50 kHz")
        ax4.grid(axis="y", alpha=0.3)
        ax4.legend()
        fig4.tight_layout()
        fig4.savefig(out_dir / "clot_tissue_phase_delta_bars_relative_50k.png", dpi=160)
        fig4.savefig(out_dir / "tissue_three_condition_phase_delta_bars_relative_50k.png", dpi=160)
        plt.close(fig4)

    # Plot 5: impedance vs frequency for each sample.
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    for sample, g in df.groupby("sample"):
        g = g.sort_values("kHz")
        ax5.plot(g["kHz"], g["ohm"], marker="o", linewidth=2, label=sample)
    ax5.set_xscale("log")
    ax5.set_xticks([5, 50, 100])
    ax5.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax5.set_xlabel("Frequency [kHz]")
    ax5.set_ylabel("|Z| [ohm]")
    ax5.set_title("Impedance vs Frequency")
    ax5.grid(True, which="both", alpha=0.3)
    ax5.legend(fontsize=8)
    fig5.tight_layout()
    fig5.savefig(out_dir / "impedance_vs_frequency.png", dpi=170)
    plt.close(fig5)

    # Plot 6: impedance feature scatter (level vs dispersion).
    fig6, ax6 = plt.subplots(figsize=(8.4, 5.6))
    xz = features["Z_100"].values
    yd = features["drop_pct_100_vs_5"].values

    # Heuristic guide lines based on this dataset.
    ax6.axvline(560, color="gray", linestyle="--", linewidth=1.1)
    ax6.axvline(700, color="gray", linestyle=":", linewidth=1.1)
    ax6.axhline(10, color="gray", linestyle="--", linewidth=1.1)
    ax6.axhline(15, color="gray", linestyle=":", linewidth=1.1)

    for sample, row in features.iterrows():
        if sample == "blood":
            color = "#2ca02c"
        elif sample == "clot":
            color = "#d62728"
        elif sample == "tissue_3_no_blood":
            color = "#1f77b4"
        else:
            color = "#ff7f0e"

        ax6.scatter(row["Z_100"], row["drop_pct_100_vs_5"], s=120, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        ax6.annotate(sample, (row["Z_100"], row["drop_pct_100_vs_5"]), fontsize=9, xytext=(6, 6), textcoords="offset points")

    ax6.set_xlabel("Z_100 [ohm]")
    ax6.set_ylabel("Impedance drop 5->100 kHz [%]")
    ax6.set_title("Impedance Feature Map: Level vs Dispersion")
    ax6.grid(True, alpha=0.25)
    fig6.tight_layout()
    fig6.savefig(out_dir / "impedance_feature_scatter.png", dpi=180)
    plt.close(fig6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze phase-angle features for blood/clot/wall separation.")
    parser.add_argument("--input", type=Path, default=Path("Lab_1.csv"), help="Input CSV path")
    parser.add_argument("--outdir", type=Path, default=Path("phase_analysis_out"), help="Output directory")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = parse_csv(args.input)
    features = build_features(df)

    cleaned_path = args.outdir / "cleaned_long.csv"
    features_path = args.outdir / "features_table.csv"
    report_path = args.outdir / "phase_threshold_report.txt"

    df.to_csv(cleaned_path, index=False)
    features.to_csv(features_path)
    save_threshold_report(features, report_path)
    make_plots(df, features, args.outdir)

    print(f"Saved: {cleaned_path}")
    print(f"Saved: {features_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {args.outdir / 'phase_vs_frequency.png'}")
    print(f"Saved: {args.outdir / 'phase_feature_scatter.png'}")
    print(f"Saved: {args.outdir / 'phase_feature_scatter_decision_zones.png'}")
    print(f"Saved: {args.outdir / 'impedance_vs_frequency.png'}")
    print(f"Saved: {args.outdir / 'impedance_feature_scatter.png'}")
    print("\nFeature summary:")
    print(features.round(3).to_string())


if __name__ == "__main__":
    main()
