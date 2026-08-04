from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set this once per run; used in titles and output filenames.
CASE_NAME = "Porcine"
OUTPUT_PREFIX = CASE_NAME.lower().replace(" ", "_")


def out_path(out_dir: Path, stem: str) -> Path:
    return out_dir / f"{OUTPUT_PREFIX}_{stem}.png"


def parse_sample_metadata(sample: str) -> dict[str, object]:
    s_raw = sample.upper().replace(" ", "_").replace("-", "_")
    s = re.sub(r"_+", "_", s_raw).strip("_")
    core = s[3:] if s.startswith("FN_") else s

    meta: dict[str, object] = {
        "sample": sample,
        "group": "other",
        "display_label": sample.replace("FN_", ""),
        "sort_key": (99, 99, 99),
    }

    blood_match = re.fullmatch(r"(?:BLOOD|B)(?:_(?P<n>\d+)|(?P<n2>\d+))?", core)
    if blood_match:
        n_text = blood_match.group("n") or blood_match.group("n2")
        n = int(n_text) if n_text is not None else 0
        label = f"Blood{n}" if n else "Blood"
        meta.update(group="blood", display_label=label, sort_key=(0, n, 0))
        return meta

    clot_match = re.fullmatch(r"C(?P<n>\d+)", core)
    if clot_match:
        n = int(clot_match.group("n"))
        meta.update(group="clot", display_label=f"C{n}", sort_key=(1, n, 0))
        return meta

    # New naming convention examples:
    #   FN_T1p   -> tissue 1, no pressure, no blood
    #   FN_T1np  -> tissue 1, with pressure, no blood
    #   FN_T1pB  -> tissue 1, no pressure, with blood
    #   FN_T1npB -> tissue 1, with pressure, with blood
    tissue_match = re.fullmatch(r"T(?P<n>\d+)(?P<press>NP|P)(?P<blood>B)?", core)
    if tissue_match:
        n = int(tissue_match.group("n"))
        # New naming convention: T?p = no pressure, T?np = with pressure.
        press = "with_pressure" if tissue_match.group("press") == "NP" else "no_pressure"
        blood = "with_blood" if tissue_match.group("blood") else "no_blood"
        display = f"T{n}{'np' if press == 'with_pressure' else 'p'}{'B' if blood == 'with_blood' else ''}"
        meta.update(
            group=f"tissue_{press}_{blood}",
            display_label=display,
            sort_key=(2, n, 1 if press == "with_pressure" else 0),
        )
        return meta

    return meta


def classify_sample(sample: str) -> str:
    return str(parse_sample_metadata(sample)["group"])


def build_group_sweeps(out_dir: Path) -> pd.DataFrame:
    rows = []
    for parsed_file in sorted(out_dir.glob("FN*_parsed.csv")):
        sample = parsed_file.stem.replace("_parsed", "")
        meta = parse_sample_metadata(sample)
        df = pd.read_csv(parsed_file)
        for _, r in df.iterrows():
            rows.append(
                {
                    "sample": sample,
                    "group": meta["group"],
                    "display_label": meta["display_label"],
                    "sort_key": meta["sort_key"],
                    "frequency_khz": float(r["frequency_hz"]) / 1000.0,
                    "impedance_ohm": float(r["impedance_ohm"]),
                    "phase_deg": float(r["phase_deg"]),
                }
            )
    if not rows:
        raise ValueError("No FN*_parsed.csv files found. Run plot_4294a_txt.py first.")
    return pd.DataFrame(rows)


def plot_group_overlays(long_df: pd.DataFrame, out_dir: Path) -> None:
    palette = {
        "blood": "#2ca02c",
        "clot": "#d62728",
        "tissue_no_pressure_no_blood": "#1f77b4",
        "tissue_with_pressure_no_blood": "#9467bd",
        "tissue_no_pressure_with_blood": "#e377c2",
        "tissue_with_pressure_with_blood": "#bcbd22",
        "tissue_other": "#7f7f7f",
        "other": "#333333",
    }

    groups = sorted(long_df["group"].unique().tolist())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 9), sharex=True)

    for g in groups:
        gdf = long_df[long_df["group"] == g].copy()
        color = palette.get(g, "#333333")

        # Thin lines for each repeat.
        for sample, sdf in gdf.groupby("sample"):
            sdf = sdf.sort_values("frequency_khz")
            ax1.plot(sdf["frequency_khz"], sdf["impedance_ohm"], color=color, alpha=0.23, linewidth=1)
            ax2.plot(sdf["frequency_khz"], sdf["phase_deg"], color=color, alpha=0.23, linewidth=1)

        # Mean and std bands per group.
        grp_stats = (
            gdf.groupby("frequency_khz")
            .agg(
                z_mean=("impedance_ohm", "mean"),
                z_std=("impedance_ohm", "std"),
                p_mean=("phase_deg", "mean"),
                p_std=("phase_deg", "std"),
            )
            .reset_index()
            .sort_values("frequency_khz")
        )

        x = grp_stats["frequency_khz"].values
        z_mean = grp_stats["z_mean"].values
        z_std = np.nan_to_num(grp_stats["z_std"].values, nan=0.0)
        p_mean = grp_stats["p_mean"].values
        p_std = np.nan_to_num(grp_stats["p_std"].values, nan=0.0)

        ax1.plot(x, z_mean, color=color, linewidth=2.4, label=g)
        ax1.fill_between(x, z_mean - z_std, z_mean + z_std, color=color, alpha=0.12)

        ax2.plot(x, p_mean, color=color, linewidth=2.4, label=g)
        ax2.fill_between(x, p_mean - p_std, p_mean + p_std, color=color, alpha=0.12)

    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax1.set_ylabel("Impedance |Z| [ohm]")
    ax2.set_ylabel("Phase [deg]")
    ax2.set_xlabel("Frequency [kHz] (log scale)")
    ax1.set_title(f"{CASE_NAME} Sweep Summary: Impedance (group mean ± std)")
    ax2.set_title(f"{CASE_NAME} Sweep Summary: Phase (group mean ± std)")
    ax1.grid(True, which="both", alpha=0.25)
    ax2.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path(out_dir, "group_overlay"), dpi=185)
    plt.close(fig)


def plot_feature_maps(summary_df: pd.DataFrame, out_dir: Path) -> None:
    summary_df = summary_df.copy()
    summary_df["group"] = summary_df["sample"].map(classify_sample)
    summary_df["drop_pct_100_vs_5"] = 100.0 * (summary_df["Z_5kHz"] - summary_df["Z_100kHz"]) / summary_df["Z_5kHz"]

    palette = {
        "blood": "#2ca02c",
        "clot": "#d62728",
        "tissue_no_pressure_no_blood": "#1f77b4",
        "tissue_with_pressure_no_blood": "#9467bd",
        "tissue_no_pressure_with_blood": "#e377c2",
        "tissue_with_pressure_with_blood": "#bcbd22",
        "tissue_other": "#7f7f7f",
        "other": "#333333",
    }

    # Feature map 1: phase separation view.
    fig1, ax1 = plt.subplots(figsize=(8.7, 5.8))
    for _, r in summary_df.iterrows():
        c = palette.get(r["group"], "#333333")
        ax1.scatter(r["phase_100kHz"], r["dphase_100_5"], s=95, color=c, edgecolor="black", linewidth=0.7)
        ax1.annotate(r["sample"], (r["phase_100kHz"], r["dphase_100_5"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax1.set_xlabel("phase_100kHz [deg]")
    ax1.set_ylabel("dphase_100_5 [deg]")
    ax1.set_title(f"{CASE_NAME} Sweep Summary: Phase Domain")
    ax1.grid(True, alpha=0.28)
    fig1.tight_layout()
    fig1.savefig(out_path(out_dir, "phase_feature_map"), dpi=185)
    plt.close(fig1)

    # Feature map 2: impedance level vs dispersion.
    fig2, ax2 = plt.subplots(figsize=(8.7, 5.8))
    for _, r in summary_df.iterrows():
        c = palette.get(r["group"], "#333333")
        ax2.scatter(r["Z_100kHz"], r["drop_pct_100_vs_5"], s=95, color=c, edgecolor="black", linewidth=0.7)
        ax2.annotate(r["sample"], (r["Z_100kHz"], r["drop_pct_100_vs_5"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax2.set_xlabel("Z_100kHz [ohm]")
    ax2.set_ylabel("Drop 5->100kHz [%]")
    ax2.set_title(f"{CASE_NAME} Sweep Summary: Impedance Domain")
    ax2.grid(True, alpha=0.28)
    fig2.tight_layout()
    fig2.savefig(out_path(out_dir, "impedance_feature_map"), dpi=185)
    plt.close(fig2)


def plot_key_frequency_bars(summary_df: pd.DataFrame, out_dir: Path) -> None:
    summary_df = summary_df.copy()
    summary_df["meta"] = summary_df["sample"].map(parse_sample_metadata)
    summary_df["display_label"] = summary_df["meta"].map(lambda m: m["display_label"])
    summary_df["sort_key"] = summary_df["meta"].map(lambda m: m["sort_key"])

    sdf = summary_df.sort_values("sort_key").copy()
    if sdf.empty:
        return

    x = np.arange(len(sdf))
    w = 0.24

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax1.bar(x - w, sdf["phase_5kHz"], width=w, label="5 kHz")
    ax1.bar(x, sdf["phase_50kHz"], width=w, label="50 kHz")
    ax1.bar(x + w, sdf["phase_100kHz"], width=w, label="100 kHz")
    ax1.set_ylabel("Phase [deg]")
    ax1.set_title(f"{CASE_NAME} Sweep Summary: Phase at Key Frequencies")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()

    ax2.bar(x - w, sdf["Z_5kHz"], width=w, label="5 kHz")
    ax2.bar(x, sdf["Z_50kHz"], width=w, label="50 kHz")
    ax2.bar(x + w, sdf["Z_100kHz"], width=w, label="100 kHz")
    ax2.set_ylabel("Impedance [ohm]")
    ax2.set_title(f"{CASE_NAME} Sweep Summary: Impedance at Key Frequencies")
    ax2.grid(axis="y", alpha=0.25)

    ax2.set_xticks(x)
    ax2.set_xticklabels(sdf["display_label"].tolist(), rotation=28, ha="right")
    ax2.set_xlabel("Sample")

    fig.tight_layout()
    fig.savefig(out_path(out_dir, "keyfreq_group_bars"), dpi=185)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create summary plots for FN sweep repeats.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("phase_analysis_out/FN_sweep_feature_summary.csv"),
        help="Summary CSV generated by plot_4294a_txt.py",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("phase_analysis_out"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.read_csv(args.summary_csv)
    long_df = build_group_sweeps(args.outdir)

    plot_group_overlays(long_df, args.outdir)
    plot_feature_maps(summary_df, args.outdir)
    plot_key_frequency_bars(summary_df, args.outdir)

    print(f"Saved: {out_path(args.outdir, 'group_overlay')}")
    print(f"Saved: {out_path(args.outdir, 'phase_feature_map')}")
    print(f"Saved: {out_path(args.outdir, 'impedance_feature_map')}")
    print(f"Saved: {out_path(args.outdir, 'keyfreq_group_bars')}")


if __name__ == "__main__":
    main()
