#!/usr/bin/env python3
"""
Analyze 2-electrode and 4-electrode impedance / phase measurements
for Blood and Tissue (TA/TB/TC), with baseline subtraction on |Z| only.

Expected folder structure / naming (case-insensitive, .TXT or .txt):
  Blood samples : B2, B4
  Blood baseline: base  (or baseline, baseB, base_blood, ...)
  Tissue samples: TA2, TA4, TB2, TB4, TC2, TC4
  Tissue bases  : baseTA, baseTB, baseTC  (or base_TA, BASE_TA, ...)

Usage (from the measurement folder or with --dir):
  python analyze_2v4_Bl_Wall.py
  python analyze_2v4_Bl_Wall.py --dir "C:/Users/.../2v4_Bl_Wall"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parser (same logic as your plot_4294a_txt.py)
# ---------------------------------------------------------------------------

def _is_data_line(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith('"'):
        return False
    first = s.split()[0]
    return first[0].isdigit() or first[0] in "+-."


def parse_4294a_txt(file_path: Path) -> pd.DataFrame:
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    current_trace = None
    in_table = False
    trace_data: dict[str, list[tuple[float, float, float]]] = {"A": [], "B": []}

    for raw in lines:
        line = raw.strip()

        if line.startswith('"TRACE:'):
            trace_label = line.replace('"', "").split(":", 1)[-1].strip()
            current_trace = trace_label if trace_label in ("A", "B") else None
            in_table = False
            continue

        if line.startswith('"Frequency"'):
            in_table = True
            continue

        if not in_table or current_trace is None:
            continue

        if not _is_data_line(line):
            in_table = False
            continue

        parts = line.replace("\t", " ").split()
        if len(parts) < 3:
            continue
        try:
            freq = float(parts[0])
            real = float(parts[1])
            imag = float(parts[2])
        except ValueError:
            continue
        trace_data[current_trace].append((freq, real, imag))

    if not trace_data["A"]:
        raise ValueError(f"Trace A not found in {file_path.name}")
    if not trace_data["B"]:
        raise ValueError(f"Trace B not found in {file_path.name}")

    df_a = pd.DataFrame(trace_data["A"], columns=["frequency_hz", "traceA_real", "traceA_imag"])
    df_b = pd.DataFrame(trace_data["B"], columns=["frequency_hz", "traceB_real", "traceB_imag"])
    df = df_a.merge(df_b, on="frequency_hz", how="inner")
    if df.empty:
        raise ValueError(f"No overlapping frequencies in {file_path.name}")

    # Convention used in your lab: Trace A = |Z| (ohm), Trace B = phase (deg)
    df["impedance_ohm"] = df["traceA_real"]
    df["phase_deg"] = df["traceB_real"]
    df = df.sort_values("frequency_hz").reset_index(drop=True)
    df["frequency_khz"] = df["frequency_hz"] / 1000.0
    return df


# ---------------------------------------------------------------------------
# File discovery & classification
# ---------------------------------------------------------------------------

def normalize_stem(stem: str) -> str:
    """Upper-case, remove spaces/underscores/dashes for matching."""
    return re.sub(r"[\s_\-]+", "", stem).upper()


def discover_files(data_dir: Path) -> Dict[str, Path]:
    """Return map: normalized_key -> Path for all TXT files."""
    files: Dict[str, Path] = {}
    for p in sorted(data_dir.glob("*.TXT")) + sorted(data_dir.glob("*.txt")):
        key = normalize_stem(p.stem)
        files[key] = p
    return files


def find_file(files: Dict[str, Path], candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        key = normalize_stem(c)
        if key in files:
            return files[key]
    return None


def build_sample_map(files: Dict[str, Path]) -> Dict[str, Dict]:
    """
    Build a dictionary of analysis jobs.

    Each entry:
      name, group (Blood/Tissue), electrodes (2/4),
      sample_path, baseline_path, tissue_id (or None)
    """
    jobs = []

    # ----- Blood -----
    base_blood = find_file(files, ["base", "baseline", "baseB", "baseBlood", "BASE", "Base"])
    for elec, name in [(2, "B2"), (4, "B4")]:
        sp = find_file(files, [name, f"B{elec}", f"Blood{elec}", f"BL{elec}"])
        if sp is not None:
            jobs.append({
                "name": name,
                "group": "Blood",
                "electrodes": elec,
                "sample_path": sp,
                "baseline_path": base_blood,
                "tissue_id": None,
            })

    # ----- Clot -----
    base_clot = find_file(files, [
        "baseClot", "base_clot", "baseC", "baselineClot", "BaseClot", "baseCLOT",
    ])
    for elec, name in [(2, "Clot2"), (4, "Clot4")]:
        sp = find_file(files, [
            name, f"Clot{elec}", f"CLOT{elec}", f"C{elec}", f"Clot_{elec}",
        ])
        if sp is not None:
            jobs.append({
                "name": name,
                "group": "Clot",
                "electrodes": elec,
                "sample_path": sp,
                "baseline_path": base_clot if base_clot is not None else base_blood,
                "tissue_id": None,
            })

    # ----- Tissue TA / TB / TC -----
    for tissue in ["TA", "TB", "TC"]:
        base_t = find_file(files, [
            f"base{tissue}", f"base_{tissue}", f"BASE{tissue}",
            f"baseline{tissue}", f"Base{tissue}",
        ])
        for elec in [2, 4]:
            candidates = [
                f"{tissue}{elec}",
                f"{tissue}_{elec}",
                f"{tissue}-{elec}",
                f"T{tissue[-1]}{elec}",
            ]
            sp = find_file(files, candidates)
            if sp is not None:
                jobs.append({
                    "name": f"{tissue}{elec}",
                    "group": "Tissue",
                    "electrodes": elec,
                    "sample_path": sp,
                    "baseline_path": base_t,
                    "tissue_id": tissue,
                })

    return jobs


# ---------------------------------------------------------------------------
# Baseline subtraction (impedance only)
# ---------------------------------------------------------------------------

def subtract_baseline_Z(
    sample: pd.DataFrame,
    baseline: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Return a copy of sample with:
      Z_corrected = Z_sample - Z_baseline  (interpolated onto sample frequencies)
      phase unchanged
    If baseline is None, Z_corrected = Z_sample and a warning is printed.
    """
    out = sample.copy()
    if baseline is None:
        print("  WARNING: no baseline found – using raw impedance")
        out["Z_corrected"] = out["impedance_ohm"]
        return out

    # Interpolate baseline |Z| onto the sample frequency grid
    z_base_interp = np.interp(
        sample["frequency_hz"].values,
        baseline["frequency_hz"].values,
        baseline["impedance_ohm"].values,
    )
    out["Z_baseline"] = z_base_interp
    out["Z_corrected"] = out["impedance_ohm"] - z_base_interp
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

GROUP_COLORS = {
    "Blood": "#1f77b4",
    "Clot":  "#9467bd",
    "TA": "#2ca02c",
    "TB": "#ff7f0e",
    "TC": "#d62728",
    "B": "#1f77b4",
}

# 4-electrode: dark, solid
# 2-electrode: dark, dashed
ELEC_STYLE = {
    4: {"linestyle": "-",  "linewidth": 2.4, "alpha": 1.0},
    2: {"linestyle": "--", "linewidth": 2.2, "alpha": 0.95},
}

# Dark colors for both 2 and 4 (distinction is by line style)
SAMPLE_COLORS = {
    "B2":    "#1f77b4",
    "B4":    "#1f77b4",
    "Clot2": "#9467bd",
    "Clot4": "#9467bd",
    "TA2":   "#2ca02c",
    "TA4":   "#2ca02c",
    "TB2":   "#ff7f0e",
    "TB4":   "#ff7f0e",
    "TC2":   "#d62728",
    "TC4":   "#d62728",
}


def _style_for_label(label: str) -> tuple:
    """Return (color, linestyle, linewidth, alpha) for a sample label."""
    color = SAMPLE_COLORS.get(label)
    if color is None:
        color = "#333333"
        for key, c in GROUP_COLORS.items():
            if key in label:
                color = c
                break
    elec = 4 if label.endswith("4") else 2
    st = ELEC_STYLE[elec]
    return color, st["linestyle"], st["linewidth"], st["alpha"]


def plot_group(
    dfs: Dict[str, pd.DataFrame],
    title: str,
    out_path: Path,
    use_corrected: bool = True,
) -> None:
    """
    dfs: {label: dataframe with frequency_khz, Z_corrected / impedance_ohm, phase_deg}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for label, df in sorted(dfs.items()):
        zcol = "Z_corrected" if use_corrected and "Z_corrected" in df.columns else "impedance_ohm"
        color, ls, lw, alpha = _style_for_label(label)
        ax1.plot(
            df["frequency_khz"], df[zcol],
            color=color, linestyle=ls, linewidth=lw, alpha=alpha, label=label,
        )
        ax2.plot(
            df["frequency_khz"], df["phase_deg"],
            color=color, linestyle=ls, linewidth=lw, alpha=alpha, label=label,
        )

    ax1.set_xscale("log")
    ax1.set_ylabel("Impedance |Z| [ohm]" + (" (baseline-subtracted)" if use_corrected else ""))
    ax1.set_title(f"{title} – Impedance vs Frequency")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.set_xscale("log")
    ax2.set_xlabel("Frequency [kHz] (log scale)")
    ax2.set_ylabel("Phase [deg] (raw)")
    ax2.set_title(f"{title} – Phase vs Frequency")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_single(df: pd.DataFrame, stem: str, out_dir: Path) -> None:
    """Individual corrected plot for one sample (style of the attachment)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    z = df["Z_corrected"] if "Z_corrected" in df.columns else df["impedance_ohm"]
    ax1.plot(df["frequency_khz"], z, color="#1f77b4", linewidth=2)
    ax1.set_xscale("log")
    ax1.set_ylabel("Impedance |Z| [ohm] (baseline-subtracted)")
    ax1.set_title(f"{stem} – Impedance vs Frequency")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(df["frequency_khz"], df["phase_deg"], color="#d62728", linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Frequency [kHz] (log scale)")
    ax2.set_ylabel("Phase [deg]")
    ax2.set_title(f"{stem} – Phase vs Frequency")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_corrected_impedance_phase.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(df: pd.DataFrame, name: str, target_khz=(1, 5, 10, 50, 100)) -> dict:
    row = {"sample": name}
    for khz in target_khz:
        z = float(np.interp(khz, df["frequency_khz"], df["Z_corrected"]))
        p = float(np.interp(khz, df["frequency_khz"], df["phase_deg"]))
        row[f"Zcorr_{int(khz)}kHz"] = z
        row[f"phase_{int(khz)}kHz"] = p

    if 5 in target_khz and 100 in target_khz:
        row["dphase_100_5"] = row["phase_100kHz"] - row["phase_5kHz"]
        row["Zratio_100_5"] = row["Zcorr_100kHz"] / row["Zcorr_5kHz"] if row["Zcorr_5kHz"] != 0 else np.nan

    x_log = np.log10(df["frequency_khz"].values)
    row["phase_slope_deg_per_dec"] = float(np.polyfit(x_log, df["phase_deg"].values, 1)[0])
    row["Zcorr_slope_ohm_per_dec"] = float(np.polyfit(x_log, df["Z_corrected"].values, 1)[0])
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="2v4 Blood/Tissue impedance analysis with baseline subtraction")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Electrodes\labMeasurements\2v4_Bl_Wall"),
        help="Folder containing the 4294A TXT files",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output folder (default: <dir>/analysis_out)",
    )
    args = parser.parse_args()

    data_dir = args.dir
    out_dir = args.outdir or (data_dir / "analysis_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory : {data_dir}")
    print(f"Output directory: {out_dir}")

    files = discover_files(data_dir)
    print(f"\nFound {len(files)} TXT files:")
    for k, p in files.items():
        print(f"  {k:20s} -> {p.name}")

    jobs = build_sample_map(files)
    if not jobs:
        raise SystemExit(
            "No samples matched. Check file names (expected B2/B4, TA2/TA4, …, base, baseTA, …)."
        )

    print(f"\nAnalysis jobs ({len(jobs)}):")
    for j in jobs:
        base_name = j["baseline_path"].name if j["baseline_path"] else "NONE"
        print(f"  {j['name']:8s}  electrodes={j['electrodes']}  "
              f"sample={j['sample_path'].name}  baseline={base_name}")

    # Cache parsed baselines
    baseline_cache: Dict[str, pd.DataFrame] = {}
    corrected: Dict[str, pd.DataFrame] = {}
    feature_rows = []

    for job in jobs:
        print(f"\nProcessing {job['name']} …")
        sample_df = parse_4294a_txt(job["sample_path"])

        base_df = None
        if job["baseline_path"] is not None:
            bkey = str(job["baseline_path"])
            if bkey not in baseline_cache:
                baseline_cache[bkey] = parse_4294a_txt(job["baseline_path"])
            base_df = baseline_cache[bkey]

        corr = subtract_baseline_Z(sample_df, base_df)
        corrected[job["name"]] = corr

        # Save corrected CSV
        csv_path = out_dir / f"{job['name']}_corrected.csv"
        corr.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path.name}")

        # Individual plot
        plot_single(corr, job["name"], out_dir)

        # Features
        feature_rows.append(extract_features(corr, job["name"]))

    # ----- Group plots: 2-electrode and 4-electrode -----
    for elec in [2, 4]:
        subset = {name: df for name, df in corrected.items()
                  if any(j["name"] == name and j["electrodes"] == elec for j in jobs)}
        if subset:
            plot_group(
                subset,
                title=f"{elec}-Electrode Blood & Tissue (baseline-subtracted |Z|)",
                out_path=out_dir / f"all_{elec}electrode_corrected.png",
            )

    # ----- Blood / Clot / Tissue overlays -----
    blood = {n: d for n, d in corrected.items() if n.startswith("B")}
    if blood:
        plot_group(blood, "Blood (B2 / B4)", out_dir / "Blood_corrected.png")

    clot = {n: d for n, d in corrected.items() if n.startswith("Clot")}
    if clot:
        plot_group(clot, "Clot (Clot2 / Clot4)", out_dir / "Clot_corrected.png")

    for tissue in ["TA", "TB", "TC"]:
        tset = {n: d for n, d in corrected.items() if n.startswith(tissue)}
        if tset:
            plot_group(tset, f"Tissue {tissue}", out_dir / f"Tissue_{tissue}_corrected.png")

    # ----- Feature summary -----
    summary = pd.DataFrame(feature_rows).sort_values("sample").reset_index(drop=True)
    summary_path = out_dir / "feature_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved feature summary: {summary_path}")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
