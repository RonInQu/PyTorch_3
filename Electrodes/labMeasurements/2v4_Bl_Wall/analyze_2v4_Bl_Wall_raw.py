#!/usr/bin/env python3
"""
Analyze 2-electrode and 4-electrode impedance / phase measurements
for Blood and Tissue (TA/TB/TC) — RAW values (NO baseline subtraction).

Expected folder / naming (case-insensitive):
  Blood samples : B2, B4
  Tissue samples: TA2, TA4, TB2, TB4, TC2, TC4
  (baselines are ignored in this script)

Usage:
  python analyze_2v4_Bl_Wall_raw.py
  python analyze_2v4_Bl_Wall_raw.py --dir "C:/Users/.../2v4_Bl_Wall"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parser (same logic as plot_4294a_txt.py)
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

    # Trace A = |Z| (ohm), Trace B = phase (deg)
    df["impedance_ohm"] = df["traceA_real"]
    df["phase_deg"] = df["traceB_real"]
    df = df.sort_values("frequency_hz").reset_index(drop=True)
    df["frequency_khz"] = df["frequency_hz"] / 1000.0
    return df


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def normalize_stem(stem: str) -> str:
    return re.sub(r"[\s_\-]+", "", stem).upper()


def discover_files(data_dir: Path) -> Dict[str, Path]:
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


def build_sample_map(files: Dict[str, Path]) -> list:
    """Return list of sample jobs (no baselines)."""
    jobs = []

    # Blood
    for elec, name in [(2, "B2"), (4, "B4")]:
        sp = find_file(files, [name, f"B{elec}", f"Blood{elec}", f"BL{elec}"])
        if sp is not None:
            jobs.append({
                "name": name,
                "group": "Blood",
                "electrodes": elec,
                "sample_path": sp,
                "tissue_id": None,
            })

    # Tissue TA / TB / TC
    for tissue in ["TA", "TB", "TC"]:
        for elec in [2, 4]:
            candidates = [
                f"{tissue}{elec}",
                f"{tissue}_{elec}",
                f"{tissue}-{elec}",
            ]
            sp = find_file(files, candidates)
            if sp is not None:
                jobs.append({
                    "name": f"{tissue}{elec}",
                    "group": "Tissue",
                    "electrodes": elec,
                    "sample_path": sp,
                    "tissue_id": tissue,
                })

    return jobs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "Blood": "#1f77b4",
    "TA": "#2ca02c",
    "TB": "#ff7f0e",
    "TC": "#d62728",
}


def plot_group(
    dfs: Dict[str, pd.DataFrame],
    title: str,
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for label, df in dfs.items():
        color = None
        for key, c in COLORS.items():
            if key in label:
                color = c
                break
        ax1.plot(df["frequency_khz"], df["impedance_ohm"], linewidth=2, label=label, color=color)
        ax2.plot(df["frequency_khz"], df["phase_deg"], linewidth=2, label=label, color=color)

    ax1.set_xscale("log")
    ax1.set_ylabel("Impedance |Z| [ohm] (raw)")
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
    """Individual raw plot (style of the attachment)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(df["frequency_khz"], df["impedance_ohm"], color="#1f77b4", linewidth=2)
    ax1.set_xscale("log")
    ax1.set_ylabel("Impedance |Z| [ohm]")
    ax1.set_title(f"{stem} – Impedance vs Frequency (raw)")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(df["frequency_khz"], df["phase_deg"], color="#d62728", linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Frequency [kHz] (log scale)")
    ax2.set_ylabel("Phase [deg]")
    ax2.set_title(f"{stem} – Phase vs Frequency (raw)")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_raw_impedance_phase.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature extraction (raw)
# ---------------------------------------------------------------------------

def extract_features(df: pd.DataFrame, name: str, target_khz=(1, 5, 10, 50, 100)) -> dict:
    row = {"sample": name}
    for khz in target_khz:
        z = float(np.interp(khz, df["frequency_khz"], df["impedance_ohm"]))
        p = float(np.interp(khz, df["frequency_khz"], df["phase_deg"]))
        row[f"Z_{int(khz)}kHz"] = z
        row[f"phase_{int(khz)}kHz"] = p

    if 5 in target_khz and 100 in target_khz:
        row["dphase_100_5"] = row["phase_100kHz"] - row["phase_5kHz"]
        row["Zratio_100_5"] = row["Z_100kHz"] / row["Z_5kHz"] if row["Z_5kHz"] != 0 else np.nan

    x_log = np.log10(df["frequency_khz"].values)
    row["phase_slope_deg_per_dec"] = float(np.polyfit(x_log, df["phase_deg"].values, 1)[0])
    row["Z_slope_ohm_per_dec"] = float(np.polyfit(x_log, df["impedance_ohm"].values, 1)[0])
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="2v4 Blood/Tissue impedance analysis – RAW (no baseline subtraction)")
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
        help="Output folder (default: <dir>/analysis_raw_out)",
    )
    args = parser.parse_args()

    data_dir = args.dir
    out_dir = args.outdir or (data_dir / "analysis_raw_out")
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
            "No samples matched. Check file names (expected B2/B4, TA2/TA4, TB2/TB4, TC2/TC4)."
        )

    print(f"\nSamples to process ({len(jobs)}):")
    for j in jobs:
        print(f"  {j['name']:8s}  electrodes={j['electrodes']}  file={j['sample_path'].name}")

    parsed: Dict[str, pd.DataFrame] = {}
    feature_rows = []

    for job in jobs:
        print(f"\nProcessing {job['name']} …")
        df = parse_4294a_txt(job["sample_path"])
        parsed[job["name"]] = df

        # Save raw CSV
        csv_path = out_dir / f"{job['name']}_raw.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path.name}")

        # Individual plot
        plot_single(df, job["name"], out_dir)

        # Features
        feature_rows.append(extract_features(df, job["name"]))

    # ----- Group plots: 2-electrode and 4-electrode -----
    for elec in [2, 4]:
        subset = {name: df for name, df in parsed.items()
                  if any(j["name"] == name and j["electrodes"] == elec for j in jobs)}
        if subset:
            plot_group(
                subset,
                title=f"{elec}-Electrode Blood & Tissue (raw)",
                out_path=out_dir / f"all_{elec}electrode_raw.png",
            )

    # ----- Blood-only and Tissue-only overlays -----
    blood = {n: d for n, d in parsed.items() if n.startswith("B")}
    if blood:
        plot_group(blood, "Blood (B2 / B4) – raw", out_dir / "Blood_raw.png")

    for tissue in ["TA", "TB", "TC"]:
        tset = {n: d for n, d in parsed.items() if n.startswith(tissue)}
        if tset:
            plot_group(tset, f"Tissue {tissue} – raw", out_dir / f"Tissue_{tissue}_raw.png")

    # ----- Feature summary -----
    summary = pd.DataFrame(feature_rows).sort_values("sample").reset_index(drop=True)
    summary_path = out_dir / "feature_summary_raw.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved feature summary: {summary_path}")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
