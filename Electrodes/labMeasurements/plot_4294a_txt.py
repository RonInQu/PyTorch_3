from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _is_data_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith('"'):
        return False
    first = s.split()[0]
    # Numeric rows in this export start with frequency (often scientific notation).
    return first[0].isdigit() or first[0] in "+-."


def parse_4294a_txt(file_path: Path) -> pd.DataFrame:
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    current_trace = None
    in_table = False
    trace_data: dict[str, list[tuple[float, float, float]]] = {"A": [], "B": []}

    for raw in lines:
        line = raw.strip()

        if line.startswith('"TRACE:'):
            # Parse explicit trace label after the colon (A or B).
            trace_label = line.replace('"', "").split(":", 1)[-1].strip()
            if trace_label == "A":
                current_trace = "A"
            elif trace_label == "B":
                current_trace = "B"
            else:
                current_trace = None
            in_table = False
            continue

        if line.startswith('"Frequency"'):
            in_table = True
            continue

        if not in_table or current_trace is None:
            continue

        if not _is_data_line(line):
            # End of this trace table section.
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
        raise ValueError("Trace A data not found in file")
    if not trace_data["B"]:
        raise ValueError("Trace B data not found in file")

    df_a = pd.DataFrame(trace_data["A"], columns=["frequency_hz", "traceA_real", "traceA_imag"])
    df_b = pd.DataFrame(trace_data["B"], columns=["frequency_hz", "traceB_real", "traceB_imag"])

    df = df_a.merge(df_b, on="frequency_hz", how="inner")
    if df.empty:
        raise ValueError("No overlapping frequency rows between Trace A and Trace B")

    # For this measurement setup: Trace A is impedance magnitude (ohm), Trace B is phase (deg).
    df["impedance_ohm"] = df["traceA_real"]
    df["phase_deg"] = df["traceB_real"]
    return df.sort_values("frequency_hz").reset_index(drop=True)


def make_plots(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    freq_khz = df["frequency_hz"].values / 1000.0

    # Combined figure: impedance and phase vs log-frequency.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(freq_khz, df["impedance_ohm"].values, color="#1f77b4", linewidth=2)
    ax1.set_xscale("log")
    ax1.set_ylabel("Impedance |Z| [ohm]")
    ax1.set_title("Impedance vs Frequency")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(freq_khz, df["phase_deg"].values, color="#d62728", linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Frequency [kHz] (log scale)")
    ax2.set_ylabel("Phase [deg]")
    ax2.set_title("Phase vs Frequency")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_impedance_phase_logfreq.png", dpi=180)
    plt.close(fig)

    # Separate plots.
    fig_i, ax_i = plt.subplots(figsize=(8.8, 4.6))
    ax_i.plot(freq_khz, df["impedance_ohm"].values, color="#1f77b4", linewidth=2)
    ax_i.set_xscale("log")
    ax_i.set_xlabel("Frequency [kHz] (log scale)")
    ax_i.set_ylabel("Impedance |Z| [ohm]")
    ax_i.set_title("Impedance vs Frequency")
    ax_i.grid(True, which="both", alpha=0.3)
    fig_i.tight_layout()
    fig_i.savefig(out_dir / f"{stem}_impedance_logfreq.png", dpi=180)
    plt.close(fig_i)

    fig_p, ax_p = plt.subplots(figsize=(8.8, 4.6))
    ax_p.plot(freq_khz, df["phase_deg"].values, color="#d62728", linewidth=2)
    ax_p.set_xscale("log")
    ax_p.set_xlabel("Frequency [kHz] (log scale)")
    ax_p.set_ylabel("Phase [deg]")
    ax_p.set_title("Phase vs Frequency")
    ax_p.grid(True, which="both", alpha=0.3)
    fig_p.tight_layout()
    fig_p.savefig(out_dir / f"{stem}_phase_logfreq.png", dpi=180)
    plt.close(fig_p)


def _value_at_khz(df: pd.DataFrame, khz: float, col: str) -> float:
    x = df["frequency_khz"].values
    y = df[col].values
    return float(np.interp(khz, x, y))


def make_feature_row(df: pd.DataFrame, stem: str, target_khz: Iterable[float]) -> dict[str, float | str]:
    row: dict[str, float | str] = {"sample": stem}
    for khz in target_khz:
        z_v = _value_at_khz(df, khz, "impedance_ohm")
        p_v = _value_at_khz(df, khz, "phase_deg")
        row[f"Z_{int(khz)}kHz"] = z_v
        row[f"phase_{int(khz)}kHz"] = p_v

    # Common dispersion-style features (closest analog to existing workflow)
    if {5.0, 100.0}.issubset(set(target_khz)):
        row["dphase_100_5"] = row["phase_100kHz"] - row["phase_5kHz"]
        row["Zratio_100_5"] = row["Z_100kHz"] / row["Z_5kHz"]

    x_log = np.log10(df["frequency_khz"].values)
    row["phase_slope_deg_per_dec"] = float(np.polyfit(x_log, df["phase_deg"].values, 1)[0])
    row["Z_slope_ohm_per_dec"] = float(np.polyfit(x_log, df["impedance_ohm"].values, 1)[0])
    return row


def process_one_file(input_file: Path, out_dir: Path, target_khz: Iterable[float]) -> dict[str, float | str]:
    df = parse_4294a_txt(input_file)
    df["frequency_khz"] = df["frequency_hz"] / 1000.0

    stem = input_file.stem
    csv_out = out_dir / f"{stem}_parsed.csv"
    df.to_csv(csv_out, index=False)
    make_plots(df, out_dir, stem)

    print(f"Saved: {csv_out}")
    print(f"Saved: {out_dir / (stem + '_impedance_phase_logfreq.png')}")
    print(f"Saved: {out_dir / (stem + '_impedance_logfreq.png')}")
    print(f"Saved: {out_dir / (stem + '_phase_logfreq.png')}")

    return make_feature_row(df, stem, target_khz)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Agilent 4294A TXT export and plot impedance/phase vs log-frequency."
    )
    parser.add_argument("--input", type=Path, help="Path to one 4294A TXT file")
    parser.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Glob pattern for multiple TXT files, e.g. 'Electrodes/labMeasurements/FN*.TXT'",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("phase_analysis_out"),
        help="Output directory for CSV and PNG files",
    )
    parser.add_argument(
        "--summary-khz",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 50.0, 100.0],
        help="Frequencies (kHz) to include in combined feature summary",
    )
    args = parser.parse_args()

    if args.input is None and args.input_glob is None:
        raise ValueError("Provide --input for one file or --input-glob for multiple files")

    args.outdir.mkdir(parents=True, exist_ok=True)

    input_files: list[Path] = []
    if args.input is not None:
        input_files.append(args.input)
    if args.input_glob is not None:
        input_files.extend(sorted(Path().glob(args.input_glob)))

    # De-duplicate while preserving order.
    seen = set()
    deduped = []
    for p in input_files:
        rp = str(p.resolve())
        if rp not in seen:
            deduped.append(p)
            seen.add(rp)

    if not deduped:
        raise ValueError("No input files matched")

    rows = []
    for input_file in deduped:
        print(f"\nProcessing: {input_file}")
        rows.append(process_one_file(input_file, args.outdir, args.summary_khz))

    summary_df = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    summary_out = args.outdir / "FN_sweep_feature_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"\nSaved: {summary_out}")
    print("\nFeature summary preview:")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
