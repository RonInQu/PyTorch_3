from __future__ import annotations

import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Agilent 4294A TXT export and plot impedance/phase vs log-frequency."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to 4294A TXT file")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("phase_analysis_out"),
        help="Output directory for CSV and PNG files",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = parse_4294a_txt(args.input)

    stem = args.input.stem
    csv_out = args.outdir / f"{stem}_parsed.csv"
    df["frequency_khz"] = df["frequency_hz"] / 1000.0
    df.to_csv(csv_out, index=False)

    make_plots(df, args.outdir, stem)

    print(f"Saved: {csv_out}")
    print(f"Saved: {args.outdir / (stem + '_impedance_phase_logfreq.png')}")
    print(f"Saved: {args.outdir / (stem + '_impedance_logfreq.png')}")
    print(f"Saved: {args.outdir / (stem + '_phase_logfreq.png')}")
    print("\nData preview:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
