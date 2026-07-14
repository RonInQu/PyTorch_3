"""RC 2-probe vs 4-probe demo.

Creates:
  - a simple circuit schematic for the measurement setup
  - |Z| vs frequency plots for 2-probe and 4-probe
  - phase vs frequency plots for 2-probe and 4-probe
  - a printed table at 5, 50, and 100 kHz

The example uses a low-impedance RC DUT so that the difference between
2-wire and 4-wire measurements is visually obvious.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


# ----------------------------------------------------------------------------
# Exact component choices used in the plots
# ----------------------------------------------------------------------------
R_DUT = 10.0  # ohms
C_DUT = 100e-9  # farads
R_LEAD_LEFT = 20.0  # ohms (2-probe extra series resistance)
R_LEAD_RIGHT = 20.0  # ohms (2-probe extra series resistance)

FREQS = np.logspace(3, 5, 400)  # 1 kHz to 100 kHz
REPORT_FREQS = np.array([5e3, 50e3, 100e3])

OUT_DIR = Path(__file__).resolve().parent / "rc_two_vs_four_probe_demo"
OUT_DIR.mkdir(exist_ok=True)


def z_series_rc(freq_hz: np.ndarray | float, r_ohm: float, c_f: float) -> np.ndarray:
    """Series RC impedance: Z = R - j/(omega C)."""
    omega = 2 * np.pi * np.asarray(freq_hz)
    return r_ohm - 1j / (omega * c_f)


def magnitude_phase(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.abs(z), np.degrees(np.angle(z))


def draw_resistor(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    n = 7
    xs = np.linspace(x0, x1, n * 2 + 1)
    ys = np.full_like(xs, y)
    amp = 0.03
    for i in range(1, len(xs) - 1):
        ys[i] = y + amp * (1 if i % 2 else -1)
    ax.plot(xs, ys, color="black", lw=2)


def draw_capacitor(ax: plt.Axes, xc: float, y: float) -> None:
    plate_gap = 0.015
    plate_h = 0.10
    ax.plot([xc - plate_gap, xc - plate_gap], [y - plate_h / 2, y + plate_h / 2], color="black", lw=2)
    ax.plot([xc + plate_gap, xc + plate_gap], [y - plate_h / 2, y + plate_h / 2], color="black", lw=2)


def draw_probe_node(ax: plt.Axes, x: float, y: float, label: str, color: str = "black") -> None:
    ax.add_patch(Circle((x, y), 0.007, color=color, zorder=5))
    ax.text(x, y + 0.04, label, ha="center", va="bottom", fontsize=9, color=color)


def draw_standard_circuit(ax: plt.Axes, four_probe: bool) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y = 0.45
    x_in, x_rlead_l0, x_rlead_l1 = 0.06, 0.14, 0.26
    x_rdut0, x_rdut1 = 0.34, 0.50
    x_c = 0.60
    x_rlead_r0, x_rlead_r1 = 0.70, 0.82
    x_out = 0.92

    # Left wire and lead resistance
    ax.plot([x_in, x_rlead_l0], [y, y], color="black", lw=2)
    draw_resistor(ax, x_rlead_l0, x_rlead_l1, y)
    ax.text((x_rlead_l0 + x_rlead_l1) / 2, y - 0.11, f"RleadL\n{R_LEAD_LEFT:.0f} Ω", ha="center", fontsize=9)

    # DUT resistor
    ax.plot([x_rlead_l1, x_rdut0], [y, y], color="black", lw=2)
    draw_resistor(ax, x_rdut0, x_rdut1, y)
    ax.text((x_rdut0 + x_rdut1) / 2, y - 0.11, f"R_DUT\n{R_DUT:.0f} Ω", ha="center", fontsize=9)

    # DUT capacitor
    ax.plot([x_rdut1, x_c - 0.03], [y, y], color="black", lw=2)
    draw_capacitor(ax, x_c, y)
    ax.plot([x_c + 0.03, x_rlead_r0], [y, y], color="black", lw=2)
    ax.text(x_c, y - 0.11, f"C_DUT\n{C_DUT*1e9:.0f} nF", ha="center", fontsize=9)

    # Right lead resistance and wire
    draw_resistor(ax, x_rlead_r0, x_rlead_r1, y)
    ax.text((x_rlead_r0 + x_rlead_r1) / 2, y - 0.11, f"RleadR\n{R_LEAD_RIGHT:.0f} Ω", ha="center", fontsize=9)
    ax.plot([x_rlead_r1, x_out], [y, y], color="black", lw=2)

    if four_probe:
        ax.set_title("4-probe (Kelvin): I on outer leads, V on inner taps")
        draw_probe_node(ax, x_in, y, "I+")
        draw_probe_node(ax, x_out, y, "I-")

        # Sense taps directly across DUT only (between start of R_DUT and end of C_DUT)
        x_vp, x_vm = x_rdut0, x_c + 0.03
        draw_probe_node(ax, x_vp, y, "V+", color="tab:blue")
        draw_probe_node(ax, x_vm, y, "V-", color="tab:blue")
        ax.plot([x_vp, x_vp], [y, 0.78], color="tab:blue", lw=1.8)
        ax.plot([x_vm, x_vm], [y, 0.78], color="tab:blue", lw=1.8)
        ax.plot([x_vp, x_vm], [0.78, 0.78], color="tab:blue", lw=1.8)
        ax.text((x_vp + x_vm) / 2, 0.82, "High-Z voltmeter", ha="center", color="tab:blue", fontsize=9)
        ax.text(0.5, 0.92, "Measured: Z_4probe = R_DUT - j/(ωC_DUT)", ha="center", fontsize=10)
    else:
        ax.set_title("2-probe: current and voltage on same leads")
        draw_probe_node(ax, x_in, y, "I+, V+")
        draw_probe_node(ax, x_out, y, "I-, V-")
        ax.text(0.5, 0.92, "Measured: Z_2probe = (RleadL + R_DUT + RleadR) - j/(ωC_DUT)", ha="center", fontsize=10)


def print_table(freqs_hz: np.ndarray, z_2: np.ndarray, z_4: np.ndarray) -> None:
    print("\nExact components used")
    print(f"  DUT: R = {R_DUT:.1f} ohm, C = {C_DUT*1e9:.0f} nF")
    print(f"  2-probe extra series resistance: {R_LEAD_LEFT:.1f} ohm + {R_LEAD_RIGHT:.1f} ohm")
    print("\nExpected impedance values")
    print(f"{'Freq':>8} | {'|Z| 2-probe':>12} {'Phase 2-probe':>14} | {'|Z| 4-probe':>12} {'Phase 4-probe':>14}")
    print("-" * 70)
    for f in freqs_hz:
        z2 = z_series_rc(f, R_DUT + R_LEAD_LEFT + R_LEAD_RIGHT, C_DUT)
        z4 = z_series_rc(f, R_DUT, C_DUT)
        mag2, ph2 = magnitude_phase(z2)
        mag4, ph4 = magnitude_phase(z4)
        print(f"{f/1e3:7.0f} kHz | {mag2:12.2f} {ph2:14.2f} | {mag4:12.2f} {ph4:14.2f}")


def main() -> None:
    z_2 = z_series_rc(FREQS, R_DUT + R_LEAD_LEFT + R_LEAD_RIGHT, C_DUT)
    z_4 = z_series_rc(FREQS, R_DUT, C_DUT)
    mag_2, phase_2 = magnitude_phase(z_2)
    mag_4, phase_4 = magnitude_phase(z_4)

    # Schematic figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    draw_standard_circuit(axes[0], four_probe=False)
    draw_standard_circuit(axes[1], four_probe=True)
    fig.suptitle("Standard RC Measurement Diagrams", fontsize=15)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05, hspace=0.30)
    fig_path = OUT_DIR / "rc_probe_schematic.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # Magnitude plot (full scale + zoom so separation is obvious)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.semilogx(FREQS, mag_2, lw=2.6, color="tab:orange", label="2-probe")
    ax1.semilogx(FREQS, mag_4, lw=2.6, color="tab:blue", label="4-probe")
    ax1.set_ylabel(r"|Z| [$\Omega$]")
    ax1.set_title("Magnitude vs Frequency (1 kHz to 100 kHz)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    zoom = FREQS >= 1e4
    ax2.semilogx(FREQS[zoom], mag_2[zoom], lw=2.6, color="tab:orange", label="2-probe")
    ax2.semilogx(FREQS[zoom], mag_4[zoom], lw=2.6, color="tab:blue", label="4-probe")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel(r"|Z| [$\Omega$]")
    ax2.set_title("Zoomed: 10 kHz to 100 kHz")
    ax2.grid(True, which="both", alpha=0.3)
    mag_path = OUT_DIR / "rc_probe_magnitude.png"
    fig.tight_layout()
    fig.savefig(mag_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Phase plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(FREQS, phase_2, lw=2.5, color="tab:orange", label="2-probe")
    ax.semilogx(FREQS, phase_4, lw=2.5, color="tab:blue", label="4-probe")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase [deg]")
    ax.set_title("Phase vs Frequency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    phase_path = OUT_DIR / "rc_probe_phase.png"
    fig.tight_layout()
    fig.savefig(phase_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved schematic: {fig_path}")
    print(f"Saved magnitude plot: {mag_path}")
    print(f"Saved phase plot: {phase_path}")
    print_table(REPORT_FREQS, z_2, z_4)


if __name__ == "__main__":
    main()