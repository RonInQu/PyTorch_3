import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------------
# 1. PHYSICAL CONSTANTS & BASELINE BVD PARAMETERS
# -------------------------------------------------------------------------
C0 = 5.0e-12      # Shunt capacitance (5 pF)
Lm_air = 15.0e-3  # Motional inductance in air (15 mH)
nominal_frequencies = [5e6, 10e6, 20e6]  # 5, 10, 20 MHz crystals
output_dir = Path(__file__).resolve().parent

# -------------------------------------------------------------------------
# 2. DEFINE THE SIMULATION SCENARIOS (blood, clot, tissue only)
# -------------------------------------------------------------------------
scenarios = {
    "Blood (Viscous Liquid)": {
        "Rm": 150.0,
        "Lm": Lm_air + 3.0e-6,
        "color": "blue",
        "linestyle": "-"
    },
    "Blood Clot (Viscoelastic Gel)": {
        "Rm": 650.0,
        "Lm": Lm_air + 15.0e-6,
        "color": "crimson",
        "linestyle": "-"
    },
    "Vessel Wall (Elastic Contact)": {
        "Rm": 250.0,
        "Lm": Lm_air + 60.0e-6,
        "color": "darkgreen",
        "linestyle": "-"
    }
}

def run_single_frequency_case(f_nominal: float):
    Cm = 1.0 / ((2 * np.pi * f_nominal) ** 2 * Lm_air)

    span = 0.002 * f_nominal
    freqs = np.linspace(f_nominal - span, f_nominal + span, 2000)
    omega = 2 * np.pi * freqs

    results = {}

    for name, params in scenarios.items():
        Rm = params["Rm"]
        Lm = params["Lm"]

        Zm = Rm + 1j * (omega * Lm - 1.0 / (omega * Cm))
        Y0 = 1j * omega * C0
        Y_tot = (1.0 / Zm) + Y0

        G = np.real(Y_tot)
        Z_tot = 1.0 / Y_tot
        phase = np.angle(Z_tot, deg=True)

        idx_max_G = np.argmax(G)
        fr = freqs[idx_max_G]
        peak_G_mS = G[idx_max_G] * 1000

        half_max = G[idx_max_G] / 2.0
        indices_above_half = np.where(G >= half_max)[0]
        if len(indices_above_half) > 1:
            f_low = freqs[indices_above_half[0]]
            f_high = freqs[indices_above_half[-1]]
            fwhm = f_high - f_low
            Q = fr / fwhm if fwhm > 0 else np.nan
        else:
            Q = np.nan

        results[name] = {
            "freqs": freqs,
            "G_mS": G * 1000,
            "phase": phase,
            "fr": fr,
            "peak_G": peak_G_mS,
            "Q": Q,
            **params
        }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, data in results.items():
        ax1.plot(data["freqs"] / 1e6, data["G_mS"],
                 label=name, color=data["color"],
                 linestyle=data["linestyle"], linewidth=2)
        ax2.plot(data["freqs"] / 1e6, data["phase"],
                 color=data["color"], linestyle=data["linestyle"], linewidth=2)

    f_mhz = f_nominal / 1e6
    ax1.set_ylabel("Conductance G (mS)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Simulated {f_mhz:.0f} MHz QCM — Blood / Clot / Tissue",
        fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper right")

    ax2.set_xlabel("Frequency (MHz)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Phase Angle (Degrees)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    f_mhz_int = int(round(f_mhz))
    out_path = output_dir / f"qcm2_response_{f_mhz_int}MHz.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_path}")

    return results


# -------------------------------------------------------------------------
# 3. RUN ALL CRYSTAL FREQUENCIES AND PRINT ANALYSIS REPORTS
# -------------------------------------------------------------------------
all_results = {}
for f_nominal in nominal_frequencies:
    all_results[f_nominal] = run_single_frequency_case(f_nominal)

for f_nominal in nominal_frequencies:
    results = all_results[f_nominal]
    print("=" * 82)
    print(f"Nominal Crystal: {f_nominal/1e6:.0f} MHz")
    print(f"{'QCM STATE':<30} | {'RESONANCE (Hz)':<15} | {'PEAK G (mS)':<12} | {'Q-FACTOR':<10}")
    print("=" * 82)
    for name, data in results.items():
        print(f"{name:<30} | {data['fr']:<15,.1f} | {data['peak_G']:<12.2f} | {data['Q']:<10.1f}")
    print("=" * 82)
