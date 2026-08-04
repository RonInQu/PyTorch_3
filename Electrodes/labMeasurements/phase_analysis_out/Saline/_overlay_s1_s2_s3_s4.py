import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).parent


def parse_txt(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"TRACE:\s*([AB])", text)
    out = {}
    for i in range(1, len(blocks), 2):
        label = blocks[i].strip()
        body = blocks[i + 1]
        rows = []
        in_data = False
        for line in body.splitlines():
            line = line.strip()
            if line.lower().startswith("frequency"):
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.replace("\t", " ").split()
            if len(parts) < 2:
                continue
            try:
                f = float(parts[0])
                v = float(parts[1])
                rows.append((f, v))
            except ValueError:
                continue
        if rows:
            out[label] = np.array(rows)
    return out


runs = ["S1", "S2", "S3", "S4"]
colors = {
    "S1": "#1f77b4",
    "S2": "#ff7f0e",
    "S3": "#2ca02c",
    "S4": "#9467bd",
}
styles = {
    "S1": "-",
    "S2": "--",
    "S3": "-",
    "S4": "--",
}

freq_khz = None
z_data = {}
p_data = {}

for run in runs:
    data = parse_txt(base / f"{run}.TXT")
    fk = data["A"][:, 0] / 1e3
    z = data["A"][:, 1]
    p = data["B"][:, 1]
    if freq_khz is None:
        freq_khz = fk
    z_data[run] = z
    p_data[run] = p

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.8), sharex=True)
fig.suptitle("HP 4294A - Saline Repeatability: S1, S2, S3, S4", fontsize=15, fontweight="bold")

for run in runs:
    ax1.plot(freq_khz, z_data[run], color=colors[run], linestyle=styles[run], linewidth=2.1, label=run)
ax1.set_xscale("log")
ax1.set_ylabel("Impedance |Z| [ohm]", fontsize=11)
ax1.set_title("Impedance Magnitude", fontsize=12)
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(ncol=4, fontsize=9)

for run in runs:
    ax2.plot(freq_khz, p_data[run], color=colors[run], linestyle=styles[run], linewidth=2.1, label=run)
ax2.set_xscale("log")
ax2.set_xlabel("Frequency [kHz]", fontsize=11)
ax2.set_ylabel("Phase [deg]", fontsize=11)
ax2.set_title("Phase Angle", fontsize=12)
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(ncol=4, fontsize=9)

fig.tight_layout()
out = base / "S1_S2_S3_S4_overlay.png"
fig.savefig(out, dpi=180)
print(f"Saved: {out}")

for fk in [1, 10, 50, 100]:
    z_vals = [float(np.interp(fk, freq_khz, z_data[r])) for r in runs]
    p_vals = [float(np.interp(fk, freq_khz, p_data[r])) for r in runs]
    print(
        f"{fk:>3} kHz | Z range={max(z_vals)-min(z_vals):.1f} ohm | "
        f"phase range={max(p_vals)-min(p_vals):.3f} deg"
    )
