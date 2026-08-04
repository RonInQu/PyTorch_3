import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).parent


def parse_txt(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r'TRACE:\s*([AB])', text)
    result = {}
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
            parts = line.replace('\t', ' ').split()
            if len(parts) < 2:
                continue
            try:
                f = float(parts[0])
                v = float(parts[1])
                rows.append((f, v))
            except ValueError:
                pass
        if rows:
            result[label] = np.array(rows)
    return result


d1 = parse_txt(base / "S1.TXT")
d2 = parse_txt(base / "S2.TXT")

freq1 = d1['A'][:, 0] / 1e3
z1 = d1['A'][:, 1]
ph1 = d1['B'][:, 1]

freq2 = d2['A'][:, 0] / 1e3
z2 = d2['A'][:, 1]
ph2 = d2['B'][:, 1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
fig.suptitle("HP 4294A — Saline Run Comparison: S1 vs S2", fontsize=13, fontweight='bold')

# Impedance magnitude
ax1.plot(freq1, z1, color='#1f77b4', linewidth=2.0, label='S1')
ax1.plot(freq2, z2, color='#ff7f0e', linewidth=2.0, linestyle='--', label='S2')
ax1.set_xscale('log')
ax1.set_ylabel('Impedance |Z| [Ω]', fontsize=11)
ax1.set_title('Impedance Magnitude', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, which='both', alpha=0.3)
ax1.set_ylim(180, 200)

for fk in [1, 10, 50, 100]:
    z1v = float(np.interp(fk, freq1, z1))
    z2v = float(np.interp(fk, freq2, z2))
    ax1.annotate(f'Δ={z1v - z2v:+.1f} Ω',
                 xy=(fk, (z1v + z2v) / 2),
                 xytext=(fk * 1.15, (z1v + z2v) / 2 + 0.4),
                 fontsize=7.5, color='#555555')

# Phase
ax2.plot(freq1, ph1, color='#1f77b4', linewidth=2.0, label='S1')
ax2.plot(freq2, ph2, color='#ff7f0e', linewidth=2.0, linestyle='--', label='S2')
ax2.set_xscale('log')
ax2.set_xlabel('Frequency [kHz]', fontsize=11)
ax2.set_ylabel('Phase [°]', fontsize=11)
ax2.set_title('Phase Angle', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, which='both', alpha=0.3)

for fk in [1, 10, 50, 100]:
    p1v = float(np.interp(fk, freq1, ph1))
    p2v = float(np.interp(fk, freq2, ph2))
    ax2.annotate(f'Δ={p1v - p2v:+.3f}°',
                 xy=(fk, (p1v + p2v) / 2),
                 xytext=(fk * 1.15, (p1v + p2v) / 2 - 0.06),
                 fontsize=7.5, color='#555555')

fig.tight_layout()
out = base / "S1_S2_overlay.png"
fig.savefig(out, dpi=180)
plt.show()
print(f"Saved: {out}")
