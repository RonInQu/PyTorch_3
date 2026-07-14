#!/usr/bin/env python3
"""
Clean, professional schematic generator:
Tetrapolar vs Bipolar bioimpedance electrode configurations
for catheter probe (lab prototype with stainless steel electrodes).

Generates high-quality PNG and PDF for presentations/docs.
Run: python tetrapolar_schematic.py
Requires: matplotlib, numpy (optional)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Ellipse
from matplotlib.lines import Line2D
import numpy as np

# Professional styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def draw_electrode_ring(ax, x_center, y_center, width, height, color, label, label_offset=0.0):
    rect = FancyBboxPatch((x_center - width/2, y_center - height/2),
                          width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=color, edgecolor='black', linewidth=1.2, zorder=5)
    ax.add_patch(rect)
    ax.text(x_center, y_center + label_offset, label, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=6)

def create_schematic():
    fig = plt.figure(figsize=(16, 9))
    
    ax_bip = fig.add_axes([0.05, 0.35, 0.42, 0.55])
    ax_tetra = fig.add_axes([0.52, 0.35, 0.42, 0.55])
    
    fig.suptitle('Tetrapolar vs Bipolar Bioimpedance Probe\nCatheter Lab Prototype (Stainless Steel Electrodes for Validation)', 
                 fontsize=14, fontweight='bold', y=0.97)
    
    # BIPOLAR (Left)
    ax = ax_bip
    ax.set_xlim(-1, 12)
    ax.set_ylim(-3.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('BIPOLAR (2-Electrode) — Lab Comparison Baseline', fontsize=11, fontweight='bold', pad=10)
    
    catheter = FancyBboxPatch((0, -1.2), 10, 2.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                              facecolor='#E8E8E8', edgecolor='#555555', linewidth=2, zorder=1)
    ax.add_patch(catheter)
    
    ax.fill_between([0,10], [-2.8,-2.8], [-1.3,-1.3], color='#FFCCCC', alpha=0.3, zorder=0)
    ax.text(5, -2.0, 'Tissue / Saline Phantom (SS electrodes in contact)', ha='center', fontsize=8, style='italic', color='#AA0000')
    
    draw_electrode_ring(ax, 3.0, 0, 1.2, 2.0, '#1f77b4', 'E1\n(I+/V+)', label_offset=1.6)
    draw_electrode_ring(ax, 7.0, 0, 1.2, 2.0, '#1f77b4', 'E2\n(I-/V-)', label_offset=1.6)
    
    ax.annotate('', xy=(3.0, 2.8), xytext=(3.0, 3.8),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5))
    ax.text(3.0, 4.1, 'I_in', ha='center', fontsize=9, color='#d62728', fontweight='bold')
    
    ax.annotate('', xy=(7.0, -2.8), xytext=(7.0, -3.8),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5))
    ax.text(7.0, -4.1, 'I_out', ha='center', fontsize=9, color='#d62728', fontweight='bold')
    
    ax.annotate('', xy=(2.3, 0), xytext=(3.7, 0),
                arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=1.5, ls='--'))
    ax.text(3.0, 0.6, 'V_sense', ha='center', fontsize=8, color='#2ca02c')
    
    ax.annotate('', xy=(3.0, -1.5), xytext=(7.0, -1.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(5.0, -1.8, 'Electrode spacing\n(typical 5–10 mm)', ha='center', fontsize=7)
    
    formula_bip = r'$Z_{bipolar} \approx Z_{bulk} + Z_{contact1} + Z_{contact2}$' + '\n' + r'$Z_{contact} \approx (Z_{bipolar} - Z_{tetra})/2$  (estimate)'
    ax.text(5, -3.2, formula_bip, ha='center', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#FFFACD', edgecolor='#666666', alpha=0.9))
    
    legend_elements_bip = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4', markersize=10, label='SS Electrode (E1/E2)'),
        Line2D([0], [0], color='#d62728', lw=2, label='Current path (I)'),
        Line2D([0], [0], color='#2ca02c', lw=1.5, linestyle='--', label='Voltage sense (V)')
    ]
    ax.legend(handles=legend_elements_bip, loc='lower right', fontsize=7, framealpha=0.9)
    
    # TETRAPOLAR (Right)
    ax = ax_tetra
    ax.set_xlim(-1, 12)
    ax.set_ylim(-3.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('TETRAPOLAR (4-Electrode) — Recommended for TrueClot™ EIS', fontsize=11, fontweight='bold', pad=10)
    
    catheter = FancyBboxPatch((0, -1.2), 10, 2.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                              facecolor='#E8E8E8', edgecolor='#555555', linewidth=2, zorder=1)
    ax.add_patch(catheter)
    
    ax.fill_between([0,10], [-2.8,-2.8], [-1.3,-1.3], color='#FFCCCC', alpha=0.3, zorder=0)
    ax.text(5, -2.0, 'Tissue / Saline Phantom — Contact impedance largely rejected', ha='center', fontsize=8, style='italic', color='#006600')
    
    draw_electrode_ring(ax, 1.5, 0, 0.9, 1.8, '#1f77b4', 'I+', label_offset=1.5)
    draw_electrode_ring(ax, 3.5, 0, 0.7, 1.6, '#2ca02c', 'V+', label_offset=1.4)
    draw_electrode_ring(ax, 6.5, 0, 0.7, 1.6, '#2ca02c', 'V-', label_offset=1.4)
    draw_electrode_ring(ax, 8.5, 0, 0.9, 1.8, '#1f77b4', 'I-', label_offset=1.5)
    
    ax.annotate('', xy=(1.5, 2.6), xytext=(1.5, 3.6),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5))
    ax.text(1.5, 3.9, 'I_in', ha='center', fontsize=9, color='#d62728', fontweight='bold')
    
    ax.plot([1.5, 8.5], [3.2, 3.2], color='#d62728', lw=1.5, linestyle=':', alpha=0.7)
    ax.text(5, 3.4, 'Current injection path (outer pair)', ha='center', fontsize=7, color='#d62728')
    
    ax.annotate('', xy=(8.5, -2.6), xytext=(8.5, -3.6),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5))
    ax.text(8.5, -3.9, 'I_out', ha='center', fontsize=9, color='#d62728', fontweight='bold')
    
    ax.annotate('', xy=(3.2, 0), xytext=(6.8, 0),
                arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=2.0))
    ax.text(5.0, 0.7, 'V_sense (high-Z, ~0 current)', ha='center', fontsize=8, color='#2ca02c', fontweight='bold')
    
    ax.annotate('', xy=(1.5, -1.5), xytext=(3.5, -1.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(2.5, -1.75, '2 mm', ha='center', fontsize=7)
    
    ax.annotate('', xy=(3.5, -1.5), xytext=(6.5, -1.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(5.0, -1.75, '3.5 mm (key sensing gap)', ha='center', fontsize=7, fontweight='bold', color='#006600')
    
    ax.annotate('', xy=(6.5, -1.5), xytext=(8.5, -1.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(7.5, -1.75, '2 mm', ha='center', fontsize=7)
    
    advantage_text = (
        'KEY ADVANTAGE:\n'
        r'$Z_{tetrapolar} \approx Z_{bulk\ tissue/fluid}$' + '\n'
        'Contact & polarization impedance\n'
        'of electrodes is largely rejected\n'
        '(voltage sense draws negligible current)'
    )
    ax.text(5, -3.0, advantage_text, ha='center', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#E6FFE6', edgecolor='#228B22', alpha=0.95),
            fontweight='bold')
    
    legend_elements_tetra = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4', markersize=10, label='Current electrodes (I+, I-) — SS'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c', markersize=10, label='Voltage sense (V+, V-) — SS'),
        Line2D([0], [0], color='#d62728', lw=2, label='Current path'),
        Line2D([0], [0], color='#2ca02c', lw=2, label='Voltage measurement')
    ]
    ax.legend(handles=legend_elements_tetra, loc='lower right', fontsize=7, framealpha=0.9)
    
    # Bottom comparison section
    ax_comp = fig.add_axes([0.05, 0.05, 0.9, 0.25])
    ax_comp.axis('off')
    
    comparison_text = (
        'COMPARISON & CONTACT IMPEDANCE CALCULATION (Stainless Steel Lab Electrodes)\n\n'
        'Bipolar: Simple (only 2 electrodes/wires) but Z_measured includes large, frequency-dependent electrode-tissue interface impedance '
        '(double-layer capacitance + charge transfer resistance). This varies with surface condition, current density, and time — bad for reliable EIS tissue differentiation.\n\n'
        'Tetrapolar: 4 electrodes/wires required. Current injected via outer pair; voltage sensed via inner pair with high-impedance amplifier '
        '(virtually zero current through voltage electrodes). Therefore Z_measured ≈ true bulk impedance of the volume between V+ and V−. '
        'Contact impedance is rejected to first order. This is essential for accurate multi-frequency (5–150 kHz) discrimination of blood vs clot vs wall.\n\n'
        'Practical lab estimation of contact impedance (same phantom, same spacing, same current amplitude):\n'
        '    Z_contact_total ≈ Z_bipolar − Z_tetrapolar     (or per electrode ≈ (Z_bipolar − Z_tetrapolar)/2 assuming symmetry)\n'
        'Use complex values (magnitude + phase). For SS electrodes in saline at 100 kHz, typical contact Z can be 10–200 Ω depending on area and surface prep; '
        'bulk blood segment Z is often comparable, so tetrapolar is mandatory for clean data. Polish SS to mirror finish + ultrasonic clean in IPA for best results.'
    )
    
    ax_comp.text(0.5, 0.95, comparison_text, transform=ax_comp.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#333333', alpha=0.95))
    
    fig.text(0.5, 0.01, 
             'Lab prototype note: Use 316L or 304 stainless steel rings/hypodermic tubing segments on PTFE or Pebax mock catheter. '
             'Scale dimensions as needed for bench (full 24 Fr or smaller test article). Final device uses PtIr. '
             'Script generates vector-quality output for docs/presentations.',
             ha='center', fontsize=7, style='italic', color='#444444')
    
    plt.savefig('/home/workdir/artifacts/lab_setup/tetrapolar_vs_bipolar_schematic.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/workdir/artifacts/lab_setup/tetrapolar_vs_bipolar_schematic.pdf', dpi=300, bbox_inches='tight')
    print("Schematic saved to /home/workdir/artifacts/lab_setup/ (PNG + PDF)")

if __name__ == "__main__":
    create_schematic()