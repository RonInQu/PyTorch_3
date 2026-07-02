"""
Current Pathway Schematic: Recessed vs Flush Electrodes
Shows why flush electrodes lose wall discrimination but retain clot discrimination.

Cross-section view: circular vessel with catheter inside.
- Clot scenario: clot fills entire lumen (no blood anywhere)
- Wall scenario: catheter pushed up to touch vessel wall; blood fills crescent below
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from pathlib import Path

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Current Pathways: Recessed vs Flush Electrodes\n(Vessel cross-section view)',
             fontsize=14, fontweight='bold')

# Colors
c_blood = '#cc3333'
c_clot = '#8B4513'
c_wall = '#4a7c4a'
c_cath = '#888888'
c_electrode = '#FFD700'
c_current = '#0066ff'
c_current_weak = '#99ccff'

# Real dimensions (mm) — drawing in mm scale
r_vessel = 8.0        # vessel inner radius
wall_thick = 1.0      # vessel wall thickness
r_cath = 4.0          # catheter radius
recess_depth = 0.5    # mm
elec_angular_half = 12  # degrees half-width of electrode on catheter surface
elec_height = 0.15    # mm (thin pad)

# Electrode angular positions (L and R are on opposite x-sides, same z)
# In this cross-section we see the axial (z) view looking end-on.
# Electrodes are at ~±30° from top of catheter
elec_L_angle = 150  # degrees from +x axis (upper-left)
elec_R_angle = 30   # degrees from +x axis (upper-right)


def draw_vessel_cross_section(ax, title, recessed=True, tissue='clot'):
    """Draw vessel cross-section with catheter inside.
    
    Geometry (cross-section, looking along catheter axis):
    - Outer circle: vessel wall (green ring)
    - Inner circle boundary: vessel lumen
    - Catheter circle: smaller, inside vessel
    - Clot: fills lumen entirely
    - Wall: catheter pushed to top, touching wall. Blood in crescent below.
    """
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Vessel wall (outer ring)
    vessel_outer = plt.Circle((0, 0), r_vessel + wall_thick, 
                               facecolor=c_wall, edgecolor='black', lw=2, alpha=0.6, zorder=1)
    ax.add_patch(vessel_outer)
    
    # Vessel lumen boundary
    vessel_inner = plt.Circle((0, 0), r_vessel, 
                               facecolor='white', edgecolor='black', lw=1.5, zorder=2)
    ax.add_patch(vessel_inner)
    
    # Catheter position
    if tissue == 'wall':
        # Catheter pushed up to touch top wall
        cath_cy = r_vessel - r_cath  # catheter center y so top touches vessel inner wall
    else:
        # Catheter centered (or slightly off-center — centered for clot)
        cath_cy = 0.0
    cath_cx = 0.0
    
    # Fill lumen with appropriate tissue BEFORE drawing catheter
    if tissue == 'clot':
        # Clot fills entire lumen
        clot_fill = plt.Circle((0, 0), r_vessel - 0.05, 
                                facecolor=c_clot, edgecolor='none', alpha=0.5, zorder=3)
        ax.add_patch(clot_fill)
        ax.text(0, -r_cath - 2.5, 'CLOT fills\nentire lumen\n(σ=0.20 S/m)', 
                ha='center', va='center', fontsize=8, color='saddlebrown', fontweight='bold')
        
    elif tissue == 'wall':
        # Blood fills the crescent-shaped lumen below the catheter
        blood_fill = plt.Circle((0, 0), r_vessel - 0.05, 
                                 facecolor=c_blood, edgecolor='none', alpha=0.35, zorder=3)
        ax.add_patch(blood_fill)
        ax.text(0, -5.5, 'BLOOD\n(lumen crescent)\nσ=0.88 S/m', 
                ha='center', va='center', fontsize=8, color='darkred', fontweight='bold')
    
    # Catheter body
    catheter = plt.Circle((cath_cx, cath_cy), r_cath, 
                           facecolor=c_cath, edgecolor='black', lw=2, zorder=5)
    ax.add_patch(catheter)
    ax.text(cath_cx, cath_cy - 0.8, 'Catheter', ha='center', va='center',
            fontsize=8, color='white', fontweight='bold', zorder=6)
    
    # Draw electrodes on catheter surface
    # L electrode: upper-left, R electrode: upper-right
    elec_half = elec_angular_half
    rec_d = recess_depth if recessed else 0.0
    
    for label, angle_center in [('L', elec_L_angle), ('R', elec_R_angle)]:
        # Electrode surface position
        a_rad = np.radians(angle_center)
        
        # Draw electrode as a thick arc on catheter surface
        # For recessed: electrode sits at r_cath - recess_depth
        # For flush: electrode sits at r_cath
        elec_r = r_cath - rec_d
        
        # Draw electrode arc (series of small patches to approximate arc)
        n_seg = 10
        angles = np.linspace(angle_center - elec_half, angle_center + elec_half, n_seg + 1)
        for i in range(n_seg):
            a1 = np.radians(angles[i])
            a2 = np.radians(angles[i + 1])
            # Inner and outer radius of electrode
            r_in = elec_r
            r_out = elec_r + elec_height
            xs = [cath_cx + r_in * np.cos(a1), cath_cx + r_out * np.cos(a1),
                  cath_cx + r_out * np.cos(a2), cath_cx + r_in * np.cos(a2)]
            ys = [cath_cy + r_in * np.sin(a1), cath_cy + r_out * np.sin(a1),
                  cath_cy + r_out * np.sin(a2), cath_cy + r_in * np.sin(a2)]
            ax.fill(xs, ys, color=c_electrode, edgecolor='black', lw=0.3, zorder=8)
        
        # Electrode label
        ex = cath_cx + (elec_r + 0.5) * np.cos(a_rad)
        ey = cath_cy + (elec_r + 0.5) * np.sin(a_rad)
        ax.text(ex, ey, label, ha='center', va='center', fontsize=9, 
                fontweight='bold', zorder=9)
        
        # Draw recess walls (insulating sides of the pocket)
        if recessed:
            for edge_angle in [angle_center - elec_half, angle_center + elec_half]:
                ea = np.radians(edge_angle)
                x1 = cath_cx + (r_cath - rec_d) * np.cos(ea)
                y1 = cath_cy + (r_cath - rec_d) * np.sin(ea)
                x2 = cath_cx + r_cath * np.cos(ea)
                y2 = cath_cy + r_cath * np.sin(ea)
                ax.plot([x1, x2], [y1, y2], color='black', lw=1.5, zorder=8)
    
    # Electrode tip positions for drawing current arrows
    elec_L_rad = np.radians(elec_L_angle)
    elec_R_rad = np.radians(elec_R_angle)
    arrow_r = r_cath + elec_height + 0.2 if not recessed else r_cath + 0.2
    
    Lx = cath_cx + arrow_r * np.cos(elec_L_rad)
    Ly = cath_cy + arrow_r * np.sin(elec_L_rad)
    Rx = cath_cx + arrow_r * np.cos(elec_R_rad)
    Ry = cath_cy + arrow_r * np.sin(elec_R_rad)
    
    # ---- CURRENT PATHWAYS ----
    
    if tissue == 'clot':
        # Clot fills entire lumen — no blood escape path anywhere.
        # All current from L to R goes through clot (high impedance).
        # Same result for recessed and flush.
        for rad_val in [0.25, 0.45, 0.65]:
            arrow = FancyArrowPatch((Lx, Ly), (Rx, Ry),
                                    connectionstyle=f'arc3,rad=-{rad_val}',
                                    arrowstyle='->', lw=2.0, color=c_current,
                                    mutation_scale=12, zorder=10)
            ax.add_patch(arrow)
        
        ax.text(0, r_vessel + 2.0, 'Z ≈ 3500 Ω', ha='center', fontsize=11, fontweight='bold',
                color='darkblue', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        desc = 'Clot fills lumen → all current through clot'
        if recessed:
            desc += '\n(recess has no effect — clot everywhere)'
        ax.text(0, r_vessel + 0.8, desc, ha='center', fontsize=7, color='navy')
        
    elif tissue == 'wall' and recessed:
        # Catheter against wall. Electrodes recessed.
        # Recess walls block lateral current spread along catheter surface.
        # Current forced radially outward through wall tissue.
        # Some current penetrates wall to blood beyond, but wall dominates.
        
        # Main paths through wall (tight arcs staying in wall region)
        for rad_val in [0.1, 0.18]:
            arrow = FancyArrowPatch((Lx, Ly), (Rx, Ry),
                                    connectionstyle=f'arc3,rad=-{rad_val}',
                                    arrowstyle='->', lw=2.5, color=c_current,
                                    mutation_scale=12, zorder=10)
            ax.add_patch(arrow)
        
        # Weak path leaking through wall into blood beyond
        arrow_leak = FancyArrowPatch((Lx, Ly), (Rx, Ry),
                                      connectionstyle='arc3,rad=-0.4',
                                      arrowstyle='->', lw=1.2, color=c_current_weak,
                                      mutation_scale=10, linestyle='dashed', zorder=10)
        ax.add_patch(arrow_leak)
        
        ax.text(0, r_vessel + 2.0, 'Z ≈ 1800 Ω', ha='center', fontsize=11, fontweight='bold',
                color='darkblue', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.text(0, r_vessel + 0.8, 'Recess blocks lateral spread\n→ current forced through wall',
                ha='center', fontsize=7, color='navy')
        
        # Label wall at contact
        ax.annotate('Wall (1mm)', xy=(0, r_vessel), 
                    xytext=(5, r_vessel + 2), fontsize=7, color=c_wall,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=c_wall, lw=1))
        
    elif tissue == 'wall' and not recessed:
        # Catheter against wall. Electrodes flush (no recess).
        # Blood is in the crescent-shaped lumen BELOW the catheter.
        # Without recess walls, current can spread along catheter surface
        # and down into the blood-filled lumen — path of least resistance.
        # Wall is above (1mm thick, sensing depth ≈ 1mm) — poor penetration.
        
        # Current path: L → along catheter surface → down into blood crescent → 
        #               around through blood → back up to R
        # This is the LOW impedance shunt path
        
        # Arrow from L electrode going down along catheter surface
        Lx_down = cath_cx + arrow_r * np.cos(np.radians(elec_L_angle - 40))
        Ly_down = cath_cy + arrow_r * np.sin(np.radians(elec_L_angle - 40))
        arrow_L_down = FancyArrowPatch((Lx, Ly), (Lx_down, Ly_down),
                                        arrowstyle='->', lw=2.5, color=c_current,
                                        connectionstyle='arc3,rad=-0.1',
                                        mutation_scale=12, zorder=10)
        ax.add_patch(arrow_L_down)
        
        # Arrow through blood crescent (below catheter)
        blood_path_y = cath_cy - r_cath - 1.0
        arrow_blood = FancyArrowPatch((Lx_down, Ly_down),
                                       (cath_cx + arrow_r * np.cos(np.radians(elec_R_angle + 40)),
                                        cath_cy + arrow_r * np.sin(np.radians(elec_R_angle + 40))),
                                       connectionstyle='arc3,rad=0.6',
                                       arrowstyle='->', lw=3.0, color='red',
                                       mutation_scale=15, zorder=10)
        ax.add_patch(arrow_blood)
        
        # Arrow from blood back up to R electrode
        Rx_down = cath_cx + arrow_r * np.cos(np.radians(elec_R_angle + 40))
        Ry_down = cath_cy + arrow_r * np.sin(np.radians(elec_R_angle + 40))
        arrow_R_up = FancyArrowPatch((Rx_down, Ry_down), (Rx, Ry),
                                      arrowstyle='->', lw=2.5, color=c_current,
                                      connectionstyle='arc3,rad=-0.1',
                                      mutation_scale=12, zorder=10)
        ax.add_patch(arrow_R_up)
        
        # Label the blood path
        ax.annotate('Blood path\n(low Z)', 
                    xy=(0, cath_cy - r_cath - 0.5),
                    xytext=(-6, -9), fontsize=8, color='darkred', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1))
        
        # Weak direct path through wall (dashed)
        arrow_wall = FancyArrowPatch((Lx, Ly), (Rx, Ry),
                                      connectionstyle='arc3,rad=-0.08',
                                      arrowstyle='->', lw=1.0, color=c_current_weak,
                                      mutation_scale=8, linestyle='dashed', zorder=10)
        ax.add_patch(arrow_wall)
        
        ax.text(0, r_vessel + 2.0, 'Z ≈ 800 Ω (≈ Blood!)', ha='center', fontsize=11, fontweight='bold',
                color='darkred', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.text(0, r_vessel + 0.8, 'No recess → current escapes\nalong surface to blood crescent below',
                ha='center', fontsize=7, color='navy')
        
        # Label wall at contact
        ax.annotate('Wall (1mm)', xy=(0, r_vessel), 
                    xytext=(5, r_vessel + 2), fontsize=7, color=c_wall,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=c_wall, lw=1))
    
    # Vessel wall label
    ax.text(-r_vessel - wall_thick - 0.3, 0, 'Vessel\nwall', ha='right', va='center',
            fontsize=7, color=c_wall, fontweight='bold', rotation=90)
    
    # Scale bar
    ax.plot([-r_vessel, -r_vessel + 2], [-r_vessel - 1.8, -r_vessel - 1.8], 
            'k-', lw=2)
    ax.text(-r_vessel + 1, -r_vessel - 2.3, '2 mm', ha='center', fontsize=7)


# Draw the 4 panels
draw_vessel_cross_section(axes[0, 0], 'Recessed + CLOT → Z=3500Ω', recessed=True, tissue='clot')
draw_vessel_cross_section(axes[0, 1], 'Recessed + WALL → Z=1800Ω', recessed=True, tissue='wall')
draw_vessel_cross_section(axes[1, 0], 'Flush + CLOT → Z=3500Ω', recessed=False, tissue='clot')
draw_vessel_cross_section(axes[1, 1], 'Flush + WALL → Z≈800Ω (Blood!)', recessed=False, tissue='wall')

# Row labels
fig.text(0.02, 0.74, 'Recessed\n(0.5mm)', ha='left', va='center', fontsize=10, 
         fontweight='bold', color='gray')
fig.text(0.02, 0.30, 'Flush\n(0mm)', ha='left', va='center', fontsize=10, 
         fontweight='bold', color='gray')

# Key insight box
fig.text(0.5, 0.01, 
         'KEY: Clot fills entire lumen → no blood escape path → Z=3500Ω (both recessed & flush).  '
         'Wall: catheter against wall, blood in crescent below.  '
         'Flush: current escapes along surface to blood crescent (Z≈800Ω).  '
         'Recessed: recess walls block surface escape → forced through wall (Z=1800Ω).',
         ha='center', va='bottom', fontsize=8, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

plt.tight_layout(rect=[0.04, 0.05, 1, 0.94])
out_dir = Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Comsol\Geom25\3D_Results_RealGeom")
plt.savefig(out_dir / "current_pathway_recessed_vs_flush.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_dir / 'current_pathway_recessed_vs_flush.png'}")
