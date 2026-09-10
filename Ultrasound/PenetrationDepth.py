import numpy as np
import matplotlib.pyplot as plt

# Frequency range in Hz (1 MHz to 25 MHz)
f = np.linspace(1e6, 25e6, 500)
f_mhz = f / 1e6

# Parameters
# Blood
eta_blood = 0.0035
rho_blood = 1050
delta_blood = np.sqrt(eta_blood / (np.pi * f * rho_blood)) * 1e6  # in um

# Clot
eta_clot = 0.0200
rho_clot = 1060
delta_clot = np.sqrt(eta_clot / (np.pi * f * rho_clot)) * 1e6  # in um

# Vessel Wall
eta_wall = 0.1000
rho_wall = 1080
delta_wall = np.sqrt(eta_wall / (np.pi * f * rho_wall)) * 1e6  # in um

# Plotting
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(f_mhz, delta_blood, color='green', linewidth=2, label='Blood ($\eta = 0.0035$ Pa·s, $\\rho = 1050$ kg/m³)')
plt.plot(f_mhz, delta_clot, color='red', linewidth=2, label='Clot ($\eta = 0.0200$ Pa·s, $\\rho = 1060$ kg/m³)')
plt.plot(f_mhz, delta_wall, color='blue', linewidth=2, label='Vessel Wall ($\eta = 0.1000$ Pa·s, $\\rho = 1080$ kg/m³)')

# Highlight discrete frequencies 5, 10, 20 MHz
discrete_f_mhz = np.array([5, 10, 20])
discrete_f = discrete_f_mhz * 1e6

d_b_pts = np.sqrt(eta_blood / (np.pi * discrete_f * rho_blood)) * 1e6
d_c_pts = np.sqrt(eta_clot / (np.pi * discrete_f * rho_clot)) * 1e6
d_w_pts = np.sqrt(eta_wall / (np.pi * discrete_f * rho_wall)) * 1e6

plt.scatter(discrete_f_mhz, d_b_pts, color='green', zorder=5)
plt.scatter(discrete_f_mhz, d_c_pts, color='red', zorder=5)
plt.scatter(discrete_f_mhz, d_w_pts, color='blue', zorder=5)

plt.title('QCM Acoustic Penetration Depth ($\delta$) vs. Frequency', fontsize=12, fontweight='bold')
plt.xlabel('Frequency (MHz)', fontsize=11)
plt.ylabel('Penetration Depth $\delta$ ($\mu m$)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()

plt.savefig('penetration_depth_plot.png')
plt.close()