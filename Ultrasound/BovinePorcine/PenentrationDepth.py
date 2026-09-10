import numpy as np
import matplotlib.pyplot as plt

# 1. Frequency Axis Setup (5 MHz to 25 MHz)
f = np.linspace(5e6, 20e6, 500)
f_mhz = f / 1e6
omega = 2 * np.pi * f

# 2. Material Parameters
# Blood (Pure Viscous Liquid)
rho_blood = 1050  # kg/m^3
eta_blood = 0.0035  # Pa*s
delta_blood = np.sqrt(eta_blood / (np.pi * f * rho_blood)) * 1e6  # um

# Blood Clot (Viscoelastic Gel)
rho_clot = 1060  # kg/m^3
G_prime_clot = 500e3  # 500 kPa storage modulus
eta_clot = 0.0200  # Pa*s
G_double_prime_clot = omega * eta_clot
G_star_clot = np.sqrt(G_prime_clot**2 + G_double_prime_clot**2)
delta_clot = (
    np.sqrt(
        (2 * G_star_clot**2)
        / (rho_clot * omega**2 * (G_star_clot - G_prime_clot))
    )
    * 1e6
)  # um

# Vessel Wall (Viscoelastic Solid)
rho_wall = 1080  # kg/m^3
G_prime_wall = 2.0e6  # 2.0 MPa storage modulus
eta_wall = 0.1000  # Pa*s
G_double_prime_wall = omega * eta_wall
G_star_wall = np.sqrt(G_prime_wall**2 + G_double_prime_wall**2)
delta_wall = (
    np.sqrt(
        (2 * G_star_wall**2)
        / (rho_wall * omega**2 * (G_star_wall - G_prime_wall))
    )
    * 1e6
)  # um

# 3. Discrete Markers at 5, 10, 20 MHz
f_pts_mhz = np.array([5, 10, 20])
f_pts = f_pts_mhz * 1e6
omega_pts = 2 * np.pi * f_pts

d_b_pts = np.sqrt(eta_blood / (np.pi * f_pts * rho_blood)) * 1e6

G_dp_c = omega_pts * eta_clot
G_s_c = np.sqrt(G_prime_clot**2 + G_dp_c**2)
d_c_pts = (
    np.sqrt((2 * G_s_c**2) / (rho_clot * omega_pts**2 * (G_s_c - G_prime_clot)))
    * 1e6
)

G_dp_w = omega_pts * eta_wall
G_s_w = np.sqrt(G_prime_wall**2 + G_dp_w**2)
d_w_pts = (
    np.sqrt((2 * G_s_w**2) / (rho_wall * omega_pts**2 * (G_s_w - G_prime_wall)))
    * 1e6
)

# 4. Plot Generation
plt.figure(figsize=(9, 5.5), dpi=150)

plt.plot(
    f_mhz,
    delta_blood,
    color="green",
    linewidth=2,
    label="Blood (Pure Viscous: $\eta = 0.0035$ Pa·s)",
)
plt.plot(
    f_mhz,
    delta_clot,
    color="red",
    linewidth=2,
    label="Clot (Viscoelastic: $G\' = 500$ kPa)",
)
plt.plot(
    f_mhz,
    delta_wall,
    color="blue",
    linewidth=2,
    label="Vessel Wall (Viscoelastic: $G\' = 2.0$ MPa)",
)

# Scatter plot for key sampling points
plt.scatter(f_pts_mhz, d_b_pts, color="green", zorder=5)
plt.scatter(f_pts_mhz, d_c_pts, color="red", zorder=5)
plt.scatter(f_pts_mhz, d_w_pts, color="blue", zorder=5)

# Annotations for 5 MHz values
# plt.annotate(
#     f"{d_b_pts[0]:.2f} $\mu m$",
#     (5, d_b_pts[0]),
#     textcoords="offset points",
#     xytext=(10, -5),
#     color="green",
#     fontweight="bold",
# )
# plt.annotate(
#     f"{d_c_pts[0]:.2f} $\mu m$",
#     (5, d_c_pts[0]),
#     textcoords="offset points",
#     xytext=(10, -5),
#     color="red",
#     fontweight="bold",
# )
# plt.annotate(
#     f"{d_w_pts[0]:.2f} $\mu m$",
#     (5, d_w_pts[0]),
#     textcoords="offset points",
#     xytext=(10, -5),
#     color="blue",
#     fontweight="bold",
# )

plt.title(
    "Viscoelastic Acoustic Penetration Depth ($\delta_{VE}$) vs. Frequency",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Frequency (MHz)", fontsize=11)
plt.ylabel("Penetration Depth $\delta$ ($\mu m$)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.show()