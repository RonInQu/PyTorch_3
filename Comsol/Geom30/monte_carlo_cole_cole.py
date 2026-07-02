"""
Monte Carlo Analysis: Cole-Cole Parameter Variability
=====================================================
Option A: Film thickness as fixed condition, Cole-Cole params as random.
Evaluates discrimination robustness at each film thickness.

Uses analytical model (no COMSOL needed):
  Z = K_eff / sigma_complex(f)
  sigma_complex = sigma_dc * (1 + sum_dispersion)
  Film: Z_film = Z_tissue * [1 - CDF(t) * (1 - sigma_tissue/sigma_blood)]
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.special import gamma

# =====================================================================
#  CONFIGURATION
# =====================================================================
N_TRIALS = 10000
SEED = 42

# Frequencies of interest [Hz]
freqs = np.array([5e3, 10e3, 20e3, 50e3, 100e3])

# Cell constant (Geom30 calibrated)
K_eff = 728 * 0.8775  # = 639 m^-1

# Target impedances
Z_target_blood = 800
Z_target_clot = 3500
Z_target_wall = 1800

# Calibrated conductivities: these are the SIMPLE K/Z values used in Geom30.
# In Geom30, the COMSOL model uses sigma_dc directly (no Cole-Cole dispersion
# in the baseline solve). The Cole-Cole is only applied in the freq sweep 
# as a scaling factor relative to 50 kHz baseline.
# So for the MC, we should NOT add Cole-Cole dispersion on top — that would
# double-count. Instead: use sigma_dc = K/Z_target as the effective 
# conductivity at 50 kHz, and apply Cole-Cole only as a RELATIVE frequency
# correction factor.
sigma_blood_50k = K_eff / Z_target_blood  # 0.799 S/m (effective at 50 kHz)
sigma_clot_50k = K_eff / Z_target_clot    # 0.183 S/m
sigma_wall_50k = K_eff / Z_target_wall    # 0.355 S/m

# Film thicknesses to evaluate [m]
film_mm_list = np.array([0, 0.1, 0.2, 0.3, 0.5])
film_m_list = film_mm_list * 1e-3

# Weibull sensing depth CDF: lambda=0.410mm, beta=1.73 (from COMSOL)
weibull_lam = 0.410e-3  # m
weibull_beta = 1.73

# =====================================================================
#  COLE-COLE PARAMETER DISTRIBUTIONS
# =====================================================================
# Nominal σ at 50 kHz = calibrated values from Geom30 (forced to Z_target).
# Cole-Cole parameters provide RELATIVE frequency dispersion.
# Variability in σ_dc captures tissue property uncertainty.
# cv = coefficient of variation (std/mean) for log-normal

params = {
    'blood': {
        'sigma_50k': {'mean': sigma_blood_50k, 'cv': 0.25},  # varies with Hct
        'tau':      {'mean': 10e-6, 'cv': 0.30},
        'alpha':    {'mean': 0.25, 'std': 0.05, 'lo': 0.05, 'hi': 0.45},
        'delta_eps': {'mean': 2.53e6, 'cv': 0.30},
        'eps_inf':  50,  # fixed
        'sigma_dc_nominal': 1.30,  # literature value for CC ratio calc
    },
    'clot': {
        'sigma_50k': {'mean': sigma_clot_50k, 'cv': 0.35},   # high variability
        'tau':      {'mean': 12e-6, 'cv': 0.30},
        'alpha':    {'mean': 0.30, 'std': 0.05, 'lo': 0.05, 'hi': 0.45},
        'delta_eps': {'mean': 7.7e5, 'cv': 0.35},
        'eps_inf':  40,  # fixed
        'sigma_dc_nominal': 0.155,
    },
    'wall': {
        'sigma_50k': {'mean': sigma_wall_50k, 'cv': 0.20},   # less variable
        'tau':      {'mean': 9e-6, 'cv': 0.25},
        'alpha':    {'mean': 0.25, 'std': 0.05, 'lo': 0.05, 'hi': 0.45},
        'delta_eps': {'mean': 1.2e6, 'cv': 0.25},
        'eps_inf':  40,  # fixed
        'sigma_dc_nominal': 0.40,
    },
}


# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================
def draw_lognormal(mean, cv, n, rng):
    """Draw from log-normal with given mean and coefficient of variation."""
    sigma_ln = np.sqrt(np.log(1 + cv**2))
    mu_ln = np.log(mean) - 0.5 * sigma_ln**2
    return rng.lognormal(mu_ln, sigma_ln, n)


def draw_truncated_normal(mean, std, lo, hi, n, rng):
    """Draw from truncated normal."""
    samples = rng.normal(mean, std, n)
    # Resample out-of-bounds
    mask = (samples < lo) | (samples > hi)
    while mask.any():
        samples[mask] = rng.normal(mean, std, mask.sum())
        mask = (samples < lo) | (samples > hi)
    return samples


def cole_cole_sigma(f, sigma_dc, tau, alpha, delta_eps, eps_inf):
    """
    Complex conductivity from Cole-Cole model.
    sigma*(f) = sigma_dc + j*omega*eps0*(eps_inf + delta_eps/(1+(j*omega*tau)^(1-alpha)))
    Returns |sigma*| for impedance calculation.
    """
    omega = 2 * np.pi * f
    eps0 = 8.854e-12
    jwt_alpha = (1j * omega * tau) ** (1 - alpha)
    eps_complex = eps_inf + delta_eps / (1 + jwt_alpha)
    sigma_complex = sigma_dc + 1j * omega * eps0 * eps_complex
    return sigma_complex


def weibull_cdf(t, lam, beta):
    """Sensing depth CDF: fraction of signal within distance t."""
    return 1.0 - np.exp(-(t / lam) ** beta)


def compute_z(f, sigma_dc, tau, alpha, delta_eps, eps_inf, K):
    """Compute impedance magnitude at frequency f."""
    sigma_c = cole_cole_sigma(f, sigma_dc, tau, alpha, delta_eps, eps_inf)
    return K / np.abs(sigma_c)


def compute_z_relative(f, sigma_50k, sigma_dc_nom, tau, alpha, delta_eps, eps_inf, K):
    """
    Compute impedance using relative Cole-Cole correction.
    Same approach as Geom30 frequency sweep:
      sigma_eff(f) = sigma_50k * |sigma_cc(f)| / |sigma_cc(50kHz)|
      Z(f) = K / sigma_eff(f)
    This ensures Z(50kHz) = K/sigma_50k = Z_target (when sigma_50k = nominal)
    """
    sigma_cc_f = np.abs(cole_cole_sigma(f, sigma_dc_nom, tau, alpha, delta_eps, eps_inf))
    sigma_cc_50k = np.abs(cole_cole_sigma(50e3, sigma_dc_nom, tau, alpha, delta_eps, eps_inf))
    sigma_eff = sigma_50k * (sigma_cc_f / sigma_cc_50k)
    return K / sigma_eff


# =====================================================================
#  DRAW MONTE CARLO SAMPLES
# =====================================================================
print(f"Monte Carlo: {N_TRIALS} trials, {len(freqs)} frequencies, "
      f"{len(film_mm_list)} film thicknesses")
print("=" * 70)

rng = np.random.default_rng(SEED)

# Draw parameters for each tissue
mc_params = {}
for tissue, p in params.items():
    mc_params[tissue] = {
        'sigma_50k': draw_lognormal(p['sigma_50k']['mean'], p['sigma_50k']['cv'], N_TRIALS, rng),
        'tau': draw_lognormal(p['tau']['mean'], p['tau']['cv'], N_TRIALS, rng),
        'alpha': draw_truncated_normal(p['alpha']['mean'], p['alpha']['std'],
                                       p['alpha']['lo'], p['alpha']['hi'], N_TRIALS, rng),
        'delta_eps': draw_lognormal(p['delta_eps']['mean'], p['delta_eps']['cv'], N_TRIALS, rng),
        'eps_inf': p['eps_inf'],
        'sigma_dc_nominal': p['sigma_dc_nominal'],
    }

# Print parameter statistics
for tissue in ['blood', 'clot', 'wall']:
    p = mc_params[tissue]
    print(f"\n{tissue.upper()}:")
    print(f"  sigma_50k: mean={p['sigma_50k'].mean():.4f}, "
          f"std={p['sigma_50k'].std():.4f}, "
          f"[{np.percentile(p['sigma_50k'], 5):.4f}, {np.percentile(p['sigma_50k'], 95):.4f}]")
    print(f"  tau [us]:  mean={p['tau'].mean()*1e6:.2f}, "
          f"std={p['tau'].std()*1e6:.2f}, "
          f"[{np.percentile(p['tau'], 5)*1e6:.2f}, {np.percentile(p['tau'], 95)*1e6:.2f}]")
    print(f"  alpha:    mean={p['alpha'].mean():.3f}, "
          f"std={p['alpha'].std():.3f}")
    print(f"  delta_eps: mean={p['delta_eps'].mean():.2e}, "
          f"std={p['delta_eps'].std():.2e}")

# =====================================================================
#  COMPUTE IMPEDANCES
# =====================================================================
# Z_tissue[tissue][freq_idx, trial] — bare impedance (no film)
Z_bare = {}
for tissue in ['blood', 'clot', 'wall']:
    p = mc_params[tissue]
    Z_bare[tissue] = np.zeros((len(freqs), N_TRIALS))
    for fi, f in enumerate(freqs):
        for trial in range(N_TRIALS):
            Z_bare[tissue][fi, trial] = compute_z_relative(
                f, p['sigma_50k'][trial], p['sigma_dc_nominal'],
                p['tau'][trial], p['alpha'][trial],
                p['delta_eps'][trial], p['eps_inf'], K_eff)

# Apply film correction: Z_film = Z_tissue * [1 - CDF(t)*(1 - sigma_tissue/sigma_blood)]
# Note: sigma_tissue/sigma_blood ≈ Z_blood/Z_tissue (at same frequency)
# More precisely: use the actual conductivity magnitudes
Z_film = {}  # Z_film[tissue][film_idx, freq_idx, trial]
for tissue in ['clot', 'wall']:
    Z_film[tissue] = np.zeros((len(film_m_list), len(freqs), N_TRIALS))
    for ti, t in enumerate(film_m_list):
        cdf_t = weibull_cdf(t, weibull_lam, weibull_beta)
        for fi in range(len(freqs)):
            # sigma ratio from impedances: sigma_tissue/sigma_blood = Z_blood/Z_tissue
            ratio = Z_bare['blood'][fi, :] / Z_bare[tissue][fi, :]
            Z_film[tissue][ti, fi, :] = Z_bare[tissue][fi, :] * (1.0 - cdf_t * (1.0 - ratio))

# =====================================================================
#  DISCRIMINATION ANALYSIS
# =====================================================================
print("\n" + "=" * 70)
print("DISCRIMINATION: Z_clot / Z_wall (>1 means can distinguish)")
print("=" * 70)

# Classification threshold
THRESHOLD = 1.3  # ratio > 1.3 = "can discriminate"

# Results table
print(f"\n{'Film':>6} {'Freq':>7} {'Median':>8} {'P5':>8} {'P95':>8} "
      f"{'P(>1.3)':>8} {'P(>1.5)':>8} {'P(>1.0)':>8}")
print("-" * 65)

results = {}  # results[(film_mm, freq_kHz)] = dict of stats

for ti, film_mm in enumerate(film_mm_list):
    for fi, f in enumerate(freqs):
        if film_mm == 0:
            ratio = Z_bare['clot'][fi, :] / Z_bare['wall'][fi, :]
        else:
            ratio = Z_film['clot'][ti, fi, :] / Z_film['wall'][ti, fi, :]

        med = np.median(ratio)
        p5 = np.percentile(ratio, 5)
        p95 = np.percentile(ratio, 95)
        p_above_13 = np.mean(ratio > 1.3) * 100
        p_above_15 = np.mean(ratio > 1.5) * 100
        p_above_10 = np.mean(ratio > 1.0) * 100

        results[(film_mm, f/1e3)] = {
            'median': med, 'p5': p5, 'p95': p95,
            'p_13': p_above_13, 'p_15': p_above_15, 'p_10': p_above_10,
            'ratio_all': ratio
        }

        print(f"{film_mm:>5.1f}mm {f/1e3:>5.0f}k {med:>8.3f} {p5:>8.3f} "
              f"{p95:>8.3f} {p_above_13:>7.1f}% {p_above_15:>7.1f}% {p_above_10:>7.1f}%")
    print()

# =====================================================================
#  ABSOLUTE Z DISTRIBUTIONS (at 50 kHz, no film)
# =====================================================================
print("\n" + "=" * 70)
print("ABSOLUTE IMPEDANCE at 50 kHz (no film)")
print("=" * 70)
fi_50 = np.where(freqs == 50e3)[0][0]
for tissue in ['blood', 'clot', 'wall']:
    z = Z_bare[tissue][fi_50, :]
    print(f"  {tissue:>5}: median={np.median(z):.0f}, "
          f"mean={z.mean():.0f}, "
          f"[P5={np.percentile(z, 5):.0f}, P95={np.percentile(z, 95):.0f}] Ohm")

# =====================================================================
#  FIGURES
# =====================================================================
out_dir = Path(__file__).parent / '3D_Results_RealGeom'
out_dir.mkdir(exist_ok=True)

# --- Figure 1: Discrimination vs film thickness at each frequency ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Median discrimination ratio
ax = axes[0]
for fi, f in enumerate(freqs):
    medians = [results[(fm, f/1e3)]['median'] for fm in film_mm_list]
    p5s = [results[(fm, f/1e3)]['p5'] for fm in film_mm_list]
    p95s = [results[(fm, f/1e3)]['p95'] for fm in film_mm_list]
    ax.plot(film_mm_list, medians, 'o-', lw=2, label=f'{f/1e3:.0f} kHz')
    ax.fill_between(film_mm_list, p5s, p95s, alpha=0.1)
ax.axhline(1.3, color='red', ls='--', alpha=0.7, label='Threshold (1.3)')
ax.axhline(1.0, color='black', ls=':', alpha=0.5)
ax.set_xlabel('Blood Film Thickness [mm]')
ax.set_ylabel('Z_clot / Z_wall')
ax.set_title('Discrimination Ratio (median ± 90% CI)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 0.52])

# Panel B: P(discrimination > 1.3) vs film
ax = axes[1]
for fi, f in enumerate(freqs):
    probs = [results[(fm, f/1e3)]['p_13'] for fm in film_mm_list]
    ax.plot(film_mm_list, probs, 'o-', lw=2, label=f'{f/1e3:.0f} kHz')
ax.axhline(90, color='red', ls='--', alpha=0.7, label='90% target')
ax.set_xlabel('Blood Film Thickness [mm]')
ax.set_ylabel('P(Z_clot/Z_wall > 1.3) [%]')
ax.set_title('Probability of Correct Discrimination')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 0.52])
ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig(out_dir / 'mc_discrimination_vs_film.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure 1 saved: {out_dir / 'mc_discrimination_vs_film.png'}")

# --- Figure 2: Z histograms at 50 kHz, no film ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fi_50 = np.where(freqs == 50e3)[0][0]
colors = {'blood': 'red', 'clot': 'saddlebrown', 'wall': 'green'}
for ax, tissue in zip(axes, ['blood', 'clot', 'wall']):
    z = Z_bare[tissue][fi_50, :]
    ax.hist(z, bins=80, color=colors[tissue], alpha=0.7, edgecolor='black', lw=0.3)
    ax.axvline(np.median(z), color='black', ls='-', lw=2, label=f'Median={np.median(z):.0f}')
    ax.axvline(np.percentile(z, 5), color='gray', ls='--', label=f'P5={np.percentile(z, 5):.0f}')
    ax.axvline(np.percentile(z, 95), color='gray', ls='--', label=f'P95={np.percentile(z, 95):.0f}')
    ax.set_xlabel('|Z| [Ω]')
    ax.set_title(f'{tissue.capitalize()} @ 50 kHz')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / 'mc_z_distributions_50kHz.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 2 saved: {out_dir / 'mc_z_distributions_50kHz.png'}")

# --- Figure 3: Discrimination ratio histogram at 50kHz for each film ---
fig, axes = plt.subplots(1, len(film_mm_list), figsize=(16, 4), sharey=True)
for ti, (ax, film_mm) in enumerate(zip(axes, film_mm_list)):
    ratio = results[(film_mm, 50.0)]['ratio_all']
    ax.hist(ratio, bins=80, color='steelblue', alpha=0.7, edgecolor='black', lw=0.3)
    ax.axvline(1.3, color='red', ls='--', lw=2)
    ax.axvline(np.median(ratio), color='black', ls='-', lw=2)
    p_above = results[(film_mm, 50.0)]['p_13']
    ax.set_title(f'Film={film_mm:.1f}mm\nP(>1.3)={p_above:.0f}%', fontsize=9)
    ax.set_xlabel('Z_clot/Z_wall')
    if ti == 0:
        ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

fig.suptitle('Discrimination Histogram @ 50 kHz (red=threshold)', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(out_dir / 'mc_discrimination_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 3 saved: {out_dir / 'mc_discrimination_histograms.png'}")

# --- Figure 4: Sensitivity analysis — which parameter dominates? ---
# Correlation of each tissue's sigma_dc with the discrimination ratio
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ratio_nofilm = results[(0.0, 50.0)]['ratio_all']

for col, (tissue, label) in enumerate(zip(['blood', 'clot', 'wall'],
                                           ['Blood σ_50k', 'Clot σ_50k', 'Wall σ_50k'])):
    ax = axes[col]
    vals = mc_params[tissue]['sigma_50k']
    ax.scatter(vals, ratio_nofilm, alpha=0.03, s=2, c='steelblue')
    corr = np.corrcoef(vals, ratio_nofilm)[0, 1]
    ax.set_xlabel(f'{label} [S/m]')
    ax.set_ylabel('Z_clot / Z_wall')
    ax.set_title(f'r = {corr:.3f}', fontsize=10)
    ax.axhline(1.3, color='red', ls='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

fig.suptitle('Sensitivity: σ_dc vs Discrimination Ratio (50 kHz, no film)', fontsize=11)
plt.tight_layout()
plt.savefig(out_dir / 'mc_sensitivity_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 4 saved: {out_dir / 'mc_sensitivity_scatter.png'}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
