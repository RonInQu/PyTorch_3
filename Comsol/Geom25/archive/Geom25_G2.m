function out = Geom25
%% COMSOL LiveLink — Real Geometry Bioimpedance Model (Gen 2.5)
%
% Full 3D model using IMPORTED catheter geometry (PRT-1173, 24FR).
% All parameters defined in COMSOL (model.param) — no magic numbers.
% Calibrated to match measured impedances:
%   Blood = 800 Ohm, Wall = 1800 Ohm, Clot = 3500 Ohm @ 50 kHz
%
% Architecture matches v6 flow:
%   1. Geometry + Materials + ec physics + Mesh + Baseline solve
%   2. Sensing depth (probe existing solution)
%   3. Frequency sweep (ec only — ht not yet added)
%   4. Blood film sensitivity (ec only)
%   5. Electrode interface (analytical)
%   6. Thermal analysis (add ht physics, solve separately)
%   7. COMSOL Result plots + MATLAB figures
%
% Run in MATLAB launched via "COMSOL with MATLAB"

import com.comsol.model.*
import com.comsol.model.util.*

close all; clc;

fprintf('=== COMSOL LiveLink - Real Geometry Model (Gen 2.5) ===\n');
fprintf('  Date: %s\n', datestr(now));

%% ====================================================================
%  CONFIGURATION
% =====================================================================
% Excitation voltages (PTP measured on oscilloscope)
% V_applied = PTP/2 (each electrode gets +/- V_applied)
V_ptp_5k   = 0.560;        % [V] PTP at 5 kHz
V_ptp_50k  = 0.860;        % [V] PTP at 50 kHz
V_ptp_100k = 0.900;        % [V] PTP at 100 kHz

freq_base = 50e3;          % [Hz] (baseline solve)
V_applied = V_ptp_50k / 2; % [V] (+/- on electrodes) for baseline
freq_list  = [5e3,   10e3,  20e3,  50e3,   100e3];
V_ptp_list = [0.560, 0.560, 0.860, 0.860,  0.900]; % interpolated for 10k/20k

% Vessel geometry [m]
vessel_r_inner = 0.008;
vessel_r_outer = 0.009;
vessel_length  = 0.040;
vessel_z_start = -0.020;

% Domain / Boundary IDs (verified in GUI)
dom_wall  = 1;
dom_blood = 2;
dom_cath  = 3;
bnd_elec_L = 64;
bnd_elec_R = 30;

% Cell constant
K_real = 702;   % m^-1

% Impedance targets [Ohm]
Z_target_blood = 800;
Z_target_clot  = 3500;
Z_target_wall  = 1800;

% Calibrated conductivities [S/m]
sigma_blood = K_real / Z_target_blood;
sigma_clot  = K_real / Z_target_clot;
sigma_wall  = K_real / Z_target_wall;
sigma_cath  = 1e-10;

% Cole-Cole parameters: [Blood, Clot, Wall]
cc_names     = {'Blood', 'Clot', 'Wall'};
cc_sigma_dc  = [1.30,   0.155,  0.40];
cc_eps_inf   = [50,     40,     40];
cc_delta_eps = [2530000, 770000, 1200000];
cc_tau       = [10e-6,  12e-6,  9e-6];
cc_alpha     = [0.25,   0.30,   0.25];
sigma_cal = [sigma_blood, sigma_clot, sigma_wall];

% Electrode interface
electrode_materials = {'SS316L smooth', 'SS316L rough', 'Pt-Ir', 'Ti'};
CPE_Q  = [0.030, 0.30, 0.15, 0.015];
CPE_n  = [0.83,  0.85, 0.87, 0.80];
R_ct   = [2e-3,  2e-4, 5e-4, 3e-3];
d_oxide_m = [3e-9, 3e-9, 0, 0];
epsr_oxide = 12;
A_elec_m2 = 1.386e-6;

% Thermal properties
k_blood_val  = 0.52;   rho_blood_val = 1060;  Cp_blood_val = 3900;
k_wall_val   = 0.42;   rho_wall_val  = 1050;  Cp_wall_val  = 3700;
k_cath_val   = 0.22;   rho_cath_val  = 900;   Cp_cath_val  = 1800;

% Output
out_dir = fullfile(pwd, '3D_Results_RealGeom');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

V_total = 2 * V_applied;  % = V_ptp for baseline (50 kHz)

fprintf('  K = %.1f m^-1, sigma: Blood=%.4f, Clot=%.4f, Wall=%.4f S/m\n', ...
    K_real, sigma_blood, sigma_clot, sigma_wall);
fprintf('  V_ptp: 5kHz=%.0fmV, 50kHz=%.0fmV, 100kHz=%.0fmV\n', ...
    V_ptp_5k*1e3, V_ptp_50k*1e3, V_ptp_100k*1e3);

%% ====================================================================
%  CREATE MODEL + GLOBAL PARAMETERS
% =====================================================================
fprintf('\n--- Creating model ---\n');
model = ModelUtil.create('Geom25');
model.modelPath(pwd);
par = model.param;

par.set('V_app', sprintf('%.2f[V]', V_applied), 'Applied voltage');
par.set('f0', sprintf('%.0f[Hz]', freq_base), 'Base frequency');
par.set('r_vessel_inner', sprintf('%.4f[m]', vessel_r_inner));
par.set('r_vessel_outer', sprintf('%.4f[m]', vessel_r_outer));
par.set('L_vessel', sprintf('%.4f[m]', vessel_length));
par.set('z_vessel_start', sprintf('%.4f[m]', vessel_z_start));
par.set('K_cell', sprintf('%.1f[1/m]', K_real), 'Cell constant');
par.set('sigma_blood', sprintf('%.6f[S/m]', sigma_blood));
par.set('sigma_clot', sprintf('%.6f[S/m]', sigma_clot));
par.set('sigma_wall', sprintf('%.6f[S/m]', sigma_wall));
par.set('sigma_cath', sprintf('%.2e[S/m]', sigma_cath));

tissues_cc = {'blood', 'clot', 'wall'};
for k = 1:3
    t = tissues_cc{k};
    par.set(['sigma_dc_' t], sprintf('%.4g[S/m]', cc_sigma_dc(k)));
    par.set(['eps_inf_' t], sprintf('%.0f', cc_eps_inf(k)));
    par.set(['delta_eps_' t], sprintf('%.0f', cc_delta_eps(k)));
    par.set(['tau_' t], sprintf('%.2e[s]', cc_tau(k)));
    par.set(['alpha_cc_' t], sprintf('%.2f', cc_alpha(k)));
end

par.set('k_blood', sprintf('%.2f[W/(m*K)]', k_blood_val));
par.set('rho_blood', sprintf('%.0f[kg/m^3]', rho_blood_val));
par.set('Cp_blood', sprintf('%.0f[J/(kg*K)]', Cp_blood_val));
par.set('k_wall_th', sprintf('%.2f[W/(m*K)]', k_wall_val));
par.set('rho_wall_th', sprintf('%.0f[kg/m^3]', rho_wall_val));
par.set('Cp_wall_th', sprintf('%.0f[J/(kg*K)]', Cp_wall_val));
par.set('k_cath', sprintf('%.2f[W/(m*K)]', k_cath_val));
par.set('rho_cath', sprintf('%.0f[kg/m^3]', rho_cath_val));
par.set('Cp_cath', sprintf('%.0f[J/(kg*K)]', Cp_cath_val));
par.set('T_body', '310.15[K]', 'Body temperature');
par.set('d_oxide', '3[nm]', 'Cr2O3 oxide thickness');
par.set('epsr_oxide', sprintf('%.0f', epsr_oxide));
par.set('A_elec', sprintf('%.6e[m^2]', A_elec_m2));

fprintf('  Parameters defined.\n');

%% ====================================================================
%  GEOMETRY
% =====================================================================
fprintf('\n--- Building geometry ---\n');
model.component.create('comp1', true);
model.component('comp1').geom.create('geom1', 3);
geom = model.component('comp1').geom('geom1');

geom.create('imp1', 'Import');
geom.feature('imp1').set('filename', ...
    fullfile(pwd, 'PRT-1173 - ASPIRATION ELT TIP 24FR.STEP'));
geom.feature('imp1').set('minimizetol', true);
geom.feature('imp1').set('healedges', true);
geom.feature('imp1').set('simplify', true);
geom.feature('imp1').set('removeredundant', true);
geom.run('imp1');

geom.create('cyl_blood', 'Cylinder');
geom.feature('cyl_blood').set('r', vessel_r_inner);
geom.feature('cyl_blood').set('h', vessel_length);
geom.feature('cyl_blood').set('pos', [0, 0, vessel_z_start]);
geom.feature('cyl_blood').set('axis', [0, 0, 1]);
geom.run('cyl_blood');

geom.create('cyl_wall', 'Cylinder');
geom.feature('cyl_wall').set('r', vessel_r_outer);
geom.feature('cyl_wall').set('h', vessel_length);
geom.feature('cyl_wall').set('pos', [0, 0, vessel_z_start]);
geom.feature('cyl_wall').set('axis', [0, 0, 1]);
geom.run('cyl_wall');

geom.run('fin');
fprintf('  Geometry: wall=%d, blood=%d, catheter=%d.\n', dom_wall, dom_blood, dom_cath);

%% ====================================================================
%  MATERIALS
% =====================================================================
fprintf('\n--- Materials ---\n');
mat_blood = model.component('comp1').material.create('mat_blood', 'Common');
mat_blood.label('Blood'); mat_blood.selection.set([dom_blood]);
mat_blood.propertyGroup('def').set('electricconductivity', 'sigma_blood');
mat_blood.propertyGroup('def').set('relpermittivity', '1');

mat_wall = model.component('comp1').material.create('mat_wall', 'Common');
mat_wall.label('Vessel Wall'); mat_wall.selection.set([dom_wall]);
mat_wall.propertyGroup('def').set('electricconductivity', 'sigma_wall');
mat_wall.propertyGroup('def').set('relpermittivity', '1');

mat_cath = model.component('comp1').material.create('mat_cath', 'Common');
mat_cath.label('Catheter'); mat_cath.selection.set([dom_cath]);
mat_cath.propertyGroup('def').set('electricconductivity', 'sigma_cath');
mat_cath.propertyGroup('def').set('relpermittivity', '3');

fprintf('  Materials assigned.\n');

%% ====================================================================
%  PHYSICS: Electric Currents (ec only — no ht yet)
% =====================================================================
fprintf('\n--- Physics (ec) ---\n');
ec = model.component('comp1').physics.create('ec', 'ConductiveMedia', 'geom1');

ec.create('epot1', 'ElectricPotential', 2);
ec.feature('epot1').selection.set([bnd_elec_L]);
ec.feature('epot1').set('V0', 'V_app');
ec.feature('epot1').label('Electrode L (+V)');

ec.create('epot2', 'ElectricPotential', 2);
ec.feature('epot2').selection.set([bnd_elec_R]);
ec.feature('epot2').set('V0', '-V_app');
ec.feature('epot2').label('Electrode R (-V)');

%% ====================================================================
%  MESH + STUDY + BASELINE SOLVE
% =====================================================================
fprintf('\n--- Mesh + Solve ---\n');
model.study.create('std1');
model.study('std1').create('freq', 'Frequency');
model.study('std1').feature('freq').set('plist', 'f0');

mesh = model.component('comp1').mesh.create('mesh1');
mesh.create('size1', 'Size');
mesh.feature('size1').set('hauto', 4);
mesh.create('ftet1', 'FreeTet');
mesh.feature('ftet1').create('size_elec', 'Size');
mesh.feature('ftet1').feature('size_elec').selection.geom('geom1', 2);
mesh.feature('ftet1').feature('size_elec').selection.set([bnd_elec_L, bnd_elec_R]);
mesh.feature('ftet1').feature('size_elec').set('custom', true);
mesh.feature('ftet1').feature('size_elec').set('hmax', 1e-4);
mesh.feature('ftet1').feature('size_elec').set('hmin', 2e-5);
mesh.feature('ftet1').feature('size_elec').set('hgrad', 1.3);
mesh.run;
fprintf('  Mesh complete.\n');

model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').feature('st1').set('study', 'std1');
model.sol('sol1').feature('st1').set('studystep', 'freq');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').set('stol', 1e-6);
model.sol('sol1').runAll;

% Baseline Z via current method
I_base = mphint2(model, 'ec.nJ', 'surface', 'selection', bnd_elec_L);
Z_blood_meas = V_total / abs(I_base);
fprintf('  Z_blood = %.1f Ohm (target: %d)\n', Z_blood_meas, Z_target_blood);

%% ====================================================================
%  SENSING DEPTH (probe existing baseline solution)
% =====================================================================
fprintf('\n--- Sensing depth ---\n');

A_elec_computed = mphint2(model, '1', 'surface', 'selection', bnd_elec_L);
x_ctr = mphint2(model, 'x', 'surface', 'selection', bnd_elec_L) / A_elec_computed;
y_ctr = mphint2(model, 'y', 'surface', 'selection', bnd_elec_L) / A_elec_computed;
z_ctr = mphint2(model, 'z', 'surface', 'selection', bnd_elec_L) / A_elec_computed;
fprintf('  Electrode L centroid: [%.4f, %.4f, %.4f] mm\n', x_ctr*1e3, y_ctr*1e3, z_ctr*1e3);

% Electrode R centroid
A_elec_R = mphint2(model, '1', 'surface', 'selection', bnd_elec_R);
x_ctr_R = mphint2(model, 'x', 'surface', 'selection', bnd_elec_R) / A_elec_R;
y_ctr_R = mphint2(model, 'y', 'surface', 'selection', bnd_elec_R) / A_elec_R;
z_ctr_R = mphint2(model, 'z', 'surface', 'selection', bnd_elec_R) / A_elec_R;
fprintf('  Electrode R centroid: [%.4f, %.4f, %.4f] mm\n', x_ctr_R*1e3, y_ctr_R*1e3, z_ctr_R*1e3);

probe_dir = [x_ctr; y_ctr; 0];
probe_dir = probe_dir / norm(probe_dir);
r_cath_surface = norm([x_ctr, y_ctr]);

n_radial = 100;
r_max_offset = vessel_r_inner - r_cath_surface - 0.0003;
r_offsets = linspace(0.00005, r_max_offset, n_radial);
r_from_surface = r_offsets * 1e3;  % mm

pd_sense = zeros(1, n_radial);
for k = 1:n_radial
    pt = [x_ctr; y_ctr; z_ctr] + probe_dir * r_offsets(k);
    try
        pd_sense(k) = mphinterp(model, 'ec.Qrh', 'coord', pt);
    catch
        pd_sense(k) = NaN;
    end
end

valid = ~isnan(pd_sense) & pd_sense > 0;
pd_valid = pd_sense(valid);
r_valid = r_from_surface(valid);
cum_power = cumsum(pd_valid);
depth_50 = NaN; depth_80 = NaN; depth_90 = NaN; depth_95 = NaN;
cum_norm = [];
if ~isempty(cum_power) && cum_power(end) > 0
    cum_norm = cum_power / cum_power(end);
    idx_50 = find(cum_norm >= 0.50, 1);
    idx_80 = find(cum_norm >= 0.80, 1);
    idx_90 = find(cum_norm >= 0.90, 1);
    idx_95 = find(cum_norm >= 0.95, 1);
    if ~isempty(idx_50), depth_50 = r_valid(idx_50); end
    if ~isempty(idx_80), depth_80 = r_valid(idx_80); end
    if ~isempty(idx_90), depth_90 = r_valid(idx_90); end
    if ~isempty(idx_95), depth_95 = r_valid(idx_95); end
    fprintf('  50%%=%.2f, 80%%=%.2f, 95%%=%.2f mm\n', depth_50, depth_80, depth_95);
end

%% ====================================================================
%  FREQUENCY SWEEP (ec only — no ht in model yet)
%  Exact v6 pattern: change mat sigma, change plist, study.run(), V/I
% =====================================================================
fprintf('\n--- Frequency sweep ---\n');

sig_cc_50k = zeros(1, 3);
for mi = 1:3
    [sig_cc_50k(mi), ~] = cole_cole_calc(50e3, cc_sigma_dc(mi), cc_eps_inf(mi), ...
        cc_delta_eps(mi), cc_tau(mi), cc_alpha(mi));
end

n_freq = length(freq_list);
n_mat = 3;
Z_sweep = zeros(n_freq, n_mat);
phase_sweep = zeros(n_freq, n_mat);

for fi = 1:n_freq
    f = freq_list(fi);
    omega = 2*pi*f;
    V_ptp_f = V_ptp_list(fi);   % PTP voltage at this frequency
    V_app_f = V_ptp_f / 2;      % amplitude on each electrode
    V_tot_f = V_ptp_f;           % total potential difference
    
    % Update study frequency and applied voltage
    model.study('std1').feature('freq').set('plist', num2str(f));
    par.set('V_app', sprintf('%.4f[V]', V_app_f), 'Applied voltage');
    
    for mi = 1:n_mat
        [sig_cc_f, epsr_cc_f] = cole_cole_calc(f, cc_sigma_dc(mi), cc_eps_inf(mi), ...
            cc_delta_eps(mi), cc_tau(mi), cc_alpha(mi));
        
        sig_scaled = sigma_cal(mi) * (sig_cc_f / sig_cc_50k(mi));
        
        % Update blood domain conductivity (numeric, not parameter name)
        mat_blood.propertyGroup('def').set('electricconductivity', ...
            num2str(sig_scaled, '%.6g'));
        
        % Force re-solve (study.run() caches; sol.runAll() always re-solves)
        model.sol('sol1').runAll();
        
        % Z from current (proven correct at baseline)
        I_meas = mphint2(model, 'ec.nJ', 'surface', 'selection', bnd_elec_L);
        Z_sweep(fi, mi) = V_tot_f / abs(I_meas);
        
        % Phase from Cole-Cole (analytical)
        phase_sweep(fi, mi) = -atand(omega * 8.854e-12 * epsr_cc_f / sig_cc_f);
    end
    fprintf('  f=%3.0f kHz: Blood=%4.0f, Clot=%5.0f, Wall=%4.0f\n', ...
        f/1e3, Z_sweep(fi,1), Z_sweep(fi,2), Z_sweep(fi,3));
end

% Restore baseline voltage for film section
par.set('V_app', sprintf('%.4f[V]', V_applied), 'Applied voltage');

%% ====================================================================
%  BLOOD FILM SENSITIVITY (analytical — no shunting)
%  Physics: Blood film on electrode surfaces only, clot/wall connects
%  electrodes. No lateral current spreading in film.
%  Method: Use sensing depth CDF from COMSOL, apply series correction.
%  Z_film = Z_tissue * [1 - CDF(t) * (1 - sigma_tissue/sigma_blood)]
% =====================================================================
fprintf('\n--- Blood film sensitivity (no-shunt, analytical) ---\n');

film_thicknesses = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0];  % mm
n_film = length(film_thicknesses);
Z_film_clot = zeros(1, n_film);
Z_film_wall = zeros(1, n_film);

% Blood baseline (V/I) — needed for figure normalization
mat_blood.propertyGroup('def').set('electricconductivity', num2str(sigma_blood, '%.6g'));
model.study('std1').feature('freq').set('plist', num2str(freq_base));
model.sol('sol1').runAll();
Z_blood_A = V_total / abs(mphint2(model, 'ec.nJ', 'surface', 'selection', bnd_elec_L));
fprintf('  Blood baseline: Z=%.0f Ohm\n', Z_blood_A);

% Pure tissue baselines from COMSOL (no film)
mat_blood.propertyGroup('def').set('electricconductivity', num2str(sigma_clot, '%.6g'));
model.sol('sol1').runAll();
Z_clot_base = V_total / abs(mphint2(model, 'ec.nJ', 'surface', 'selection', bnd_elec_L));

mat_blood.propertyGroup('def').set('electricconductivity', num2str(sigma_wall, '%.6g'));
model.sol('sol1').runAll();
Z_wall_base = V_total / abs(mphint2(model, 'ec.nJ', 'surface', 'selection', bnd_elec_L));

fprintf('  Clot baseline: Z=%.0f Ohm\n', Z_clot_base);
fprintf('  Wall baseline: Z=%.0f Ohm\n', Z_wall_base);

% Fit Weibull CDF to sensing depth data
% CDF(r) = 1 - exp(-(r/lambda)^beta)
% Data from COMSOL sensing depth probe: 50%=0.34mm, 80%=0.53mm, 95%=0.89mm
depth_r = [0.34, 0.53, 0.89] * 1e-3;  % m
depth_cdf = [0.50, 0.80, 0.95];

% Analytical fit: from 50% point, lambda = depth_50 / (-ln(0.5))^(1/beta)
% Iterative fit for beta
best_err = inf;
best_lam = 0.4e-3; best_beta = 1.7;
for beta_try = 1.0:0.01:3.0
    for lam_try = 0.1e-3:0.01e-3:1.0e-3
        cdf_pred = 1 - exp(-(depth_r/lam_try).^beta_try);
        err = sum((cdf_pred - depth_cdf).^2);
        if err < best_err
            best_err = err;
            best_lam = lam_try;
            best_beta = beta_try;
        end
    end
end
lam_w = best_lam;
beta_w = best_beta;
fprintf('  Weibull fit: lambda=%.3f mm, beta=%.2f\n', lam_w*1e3, beta_w);

% Verify fit
for k = 1:3
    cdf_chk = 1 - exp(-(depth_r(k)/lam_w)^beta_w);
    fprintf('    r=%.2fmm: data=%.2f, fit=%.3f\n', depth_r(k)*1e3, depth_cdf(k), cdf_chk);
end

% Compute film effect analytically (no shunting)
% Z_film = Z_tissue * [1 - CDF(t) * (1 - sigma_tissue/sigma_blood)]
% As t->0: Z = Z_tissue (no film)
% As t->inf: Z = Z_tissue * sigma_tissue/sigma_blood = K/sigma_blood = Z_blood
for fi_f = 1:n_film
    t = film_thicknesses(fi_f) * 1e-3;  % m
    cdf_t = 1 - exp(-(t/lam_w)^beta_w);
    if t == 0, cdf_t = 0; end
    
    Z_film_clot(fi_f) = Z_clot_base * (1 - cdf_t * (1 - sigma_clot/sigma_blood));
    Z_film_wall(fi_f) = Z_wall_base * (1 - cdf_t * (1 - sigma_wall/sigma_blood));
    
    fprintf('  Film %.2fmm: CDF=%.3f  Clot=%5.0f  Wall=%5.0f  Ratio=%.2f\n', ...
        film_thicknesses(fi_f), cdf_t, Z_film_clot(fi_f), Z_film_wall(fi_f), ...
        Z_film_clot(fi_f)/Z_film_wall(fi_f));
end

% Restore blood for subsequent sections
mat_blood.propertyGroup('def').set('electricconductivity', num2str(sigma_blood, '%.6g'));
model.sol('sol1').runAll();

%% ====================================================================
%  ELECTRODE INTERFACE (analytical — no COMSOL solve needed)
% =====================================================================
fprintf('\n--- Electrode interface ---\n');

C_oxide_val = 8.854e-12 * epsr_oxide ./ max(d_oxide_m, 1e-20);
C_oxide_val(d_oxide_m == 0) = Inf;

Z_interface_results = zeros(length(electrode_materials), n_freq);
for ei = 1:length(electrode_materials)
    for fi = 1:n_freq
        omega_fi = 2*pi*freq_list(fi);
        Z_cpe = 1 / (CPE_Q(ei) * (1j*omega_fi)^CPE_n(ei));
        if isinf(C_oxide_val(ei)), Z_ox = 0;
        else, Z_ox = 1 / (1j*omega_fi*C_oxide_val(ei)); end
        Z_both = 2 * (Z_cpe + Z_ox + R_ct(ei)) / A_elec_m2;
        Z_interface_results(ei, fi) = abs(Z_sweep(fi, 1) + Z_both);
    end
end
fprintf('  SS316L smooth at 50 kHz: %.0f Ohm (bulk: %.0f)\n', ...
    Z_interface_results(1, 4), Z_sweep(4, 1));

%% ====================================================================
%  THERMAL ANALYSIS (add ht physics NOW — after all ec sweeps)
% =====================================================================
fprintf('\n--- Thermal analysis ---\n');

% Adiabatic
rho_cp = rho_blood_val * Cp_blood_val;
alpha_th = k_blood_val / rho_cp;
dT_1s  = pd_sense / rho_cp;
dT_10s = pd_sense * 10 / rho_cp;
if any(valid)
    fprintf('  Adiabatic: 1s=%.4f C, 10s=%.4f C\n', dT_1s(find(valid,1)), dT_10s(find(valid,1)));
end

% COMSOL Heat Transfer
has_comsol_thermal = false;
T_comsol_1s = zeros(1, n_radial);
T_comsol_10s = zeros(1, n_radial);
dset_th = '';
try
    ht = model.component('comp1').physics.create('ht', 'HeatTransfer', 'geom1');
    ht.feature('solid1').set('k_mat', 'userdef');
    ht.feature('solid1').set('k', 'k_blood');
    ht.feature('solid1').set('rho_mat', 'userdef');
    ht.feature('solid1').set('rho', 'rho_blood');
    ht.feature('solid1').set('Cp_mat', 'userdef');
    ht.feature('solid1').set('Cp', 'Cp_blood');
    
    ht_wall = ht.create('solid2', 'SolidHeatTransferModel', 3);
    ht_wall.selection.set([dom_wall]);
    ht_wall.set('k_mat', 'userdef'); ht_wall.set('k', 'k_wall_th');
    ht_wall.set('rho_mat', 'userdef'); ht_wall.set('rho', 'rho_wall_th');
    ht_wall.set('Cp_mat', 'userdef'); ht_wall.set('Cp', 'Cp_wall_th');
    
    ht_cath = ht.create('solid3', 'SolidHeatTransferModel', 3);
    ht_cath.selection.set([dom_cath]);
    ht_cath.set('k_mat', 'userdef'); ht_cath.set('k', 'k_cath');
    ht_cath.set('rho_mat', 'userdef'); ht_cath.set('rho', 'rho_cath');
    ht_cath.set('Cp_mat', 'userdef'); ht_cath.set('Cp', 'Cp_cath');
    
    hs1 = ht.create('hs1', 'HeatSource', 3);
    hs1.selection.set([dom_blood]);
    hs1.set('Q', 'ec.Qrh');
    
    ht.feature('init1').set('Tinit', 'T_body');
    
    temp_bc = ht.create('temp1', 'TemperatureBoundary', 2);
    wall_bnds = mphgetadj(model, 'geom1', 'boundary', 'domain', dom_wall);
    temp_bc.selection.set(wall_bnds);
    temp_bc.set('T0', 'T_body');
    
    % Combined study (matches v6: freq ec + transient ht)
    std_th = model.study.create('std_th');
    step_ec = std_th.create('freq2', 'Frequency');
    step_ec.set('plist', num2str(freq_base));
    step_ec.setSolveFor('/physics/ec', true);
    step_ec.setSolveFor('/physics/ht', false);
    step_ht = std_th.create('time', 'Transient');
    step_ht.set('tlist', 'range(0,0.1,1) range(2,1,10)');
    step_ht.setSolveFor('/physics/ec', false);
    step_ht.setSolveFor('/physics/ht', true);
    
    fprintf('  Solving coupled thermal...\n');
    std_th.run();
    fprintf('  Thermal solve complete.\n');
    
    % Find thermal dataset
    for dset_try = {'dset4', 'dset3', 'dset5', 'dset2', 'dset6'}
        try
            pt_test = [x_ctr; y_ctr; z_ctr] + probe_dir * r_offsets(1);
            mphinterp(model, 'T', 'coord', pt_test, 'dataset', dset_try{1}, 'solnum', 11);
            dset_th = dset_try{1};
            break;
        catch
        end
    end
    
    if ~isempty(dset_th)
        for k = 1:n_radial
            pt = [x_ctr; y_ctr; z_ctr] + probe_dir * r_offsets(k);
            try
                T_comsol_1s(k) = mphinterp(model, 'T', 'coord', pt, ...
                    'dataset', dset_th, 'solnum', 11) - 310.15;
                T_comsol_10s(k) = mphinterp(model, 'T', 'coord', pt, ...
                    'dataset', dset_th, 'solnum', 20) - 310.15;
            catch
                T_comsol_1s(k) = NaN; T_comsol_10s(k) = NaN;
            end
        end
        has_comsol_thermal = true;
        fprintf('  COMSOL dT surface: 1s=%.4f, 10s=%.4f C\n', ...
            T_comsol_1s(find(valid,1)), T_comsol_10s(find(valid,1)));
    end
catch ME
    fprintf('  COMSOL thermal failed: %s\n', ME.message);
end

% 1D FD thermal (always runs)
r_abs = (r_cath_surface + r_offsets)';
dr_m = r_abs(2) - r_abs(1);
Q_src = pd_sense(:); Q_src(isnan(Q_src)) = 0;
dt_fd = 0.9 * dr_m^2 / (2 * alpha_th);
T_fd = zeros(n_radial, 1);  T_fd_1s = zeros(n_radial, 1);
for step = 1:round(10/dt_fd)
    T_new = T_fd;
    for i = 2:n_radial-1
        d2T = (T_fd(i+1) - 2*T_fd(i) + T_fd(i-1)) / dr_m^2;
        dTdr = (T_fd(i+1) - T_fd(i-1)) / (2*dr_m);
        T_new(i) = T_fd(i) + dt_fd * (alpha_th*(d2T + dTdr/r_abs(i)) + Q_src(i)/rho_cp);
    end
    T_new(1) = T_new(2);
    T_new(end) = T_new(end-1);
    T_fd = T_new;
    if step == round(1/dt_fd), T_fd_1s = T_fd; end
end
T_fd_10s = T_fd;
fprintf('  1D FD: 1s=%.4f, 10s=%.4f C\n', T_fd_1s(1), T_fd_10s(1));

%% ====================================================================
%  COMSOL RESULT PLOT GROUPS
% =====================================================================
fprintf('\n--- Result plot groups ---\n');

pg1 = model.result.create('pg1', 'PlotGroup3D');
pg1.label('1. Electric Potential V [V]');
ms1 = pg1.create('mslc1', 'Multislice');
ms1.set('expr', 'real(V)'); ms1.set('unit', 'V');
ms1.set('multiplanexmethod', 'coord'); ms1.set('xcoord', '0');
ms1.set('multiplaneymethod', 'coord'); ms1.set('ycoord', '0');
ms1.set('multiplanezmethod', 'coord'); ms1.set('zcoord', num2str(z_ctr, '%.4f'));
ms1.active(false);
pg1.run;

pg2 = model.result.create('pg2', 'PlotGroup3D');
pg2.label('2. E-field |E| [V/m] + Field Lines');
ms2 = pg2.create('mslc2', 'Multislice');
ms2.set('expr', 'ec.normE'); ms2.set('unit', 'V/m');
ms2.set('multiplanexmethod', 'coord'); ms2.set('xcoord', '0');
ms2.set('multiplaneymethod', 'coord'); ms2.set('ycoord', '0');
ms2.set('multiplanezmethod', 'coord'); ms2.set('zcoord', num2str(z_ctr, '%.4f'));
ms2.active(false);
% Electric field streamlines — default coloring
sl2 = pg2.create('str1', 'Streamline');
sl2.set('expr', {'ec.Ex', 'ec.Ey', 'ec.Ez'});
sl2.set('posmethod', 'start');
sl2.set('startmethod', 'coord');
% Seed points: fan out from electrode L centroid
n_sl = 12;
th_sl = linspace(0, 2*pi*(1-1/n_sl), n_sl);
sl_r = 0.0005;  % 0.5mm from centroid
sl_x = x_ctr + sl_r*cos(th_sl)*probe_dir(1) - sl_r*sin(th_sl)*probe_dir(2);
sl_y = y_ctr + sl_r*cos(th_sl)*probe_dir(2) + sl_r*sin(th_sl)*probe_dir(1);
sl_z = z_ctr * ones(1, n_sl);
sl2.set('xcoord', num2str(sl_x, '%.6f '));
sl2.set('ycoord', num2str(sl_y, '%.6f '));
sl2.set('zcoord', num2str(sl_z, '%.6f '));
pg2.run;

pg3 = model.result.create('pg3', 'PlotGroup3D');
pg3.label('3. Current Density |J| [A/m^2]');
ms3 = pg3.create('mslc3', 'Multislice');
ms3.set('expr', 'ec.normJ'); ms3.set('unit', 'A/m^2');
ms3.set('multiplanexmethod', 'coord'); ms3.set('xcoord', '0');
ms3.set('multiplaneymethod', 'coord'); ms3.set('ycoord', '0');
ms3.set('multiplanezmethod', 'coord'); ms3.set('zcoord', num2str(z_ctr, '%.4f'));
ms3.active(false);
pg3.run;

pg4 = model.result.create('pg4', 'PlotGroup3D');
pg4.label('4. Joule Heating Qrh [W/m^3]');
ms4 = pg4.create('mslc4', 'Multislice');
ms4.set('expr', 'ec.Qrh'); ms4.set('unit', 'W/m^3');
ms4.set('multiplanexmethod', 'coord'); ms4.set('xcoord', '0');
ms4.set('multiplaneymethod', 'coord'); ms4.set('ycoord', '0');
ms4.set('multiplanezmethod', 'coord'); ms4.set('zcoord', num2str(z_ctr, '%.4f'));
ms4.active(false);
pg4.run;

% --- 2D contour plots on cut planes ---
cp1 = model.result.dataset.create('cp1', 'CutPlane');
cp1.set('quickplane', 'xz'); cp1.set('quicky', num2str(y_ctr, '%.4f'));

cp2 = model.result.dataset.create('cp2', 'CutPlane');
cp2.set('quickplane', 'xy'); cp2.set('quickz', num2str(z_ctr, '%.4f'));

pg5 = model.result.create('pg5', 'PlotGroup2D');
pg5.label('5. Contour: Potential (xz plane)');
pg5.set('data', 'cp1');
surf5 = pg5.create('surf1', 'Surface');
surf5.set('expr', 'real(V)');
con5 = pg5.create('con1', 'Contour');
con5.set('expr', 'real(V)'); con5.set('levelmethod', 'levels');
con5.set('levels', '-0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3');
con5.set('coloring', 'uniform'); con5.set('color', 'black');
pg5.run;

pg6 = model.result.create('pg6', 'PlotGroup2D');
pg6.label('6. Contour: |E| (xy plane at electrode z)');
pg6.set('data', 'cp2');
surf6 = pg6.create('surf1', 'Surface');
surf6.set('expr', 'ec.normE'); surf6.set('unit', 'V/m');
con6 = pg6.create('con1', 'Contour');
con6.set('expr', 'ec.normE'); con6.set('levelmethod', 'levels');
con6.set('levels', '50 100 200 400 800');
con6.set('coloring', 'uniform'); con6.set('color', 'black');
pg6.run;

pg7 = model.result.create('pg7', 'PlotGroup2D');
pg7.label('7. Contour: |J| (xz plane)');
pg7.set('data', 'cp1');
surf7 = pg7.create('surf1', 'Surface');
surf7.set('expr', 'ec.normJ'); surf7.set('unit', 'A/m^2');
con7 = pg7.create('con1', 'Contour');
con7.set('expr', 'ec.normJ'); con7.set('levelmethod', 'levels');
con7.set('levels', '10 50 100 200 500');
con7.set('coloring', 'uniform'); con7.set('color', 'black');
pg7.run;

pg8 = model.result.create('pg8', 'PlotGroup2D');
pg8.label('8. Contour: Joule Heating (xz plane)');
pg8.set('data', 'cp1');
surf8 = pg8.create('surf1', 'Surface');
surf8.set('expr', 'ec.Qrh'); surf8.set('unit', 'W/m^3');
con8 = pg8.create('con1', 'Contour');
con8.set('expr', 'ec.Qrh'); con8.set('levelmethod', 'levels');
con8.set('levels', '100 500 1000 5000 10000');
con8.set('coloring', 'uniform'); con8.set('color', 'black');
pg8.run;

if has_comsol_thermal && ~isempty(dset_th)
    pg9 = model.result.create('pg9', 'PlotGroup3D');
    pg9.label('9. Temperature Rise t=10s');
    pg9.set('data', dset_th); pg9.set('solnum', 20);
    ms9 = pg9.create('mslc9', 'Multislice');
    ms9.set('expr', 'T - T_body'); ms9.set('unit', 'K');
    ms9.set('multiplanexmethod', 'coord'); ms9.set('xcoord', '0');
    ms9.set('multiplaneymethod', 'coord'); ms9.set('ycoord', '0');
    ms9.set('multiplanezmethod', 'coord'); ms9.set('zcoord', num2str(z_ctr, '%.4f'));
    pg9.run;
    fprintf('  pg9: Temperature rise\n');
end
fprintf('  pg1-pg8 created.\n');

%% ====================================================================
%  SAVE MODEL
% =====================================================================
try
    mphsave(model, fullfile(pwd, 'Geom25_complete.mph'));
    fprintf('  Model saved: Geom25_complete.mph\n');
catch
    alt_name = sprintf('Geom25_%s.mph', datestr(now, 'yyyymmdd_HHMMSS'));
    mphsave(model, fullfile(pwd, alt_name));
    fprintf('  Model saved: %s\n', alt_name);
end

%% ====================================================================
%  MATLAB FIGURES
% =====================================================================
fprintf('\n--- Figures ---\n');

clr_blood = [0.17 0.63 0.17];
clr_clot  = [0.84 0.15 0.16];
clr_wall  = [0.12 0.47 0.71];

% ---- Figure 1: Frequency Discrimination Summary (4 subplots) ----
fig1 = figure('Position', [50 50 1200 900]);

subplot(2,2,1);
b1 = bar([Z_sweep(:,2), Z_sweep(:,3)]);
b1(1).FaceColor = clr_clot; b1(2).FaceColor = clr_wall;
set(gca, 'XTickLabel', arrayfun(@(f) sprintf('%.0f kHz', f/1e3), freq_list, 'UniformOutput', false));
ylabel('|Z| [\Omega]'); title('Impedance: Clot vs Wall');
legend({'Clot', 'Wall'}, 'Location', 'northeast'); grid on;

subplot(2,2,2);
bar(Z_sweep(:,2) ./ Z_sweep(:,3), 'FaceColor', [0.5 0 0.5]);
set(gca, 'XTickLabel', arrayfun(@(f) sprintf('%.0f kHz', f/1e3), freq_list, 'UniformOutput', false));
ylabel('Z_{clot} / Z_{wall}'); title('Clot/Wall Ratio');
yline(1, '--k'); grid on;

subplot(2,2,3);
bar(Z_sweep(:,2) - Z_sweep(:,3), 'FaceColor', [0.17 0.63 0.17]);
set(gca, 'XTickLabel', arrayfun(@(f) sprintf('%.0f kHz', f/1e3), freq_list, 'UniformOutput', false));
ylabel('\DeltaZ [\Omega]'); title('Absolute Contrast (Z_{clot} - Z_{wall})'); grid on;

subplot(2,2,4);
bar(phase_sweep(:,2) - phase_sweep(:,3), 'FaceColor', [0.85 0.55 0.13]);
set(gca, 'XTickLabel', arrayfun(@(f) sprintf('%.0f kHz', f/1e3), freq_list, 'UniformOutput', false));
ylabel('\Delta\phi [deg]'); title('Phase Difference (\phi_{clot} - \phi_{wall})'); grid on;

sgtitle('Frequency Discrimination Summary - 3D COMSOL Model');
saveas(fig1, fullfile(out_dir, 'frequency_discrimination_summary.png'));
fprintf('  Saved: frequency_discrimination_summary.png\n');

% ---- Figure 2: 3-Frequency Feature Set ----
fig2 = figure('Position', [100 100 1100 500]);
fi_set = [1, 4, 5];
labels_3f = {'5 kHz', '50 kHz', '100 kHz'};

subplot(1,2,1);
b2 = bar([Z_sweep(fi_set,1)/Z_sweep(4,1), Z_sweep(fi_set,2)/Z_sweep(4,2), Z_sweep(fi_set,3)/Z_sweep(4,3)]);
b2(1).FaceColor = clr_blood; b2(2).FaceColor = clr_clot; b2(3).FaceColor = clr_wall;
set(gca, 'XTickLabel', labels_3f);
ylabel('|Z_f| / |Z_{50}|'); title('Magnitude Ratios (normalized to 50 kHz)');
legend({'Blood', 'Clot', 'Wall'}, 'Location', 'northwest'); grid on;

subplot(1,2,2);
b2b = bar([phase_sweep(fi_set,1)-phase_sweep(4,1), phase_sweep(fi_set,2)-phase_sweep(4,2), phase_sweep(fi_set,3)-phase_sweep(4,3)]);
b2b(1).FaceColor = clr_blood; b2b(2).FaceColor = clr_clot; b2b(3).FaceColor = clr_wall;
set(gca, 'XTickLabel', labels_3f);
ylabel('\Delta\phi [deg]'); title('Phase Deltas (relative to 50 kHz)');
legend({'Blood', 'Clot', 'Wall'}, 'Location', 'northwest'); grid on;

sgtitle('3-Frequency Feature Set: 5, 50, 100 kHz - 3D COMSOL');
saveas(fig2, fullfile(out_dir, '3freq_feature_set.png'));
fprintf('  Saved: 3freq_feature_set.png\n');

% ---- Figure 3: Sensing Depth ----
fig3 = figure('Position', [100 100 1000 500]);
subplot(1,2,1);
if any(valid), semilogy(r_valid, pd_valid/1e3, 'b-', 'LineWidth', 2); end
xlabel('Distance from electrode [mm]'); ylabel('Q_{rh} [kW/m^3]');
title('Joule Heating vs Distance'); grid on;

subplot(1,2,2);
if ~isempty(cum_norm)
    plot(r_valid, cum_norm*100, 'b-', 'LineWidth', 2); hold on;
    yline(50, '--r', '50%'); yline(80, '--', '80%', 'Color', [0.8 0.5 0]);
    yline(95, '--k', '95%');
    if ~isnan(depth_50), xline(depth_50, ':r', sprintf('%.2fmm', depth_50)); end
    if ~isnan(depth_80), xline(depth_80, ':', sprintf('%.2fmm', depth_80), 'Color', [0.8 0.5 0]); end
    if ~isnan(depth_95), xline(depth_95, ':k', sprintf('%.2fmm', depth_95)); end
end
xlabel('Distance from electrode [mm]'); ylabel('Cumulative power [%]');
title(sprintf('Sensing Depth: 50%%=%.2f, 80%%=%.2f, 95%%=%.2f mm', depth_50, depth_80, depth_95));
grid on; ylim([0 105]);
saveas(fig3, fullfile(out_dir, 'sensing_depth.png'));
fprintf('  Saved: sensing_depth.png\n');

% ---- Figure 4: Heating Profile ----
fig4 = figure('Position', [100 100 1000 600]);
if any(valid)
    semilogy(r_from_surface, max(dT_1s,1e-10), 'b--', 'LineWidth', 1.5); hold on;
    semilogy(r_from_surface, max(dT_10s,1e-10), 'r--', 'LineWidth', 1.5);
    semilogy(r_from_surface, max(T_fd_1s,1e-10), 'b-', 'LineWidth', 2);
    semilogy(r_from_surface, max(T_fd_10s,1e-10), 'r-', 'LineWidth', 2);
    leg = {'Adiabatic 1s', 'Adiabatic 10s', '1D FD conduction 1s', '1D FD conduction 10s'};
    if has_comsol_thermal
        semilogy(r_from_surface, max(T_comsol_1s,1e-10), 'b-.', 'LineWidth', 2.5);
        semilogy(r_from_surface, max(T_comsol_10s,1e-10), 'r-.', 'LineWidth', 2.5);
        leg = [leg, {'COMSOL 3D 1s', 'COMSOL 3D 10s'}];
    end
    yline(2, '--k', 'LineWidth', 1.5);
    leg = [leg, {'IEC 60601 limit'}];
    legend(leg, 'Location', 'northeast');
end
xlabel('Distance from electrode surface [mm]'); ylabel('Temperature rise [degC]');
title(sprintf('Heating Profile (Blood, 50 kHz, +/-%.1fV)', V_applied));
grid on; xlim([0, max(r_from_surface)]);
saveas(fig4, fullfile(out_dir, 'heating_profile.png'));
fprintf('  Saved: heating_profile.png\n');

% ---- Figure 5: Electrode Interface ----
fig5 = figure('Position', [100 100 1000 550]);
colors_ei = {[0.7 0.2 0.2], [0.2 0.7 0.2], [0.5 0.5 0.5], [0 0.6 0.8]};
loglog(freq_list/1e3, Z_sweep(:,1), 'k-', 'LineWidth', 3); hold on;
for ei = 1:length(electrode_materials)
    loglog(freq_list/1e3, Z_interface_results(ei,:), '--', 'LineWidth', 2, 'Color', colors_ei{ei});
end
plot(50, Z_target_blood, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'k');
xlabel('Frequency [kHz]'); ylabel('|Z| [\Omega]');
title('Electrode Interface Effect');
legend(['Bulk tissue', electrode_materials, 'Measured'], 'Location', 'northeast');
grid on; xlim([4 110]);
saveas(fig5, fullfile(out_dir, 'electrode_interface.png'));
fprintf('  Saved: electrode_interface.png\n');

% ---- Figure 6: Blood Film Sensitivity (no-shunt analytical) ----
fig6 = figure('Position', [50 50 1400 500]);

subplot(1,3,1);
plot(film_thicknesses, Z_film_clot, 'r-o', 'LineWidth', 2, 'MarkerSize', 6); hold on;
plot(film_thicknesses, Z_film_wall, 'b-s', 'LineWidth', 2, 'MarkerSize', 6);
yline(Z_blood_A, '--g', 'Blood');
xlabel('Film thickness [mm]'); ylabel('|Z| [\Omega]');
title('Impedance vs Blood Film'); legend({'Clot','Wall'}, 'Location', 'east'); grid on;

subplot(1,3,2);
plot(film_thicknesses, Z_film_clot/Z_blood_A, 'r-o', 'LineWidth', 2, 'MarkerSize', 6); hold on;
plot(film_thicknesses, Z_film_wall/Z_blood_A, 'b-s', 'LineWidth', 2, 'MarkerSize', 6);
yline(1, '--k', 'Blood level');
xlabel('Film thickness [mm]'); ylabel('Z / Z_{blood}');
title('Normalized Impedance'); legend({'Clot','Wall'}, 'Location', 'east'); grid on;

subplot(1,3,3);
plot(film_thicknesses, Z_film_clot ./ Z_film_wall, 'k-d', 'LineWidth', 2, 'MarkerSize', 8);
yline(1, '--k');
xlabel('Film thickness [mm]'); ylabel('Z_{clot} / Z_{wall}');
ratio_02 = Z_film_clot(2)/Z_film_wall(2);
title(sprintf('Clot/Wall Discrimination\n(%.2f no film \\rightarrow %.2f at 0.02mm)', ...
    Z_film_clot(1)/Z_film_wall(1), ratio_02));
grid on;

sgtitle('Blood Film Sensitivity - No Shunt (Analytical + COMSOL Sensing Depth)');
saveas(fig6, fullfile(out_dir, 'blood_film_sensitivity.png'));
fprintf('  Saved: blood_film_sensitivity.png\n');

% ---- Figure 7: Impedance Summary ----
fig7 = figure('Position', [50 50 1400 700]);
subplot(2,2,1);
Z_bar = [Z_sweep(4,1), Z_sweep(4,2), Z_sweep(4,3)];
b7 = bar(Z_bar); b7.FaceColor = 'flat';
b7.CData(1,:) = clr_blood; b7.CData(2,:) = clr_clot; b7.CData(3,:) = clr_wall;
set(gca, 'XTickLabel', cc_names); ylabel('|Z| [\Omega]');
title(sprintf('50 kHz (+/-%.1fV)', V_applied)); grid on;

subplot(2,2,2);
loglog(freq_list/1e3, Z_sweep(:,1), '-o', 'Color', clr_blood, 'LineWidth', 2); hold on;
loglog(freq_list/1e3, Z_sweep(:,2), '-o', 'Color', clr_clot, 'LineWidth', 2);
loglog(freq_list/1e3, Z_sweep(:,3), '-o', 'Color', clr_wall, 'LineWidth', 2);
xlabel('Frequency [kHz]'); ylabel('|Z| [\Omega]'); title('|Z| vs Frequency');
legend(cc_names); grid on; xlim([4 110]);

subplot(2,2,3);
semilogx(freq_list/1e3, phase_sweep(:,1), '-o', 'Color', clr_blood, 'LineWidth', 2); hold on;
semilogx(freq_list/1e3, phase_sweep(:,2), '-o', 'Color', clr_clot, 'LineWidth', 2);
semilogx(freq_list/1e3, phase_sweep(:,3), '-o', 'Color', clr_wall, 'LineWidth', 2);
xlabel('Frequency [kHz]'); ylabel('Phase [deg]'); title('Phase vs Frequency');
legend(cc_names, 'Location', 'southwest'); grid on; xlim([4 110]);

subplot(2,2,4);
yyaxis left;
plot(freq_list/1e3, Z_sweep(:,2)./Z_sweep(:,1), 'r-o', 'LineWidth', 2); hold on;
plot(freq_list/1e3, Z_sweep(:,3)./Z_sweep(:,1), 'b-s', 'LineWidth', 2);
ylabel('Z / Z_{blood}');
yyaxis right;
plot(freq_list/1e3, Z_sweep(:,2)./Z_sweep(:,3), 'm-d', 'LineWidth', 2);
ylabel('Clot/Wall');
xlabel('Frequency [kHz]'); title('Discrimination Ratios');
legend({'Clot/Blood','Wall/Blood','Clot/Wall'}, 'Location', 'east'); grid on;

sgtitle('3D COMSOL Real Geometry: Impedance Summary');
saveas(fig7, fullfile(out_dir, 'impedance_spectra_summary.png'));
fprintf('  Saved: impedance_spectra_summary.png\n');

%% ====================================================================
%  SUMMARY
% =====================================================================
fprintf('\n========================================\n');
fprintf('  SUMMARY\n');
fprintf('========================================\n');
fprintf('  Z_blood = %.1f Ohm (target: %d)\n', Z_blood_meas, Z_target_blood);
fprintf('  Sensing: 50%%=%.2f, 80%%=%.2f, 95%%=%.2f mm\n', depth_50, depth_80, depth_95);
fprintf('  Heating (10s): Adiabatic=%.2f, 1D_FD=%.4f, COMSOL=%.4f C\n', ...
    dT_10s(find(valid,1)), T_fd_10s(1), T_comsol_10s(find(valid,1)));
fprintf('  Discrim @50kHz: Clot/Blood=%.2fx, Wall/Blood=%.2fx\n', ...
    Z_sweep(4,2)/Z_sweep(4,1), Z_sweep(4,3)/Z_sweep(4,1));
fprintf('  Film sensitivity (Clot/Wall discrimination, no-shunt):\n');
fprintf('  Film [mm]  Z_clot  Z_wall  Ratio\n');
for fi_f = 1:n_film
    fprintf('  %5.2f      %5.0f   %5.0f   %.2f\n', ...
        film_thicknesses(fi_f), ...
        Z_film_clot(fi_f), Z_film_wall(fi_f), ...
        Z_film_clot(fi_f)/Z_film_wall(fi_f));
end
fprintf('========================================\n');

% Save workspace
save(fullfile(out_dir, 'results_workspace.mat'), ...
    'Z_blood_meas', 'K_real', 'Z_sweep', 'phase_sweep', 'freq_list', ...
    'depth_50', 'depth_80', 'depth_90', 'depth_95', ...
    'pd_sense', 'r_from_surface', 'dT_1s', 'dT_10s', 'T_fd_1s', 'T_fd_10s', ...
    'T_comsol_1s', 'T_comsol_10s', 'has_comsol_thermal', ...
    'Z_film_clot', 'Z_film_wall', 'film_thicknesses', ...
    'Z_interface_results', 'sigma_cal', 'cc_names', ...
    'x_ctr', 'y_ctr', 'z_ctr', 'r_cath_surface');
fprintf('  Workspace saved.\n');

out = model;
end

%% ====================================================================
function [sigma_eff, epsr_eff] = cole_cole_calc(freq, sigma_dc, eps_inf, delta_eps, tau, alpha)
    eps0 = 8.854e-12;
    omega = 2*pi*freq;
    jwt = 1j * omega * tau;
    eps_star = eps_inf + delta_eps / (1 + jwt^(1-alpha));
    epsr_eff = real(eps_star);
    sigma_eff = sigma_dc - omega * eps0 * imag(eps_star);
end
