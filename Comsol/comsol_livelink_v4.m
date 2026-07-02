%% COMSOL LiveLink - Bioimpedance Model v4
% Two rectangular electrode pads inside a blood-filled cylinder.
% Based on working test (comsol_test_simple.m) pattern.
%
% Run in MATLAB launched via "COMSOL with MATLAB"
% Working directory: ..\Comsol\

clear; close all; clc;

import com.comsol.model.*
import com.comsol.model.util.*

%% ====================================================================
%  CONFIGURATION
% =====================================================================
V_applied = 1.5;           % [V] (+/- on electrodes)
freq_base = 50e3;          % [Hz]

% Blood bath cylinder [mm]
cyl_radius = 8.0;
cyl_length = 40.0;
cyl_z_start = -15.0;

% Catheter body [mm]
% Electrodes sit at radial distance ~3.3-3.9 mm from center.
% Catheter outer radius = min radial distance of electrode corners.
cath_radius = 3.3;         % Outer radius — electrodes on this surface
cath_length = 25.0;        % z: -5 to 20 mm
cath_z_start = -5.0;

% Electrode dimensions [mm] (measured from CAD)
elec_width  = 0.693;   % narrow dimension
elec_length = 2.000;   % long dimension
elec_thickness = 0.1;  % thin pad

% Electrode centers (measured from CAD, mm)
% Left electrode center
elec_L_corners = [1.21939704,  3.455094017, 1.629613301;
                  1.69325454,  3.907606382, 1.402932355;
                  2.985116917, 3.243491046, 2.777720491;
                  2.511259417, 2.790978681, 3.004401437];
ctr_L = mean(elec_L_corners, 1);

% Right electrode center
elec_R_corners = [-1.227925272, 3.463238091, 1.625533619;
                  -1.701782772, 3.915750456, 1.398852673;
                  -2.993645150, 3.251635121, 2.773640809;
                  -2.519787650, 2.799122755, 3.000321755];
ctr_R = mean(elec_R_corners, 1);

% Cole-Cole parameters: [Saline, Blood, Clot, Wall]
cc_sigma_dc  = [9.36,   1.30,   0.155,  0.40];
cc_eps_inf   = [76,     50,     40,     40];
cc_delta_eps = [0,      2530000, 770000, 1200000];
cc_tau       = [1e-12,  10e-6,  12e-6,  9e-6];
cc_alpha     = [0.0,    0.25,   0.30,   0.25];

% Impedance targets
Z_target_blood = 800;
Z_target_clot  = 3500;
Z_target_wall  = 1800;

fprintf('=== COMSOL LiveLink - Bioimpedance Model v4 ===\n');
fprintf('  Frequency: %.0f kHz, V = +/-%.1f V\n', freq_base/1e3, V_applied);
fprintf('  Left electrode center:  [%.2f, %.2f, %.2f] mm\n', ctr_L);
fprintf('  Right electrode center: [%.2f, %.2f, %.2f] mm\n', ctr_R);

%% ====================================================================
%  CREATE MODEL
% =====================================================================
model = ModelUtil.create('BioZ');
comp1 = model.component.create('comp1', true);
geom1 = comp1.geom.create('geom1', 3);
geom1.lengthUnit('mm');

%% ====================================================================
%  GLOBAL DEFINITIONS: Cole-Cole Parameters
% =====================================================================
fprintf('\n--- Global Definitions ---\n');

% Add Cole-Cole parameters to model (visible in COMSOL GUI)
par = model.param;

% Tissue labels for reference
tissues = {'saline', 'blood', 'clot', 'wall'};

for k = 1:4
    t = tissues{k};
    par.set(['sigma_dc_' t], num2str(cc_sigma_dc(k), '%.4g'), ...
        ['DC conductivity ' t ' [S/m]']);
    par.set(['eps_inf_' t], num2str(cc_eps_inf(k)), ...
        ['High-freq permittivity ' t]);
    par.set(['delta_eps_' t], num2str(cc_delta_eps(k), '%.4g'), ...
        ['Cole-Cole delta_eps ' t]);
    par.set(['tau_' t], num2str(cc_tau(k), '%.2e'), ...
        ['Cole-Cole tau ' t ' [s]']);
    par.set(['alpha_' t], num2str(cc_alpha(k), '%.2f'), ...
        ['Cole-Cole alpha ' t]);
end

% Operating parameters
par.set('f0', [num2str(freq_base) '[Hz]'], 'Excitation frequency');
par.set('V_app', [num2str(V_applied) '[V]'], 'Applied voltage amplitude');
par.set('sigma_blood', '0.55[S/m]', 'Blood conductivity at 50 kHz');
par.set('R_bath', [num2str(cyl_radius) '[mm]'], 'Bath cylinder radius');
par.set('R_cath', [num2str(cath_radius) '[mm]'], 'Catheter outer radius');

fprintf('  Cole-Cole parameters added for: %s\n', strjoin(tissues, ', '));
fprintf('  f0=%.0f kHz, V_app=%.1f V, sigma_blood=0.55 S/m\n', freq_base/1e3, V_applied);

%% ====================================================================
%  GEOMETRY
% =====================================================================
fprintf('\n--- Building Geometry ---\n');

% Blood cylinder (large bath, axis along z)
cyl1 = geom1.create('cyl1', 'Cylinder');
cyl1.label('Blood Bath');
cyl1.set('r', cyl_radius);
cyl1.set('h', cyl_length);
cyl1.set('pos', [0, 0, cyl_z_start]);
cyl1.set('axis', [0, 0, 1]);
fprintf('  Blood cylinder: R=%.0f mm, z=[%.0f, %.0f] mm\n', ...
    cyl_radius, cyl_z_start, cyl_z_start+cyl_length);

% Catheter body (solid insulating cylinder, axis along z)
cyl2 = geom1.create('cyl2', 'Cylinder');
cyl2.label('Catheter Body');
cyl2.set('r', cath_radius);
cyl2.set('h', cath_length);
cyl2.set('pos', [0, 0, cath_z_start]);
cyl2.set('axis', [0, 0, 1]);
fprintf('  Catheter body: R=%.1f mm, z=[%.0f, %.0f] mm\n', ...
    cath_radius, cath_z_start, cath_z_start+cath_length);

% Electrode pads: use Work Planes from measured corner coordinates,
% draw rectangle in local 2D, extrude to create oriented thin pad.

% Compute electrode edge vectors and dimensions
eL_AB = elec_L_corners(2,:) - elec_L_corners(1,:);  % width direction
eL_AD = elec_L_corners(4,:) - elec_L_corners(1,:);  % length direction
eL_normal = cross(eL_AB, eL_AD);
eL_normal = eL_normal / norm(eL_normal);
w_L = norm(eL_AB);  l_L = norm(eL_AD);

eR_AB = elec_R_corners(2,:) - elec_R_corners(1,:);
eR_AD = elec_R_corners(4,:) - elec_R_corners(1,:);
eR_normal = cross(eR_AB, eR_AD);
eR_normal = eR_normal / norm(eR_normal);
w_R = norm(eR_AB);  l_R = norm(eR_AD);

fprintf('  Left electrode:  %.3f x %.3f mm, normal=[%.2f, %.2f, %.2f]\n', ...
    w_L, l_L, eL_normal);
fprintf('  Right electrode: %.3f x %.3f mm, normal=[%.2f, %.2f, %.2f]\n', ...
    w_R, l_R, eR_normal);

% Left Electrode: Work Plane → Rectangle → Extrude
wp1 = geom1.create('wp1', 'WorkPlane');
wp1.label('Left Electrode Plane');
wp1.set('planetype', 'coordinates');
wp1.set('genpoints', [elec_L_corners(1,:); elec_L_corners(2,:); elec_L_corners(4,:)]);
wp1_geom = wp1.geom;
rect1 = wp1_geom.create('rect1', 'Rectangle');
rect1.set('size', [w_L, l_L]);
rect1.set('pos', [0, 0]);
ext1 = geom1.create('ext1', 'Extrude');
ext1.label('Left Electrode Pad');
ext1.setIndex('distance', num2str(elec_thickness), 0);
ext1.selection('input').set({'wp1'});

% Right Electrode: Work Plane → Rectangle → Extrude
wp2 = geom1.create('wp2', 'WorkPlane');
wp2.label('Right Electrode Plane');
wp2.set('planetype', 'coordinates');
wp2.set('genpoints', [elec_R_corners(1,:); elec_R_corners(2,:); elec_R_corners(4,:)]);
wp2_geom = wp2.geom;
rect2 = wp2_geom.create('rect2', 'Rectangle');
rect2.set('size', [w_R, l_R]);
rect2.set('pos', [0, 0]);
ext2 = geom1.create('ext2', 'Extrude');
ext2.label('Right Electrode Pad');
ext2.setIndex('distance', num2str(elec_thickness), 0);
ext2.selection('input').set({'wp2'});

% Build — Form Union creates shared boundaries between blocks and cylinder
geom1.run('fin');

ndom = model.geom('geom1').getNDomains();
nbnd = model.geom('geom1').getNBoundaries();
fprintf('  Geometry built: %d domains, %d boundaries\n', ndom, nbnd);

% Expected: 4 domains (blood=annular, catheter=solid cylinder, 2 electrodes)
fprintf('\n  Identifying domains...\n');

% Electrode volume (tiny)
V_elec = elec_thickness * elec_width * elec_length;
fprintf('  Expected electrode volume: %.4f mm^3\n', V_elec);

for d = 1:ndom
    bnds = mphgetadj(model, 'geom1', 'boundary', 'domain', d);
    fprintf('  Domain %d: %d boundaries [%s]\n', d, length(bnds), num2str(bnds(1:min(6,end))));
end

% Identify domains by boundary count:
%   Blood (annular): most boundaries (wraps around catheter + electrodes)
%   Catheter: second most (solid cylinder, many curved boundaries)
%   Electrodes: fewest (small blocks, ~6 boundaries each)
bnd_counts = zeros(1, ndom);
for d = 1:ndom
    bnd_counts(d) = length(mphgetadj(model, 'geom1', 'boundary', 'domain', d));
end
[sorted_counts, sorted_idx] = sort(bnd_counts, 'descend');

dom_blood = sorted_idx(1);      % Most boundaries
dom_catheter = sorted_idx(2);   % Second most
% Electrode domains: the two with fewest boundaries
dom_elec_L = sorted_idx(3);
dom_elec_R = sorted_idx(4);

fprintf('  Assignment: blood=dom%d, catheter=dom%d, elec_L=dom%d, elec_R=dom%d\n', ...
    dom_blood, dom_catheter, dom_elec_L, dom_elec_R);

%% ====================================================================
%  MATERIALS
% =====================================================================
fprintf('\n--- Materials ---\n');

% Blood conductivity and permittivity from Cole-Cole at 50 kHz
eps0 = 8.854e-12;
omega = 2*pi*freq_base;
% Cole-Cole: eps*(w) = eps_inf + delta_eps / (1 + (jw*tau)^(1-alpha))
jwt = 1j * omega * cc_tau(2);
eps_star = cc_eps_inf(2) + cc_delta_eps(2) / (1 + jwt^(1-cc_alpha(2)));
% Use standard whole blood conductivity (0.55 S/m at 50 kHz)
% cc_sigma_dc(2)=1.30 is for diluted/model blood in saline
sigma_blood = 0.55;                   % S/m (whole blood, literature value)
epsr_blood = real(eps_star);           % Real part of relative permittivity
fprintf('  Blood at %.0f kHz: sigma=%.4f S/m, epsr=%.0f\n', ...
    freq_base/1e3, sigma_blood, epsr_blood);

mat1 = comp1.material.create('mat_blood', 'Common');
mat1.label('Blood');
mat1.selection.set(dom_blood);
mat1.propertyGroup('def').set('electricconductivity', num2str(sigma_blood, '%.6g'));
mat1.propertyGroup('def').set('relpermittivity', num2str(epsr_blood, '%.6g'));

% Electrode + catheter domains are geometric only (not solved)
fprintf('  (Electrode + catheter domains are geometric only)\n');

%% ====================================================================
%  PHYSICS
% =====================================================================
fprintf('\n--- Physics ---\n');

ec = comp1.physics.create('ec', 'ConductiveMedia', 'geom1');
% CRITICAL: Only solve in the blood domain. Electrode domains are just
% geometric — they define boundary faces for BCs.
ec.selection.set(dom_blood);
fprintf('  Physics applied to domain %d (blood) only.\n', dom_blood);

% Get boundaries shared between blood and electrode domains
% These are the electrode-blood interface faces
bnd_L = mphgetadj(model, 'geom1', 'boundary', 'domain', dom_elec_L);
bnd_R = mphgetadj(model, 'geom1', 'boundary', 'domain', dom_elec_R);
fprintf('  Left electrode boundaries:  [%s]\n', num2str(bnd_L));
fprintf('  Right electrode boundaries: [%s]\n', num2str(bnd_R));

pot1 = ec.create('pot1', 'ElectricPotential', 2);
pot1.selection.set(bnd_L);
pot1.set('V0', V_applied);
pot1.label('Left Electrode (+V)');

pot2 = ec.create('pot2', 'ElectricPotential', 2);
pot2.selection.set(bnd_R);
pot2.set('V0', -V_applied);
pot2.label('Right Electrode (-V)');

%% ====================================================================
%  MESH
% =====================================================================
fprintf('\n--- Mesh ---\n');

mesh1 = comp1.mesh.create('mesh1');
mesh1.feature('size').set('hauto', 3);  % Finer

mesh1.run;
fprintf('  Mesh complete.\n');

%% ====================================================================
%  SOLVE (Frequency Domain at 50 kHz)
% =====================================================================
fprintf('\n--- Solve ---\n');

std1 = model.study.create('std1');
std1.create('freq', 'Frequency');
std1.feature('freq').set('plist', num2str(freq_base));
fprintf('  Frequency domain study at %.0f kHz\n', freq_base/1e3);

tic;
model.study('std1').run();
fprintf('  Solved in %.1f s\n', toc);

%% ====================================================================
%  RESULTS
% =====================================================================
fprintf('\n--- Results ---\n');

% For frequency domain, V is complex. Check magnitude.
pd = mpheval(model, 'abs(V)');
fprintf('  |V| range: [%.4f, %.4f] V\n', min(pd.d1), max(pd.d1));

% Compute impedance from power:
% In AC: P = 0.5 * Re(integral(J.E*)) but COMSOL ec.Qrh gives time-avg
% For frequency domain: Z = |V_total|^2 / (2*P_avg)
% Or simply: integrate ec.Qrh which is the time-averaged dissipation
P_blood = mphint2(model, 'ec.Qrh', 'volume', 'selection', dom_blood);
fprintf('  Time-avg power in blood: %.6e W\n', P_blood);

% For voltage BCs with peak amplitude V_applied:
% Z = V_total_peak^2 / (2 * P_avg)  [for sinusoidal excitation]
V_total = 2 * V_applied;  % peak-to-peak across electrodes
Z_sim = V_total^2 / (2 * P_blood);
fprintf('  Impedance |Z| = V_peak^2/(2P) = %.1f Ohm\n', Z_sim);
fprintf('  Target (blood): %d Ohm\n', Z_target_blood);

% Create result plots
pg1 = model.result.create('pg1', 'PlotGroup3D');
pg1.label('Electric Potential');
ms1 = pg1.create('mslc1', 'Multislice');
ms1.set('expr', 'V');
ms1.set('multiplanexmethod', 'coord');
ms1.set('xcoord', '0');
ms1.set('multiplaneymethod', 'coord');
ms1.set('ycoord', num2str(ctr_L(2), '%.2f'));
ms1.set('multiplanezmethod', 'coord');
ms1.set('zcoord', num2str(ctr_L(3), '%.2f'));
pg1.run;

pg2 = model.result.create('pg2', 'PlotGroup3D');
pg2.label('Current Density');
ms2 = pg2.create('mslc2', 'Multislice');
ms2.set('expr', 'ec.normJ');
ms2.set('multiplanexmethod', 'coord');
ms2.set('xcoord', '0');
ms2.set('multiplaneymethod', 'coord');
ms2.set('ycoord', num2str(ctr_L(2), '%.2f'));
ms2.set('multiplanezmethod', 'coord');
ms2.set('zcoord', num2str(ctr_L(3), '%.2f'));
pg2.run;

% Field lines (current density streamlines)
pg3 = model.result.create('pg3', 'PlotGroup3D');
pg3.label('Current Field Lines');
str1 = pg3.create('str1', 'Streamline');
str1.set('expr', {'ec.Jx', 'ec.Jy', 'ec.Jz'});
str1.set('posmethod', 'start');
str1.set('startmethod', 'coord');
% Start points on left electrode face (grid across its surface)
npts = 5;
x_start = linspace(ctr_L(1)-0.3, ctr_L(1)+0.3, npts);
z_start = linspace(ctr_L(3)-0.8, ctr_L(3)+0.8, npts);
[XS, ZS] = meshgrid(x_start, z_start);
YS = ctr_L(2) * ones(size(XS));
str1.set('xcoord', XS(:)');
str1.set('ycoord', YS(:)');
str1.set('zcoord', ZS(:)');
pg3.run;

% Save
mph_out = fullfile(pwd, 'bioimpedance_v4_solved.mph');
mphsave(model, mph_out);
fprintf('\n  Model saved: %s\n', mph_out);
fprintf('  Open in COMSOL GUI → Results for plots\n');
fprintf('\n=== COMPLETE ===\n');
