%% COMSOL LiveLink - Primitive Geometry Bioimpedance Model
% Builds catheter + blood domain from COMSOL primitives (no STEP import).
% Electrodes placed at measured coordinates using Work Planes.
%
% Run in MATLAB launched via "COMSOL with MATLAB"
% Working directory: ..\Comsol\
%
% Author: R. Kurnik / Inquis Medical
% Date: June 2026

clear; close all; clc;

%% ====================================================================
%  CONFIGURATION
% =====================================================================

% Operating parameters
V_applied = 1.5;           % [V] (+/- on electrodes)
freq_base = 50e3;          % [Hz]

% --- Measured electrode corner coordinates [mm] ---
% Left electrode (SA2000_REV_1_TEE_ELECTRODE_LEFT)
elec_L = [1.21939704,  3.455094017, 1.629613301;   % P1 (corner A)
           1.69325454,  3.907606382, 1.402932355;   % P13 (corner B)
           2.985116917, 3.243491046, 2.777720491;   % P33 (corner C)
           2.511259417, 2.790978681, 3.004401437];  % P21 (corner D)

% Right electrode (SA2000_REV_1_TEE_ELECTRODE_RITE)
elec_R = [-1.227925272, 3.463238091, 1.625533619;  % P40 (corner A)
           -1.701782772, 3.915750456, 1.398852673;  % P28 (corner B)
           -2.993645150, 3.251635121, 2.773640809;  % P8  (corner C)
           -2.519787650, 2.799122755, 3.000321755]; % P20 (corner D)

% (No electrode volumes — just partition faces for BCs)

% Blood bath cylinder [mm]
cyl_radius = 8.0;
cyl_length = 40.0;
cyl_center_z = 5.0;   % Center z-coordinate of catheter (~midpoint of -5 to 20)

% (No separate catheter body — just electrodes in medium)

% Cole-Cole parameters: [Saline, Blood, Clot, Wall]
cc_params.sigma_dc  = [9.36,   1.30,   0.155,  0.40];
cc_params.eps_inf   = [76,     50,     40,     40];
cc_params.delta_eps = [0,      2530000, 770000, 1200000];
cc_params.tau       = [1e-12,  10e-6,  12e-6,  9e-6];
cc_params.alpha     = [0.0,    0.25,   0.30,   0.25];

% (No electrode material needed — single domain model)

% Impedance targets
Z_target.blood  = 800;
Z_target.clot   = 3500;
Z_target.wall   = 1800;

fprintf('=== COMSOL LiveLink - Primitive Geometry Model ===\n');
fprintf('  Frequency: %.0f kHz, V = +/-%.1f V\n', freq_base/1e3, V_applied);

%% ====================================================================
%  CREATE MODEL
% =====================================================================
import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('BioimpedanceModel');
model.label('Catheter_Bioimpedance_Primitive');

comp1 = model.component.create('comp1', true);
geom1 = comp1.geom.create('geom1', 3);
geom1.lengthUnit('mm');

fprintf('\n--- Building Geometry ---\n');

%% ====================================================================
%  GEOMETRY: BLOOD CYLINDER
% =====================================================================
cyl_z_start = cyl_center_z - cyl_length/2;

cyl1 = geom1.create('cyl1', 'Cylinder');
cyl1.label('Blood Bath');
cyl1.set('r', cyl_radius);
cyl1.set('h', cyl_length);
cyl1.set('pos', [0, 0, cyl_z_start]);
cyl1.set('axis', [0, 0, 1]);
fprintf('  Blood cylinder: R=%.0f mm, z=[%.0f, %.0f]\n', cyl_radius, cyl_z_start, cyl_z_start+cyl_length);

% (No catheter body — electrodes sit directly in medium)

%% ====================================================================
%  GEOMETRY: ELECTRODE FACES (Work Planes to partition cylinder)
% =====================================================================
% Create work planes at electrode positions, draw rectangles, then use
% Partition to imprint internal boundary faces on the cylinder.
% No electrode volumes needed — just boundary faces for BCs.

fprintf('  Creating electrode face partitions...\n');

% --- Helper: compute electrode local frame ---
eL_width  = elec_L(2,:) - elec_L(1,:);  % A→B (width direction)
eL_length = elec_L(4,:) - elec_L(1,:);  % A→D (length direction)
eL_normal = cross(eL_width, eL_length);
eL_normal = eL_normal / norm(eL_normal);

eR_width  = elec_R(2,:) - elec_R(1,:);
eR_length = elec_R(4,:) - elec_R(1,:);
eR_normal = cross(eR_width, eR_length);
eR_normal = eR_normal / norm(eR_normal);

w_L = norm(eL_width);   l_L = norm(eL_length);
w_R = norm(eR_width);   l_R = norm(eR_length);

fprintf('    Left electrode:  %.3f x %.3f mm, normal=[%.2f, %.2f, %.2f]\n', ...
    w_L, l_L, eL_normal(1), eL_normal(2), eL_normal(3));
fprintf('    Right electrode: %.3f x %.3f mm, normal=[%.2f, %.2f, %.2f]\n', ...
    w_R, l_R, eR_normal(1), eR_normal(2), eR_normal(3));

% --- Left Electrode Work Plane ---
wp1 = geom1.create('wp1', 'WorkPlane');
wp1.label('Left Electrode Plane');
wp1.set('planetype', 'coordinates');
wp1.set('genpoints', [elec_L(1,:); elec_L(2,:); elec_L(4,:)]);
wp1_geom = wp1.geom;
rect1 = wp1_geom.create('rect1', 'Rectangle');
rect1.set('size', [w_L, l_L]);
rect1.set('pos', [0, 0]);

% --- Right Electrode Work Plane ---
wp2 = geom1.create('wp2', 'WorkPlane');
wp2.label('Right Electrode Plane');
wp2.set('planetype', 'coordinates');
wp2.set('genpoints', [elec_R(1,:); elec_R(2,:); elec_R(4,:)]);
wp2_geom = wp2.geom;
rect2 = wp2_geom.create('rect2', 'Rectangle');
rect2.set('size', [w_R, l_R]);
rect2.set('pos', [0, 0]);

%% ====================================================================
%  PARTITION: Imprint electrode faces onto cylinder
% =====================================================================
fprintf('  Partitioning cylinder with electrode faces...\n');

% Partition cylinder with left electrode work plane
par1 = geom1.create('par1', 'Partition');
par1.label('Partition Left Electrode');
par1.selection('input').set({'cyl1'});
par1.selection('tool').set({'wp1'});

% Partition result with right electrode work plane
par2 = geom1.create('par2', 'Partition');
par2.label('Partition Right Electrode');
par2.selection('input').set({'par1'});
par2.selection('tool').set({'wp2'});

%% ====================================================================
%  BUILD GEOMETRY
% =====================================================================

% Build geometry (Form Union finalizes)
geom1.run('fin');
fprintf('  Geometry built.\n');

% The Partition operations split the cylinder with internal face boundaries.
% After building, print domain/boundary counts for reference.
ndom = model.geom('geom1').getNDomains();
nbnd = model.geom('geom1').getNBoundaries();
fprintf('  Domains: %d, Boundaries: %d\n', ndom, nbnd);

% With partition, internal electrode faces become specific boundaries.
% Run with RUN_PHYSICS=false first, inspect in GUI, then set boundary numbers.
% The electrode faces are the ONLY internal boundaries (flat rectangles).
%
% Expected structure after 2 partitions of a cylinder:
%   - Boundaries 1-3: cylinder shell + 2 end caps (or split shell)
%   - Boundaries 4+: internal partition faces (electrode rectangles)
%
% Set these after GUI inspection (or use the print below to identify):
bnd_L_electrode = [];  % Will be auto-set below
bnd_R_electrode = [];

% Use adjacency: internal boundaries border the same domain on both sides
% For a single-domain model with partitions, ALL boundaries border domain 1
% Internal faces are those NOT on the exterior (cylinder shell/caps)
% Exterior boundaries: area >> electrode area
% The partition faces from wp1 and wp2 are the electrode boundaries.
%
% COMSOL assigns boundary numbers sequentially. With Partition operations:
% par1 creates internal face → specific boundary number
% par2 creates another internal face → next boundary number
% The last two boundaries are typically the partition faces.

% Use the fact that internal partition faces are always the highest-numbered
% boundaries (created last by the Partition operations)
bnd_L_electrode = nbnd - 1;  % Second-to-last (from par1/wp1 = left)
bnd_R_electrode = nbnd;      % Last boundary (from par2/wp2 = right)

fprintf('    Left electrode face (boundary %d)\n', bnd_L_electrode);
fprintf('    Right electrode face (boundary %d)\n', bnd_R_electrode);
fprintf('    (Verify in COMSOL GUI if results seem wrong)\n');

%% ====================================================================
%  SAVE STAGE 1 — INSPECT IN GUI
% =====================================================================
mph_file = fullfile(pwd, 'catheter_primitive_stage1.mph');
mphsave(model, mph_file);
fprintf('\n========================================\n');
fprintf('  GEOMETRY COMPLETE\n');
fprintf('  Saved: %s\n', mph_file);
fprintf('========================================\n');
fprintf('\n  Open in COMSOL GUI to verify:\n');
fprintf('  1. Blood domain surrounds catheter correctly\n');
fprintf('  2. Electrode pads are visible at correct positions\n');
fprintf('  3. Note domain numbers (blood, catheter, elec_L, elec_R)\n');
fprintf('  4. Note boundary numbers for electrode faces exposed to blood\n');
fprintf('\n  Then update domain/boundary indices below and set RUN_PHYSICS=true\n');

%% ====================================================================
%  PHYSICS (set RUN_PHYSICS = true after geometry verification)
% =====================================================================
RUN_PHYSICS = true;

if ~RUN_PHYSICS
    fprintf('\n  Set RUN_PHYSICS = true after verifying geometry.\n');
    return;
end

% Single domain — blood medium fills the entire cylinder
dom_medium = 1;

% --- Compute blood properties ---
eps0 = 8.854e-12;
omega = 2*pi*freq_base;
sigma_blood_star = cole_cole_sigma(freq_base, ...
    cc_params.sigma_dc(2), cc_params.eps_inf(2), ...
    cc_params.delta_eps(2), cc_params.tau(2), cc_params.alpha(2));
sigma_blood = real(sigma_blood_star);
epsr_blood  = imag(sigma_blood_star) / (omega * eps0);
fprintf('  Blood: sigma=%.4f S/m, epsr=%.0f\n', sigma_blood, epsr_blood);

% --- Material: Blood ---
mat1 = comp1.material.create('mat_blood', 'Common');
mat1.label('Blood');
mat1.selection.set(dom_medium);
mat1.propertyGroup('def').set('electricconductivity', num2str(sigma_blood, '%.6g'));
mat1.propertyGroup('def').set('relpermittivity', num2str(epsr_blood, '%.6g'));

% --- Physics: Electric Currents ---
fprintf('  Physics: Electric Currents...\n');
ec = comp1.physics.create('ec', 'ConductiveMedia', 'geom1');

% Electric Potential on left electrode face
pot1 = ec.create('pot1', 'ElectricPotential', 2);
pot1.selection.set(bnd_L_electrode);
pot1.set('V0', V_applied);
pot1.label('Left Electrode (+V)');

% Ground on right electrode face
pot2 = ec.create('pot2', 'ElectricPotential', 2);
pot2.selection.set(bnd_R_electrode);
pot2.set('V0', -V_applied);
pot2.label('Right Electrode (-V)');

% --- Mesh ---
fprintf('  Meshing...\n');
mesh1 = comp1.mesh.create('mesh1');
mesh1.feature('size').set('hauto', 4);  % Fine

% Refine near electrode faces
sz1 = mesh1.create('sz_elec', 'Size');
sz1.selection.geom('geom1', 2).set([bnd_L_electrode, bnd_R_electrode]);
sz1.set('custom', true);
sz1.set('hmax', '0.15');
sz1.set('hmin', '0.02');

mesh1.run;
fprintf('  Mesh complete.\n');

% Print mesh stats
fprintf('  Mesh complete (stats available in GUI).\n');

% --- Study & Solve ---
fprintf('  Solving...\n');
std1 = model.study.create('std1');
std1.label('Bioimpedance 50kHz');
std1.create('stat', 'Stationary');

tic;
model.study('std1').run();
fprintf('  Solved in %.1f s\n', toc);

% --- Diagnostics ---
fprintf('\n--- Diagnostics ---\n');

% Check solution range (min/max V in model)
d = mpheval(model, 'V', 'edim', 0, 'selection', 'all');
fprintf('  V range: [%.6f, %.6f] V\n', min(d.d1), max(d.d1));
fprintf('  (expect range [-1.5, 1.5] if BCs applied correctly)\n');

% --- Results ---
fprintf('\n--- Results ---\n');

% Compute impedance: integrate total current density
% Use mphint2 without selection (integrates over all domains)
P_total = mphint2(model, 'ec.Qrh', 'volume');
fprintf('  Total dissipated power (ec.Qrh): %.6e W\n', P_total);

if P_total > 0
    Z_magnitude = (2*V_applied)^2 / P_total;  % V_total = 3V (+1.5 to -1.5)
    fprintf('  Impedance |Z| = V_total^2/P = %.1f Ohm  (V_total=%.1fV)\n', Z_magnitude, 2*V_applied);
else
    fprintf('  WARNING: P=0, no current flowing. Checking alternative...\n');
    % Try direct current integration on a boundary
    I_val = mphint2(model, 'ec.nJ', 'surface', 'selection', bnd_L_electrode);
    fprintf('  Current at left electrode (ec.nJ): %.6e A\n', I_val);
    if I_val ~= 0
        Z_magnitude = (2*V_applied) / abs(I_val);
        fprintf('  Impedance |Z| = V/I = %.1f Ohm\n', Z_magnitude);
    end
end
fprintf('  Target (blood) = %d Ohm\n', Z_target.blood);

% Create 3D plot: Electric potential
pg1 = model.result.create('pg1', 'PlotGroup3D');
pg1.label('Electric Potential');
surf1 = pg1.create('surf1', 'Surface');
surf1.set('expr', 'V');
pg1.run;

% Create 3D plot: Current density
pg2 = model.result.create('pg2', 'PlotGroup3D');
pg2.label('Current Density');
surf2 = pg2.create('surf2', 'Surface');
surf2.set('expr', 'ec.normJ');
pg2.run;

% Save model
mph_out = fullfile(pwd, 'catheter_primitive_solved.mph');
mphsave(model, mph_out);
fprintf('  Model saved: %s\n', mph_out);
fprintf('  Open in COMSOL GUI → Results\n');
fprintf('\n=== COMPLETE ===\n');

%% ====================================================================
function sigma_star = cole_cole_sigma(freq, sigma_dc, eps_inf, delta_eps, tau, alpha)
    eps0 = 8.854e-12;
    omega = 2*pi*freq;
    eps_star = eps_inf + delta_eps ./ (1 + (1j*omega*tau).^(1-alpha));
    sigma_star = sigma_dc + 1j*omega*eps0*eps_star;
end
