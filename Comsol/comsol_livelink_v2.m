%% COMSOL LiveLink - 3D Bioimpedance Catheter (Two-Stage)
% Stage 1: Build geometry (STEP + blood cylinder), save .mph for inspection
% Stage 2: Assign physics, mesh, solve, extract impedance
%
% Geometry:
%   - G25.STEP = hollow catheter body (polypropylene insulator)
%   - Surrounding cylinder = blood domain
%   - Electrodes are faces on the catheter surface (boundaries 106, 40 pre-Boolean)
%   - Interior of catheter is open → blood fills it through open ends
%
% Run in MATLAB launched via "COMSOL with MATLAB"
% Working directory: ..\Comsol\ (where G25.STEP lives)
%
% Author: R. Kurnik / Inquis Medical
% Date: June 2026

clear; close all; clc;

%% ====================================================================
%  CONFIGURATION
% =====================================================================
step_file = fullfile(pwd, 'G25.STEP');

% Operating parameters
V_applied = 1.5;           % Voltage amplitude [V] (+/- on electrodes)
freq_base = 50e3;          % Excitation frequency [Hz]

% Blood bath cylinder dimensions [mm] (geometry units)
% Catheter spans z: -5 to 20 mm, cross-section ±4 mm (x,y)
% Cylinder must fully enclose with margin
cyl_radius = 8.0;          % [mm] (catheter cross-section is ±4, add margin)
cyl_length = 40.0;         % [mm] (catheter is 25mm, add 15mm margin)
cyl_pos    = [0, 0, -10];  % [mm] base of cylinder (starts before catheter)
cyl_axis   = [0, 0, 1];    % along z-axis (catheter long axis)

% Cole-Cole parameters: [Saline, Blood, Clot, Wall]
cc_params.sigma_dc  = [9.36,   1.30,   0.155,  0.40];
cc_params.eps_inf   = [76,     50,     40,     40];
cc_params.delta_eps = [0,      2530000, 770000, 1200000];
cc_params.tau       = [1e-12,  10e-6,  12e-6,  9e-6];
cc_params.alpha     = [0.0,    0.25,   0.30,   0.25];

% Catheter material
sigma_catheter = 1e-10;   % [S/m]
epsr_catheter  = 2.2;

% Impedance targets at 50 kHz
Z_target.blood  = 800;
Z_target.clot   = 3500;
Z_target.wall   = 1800;

% =====================================================================
% BOUNDARY INDICES (update after Stage 1 inspection)
% Before Boolean: electrode faces were 106, 40
% After Boolean they may change. Set to 0 until verified.
% =====================================================================
bnd_electrode1 = 112;  % Left electrode (verified Stage 1)
bnd_electrode2 = 44;   % Right electrode (verified Stage 1)
dom_blood      = 1;    % Blood domain (verify: click exterior space in Domain mode)
dom_catheter   = 2;    % Catheter body (verified Stage 1)

% =====================================================================
% STAGE CONTROL
%   RUN_STAGE = 1  →  geometry only, saves .mph
%   RUN_STAGE = 2  →  full solve (needs correct indices above)
% =====================================================================
RUN_STAGE = 1;

fprintf('=== COMSOL LiveLink - Catheter Bioimpedance ===\n');
fprintf('  STEP: %s\n', step_file);
fprintf('  Stage: %d\n\n', RUN_STAGE);

%% ====================================================================
%  BUILD MODEL + GEOMETRY
% =====================================================================
import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('CatheterModel');
model.label('Catheter_Bioimpedance_3D');

comp1 = model.component.create('comp1', true);
geom1 = comp1.geom.create('geom1', 3);
geom1.lengthUnit('mm');

% --- Import STEP ---
fprintf('  Importing STEP...\n');
imp1 = geom1.create('imp1', 'Import');
imp1.set('filename', step_file);
imp1.set('type', 'cad');
imp1.importData();

% --- Blood bath cylinder ---
fprintf('  Creating blood cylinder (R=%.1f, L=%.0f mm)...\n', cyl_radius, cyl_length);
cyl1 = geom1.create('cyl1', 'Cylinder');
cyl1.set('r', cyl_radius);
cyl1.set('h', cyl_length);
cyl1.set('pos', cyl_pos);
cyl1.set('axis', cyl_axis);

% --- No explicit Boolean ---
% Let Form Union (default finalize) handle domain partitioning.
% This creates separate domains where objects overlap:
%   - Catheter walls (overlap region) = insulator domain
%   - Everything else inside cylinder (incl. catheter interior) = blood domain
fprintf('  Using Form Union (no explicit Boolean)...\n');

% --- Build ---
geom1.run('fin');
fprintf('  Geometry built.\n');

%% ====================================================================
%  STAGE 1: SAVE & EXIT
% =====================================================================
if RUN_STAGE == 1
    mph_file = fullfile(pwd, 'catheter_stage1.mph');
    mphsave(model, mph_file);
    fprintf('\n========================================\n');
    fprintf('  STAGE 1 COMPLETE\n');
    fprintf('  Saved: %s\n', mph_file);
    fprintf('========================================\n');
    fprintf('\n  NEXT STEPS:\n');
    fprintf('  1. Open catheter_stage1.mph in COMSOL Desktop\n');
    fprintf('  2. Switch to "Select Boundary" mode\n');
    fprintf('  3. Click each electrode face → note boundary numbers\n');
    fprintf('  4. Switch to "Select Domain" → click blood region, note domain #\n');
    fprintf('  5. Click catheter body → note domain #\n');
    fprintf('  6. Update bnd_electrode1, bnd_electrode2, dom_blood, dom_catheter\n');
    fprintf('  7. Set RUN_STAGE = 2 and re-run this script\n');
    fprintf('\n  TIP: The cylinder may not fully enclose the catheter.\n');
    fprintf('       If so, adjust cyl_radius, cyl_length, cyl_pos and re-run Stage 1.\n');
    return;
end

%% ====================================================================
%  STAGE 2: MATERIALS + PHYSICS + MESH + SOLVE
% =====================================================================
if bnd_electrode1 == 0 || bnd_electrode2 == 0 || dom_blood == 0
    error('Indices not set! Complete Stage 1 inspection first.');
end

fprintf('\n--- Stage 2: Full Solve ---\n');

% --- Compute blood properties from Cole-Cole ---
eps0 = 8.854e-12;
omega = 2*pi*freq_base;
sigma_blood_star = cole_cole_sigma(freq_base, ...
    cc_params.sigma_dc(2), cc_params.eps_inf(2), ...
    cc_params.delta_eps(2), cc_params.tau(2), cc_params.alpha(2));
sigma_blood = real(sigma_blood_star);
epsr_blood  = imag(sigma_blood_star) / (omega * eps0);

fprintf('  Blood: sigma=%.4f S/m, epsr=%.0f\n', sigma_blood, epsr_blood);

% --- Materials ---
mat1 = comp1.material.create('mat_catheter', 'Common');
mat1.label('Polypropylene');
mat1.selection.set(dom_catheter);
mat1.propertyGroup('def').set('electricconductivity', num2str(sigma_catheter));
mat1.propertyGroup('def').set('relpermittivity', num2str(epsr_catheter));

mat2 = comp1.material.create('mat_blood', 'Common');
mat2.label('Blood (50 kHz)');
mat2.selection.set(dom_blood);
mat2.propertyGroup('def').set('electricconductivity', num2str(sigma_blood, '%.6g'));
mat2.propertyGroup('def').set('relpermittivity', num2str(epsr_blood, '%.6g'));

% --- Physics: Electric Currents ---
fprintf('  Physics: Electric Currents...\n');
ec = comp1.physics.create('ec', 'ConductiveMedia', 'geom1');

% Electrode BCs
ep1 = ec.create('ep1', 'ElectricPotential', 2);
ep1.selection.set(bnd_electrode1);
ep1.set('V0', V_applied);
ep1.label('Electrode 1 (+1.5V)');

ep2 = ec.create('ep2', 'ElectricPotential', 2);
ep2.selection.set(bnd_electrode2);
ep2.set('V0', -V_applied);
ep2.label('Electrode 2 (-1.5V)');

% All other boundaries = insulation (natural Neumann, automatic)

% --- Mesh ---
fprintf('  Meshing (Fine + electrode refinement)...\n');
mesh1 = comp1.mesh.create('mesh1');
mesh1.feature('size').set('hauto', 4);  % Fine

sz1 = mesh1.create('sz_elec', 'Size');
sz1.selection.geom('geom1', 2).set([bnd_electrode1, bnd_electrode2]);
sz1.set('custom', true);
sz1.set('hmax', '0.3');
sz1.set('hmin', '0.05');
sz1.set('hgrad', '1.3');

mesh1.run;
fprintf('  Mesh complete.\n');

% --- Study: Stationary ---
fprintf('  Creating study + solving...\n');
std1 = model.study.create('std1');
std1.label('Bioimpedance_50kHz');
std1.create('stat', 'Stationary');

% --- Solve ---
tic;
model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').set('stol', '1e-6');
model.sol('sol1').runAll;
t_solve = toc;
fprintf('  Solved in %.1f s\n', t_solve);

% --- Post-processing: Impedance ---
fprintf('\n--- Results ---\n');
intop = model.result.numerical.create('int1', 'IntSurface');
intop.selection.set(bnd_electrode1);
intop.set('expr', 'ec.nJ');
I_val = intop.getReal();
I_electrode = I_val(1);

V_total = 2 * V_applied;
Z_mag = abs(V_total / I_electrode);

fprintf('  I (electrode 1) = %.4e A\n', I_electrode);
fprintf('  |Z| = %.1f Ohm\n', Z_mag);
fprintf('  Target (blood) = %.0f Ohm\n', Z_target.blood);
fprintf('  Error = %.1f%%\n', (Z_mag - Z_target.blood)/Z_target.blood*100);

% --- Plots ---
pg1 = model.result.create('pg1', 'PlotGroup3D');
pg1.label('Electric Potential');
s1 = pg1.create('surf1', 'Surface');
s1.set('expr', 'V');
s1.set('colortable', 'RainbowLight');

% --- Save ---
mph_out = fullfile(pwd, 'catheter_solved.mph');
mphsave(model, mph_out);
fprintf('\n  Model saved: %s\n', mph_out);
fprintf('\n=== STAGE 2 COMPLETE ===\n');

%% ====================================================================
function sigma_star = cole_cole_sigma(freq, sigma_dc, eps_inf, delta_eps, tau, alpha)
    eps0 = 8.854e-12;
    omega = 2*pi*freq;
    eps_star = eps_inf + delta_eps ./ (1 + (1j*omega*tau).^(1-alpha));
    sigma_star = sigma_dc + 1j*omega*eps0*eps_star;
end
