%% COMSOL LiveLink for MATLAB - 3D Bioimpedance Catheter Model
% Imports STEP geometry, assigns materials (Cole-Cole), applies BCs,
% solves AC Electric Currents at 50 kHz, extracts impedance.
%
% Prerequisites:
%   - COMSOL Multiphysics 6.x with AC/DC Module, CAD Import, LiveLink for MATLAB
%   - Start COMSOL Server: comsolmphserver (or use COMSOL with MATLAB)
%   - test.STEP in same folder as this script
%
% Usage:
%   1. Launch COMSOL with MATLAB (or connect via mphstart)
%   2. Run this script
%
% Author: R. Kurnik / Inquis Medical
% Date: June 2026

clear; close all; clc;

%% ====================================================================
%  CONFIGURATION
% =====================================================================
% File paths
step_file = fullfile(fileparts(mfilename('fullpath')), 'G25.STEP');

% Operating parameters
V_applied = 1.5;           % Applied voltage amplitude [V] (+/- on electrodes)
freq_base = 50e3;          % Base excitation frequency [Hz]
freq_sweep = [5e3, 50e3, 100e3];  % Frequencies for multi-freq study [Hz]

% Electrode geometry (for reference / BC identification)
elec_w = 0.78e-3;         % Electrode width [m]
elec_l = 2.1e-3;          % Electrode length [m]
elec_spacing = 5.48e-3;   % Center-to-center spacing [m]

% Domain - PA lumen surrounding catheter
lumen_radius = 3.0e-3;    % PA lumen radius [m] (adjust to anatomy)
lumen_length = 40e-3;     % PA lumen axial length [m]

% Cole-Cole parameters: [Saline, Blood, Clot, Wall]
%   sigma*(w) = sigma_dc + jw*eps0*(eps_inf + delta_eps/(1+(jw*tau)^(1-alpha)))
cc_names = {'Saline', 'Blood', 'Clot', 'Wall'};
cc_params.sigma_dc  = [9.36,   1.30,   0.155,  0.40];
cc_params.eps_inf   = [76,     50,     40,     40];
cc_params.delta_eps = [0,      2530000, 770000, 1200000];
cc_params.tau       = [1e-12,  10e-6,  12e-6,  9e-6];
cc_params.alpha     = [0.0,    0.25,   0.30,   0.25];

% Calibrated impedance targets at 50 kHz (for validation)
Z_target.blood = 800;     % [Ohm]
Z_target.clot  = 3500;    % [Ohm]
Z_target.wall  = 1800;    % [Ohm]
Z_target.saline = 300;    % [Ohm]

% Contact model parameters
contact_type = 'blood';   % 'blood', 'clot', or 'wall'
coverage_frac = 1.0;      % Fraction of cavity covered by contact material [0-1]

% Catheter material
sigma_catheter = 1e-10;   % Polypropylene conductivity [S/m]
epsr_catheter = 2.2;      % Polypropylene relative permittivity

fprintf('=== COMSOL LiveLink - 3D Bioimpedance Model ===\n');
fprintf('STEP file: %s\n', step_file);
fprintf('Frequency: %.0f kHz, V_applied: +/-%.1f V\n', freq_base/1e3, V_applied);
fprintf('Contact type: %s (coverage: %.0f%%)\n', contact_type, coverage_frac*100);

%% ====================================================================
%  COLE-COLE HELPER FUNCTION (compute complex conductivity at frequency)
% =====================================================================
% Embedded as local function at end of file - see cole_cole_sigma()

%% ====================================================================
%  CREATE COMSOL MODEL
% =====================================================================
import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('BioimpedanceModel');
model.label('Catheter_Bioimpedance_3D');
model.comments('3D bioimpedance model - catheter with electrodes in PA lumen');

% Set length unit to mm for geometry operations (internal SI for physics)
comp1 = model.component.create('comp1', true);
geom1 = comp1.geom.create('geom1', 3);
geom1.lengthUnit('mm');

fprintf('\n--- Geometry Setup ---\n');

%% ====================================================================
%  GEOMETRY: IMPORT STEP + CREATE SURROUNDING LUMEN
% =====================================================================
% Import catheter STEP file
imp1 = geom1.create('imp1', 'Import');
imp1.set('filename', step_file);
imp1.set('type', 'cad');
imp1.importData();
fprintf('  STEP imported: %s\n', step_file);

% Create surrounding cylindrical PA lumen
cyl1 = geom1.create('cyl1', 'Cylinder');
cyl1.set('r', lumen_radius * 1e3);          % Convert to mm (geometry unit)
cyl1.set('h', lumen_length * 1e3);
cyl1.set('pos', [0, 0, -lumen_length/2 * 1e3]);  % Centered on catheter
cyl1.set('axis', [0, 0, 1]);                % Along z-axis
fprintf('  Lumen cylinder: R=%.1f mm, L=%.0f mm\n', lumen_radius*1e3, lumen_length*1e3);

% Boolean: subtract catheter from lumen to get blood domain
% Keep catheter as separate domain (insulator)
% Use Fragment (partition) to share boundaries between domains
frag1 = geom1.create('frag1', 'Partition');
frag1.selection('input').set({'cyl1'});
frag1.selection('tool').set({'imp1'});

% Build geometry
geom1.run('fin');
fprintf('  Geometry built (Fragment operation complete)\n');

%% ====================================================================
%  IDENTIFY DOMAINS AND BOUNDARIES
% =====================================================================
% NOTE: After STEP import and boolean operations, domain/boundary numbers
% depend on the specific geometry. You MUST verify these interactively
% in COMSOL GUI on first run, then update the numbers below.
%
% Expected domains after Fragment:
%   - Domain for catheter body (polypropylene insulator)
%   - Domain for cavity (between electrodes) - blood/clot/wall
%   - Domain for surrounding blood (PA lumen)
%
% Expected boundaries:
%   - Electrode 1 surface (apply V = +V_applied)
%   - Electrode 2 surface (apply V = -V_applied)
%   - Outer lumen wall (insulating or ground)

% PLACEHOLDER indices - UPDATE after first geometry inspection in GUI
% Use: mphgeom(model, 'geom1', 'FaceLabels', 'on') to identify
dom_catheter = 1;    % Catheter body domain index
dom_cavity   = 2;    % Cavity domain (material varies: blood/clot/wall)
dom_lumen    = 3;    % Surrounding PA lumen (flowing blood)

bnd_electrode1 = 106;  % Electrode 1 boundary index (verified in GUI)
bnd_electrode2 = 40;   % Electrode 2 boundary index (verified in GUI)
bnd_outer      = 20;   % Outer lumen boundary - UPDATE after adding cylinder

fprintf('\n  *** IMPORTANT: Verify domain/boundary indices in COMSOL GUI ***\n');
fprintf('  Use: mphgeom(model, ''geom1'', ''FaceLabels'', ''on'')\n');
fprintf('  Then update dom_* and bnd_* variables above.\n');

%% ====================================================================
%  MATERIALS (Cole-Cole frequency-dependent)
% =====================================================================
fprintf('\n--- Materials Setup ---\n');

% Compute complex conductivity at base frequency for each tissue
eps0 = 8.854e-12;  % [F/m]
omega = 2*pi*freq_base;

sigma_star = zeros(1, 4);
for k = 1:4
    sigma_star(k) = cole_cole_sigma(freq_base, ...
        cc_params.sigma_dc(k), cc_params.eps_inf(k), ...
        cc_params.delta_eps(k), cc_params.tau(k), cc_params.alpha(k));
    fprintf('  %s: sigma* = %.4f + j%.4f S/m (|Z_cell| ~ %.0f Ohm)\n', ...
        cc_names{k}, real(sigma_star(k)), imag(sigma_star(k)), ...
        2812/abs(sigma_star(k)));  % K_cell from 2D model
end

% --- Material: Catheter (Polypropylene) ---
mat_cath = comp1.material.create('mat_cath', 'Common');
mat_cath.label('Polypropylene (Catheter)');
mat_cath.selection.set(dom_catheter);
mat_cath.propertyGroup('def').set('electricconductivity', num2str(sigma_catheter));
mat_cath.propertyGroup('def').set('relpermittivity', num2str(epsr_catheter));

% --- Material: Cavity (parameterized - depends on contact_type) ---
% Select Cole-Cole index based on contact_type
switch lower(contact_type)
    case 'blood',  cc_idx = 2;
    case 'clot',   cc_idx = 3;
    case 'wall',   cc_idx = 4;
    case 'saline', cc_idx = 1;
    otherwise, error('Unknown contact_type: %s', contact_type);
end

sigma_cavity = real(sigma_star(cc_idx));
epsr_cavity  = imag(sigma_star(cc_idx)) / (omega * eps0);

mat_cavity = comp1.material.create('mat_cavity', 'Common');
mat_cavity.label(sprintf('Cavity (%s)', cc_names{cc_idx}));
mat_cavity.selection.set(dom_cavity);
mat_cavity.propertyGroup('def').set('electricconductivity', num2str(sigma_cavity, '%.6g'));
mat_cavity.propertyGroup('def').set('relpermittivity', num2str(epsr_cavity, '%.6g'));

% --- Material: PA Lumen (Blood) ---
sigma_lumen = real(sigma_star(2));
epsr_lumen  = imag(sigma_star(2)) / (omega * eps0);

mat_lumen = comp1.material.create('mat_lumen', 'Common');
mat_lumen.label('PA Lumen (Blood)');
mat_lumen.selection.set(dom_lumen);
mat_lumen.propertyGroup('def').set('electricconductivity', num2str(sigma_lumen, '%.6g'));
mat_lumen.propertyGroup('def').set('relpermittivity', num2str(epsr_lumen, '%.6g'));

fprintf('  Cavity material: %s (sigma=%.4f S/m, epsr=%.0f)\n', ...
    cc_names{cc_idx}, sigma_cavity, epsr_cavity);

%% ====================================================================
%  PHYSICS: AC/DC Electric Currents (ec)
% =====================================================================
fprintf('\n--- Physics Setup (Electric Currents) ---\n');

ec = comp1.physics.create('ec', 'ConductiveMedia', 'geom1');
ec.label('Electric Currents');

% Electrode 1: Electric Potential = +V_applied
ep1 = ec.create('ep1', 'ElectricPotential', 2);  % 2 = boundary
ep1.selection.set(bnd_electrode1);
ep1.set('V0', V_applied);
ep1.label('Electrode 1 (+V)');

% Electrode 2: Ground (V = 0) or Electric Potential = -V_applied
% Using symmetric +/- drive:
ep2 = ec.create('ep2', 'ElectricPotential', 2);
ep2.selection.set(bnd_electrode2);
ep2.set('V0', -V_applied);
ep2.label('Electrode 2 (-V)');

% Outer boundary: Electrical Insulation (default Neumann: n·J = 0)
% This is the natural BC, no explicit assignment needed unless you want Ground
ins1 = ec.create('ins1', 'ElectricalInsulation', 2);
ins1.selection.set(bnd_outer);
ins1.label('Outer Insulation');

fprintf('  BC: Electrode 1 = +%.1f V (bnd %d)\n', V_applied, bnd_electrode1);
fprintf('  BC: Electrode 2 = -%.1f V (bnd %d)\n', V_applied, bnd_electrode2);
fprintf('  BC: Outer = Insulation (bnd %d)\n', bnd_outer);

%% ====================================================================
%  MESH
% =====================================================================
fprintf('\n--- Mesh Setup ---\n');

mesh1 = comp1.mesh.create('mesh1');

% Free tetrahedral mesh with size controls
ftet = mesh1.create('ftet1', 'FreeTet');

% Global mesh size
mesh1.feature('size').set('hauto', 4);  % Fine (1=Extremely fine, 9=Extremely coarse)

% Electrode boundary refinement
sz_elec = mesh1.create('sz_elec', 'Size');
sz_elec.selection.geom('geom1', 2).set([bnd_electrode1, bnd_electrode2]);
sz_elec.set('custom', true);
sz_elec.set('hmax', 0.2e-3);   % 0.2 mm max on electrodes
sz_elec.set('hmin', 0.05e-3);  % 0.05 mm min on electrodes
sz_elec.set('hgrad', 1.3);     % Growth rate

% Cavity refinement
sz_cav = mesh1.create('sz_cav', 'Size');
sz_cav.selection.geom('geom1', 3).set(dom_cavity);
sz_cav.set('custom', true);
sz_cav.set('hmax', 0.5e-3);    % 0.5 mm max in cavity

mesh1.run;
fprintf('  Mesh generated (Fine preset + electrode refinement)\n');

%% ====================================================================
%  STUDY: FREQUENCY DOMAIN at 50 kHz
% =====================================================================
fprintf('\n--- Study Setup ---\n');

std1 = model.study.create('std1');
std1.label('Single Frequency (50 kHz)');

% Frequency domain study step
freq_step = std1.create('freq', 'Frequency');
freq_step.set('plist', num2str(freq_base));
fprintf('  Study: Frequency domain at %.0f kHz\n', freq_base/1e3);

%% ====================================================================
%  SOLVE
% =====================================================================
fprintf('\n--- Solving ---\n');
tic;
model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').set('stol', 1e-6);
model.sol('sol1').runAll;
solve_time = toc;
fprintf('  Solved in %.1f seconds\n', solve_time);

%% ====================================================================
%  POST-PROCESSING: IMPEDANCE EXTRACTION
% =====================================================================
fprintf('\n--- Post-Processing ---\n');

% Method: Z = V_total / I_electrode
% V_total = 2 * V_applied (differential drive)
% I_electrode = integral of J·n over electrode 1 surface

% Create integration operator over electrode 1
intop = model.result.numerical.create('intop1', 'IntSurface');
intop.selection.set(bnd_electrode1);
intop.set('expr', 'ec.nJ');  % Normal current density

% Evaluate current
I_data = intop.getReal();
I_electrode = I_data(1);

% Impedance
V_total = 2 * V_applied;  % Differential voltage
Z_measured = V_total / I_electrode;
Z_magnitude = abs(Z_measured);
Z_phase = angle(Z_measured) * 180/pi;

fprintf('\n========== RESULTS ==========\n');
fprintf('  Contact type: %s\n', contact_type);
fprintf('  Frequency: %.0f kHz\n', freq_base/1e3);
fprintf('  Current (electrode 1): %.4e A\n', I_electrode);
fprintf('  Impedance: %.1f Ohm (phase: %.1f deg)\n', Z_magnitude, Z_phase);
fprintf('  Target: %.0f Ohm\n', Z_target.(lower(contact_type)));
fprintf('  Error: %.1f%%\n', (Z_magnitude - Z_target.(lower(contact_type))) / Z_target.(lower(contact_type)) * 100);
fprintf('=============================\n');

%% ====================================================================
%  MULTI-FREQUENCY SWEEP (Optional - uncomment to run)
% =====================================================================
% Uncomment the block below after baseline validation

% fprintf('\n--- Multi-Frequency Sweep ---\n');
% std2 = model.study.create('std2');
% std2.label('Multi-Frequency Sweep');
% freq_step2 = std2.create('freq', 'Frequency');
% freq_step2.set('plist', num2str(freq_sweep));
% 
% % For frequency-dependent materials, update sigma/epsr at each freq
% % This requires a parametric approach or COMSOL's built-in Cole-Cole
% % material model (available in AC/DC Module v6.0+)
% %
% % Alternative: loop over frequencies, updating material properties
% Z_sweep = zeros(length(freq_sweep), length(cc_names));
% for fi = 1:length(freq_sweep)
%     f = freq_sweep(fi);
%     for ci = 1:4
%         sigma_f = cole_cole_sigma(f, cc_params.sigma_dc(ci), ...
%             cc_params.eps_inf(ci), cc_params.delta_eps(ci), ...
%             cc_params.tau(ci), cc_params.alpha(ci));
%         % Update material properties...
%     end
%     % Re-solve and extract impedance
% end

%% ====================================================================
%  PARAMETRIC CONTACT SWEEP (Optional - uncomment to run)
% =====================================================================
% Sweep cavity material: blood -> clot -> wall

% fprintf('\n--- Contact Material Sweep ---\n');
% contact_types = {'blood', 'clot', 'wall'};
% Z_results = zeros(1, 3);
% 
% for ci = 1:3
%     ct = contact_types{ci};
%     switch ct
%         case 'blood', idx = 2;
%         case 'clot',  idx = 3;
%         case 'wall',  idx = 4;
%     end
%     sig = real(sigma_star(idx));
%     epr = imag(sigma_star(idx)) / (omega * eps0);
%     
%     % Update cavity material
%     mat_cavity.propertyGroup('def').set('electricconductivity', num2str(sig, '%.6g'));
%     mat_cavity.propertyGroup('def').set('relpermittivity', num2str(epr, '%.6g'));
%     
%     % Re-solve
%     model.sol('sol1').runAll;
%     
%     % Extract impedance
%     I_data = intop.getReal();
%     Z_results(ci) = V_total / I_data(1);
%     
%     fprintf('  %s: Z = %.0f Ohm (target: %.0f)\n', ct, abs(Z_results(ci)), ...
%         Z_target.(ct));
% end

%% ====================================================================
%  VISUALIZATION
% =====================================================================
fprintf('\n--- Creating Plots ---\n');

% Electric potential slice plot
pg1 = model.result.create('pg1', 'PlotGroup3D');
pg1.label('Electric Potential');
pg1.set('data', 'dset1');
surf1 = pg1.create('surf1', 'Surface');
surf1.set('expr', 'V');
surf1.set('colortable', 'RainbowLight');
surf1.set('colorlegend', true);

% Slice through z=0 plane
slc1 = pg1.create('slc1', 'Slice');
slc1.set('expr', 'V');
slc1.set('quickplane', 'xz');
slc1.set('quickx', 0);

% Current density arrow plot
pg2 = model.result.create('pg2', 'PlotGroup3D');
pg2.label('Current Density');
arr1 = pg2.create('arr1', 'ArrowVolume');
arr1.set('expr', {'ec.Jx', 'ec.Jy', 'ec.Jz'});
arr1.set('arrowcount', 500);

% Electric field magnitude on electrode plane
pg3 = model.result.create('pg3', 'PlotGroup3D');
pg3.label('E-Field Magnitude');
slc2 = pg3.create('slc1', 'Slice');
slc2.set('expr', 'ec.normE');
slc2.set('quickplane', 'xz');
slc2.set('quickx', 0);
slc2.set('colortable', 'ThermalLight');

fprintf('  3 plot groups created (Potential, J-field, E-field)\n');

%% ====================================================================
%  EXPORT MODEL
% =====================================================================
output_file = fullfile(fileparts(mfilename('fullpath')), ...
    'catheter_bioimpedance_3D.mph');
mphsave(model, output_file);
fprintf('\n  Model saved: %s\n', output_file);

%% ====================================================================
%  SUMMARY TABLE
% =====================================================================
fprintf('\n\n====== SESSION SUMMARY ======\n');
fprintf('%-12s %-12s %-12s %-10s\n', 'Material', '|Z| [Ohm]', 'Target', 'Status');
fprintf('%-12s %-12s %-12s %-10s\n', '--------', '---------', '------', '------');
if exist('Z_magnitude', 'var')
    target_val = Z_target.(lower(contact_type));
    err_pct = abs(Z_magnitude - target_val)/target_val * 100;
    if err_pct < 20
        status = 'PASS';
    else
        status = 'NEEDS CAL';
    end
    fprintf('%-12s %-12.0f %-12.0f %-10s\n', contact_type, Z_magnitude, target_val, status);
end
fprintf('=============================\n');
fprintf('\nNext steps:\n');
fprintf('  1. Verify domain/boundary indices in GUI\n');
fprintf('  2. Run contact material sweep (uncomment block above)\n');
fprintf('  3. Run multi-frequency sweep\n');
fprintf('  4. Compare 3D K_cell to 2D value (2812 m^-1)\n');

%% ====================================================================
%  LOCAL FUNCTIONS
% =====================================================================

function sigma_star = cole_cole_sigma(freq, sigma_dc, eps_inf, delta_eps, tau, alpha)
% COLE_COLE_SIGMA Compute complex conductivity from Cole-Cole parameters
%   sigma*(w) = sigma_dc + jw*eps0*(eps_inf + delta_eps/(1+(jw*tau)^(1-alpha)))
%
% Inputs:
%   freq      - Frequency [Hz]
%   sigma_dc  - DC conductivity [S/m]
%   eps_inf   - High-frequency permittivity
%   delta_eps - Dielectric decrement
%   tau       - Relaxation time [s]
%   alpha     - Cole-Cole exponent (0 = Debye)
%
% Output:
%   sigma_star - Complex conductivity [S/m]

    eps0 = 8.854e-12;
    omega = 2 * pi * freq;
    
    % Cole-Cole permittivity
    eps_star = eps_inf + delta_eps ./ (1 + (1j*omega*tau).^(1-alpha));
    
    % Complex conductivity: sigma* = sigma_dc + jw*eps0*eps_star
    sigma_star = sigma_dc + 1j*omega*eps0*eps_star;
end
