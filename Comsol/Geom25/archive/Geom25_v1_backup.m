function out = Geom25
%
% Geom25.m — Real catheter geometry with surrounding blood vessel
%
% Workflow:
%   1. Import STEP catheter (1 domain, electrodes = boundaries)
%   2. Remove small details (repair)
%   3. Add blood cylinder + vessel wall cylinder
%   4. Boolean difference: blood = inner_cyl - catheter
%   5. Assign materials, physics, mesh, solve
%
% Units: SI (meters). Catheter along z-axis.

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath('C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Comsol\Geom25');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 3);

%% ====================================================================
%  PARAMETERS
% =====================================================================
% Vessel geometry (units: meters)
vessel_r_inner = 0.008;    % 8 mm vessel lumen radius (IVC-sized for 24FR)
vessel_r_outer = 0.009;    % 9 mm (1 mm wall thickness)
vessel_length  = 0.040;    % 40 mm long cylinder
vessel_z_start = -0.020;   % center the cylinder on the catheter tip region

% Catheter info (24FR = 8mm OD = 4mm radius)
% Catheter tip at approximately z = -0.012 to z = 0.002 from image
% Center the vessel around z ≈ -0.005 (midpoint of electrode region)

% Materials (calibrated at 50 kHz)
sigma_blood = 0.4177;   % S/m
sigma_wall  = 0.1856;   % S/m

% Excitation
V_drive = 1.5;  % ±1.5V on electrodes
freq    = 50e3; % 50 kHz

%% ====================================================================
%  GEOMETRY: Import + Blood Vessel
% =====================================================================
geom = model.component('comp1').geom('geom1');

% Step 1: Import catheter STEP file
geom.create('imp1', 'Import');
geom.feature('imp1').set('filename', ...
    'C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Comsol\Geom25\PRT-1173 - ASPIRATION ELT TIP 24FR.STEP');
geom.feature('imp1').set('minimizetol', true);
geom.feature('imp1').set('healedges', true);
geom.feature('imp1').set('simplify', true);
geom.feature('imp1').set('removeredundant', true);
geom.run('imp1');

% Step 2: Add inner blood cylinder (lumen)
% (Remove Details omitted — import healing options handle STEP quality,
%  and rmd1 was ending up after Form Union, collapsing domains)
geom.create('cyl_blood', 'Cylinder');
geom.feature('cyl_blood').set('r', vessel_r_inner);
geom.feature('cyl_blood').set('h', vessel_length);
geom.feature('cyl_blood').set('pos', [0, 0, vessel_z_start]);
geom.feature('cyl_blood').set('axis', [0, 0, 1]);
geom.run('cyl_blood');

% Step 4: Add outer wall cylinder
geom.create('cyl_wall', 'Cylinder');
geom.feature('cyl_wall').set('r', vessel_r_outer);
geom.feature('cyl_wall').set('h', vessel_length);
geom.feature('cyl_wall').set('pos', [0, 0, vessel_z_start]);
geom.feature('cyl_wall').set('axis', [0, 0, 1]);
geom.run('cyl_wall');

% Step 5: Form Union — let COMSOL automatically create separate domains
% where objects overlap. This avoids explicit boolean issues.
% Result: catheter=1 domain, blood=1 domain (annular), wall=1 domain (shell)
geom.run('fin');

fprintf('Geometry built (Form Union complete).\n');

%% ====================================================================
%  DOMAIN AND BOUNDARY ASSIGNMENTS (verified in GUI)
% =====================================================================
dom_wall = 1;
dom_blood = 2;
dom_cath = 3;
% Electrode boundaries (verified in GUI after Form Union with 78 bnds)
bnd_elec_L = 64;   % Left electrode boundary
bnd_elec_R = 30;   % Right electrode boundary

fprintf('  Domain assignments: blood=%d, wall=%d, catheter=%d\n', dom_blood, dom_wall, dom_cath);
fprintf('  Electrode boundaries: L=%d, R=%d\n', bnd_elec_L, bnd_elec_R);

%% ====================================================================
%  MATERIALS (assigned to specific domains)
% =====================================================================
fprintf('\n--- Setting up materials ---\n');

% Blood — domain 2
mat_blood = model.component('comp1').material.create('mat_blood', 'Common');
mat_blood.propertyGroup('def').set('electricconductivity', num2str(sigma_blood));
mat_blood.propertyGroup('def').set('relpermittivity', '1');
mat_blood.label('Blood');
mat_blood.selection.set([dom_blood]);

% Vessel Wall — domain 1
mat_wall = model.component('comp1').material.create('mat_wall', 'Common');
mat_wall.propertyGroup('def').set('electricconductivity', num2str(sigma_wall));
mat_wall.propertyGroup('def').set('relpermittivity', '1');
mat_wall.label('Vessel Wall');
mat_wall.selection.set([dom_wall]);

% Catheter (insulating polymer) — domain 3
mat_cath = model.component('comp1').material.create('mat_cath', 'Common');
mat_cath.propertyGroup('def').set('electricconductivity', '1e-10');
mat_cath.propertyGroup('def').set('relpermittivity', '3');
mat_cath.label('Catheter (insulator)');
mat_cath.selection.set([dom_cath]);

fprintf('  Materials assigned to domains.\n');

%% ====================================================================
%  PHYSICS: Electric Currents (ec) with electrode BCs
% =====================================================================
fprintf('\n--- Setting up Electric Currents physics ---\n');

ec = model.component('comp1').physics.create('ec', 'ConductiveMedia', 'geom1');

% Electrode +V (boundary 59)
ec.create('epot1', 'ElectricPotential', 2);
ec.feature('epot1').selection.set([bnd_elec_L]);
ec.feature('epot1').set('V0', V_drive);
ec.feature('epot1').label('Electrode +V');

% Electrode -V (boundary 26)
ec.create('epot2', 'ElectricPotential', 2);
ec.feature('epot2').selection.set([bnd_elec_R]);
ec.feature('epot2').set('V0', -V_drive);
ec.feature('epot2').label('Electrode -V');

fprintf('  Electrode BCs: +%.1fV on bnd %d, -%.1fV on bnd %d\n', ...
    V_drive, bnd_elec_L, V_drive, bnd_elec_R);

%% ====================================================================
%  STUDY: Frequency domain at 50 kHz
% =====================================================================
fprintf('\n--- Setting up study ---\n');

model.study.create('std1');
model.study('std1').create('freq', 'Frequency');
model.study('std1').feature('freq').set('plist', num2str(freq));
fprintf('  Frequency domain study at %.0f kHz.\n', freq/1e3);

%% ====================================================================
%  MESH (fine near electrodes, coarser in bulk)
% =====================================================================
fprintf('\n--- Mesh setup ---\n');

model.component('comp1').mesh.create('mesh1');
mesh = model.component('comp1').mesh('mesh1');

% Global mesh size
mesh.create('size1', 'Size');
mesh.feature('size1').set('hauto', 4);  % Normal globally

% Fine mesh on electrode boundaries — explicit max element size
mesh.create('ftet1', 'FreeTet');
mesh.feature('ftet1').create('size_elec', 'Size');
mesh.feature('ftet1').feature('size_elec').selection.geom('geom1', 2);
mesh.feature('ftet1').feature('size_elec').selection.set([bnd_elec_L, bnd_elec_R]);
mesh.feature('ftet1').feature('size_elec').set('custom', true);
mesh.feature('ftet1').feature('size_elec').set('hmax', 1e-4);   % 0.1 mm max on electrodes
mesh.feature('ftet1').feature('size_elec').set('hmin', 2e-5);   % 0.02 mm min on electrodes
mesh.feature('ftet1').feature('size_elec').set('hgrad', 1.3);   % growth rate from electrode

fprintf('  Mesh: hauto=4 global, hmax=0.1mm on electrodes.\n');

% Build mesh
fprintf('  Building mesh...\n');
mesh.run;
fprintf('  Mesh built successfully.\n');

%% ====================================================================
%  SOLVE
% =====================================================================
fprintf('\n--- Solving ---\n');
model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').feature('st1').set('study', 'std1');
model.sol('sol1').feature('st1').set('studystep', 'freq');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').set('stol', 1e-6);
model.sol('sol1').runAll;
fprintf('  Solve complete.\n');

%% ====================================================================
%  POST-PROCESSING: Extract impedance
% =====================================================================
fprintf('\n--- Post-processing ---\n');

% Impedance: Z = V_total / I, where I = integral of J·n over electrode
% Current through electrode L (boundary 59)
I_elec = mphint2(model, 'ec.nJ', 'surface', 'selection', bnd_elec_L);
Z_measured = (2 * V_drive) / abs(I_elec);
fprintf('  Current through electrode: I = %.4e + j%.4e A\n', real(I_elec), imag(I_elec));
fprintf('  |Z| = %.1f Ohm\n', Z_measured);
fprintf('  Phase = %.1f deg\n', angle(I_elec)*180/pi);

%% ====================================================================
%  SAVE
% =====================================================================
mphsave(model, fullfile(pwd, 'Geom25_solved.mph'));
fprintf('\n  Model saved: Geom25_solved.mph\n');

out = model;
