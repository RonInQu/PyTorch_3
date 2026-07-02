%% COMSOL LiveLink - SIMPLE TEST
% Minimal model: cylinder with end-cap electrodes.
% Purpose: verify meshing, solving, and result extraction work.

clear; close all; clc;

import com.comsol.model.*
import com.comsol.model.util.*

fprintf('=== Simple Cylinder Test ===\n');

%% Create model
model = ModelUtil.create('TestModel');
comp1 = model.component.create('comp1', true);
geom1 = comp1.geom.create('geom1', 3);
geom1.lengthUnit('mm');

%% Geometry: single cylinder
cyl1 = geom1.create('cyl1', 'Cylinder');
cyl1.set('r', 4);     % 4 mm radius
cyl1.set('h', 10);    % 10 mm long
cyl1.set('pos', [0, 0, 0]);
cyl1.set('axis', [0, 0, 1]);

geom1.run('fin');
fprintf('  Geometry built.\n');

% Print geometry info
ndom = model.geom('geom1').getNDomains();
nbnd = model.geom('geom1').getNBoundaries();
fprintf('  Domains: %d, Boundaries: %d\n', ndom, nbnd);

%% Material
sigma_blood = 3.13;   % S/m (simplified)
epsr_blood = 5750;

mat1 = comp1.material.create('mat1', 'Common');
mat1.label('Blood');
mat1.selection.set(1);
mat1.propertyGroup('def').set('electricconductivity', num2str(sigma_blood));
mat1.propertyGroup('def').set('relpermittivity', num2str(epsr_blood));

%% Physics
ec = comp1.physics.create('ec', 'ConductiveMedia', 'geom1');

% Apply +1.5V to boundary 2 (bottom cap) and -1.5V to boundary 3 (top cap)
% For a cylinder: boundary 1 = shell, 2 = bottom cap, 3 = top cap
% (May vary — check GUI if wrong)

% Try to figure out which boundaries are end caps by listing them
fprintf('  Applying BCs to end-cap boundaries...\n');

pot1 = ec.create('pot1', 'ElectricPotential', 2);
pot1.selection.set(2);  % bottom end cap
pot1.set('V0', 1.5);
pot1.label('+1.5V');

pot2 = ec.create('pot2', 'ElectricPotential', 2);
pot2.selection.set(3);  % top end cap
pot2.set('V0', -1.5);
pot2.label('-1.5V');

%% Mesh
mesh1 = comp1.mesh.create('mesh1');
mesh1.feature('size').set('hauto', 5);  % Normal
mesh1.run;
fprintf('  Mesh complete.\n');

%% Solve
fprintf('  Solving...\n');
std1 = model.study.create('std1');
std1.create('stat', 'Stationary');

tic;
model.study('std1').run();
fprintf('  Solved in %.1f s\n', toc);

%% Results
fprintf('\n--- Results ---\n');

% Check solution
try
    pd = mpheval(model, 'V');
    fprintf('  V range: [%.4f, %.4f]\n', min(pd.d1), max(pd.d1));
catch e
    fprintf('  mpheval failed: %s\n', e.message);
end

% Impedance: Z = V_total / I
% For uniform cylinder: Z = L / (sigma * A) = 10e-3 / (3.13 * pi*(4e-3)^2)
Z_theory = 10e-3 / (sigma_blood * pi * (4e-3)^2);
fprintf('  Theoretical Z (uniform cylinder): %.1f Ohm\n', Z_theory);

% Compute from power
try
    P = mphint2(model, 'ec.Qrh', 'volume');
    Z_sim = (2*1.5)^2 / P;
    fprintf('  Simulated P = %.6e W, Z = %.1f Ohm\n', P, Z_sim);
catch e
    fprintf('  mphint2 volume failed: %s\n', e.message);
    % Try alternative
    try
        P = mphint2(model, 'ec.normJ^2/sigma_blood', 'volume');
        fprintf('  Alt power calc: %.6e\n', P);
    catch e2
        fprintf('  Alt also failed: %s\n', e2.message);
    end
end

% Save
mph_out = fullfile(pwd, 'test_simple.mph');
mphsave(model, mph_out);
fprintf('  Saved: %s\n', mph_out);
fprintf('\n=== TEST COMPLETE ===\n');
