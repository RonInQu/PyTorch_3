%% COMSOL LiveLink - Bioimpedance Model v5
% Full 3D model: catheter (polypropylene), blood medium, vessel wall.
% Parametric sweep: Blood, Clot, Wall materials.
% Exports images for management presentation.
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
cath_radius = 3.3;         % Outer radius
cath_length = 25.0;        % z: -5 to 20 mm
cath_z_start = -5.0;

% Vessel wall shell [mm] — at outer boundary of blood bath
wall_thickness = 1.0;      % Arterial wall ~1 mm thick
wall_radius = cyl_radius + wall_thickness;  % Outer edge of wall

% Electrode dimensions [mm] (measured from CAD)
elec_width  = 0.693;
elec_length = 2.000;
elec_thickness = 0.1;

% Electrode corners (measured from CAD, mm)
elec_L_corners = [1.21939704,  3.455094017, 1.629613301;
                  1.69325454,  3.907606382, 1.402932355;
                  2.985116917, 3.243491046, 2.777720491;
                  2.511259417, 2.790978681, 3.004401437];
ctr_L = mean(elec_L_corners, 1);

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

% Calibrated conductivities at 50 kHz (from 2D model matching targets)
% These give: Blood ~800, Clot ~3500, Wall ~1800 Ohm
sigma_tissues = [0.55, 0.55, 0.10, 0.25];  % [Saline, Blood, Clot, Wall] S/m
epsr_tissues  = [76,   574977, 175000, 273000]; % relative permittivity

% Material properties
sigma_catheter = 1e-10;   % Polypropylene [S/m]
epsr_catheter  = 2.2;

% Impedance targets
Z_target = [NaN, 800, 3500, 1800]; % [Saline, Blood, Clot, Wall]

% Output directory for images
out_dir = fullfile(pwd, '3D_Results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

fprintf('=== COMSOL LiveLink - Bioimpedance Model v5 ===\n');
fprintf('  Frequency: %.0f kHz, V = +/-%.1f V\n', freq_base/1e3, V_applied);

%% ====================================================================
%  CREATE MODEL
% =====================================================================
model = ModelUtil.create('BioZ_v5');
comp1 = model.component.create('comp1', true);
geom1 = comp1.geom.create('geom1', 3);
geom1.lengthUnit('mm');

%% ====================================================================
%  GLOBAL DEFINITIONS
% =====================================================================
fprintf('\n--- Global Definitions ---\n');
par = model.param;
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
    par.set(['sigma_' t], num2str(sigma_tissues(k), '%.4g'), ...
        ['Effective sigma ' t ' at 50 kHz [S/m]']);
    par.set(['epsr_' t], num2str(epsr_tissues(k), '%.4g'), ...
        ['Relative permittivity ' t ' at 50 kHz']);
end
par.set('f0', [num2str(freq_base) '[Hz]'], 'Excitation frequency');
par.set('V_app', [num2str(V_applied) '[V]'], 'Applied voltage amplitude');
par.set('sigma_cath', '1e-10[S/m]', 'Polypropylene conductivity');
par.set('epsr_cath', '2.2', 'Polypropylene permittivity');
fprintf('  Parameters added for all tissues + catheter.\n');

%% ====================================================================
%  GEOMETRY
% =====================================================================
fprintf('\n--- Building Geometry ---\n');

% Outer vessel wall cylinder
cyl_wall = geom1.create('cyl_wall', 'Cylinder');
cyl_wall.label('Vessel Wall');
cyl_wall.set('r', wall_radius);
cyl_wall.set('h', cyl_length);
cyl_wall.set('pos', [0, 0, cyl_z_start]);
cyl_wall.set('axis', [0, 0, 1]);
fprintf('  Vessel wall: R=%.1f mm (outer)\n', wall_radius);

% Blood lumen cylinder (inside vessel wall)
cyl1 = geom1.create('cyl1', 'Cylinder');
cyl1.label('Blood Lumen');
cyl1.set('r', cyl_radius);
cyl1.set('h', cyl_length);
cyl1.set('pos', [0, 0, cyl_z_start]);
cyl1.set('axis', [0, 0, 1]);
fprintf('  Blood lumen: R=%.0f mm, z=[%.0f, %.0f] mm\n', ...
    cyl_radius, cyl_z_start, cyl_z_start+cyl_length);

% Catheter body (solid insulating cylinder)
cyl2 = geom1.create('cyl2', 'Cylinder');
cyl2.label('Catheter Body');
cyl2.set('r', cath_radius);
cyl2.set('h', cath_length);
cyl2.set('pos', [0, 0, cath_z_start]);
cyl2.set('axis', [0, 0, 1]);
fprintf('  Catheter body: R=%.1f mm, z=[%.0f, %.0f] mm\n', ...
    cath_radius, cath_z_start, cath_z_start+cath_length);

% Electrode pads (work planes from measured coordinates)
eL_AB = elec_L_corners(2,:) - elec_L_corners(1,:);
eL_AD = elec_L_corners(4,:) - elec_L_corners(1,:);
eL_normal = cross(eL_AB, eL_AD); eL_normal = eL_normal / norm(eL_normal);
w_L = norm(eL_AB); l_L = norm(eL_AD);

eR_AB = elec_R_corners(2,:) - elec_R_corners(1,:);
eR_AD = elec_R_corners(4,:) - elec_R_corners(1,:);
eR_normal = cross(eR_AB, eR_AD); eR_normal = eR_normal / norm(eR_normal);
w_R = norm(eR_AB); l_R = norm(eR_AD);

fprintf('  Left electrode:  %.3f x %.3f mm\n', w_L, l_L);
fprintf('  Right electrode: %.3f x %.3f mm\n', w_R, l_R);

% Left Electrode
wp1 = geom1.create('wp1', 'WorkPlane');
wp1.label('Left Electrode Plane');
wp1.set('planetype', 'coordinates');
wp1.set('genpoints', [elec_L_corners(1,:); elec_L_corners(2,:); elec_L_corners(4,:)]);
wp1.geom.create('rect1', 'Rectangle').set('size', [w_L, l_L]).set('pos', [0, 0]);
ext1 = geom1.create('ext1', 'Extrude');
ext1.label('Left Electrode Pad');
ext1.setIndex('distance', num2str(elec_thickness), 0);
ext1.selection('input').set({'wp1'});

% Right Electrode
wp2 = geom1.create('wp2', 'WorkPlane');
wp2.label('Right Electrode Plane');
wp2.set('planetype', 'coordinates');
wp2.set('genpoints', [elec_R_corners(1,:); elec_R_corners(2,:); elec_R_corners(4,:)]);
wp2.geom.create('rect2', 'Rectangle').set('size', [w_R, l_R]).set('pos', [0, 0]);
ext2 = geom1.create('ext2', 'Extrude');
ext2.label('Right Electrode Pad');
ext2.setIndex('distance', num2str(elec_thickness), 0);
ext2.selection('input').set({'wp2'});

% Build
geom1.run('fin');
ndom = model.geom('geom1').getNDomains();
nbnd = model.geom('geom1').getNBoundaries();
fprintf('  Geometry built: %d domains, %d boundaries\n', ndom, nbnd);

%% ====================================================================
%  IDENTIFY DOMAINS
% =====================================================================
fprintf('\n  Identifying domains...\n');
bnd_counts = zeros(1, ndom);
for d = 1:ndom
    bnds = mphgetadj(model, 'geom1', 'boundary', 'domain', d);
    bnd_counts(d) = length(bnds);
    fprintf('  Domain %d: %d boundaries\n', d, bnd_counts(d));
end

% Sort: most boundaries → blood (annular, touches wall + catheter + electrodes)
[~, sorted_idx] = sort(bnd_counts, 'descend');
% With 5 objects (wall_cyl, blood_cyl, catheter, 2 electrodes) → 5 domains
% Blood (annular between catheter and blood boundary) = most boundaries
%   (it shares faces with catheter, wall, and both electrodes)
% Vessel wall (annular shell) = second most (shares outer + inner face)
% Catheter / Electrodes = fewest (6 each)
dom_blood = sorted_idx(1);         % Most boundaries
dom_wall_vessel = sorted_idx(2);   % Second most
dom_catheter = sorted_idx(3);
dom_elec_L = sorted_idx(4);
dom_elec_R = sorted_idx(5);

fprintf('  Assignment: wall=%d, blood=%d, cath=%d, elecL=%d, elecR=%d\n', ...
    dom_wall_vessel, dom_blood, dom_catheter, dom_elec_L, dom_elec_R);

%% ====================================================================
%  MATERIALS
% =====================================================================
fprintf('\n--- Materials ---\n');

% Blood
mat_blood = comp1.material.create('mat_blood', 'Common');
mat_blood.label('Blood');
mat_blood.selection.set(dom_blood);
mat_blood.propertyGroup('def').set('electricconductivity', ...
    num2str(sigma_tissues(2), '%.6g'));
mat_blood.propertyGroup('def').set('relpermittivity', ...
    num2str(epsr_tissues(2), '%.6g'));
fprintf('  Blood: sigma=%.4f S/m, epsr=%.0f\n', sigma_tissues(2), epsr_tissues(2));

% Catheter (polypropylene — insulating)
mat_cath = comp1.material.create('mat_cath', 'Common');
mat_cath.label('Polypropylene');
mat_cath.selection.set(dom_catheter);
mat_cath.propertyGroup('def').set('electricconductivity', '1e-10');
mat_cath.propertyGroup('def').set('relpermittivity', '2.2');
fprintf('  Catheter: sigma=1e-10 S/m (insulator)\n');

% Vessel wall (tissue — conductive)
mat_vwall = comp1.material.create('mat_vwall', 'Common');
mat_vwall.label('Vessel Wall Tissue');
mat_vwall.selection.set(dom_wall_vessel);
mat_vwall.propertyGroup('def').set('electricconductivity', ...
    num2str(sigma_tissues(4), '%.6g'));
mat_vwall.propertyGroup('def').set('relpermittivity', ...
    num2str(epsr_tissues(4), '%.6g'));
fprintf('  Vessel wall: sigma=%.4f S/m, epsr=%.0f\n', sigma_tissues(4), epsr_tissues(4));

%% ====================================================================
%  PHYSICS
% =====================================================================
fprintf('\n--- Physics ---\n');

ec = comp1.physics.create('ec', 'ConductiveMedia', 'geom1');
% Solve in blood + catheter + vessel wall (all volumetric domains)
% Electrode domains are just geometric (for boundary faces)
ec.selection.set([dom_blood, dom_catheter, dom_wall_vessel]);
fprintf('  Physics applied to blood + catheter + vessel wall domains.\n');

% Electrode boundary conditions
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
%  SOLVE — BLOOD BASELINE (Frequency Domain at 50 kHz)
% =====================================================================
fprintf('\n--- Solve: Blood Baseline ---\n');

std1 = model.study.create('std1');
std1.label('Blood Baseline 50kHz');
std1.create('freq', 'Frequency');
std1.feature('freq').set('plist', num2str(freq_base));

tic;
model.study('std1').run();
t_solve = toc;
fprintf('  Solved in %.1f s\n', t_solve);

%% ====================================================================
%  RESULTS — BLOOD BASELINE
% =====================================================================
fprintf('\n--- Results: Blood ---\n');

pd = mpheval(model, 'abs(V)');
fprintf('  |V| range: [%.4f, %.4f] V\n', min(pd.d1), max(pd.d1));

P_blood = mphint2(model, 'ec.Qrh', 'volume', 'selection', dom_blood);
V_total = 2 * V_applied;
Z_blood = V_total^2 / (2 * P_blood);
fprintf('  Z_blood = %.1f Ohm (target: %d)\n', Z_blood, Z_target(2));

%% ====================================================================
%  CREATE PLOT GROUPS
% =====================================================================
fprintf('\n--- Creating Plots ---\n');

% 1. Electric Potential (multislice)
pg1 = model.result.create('pg1', 'PlotGroup3D');
pg1.label('Electric Potential - Blood');
ms1 = pg1.create('mslc1', 'Multislice');
ms1.set('expr', 'abs(V)');
ms1.set('multiplanexmethod', 'coord');
ms1.set('xcoord', '0');
ms1.set('multiplaneymethod', 'coord');
ms1.set('ycoord', num2str(ctr_L(2), '%.2f'));
ms1.set('multiplanezmethod', 'coord');
ms1.set('zcoord', num2str(ctr_L(3), '%.2f'));
pg1.run;

% 2. Current Density (multislice)
pg2 = model.result.create('pg2', 'PlotGroup3D');
pg2.label('Current Density - Blood');
ms2 = pg2.create('mslc2', 'Multislice');
ms2.set('expr', 'ec.normJ');
ms2.set('multiplanexmethod', 'coord');
ms2.set('xcoord', '0');
ms2.set('multiplaneymethod', 'coord');
ms2.set('ycoord', num2str(ctr_L(2), '%.2f'));
ms2.set('multiplanezmethod', 'coord');
ms2.set('zcoord', num2str(ctr_L(3), '%.2f'));
pg2.run;

% 3. Streamlines (field lines)
pg3 = model.result.create('pg3', 'PlotGroup3D');
pg3.label('Current Field Lines - Blood');
str1 = pg3.create('str1', 'Streamline');
str1.set('expr', {'ec.Jx', 'ec.Jy', 'ec.Jz'});
str1.set('posmethod', 'start');
str1.set('startmethod', 'coord');
npts = 5;
x_s = linspace(ctr_L(1)-0.3, ctr_L(1)+0.3, npts);
z_s = linspace(ctr_L(3)-0.8, ctr_L(3)+0.8, npts);
[XS, ZS] = meshgrid(x_s, z_s);
YS = ctr_L(2) * ones(size(XS));
str1.set('xcoord', XS(:)');
str1.set('ycoord', YS(:)');
str1.set('zcoord', ZS(:)');
pg3.run;

% Export blood images
fprintf('  Exporting blood images...\n');

exp1 = model.result.export.create('img_pot_blood', 'Image');
exp1.set('plotgroup', 'pg1');
exp1.set('pngfilename', fullfile(out_dir, 'potential_blood.png'));
exp1.set('size', 'current');
exp1.run;

exp2 = model.result.export.create('img_J_blood', 'Image');
exp2.set('plotgroup', 'pg2');
exp2.set('pngfilename', fullfile(out_dir, 'current_density_blood.png'));
exp2.set('size', 'current');
exp2.run;

exp3 = model.result.export.create('img_stream_blood', 'Image');
exp3.set('plotgroup', 'pg3');
exp3.set('pngfilename', fullfile(out_dir, 'streamlines_blood.png'));
exp3.set('size', 'current');
exp3.run;

fprintf('  Blood images saved to %s\n', out_dir);

% Save blood model
mph_out = fullfile(pwd, 'bioimpedance_v5_blood.mph');
mphsave(model, mph_out);
fprintf('  Blood model saved: %s\n', mph_out);

%% ====================================================================
%  PARAMETRIC SWEEP: CLOT AND WALL
% =====================================================================
fprintf('\n--- Parametric Material Sweep ---\n');

% Store results
material_names = {'Blood', 'Clot', 'Wall'};
sigma_sweep    = [sigma_tissues(2), sigma_tissues(3), sigma_tissues(4)];
epsr_sweep     = [epsr_tissues(2),  epsr_tissues(3),  epsr_tissues(4)];
Z_results      = zeros(1, 3);
Z_results(1)   = Z_blood;

for m = 2:3
    fprintf('\n  --- %s ---\n', material_names{m});
    
    % Update blood domain material to clot or wall properties
    mat_blood.propertyGroup('def').set('electricconductivity', ...
        num2str(sigma_sweep(m), '%.6g'));
    mat_blood.propertyGroup('def').set('relpermittivity', ...
        num2str(epsr_sweep(m), '%.6g'));
    fprintf('    sigma=%.4f S/m, epsr=%.0f\n', sigma_sweep(m), epsr_sweep(m));
    
    % Re-solve
    tic;
    model.study('std1').run();
    fprintf('    Solved in %.1f s\n', toc);
    
    % Compute impedance
    P_m = mphint2(model, 'ec.Qrh', 'volume', 'selection', dom_blood);
    Z_m = V_total^2 / (2 * P_m);
    Z_results(m) = Z_m;
    fprintf('    Z_%s = %.1f Ohm (target: %d)\n', ...
        material_names{m}, Z_m, Z_target(m+1));
    
    % Update plots and export for this material
    pg1.label(sprintf('Electric Potential - %s', material_names{m}));
    pg1.run;
    pg2.label(sprintf('Current Density - %s', material_names{m}));
    pg2.run;
    pg3.label(sprintf('Current Field Lines - %s', material_names{m}));
    pg3.run;
    
    % Export images
    exp1.set('pngfilename', fullfile(out_dir, ...
        sprintf('potential_%s.png', lower(material_names{m}))));
    exp1.run;
    exp2.set('pngfilename', fullfile(out_dir, ...
        sprintf('current_density_%s.png', lower(material_names{m}))));
    exp2.run;
    exp3.set('pngfilename', fullfile(out_dir, ...
        sprintf('streamlines_%s.png', lower(material_names{m}))));
    exp3.run;
    
    % Save model for this material
    mph_mat = fullfile(pwd, sprintf('bioimpedance_v5_%s.mph', lower(material_names{m})));
    mphsave(model, mph_mat);
end

%% ====================================================================
%  SUMMARY
% =====================================================================
fprintf('\n========================================\n');
fprintf('  3D IMPEDANCE RESULTS SUMMARY\n');
fprintf('========================================\n');
fprintf('  %-8s  %8s  %8s  %8s\n', 'Material', 'Z_sim', 'Z_target', 'Ratio');
fprintf('  %-8s  %8s  %8s  %8s\n', '--------', '------', '--------', '-----');
for m = 1:3
    ratio = Z_results(m) / Z_results(1);
    idx = m + 1; if m == 1, idx = 2; end
    fprintf('  %-8s  %7.1f   %7d    %5.2fx\n', ...
        material_names{m}, Z_results(m), Z_target(idx), ratio);
end
fprintf('========================================\n');
fprintf('  Clot/Blood = %.2fx  (target: %.2fx)\n', ...
    Z_results(2)/Z_results(1), Z_target(3)/Z_target(2));
fprintf('  Wall/Blood = %.2fx  (target: %.2fx)\n', ...
    Z_results(3)/Z_results(1), Z_target(4)/Z_target(2));
fprintf('========================================\n');

% Save results to CSV for Python presentation script
results_file = fullfile(out_dir, 'impedance_results.csv');
fid = fopen(results_file, 'w');
fprintf(fid, 'Material,Z_sim_Ohm,Z_target_Ohm,Ratio_to_Blood\n');
for m = 1:3
    idx = m + 1; if m == 1, idx = 2; end
    fprintf(fid, '%s,%.2f,%d,%.4f\n', material_names{m}, ...
        Z_results(m), Z_target(idx), Z_results(m)/Z_results(1));
end
fclose(fid);
fprintf('  Results saved to %s\n', results_file);

% Reset blood material and re-solve for final presentation model
mat_blood.propertyGroup('def').set('electricconductivity', ...
    num2str(sigma_sweep(1), '%.6g'));
mat_blood.propertyGroup('def').set('relpermittivity', ...
    num2str(epsr_sweep(1), '%.6g'));
model.study('std1').run();

% Mesh visualization
pg_mesh = model.result.create('pg_mesh', 'PlotGroup3D');
pg_mesh.label('Mesh Visualization');
mesh_plt = pg_mesh.create('mesh1', 'Mesh');
mesh_plt.set('filteractive', 'on');
pg_mesh.run;
exp_mesh = model.result.export.create('img_mesh', 'Image');
exp_mesh.set('plotgroup', 'pg_mesh');
exp_mesh.set('pngfilename', fullfile(out_dir, 'mesh_3d.png'));
exp_mesh.set('size', 'current');
exp_mesh.run;

% Geometry visualization (wireframe style)
pg_geom = model.result.create('pg_geom', 'PlotGroup3D');
pg_geom.label('Geometry');
ms_geom = pg_geom.create('mslc_geom', 'Multislice');
ms_geom.set('expr', '1');
ms_geom.set('multiplanexmethod', 'coord');
ms_geom.set('xcoord', '0');
ms_geom.set('multiplaneymethod', 'number');
ms_geom.set('ynumber', '0');
ms_geom.set('multiplanezmethod', 'number');
ms_geom.set('znumber', '0');
pg_geom.run;
exp_geom = model.result.export.create('img_geom', 'Image');
exp_geom.set('plotgroup', 'pg_geom');
exp_geom.set('pngfilename', fullfile(out_dir, 'geometry_3d.png'));
exp_geom.set('size', 'current');
exp_geom.run;

% Final save
mph_final = fullfile(pwd, 'bioimpedance_v5_final.mph');
mphsave(model, mph_final);
fprintf('  Final model saved: %s\n', mph_final);

fprintf('\n=== COMPLETE ===\n');
fprintf('  All images and results in: %s\n', out_dir);
