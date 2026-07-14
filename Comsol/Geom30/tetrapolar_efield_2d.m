%% tetrapolar_efield_2d.m
% =========================================================================
% 2-D E-field simulation of a 4-electrode linear tetrapolar array
% using MATLAB PDE Toolbox (no COMSOL required).
%
% Geometry (2-D cross-section, longitudinal x  x  depth y):
%
%   Electrodes on bottom boundary (y = 0):
%       E1 ── E2 ── E3 ── E4       (left to right)
%       I+    V+    V−    I−
%   Outer pair (E1, E4): current injection   ← Dirichlet BCs
%   Inner pair (E2, E3): voltage sensing     ← zero-flux Neumann (floating)
%
% Outputs
%   Figure 1 : Electric potential + white E-field streamlines
%   Figure 2 : |E| magnitude heat map
%   Figure 3 : Side-by-side comparison for blood / clot / wall
%   Console  : Z_4probe  (transfer impedance)
%              Z_2probe  (apparent 2-electrode impedance)
%
% Requirements: MATLAB PDE Toolbox  (R2019b or later recommended)
% =========================================================================

clear; close all; clc;

%% =========================================================================
%  USER PARAMETERS  ← change these
% =========================================================================
elec_spacing_mm  = 2.0;    % centre-to-centre spacing between adjacent electrodes [mm]
elec_width_mm    = 0.78;   % electrode width along longitudinal (x) axis [mm]
elec_height_mm   = 2.1;    % electrode height, transverse axis [mm]  (annotation only in 2-D)
domain_height_mm = 10.0;   % depth of medium above electrodes [mm]
domain_margin_mm = 4.0;    % domain extension beyond outer electrodes, each side [mm]
V_drive          = 1.0;    % voltage applied to E1 (+V_drive) and E4 (−V_drive) [V]

% Medium conductivities at 50 kHz [S/m]  (DC solve only — used for Figs 1–3)
sigma_blood = 0.88;
sigma_clot  = 0.23;
sigma_wall  = 0.35;

% Tissue to show in Figures 1 & 2 (change to sigma_clot or sigma_wall)
sigma_single = sigma_blood;

% ── Cole-Cole tissue parameters (for frequency sweep) ──────────────────
% sigma*(f) = sigma_dc + j*omega*eps0*(eps_inf + delta_eps/(1+(j*omega*tau)^(1-alpha)))
cc_params = struct( ...
    'blood', struct('sigma_dc', 1.30,  'delta_eps', 2.53e6, 'tau', 10e-6, 'alpha', 0.25, 'eps_inf', 50), ...
    'clot',  struct('sigma_dc', 0.155, 'delta_eps', 7.7e5,  'tau', 12e-6, 'alpha', 0.30, 'eps_inf', 40), ...
    'wall',  struct('sigma_dc', 0.40,  'delta_eps', 1.2e6,  'tau',  9e-6, 'alpha', 0.25, 'eps_inf', 40)  ...
);

% Frequency sweep
freq_list     = logspace(3, 5, 40);        % 1 kHz to 100 kHz, 40 points
freq_highlight = [5e3, 50e3, 100e3];       % print spot-check values

% =========================================================================

%% ── DERIVED GEOMETRY (all in metres) ────────────────────────────────────
sp = elec_spacing_mm  * 1e-3;
ew = elec_width_mm    * 1e-3;
dh = domain_height_mm * 1e-3;
dm = domain_margin_mm * 1e-3;

% Electrode centres: E1 leftmost, E4 rightmost, symmetric about origin
xc   = [-1.5, -0.5, 0.5, 1.5] * sp;   % [m]
E_x0 = xc - ew/2;                      % left  edge of each electrode
E_x1 = xc + ew/2;                      % right edge of each electrode

% Domain extent
x_L = xc(1) - dm;
x_R = xc(4) + dm;

m2mm = 1e3;   % scale factor: m → mm for plotting

%% ── HELPER: Cole-Cole complex conductivity ──────────────────────────────
function sc = cole_cole_sigma(f, p)
    % Returns complex conductivity sigma*(f) [S/m] using Cole-Cole model.
    % p must have fields: sigma_dc, delta_eps, tau, alpha, eps_inf
    eps0  = 8.854e-12;
    omega = 2 * pi * f;
    jwt_a = (1j * omega * p.tau) .^ (1 - p.alpha);
    eps_c = p.eps_inf + p.delta_eps ./ (1 + jwt_a);
    sc    = p.sigma_dc + 1j * omega * eps0 .* eps_c;
end

%% ── HELPER: build & solve PDE for one conductivity ──────────────────────
function [result, model, elec_edge] = solve_laplace(x_L, x_R, dh, ...
                                         E_x0, E_x1, xc, ew, sp, sigma, V_drive, drive_idx)
    % drive_idx: [i j] — which two electrodes carry ±V_drive.
    %   [1 4] = outer pair (4-probe tetrapolar)
    %   [2 3] = inner pair (2-probe, matches 4294A 2-wire mode)
    if nargin < 11, drive_idx = [1 4]; end
    % ── Polygon vertices (CCW) ─────────────────────────────────────────
    % The bottom boundary is split at every electrode edge so each
    % electrode maps to exactly one polygon edge (needed for per-edge BCs).
    %
    % Vertex layout (n_b + 2 total):
    %   1 … n_b : bottom points (x_L … x_R) at y = 0
    %   n_b+1   : top-right  (x_R, dh)     — same x as btm_x(end), new y
    %   n_b+2   : top-left   (x_L, dh)
    %   polygon auto-closes back to vertex 1
    btm_x = sort(unique([x_L, E_x0(:)', E_x1(:)', x_R]));
    n_b   = numel(btm_x);
    vx    = [btm_x(:)', x_R, x_L];    % n_b + 2
    vy    = [zeros(1,n_b), dh, dh];   % n_b + 2
    nv    = numel(vx);

    % ── decsg call ────────────────────────────────────────────────────
    % Bug fix: passing ns as a plain string '1×2 char' is interpreted as
    % two 1-character shape names.  char('P1')' gives a 2×1 column = one
    % shape named 'P1'.
    gd = [3; nv; vx(:); vy(:)];
    dl = decsg(gd, 'P1', char('P1')');

    model = createpde(1);
    geometryFromEdges(model, dl);

    % ── Identify electrode edges ───────────────────────────────────────
    % Version-compatible approach: use nearestEdge from points just above
    % each electrode centre. (evaluateGeometry is unavailable in older MATLAB.)
    nEdges    = model.Geometry.NumEdges;
    elec_edge = zeros(1, 4);
    yq = max(1e-9, min(dh*1e-3, ew*0.05));
    for k = 1:4
        elec_edge(k) = nearestEdge(model.Geometry, [xc(k), yq]);
    end

    % Guard against accidental duplicate mapping (rare, but can happen if
    % query points are too close to vertices in some releases).
    if numel(unique(elec_edge)) < 4
        for k = 1:4
            x_candidates = [xc(k), (E_x0(k)+xc(k))/2, (xc(k)+E_x1(k))/2];
            for q = 1:numel(x_candidates)
                cand = nearestEdge(model.Geometry, [x_candidates(q), yq]);
                if ~ismember(cand, elec_edge(1:max(k-1,1))) || k == 1
                    elec_edge(k) = cand;
                    break;
                end
            end
        end
    end

    if numel(unique(elec_edge)) < 4
        error(['Could not uniquely map E1..E4 to boundary edges. ' ...
               'Try increasing domain_margin_mm or elec_spacing_mm, then rerun.']);
    end

    % ── Boundary conditions ───────────────────────────────────────────
    % drive_idx(1) = +V, drive_idx(2) = -V.  All others: default zero-flux.
    applyBoundaryCondition(model, 'dirichlet', 'Edge', elec_edge(drive_idx(1)), 'u', +V_drive);
    applyBoundaryCondition(model, 'dirichlet', 'Edge', elec_edge(drive_idx(2)), 'u', -V_drive);

    % ── PDE: −div(σ ∇V) = 0 ──────────────────────────────────────────
    specifyCoefficients(model, 'm', 0, 'd', 0, 'c', sigma, 'a', 0, 'f', 0);

    % ── Mesh ──────────────────────────────────────────────────────────
    generateMesh(model, 'Hmax', sp/10, 'GeometricOrder', 'quadratic');

    result = solvepde(model);
end

%% ── HELPER: compute 4-probe COMPLEX Z from tetrapolar solve ─────────────
function [Z4, V_transfer, I_inject] = compute_Z4(result, E_x0, E_x1, xc, sigma)
    V2 = interpolateSolution(result, xc(2), 1e-8);
    V3 = interpolateSolution(result, xc(3), 1e-8);
    V_transfer = V2 - V3;
    n_int  = 120;
    x_e1   = linspace(E_x0(1), E_x1(1), n_int);
    delta  = (E_x1(1) - E_x0(1)) / 20;
    V_lo   = interpolateSolution(result, x_e1,          zeros(1,n_int));
    V_hi   = interpolateSolution(result, x_e1, delta*ones(1,n_int));
    I_inject = sigma * trapz(x_e1, (V_hi - V_lo) / delta);
    % Return complex Z so caller gets both magnitude and phase
    Z4 = V_transfer / I_inject;
end

%% ── HELPER: compute 2-probe Z from inner-electrode (E2/E3) solve ─────────
% result2 : solvepde output with E2/E3 as drive (±V_drive)
% Z2 = |V_E2 - V_E3| / |I_through_E2|  — exactly what the 4294A measures
% in 2-wire mode: same terminals drive current and sense voltage.
function [Z2, I_inject2] = compute_Z2(result2, E_x0, E_x1, xc, sigma)
    % Voltage across the drive pair (E2–E3)
    V2 = interpolateSolution(result2, xc(2), 1e-8);
    V3 = interpolateSolution(result2, xc(3), 1e-8);
    V_drive_meas = V2 - V3;   % should equal 2*V_drive by construction

    % Current injected through E2
    n_int = 120;
    x_e2  = linspace(E_x0(2), E_x1(2), n_int);
    delta = (E_x1(2) - E_x0(2)) / 20;
    V_lo  = interpolateSolution(result2, x_e2,          zeros(1,n_int));
    V_hi  = interpolateSolution(result2, x_e2, delta*ones(1,n_int));
    I_inject2 = sigma * trapz(x_e2, (V_hi - V_lo) / delta);

    Z2 = abs(V_drive_meas) / abs(I_inject2);
end

%% ── HELPER: interpolate V and E on a regular grid ───────────────────────
function [Xg, Yg, Vg, Ex, Ey] = grid_fields(result, x_L, x_R, dh, nx, ny)
    xg = linspace(x_L, x_R, nx);
    yg = linspace(0,   dh,  ny);
    [Xg, Yg] = meshgrid(xg, yg);
    Vg = reshape(interpolateSolution(result, Xg(:)', Yg(:)'), ny, nx);
    [dVdx, dVdy] = gradient(Vg, xg(2)-xg(1), yg(2)-yg(1));
    Ex = -dVdx;
    Ey = -dVdy;
end

%% ── HELPER: draw electrode rectangles on axes ────────────────────────────
function draw_electrodes(ax, E_x0, E_x1, xc, ew, m2mm)
    col_I = [0.85 0.22 0.10];   % red  : current electrodes
    col_V = [0.10 0.38 0.85];   % blue : sense electrodes
    elec_col = {col_I, col_V, col_V, col_I};
    elec_lbl = {'E1 (I+)', 'E2 (V+)', 'E3 (V−)', 'E4 (I−)'};
    for k = 1:4
        rectangle(ax, 'Position', [E_x0(k)*m2mm, -0.8, ew*m2mm, 0.65], ...
                  'FaceColor', elec_col{k}, 'EdgeColor', 'k', 'LineWidth', 1.5);
        text(ax, xc(k)*m2mm, -1.9, elec_lbl{k}, ...
             'HorizontalAlignment', 'center', 'FontSize', 9, ...
             'Color', elec_col{k}, 'FontWeight', 'bold');
    end
    % Spacing annotations between adjacent electrode centres
    for k = 1:3
        ymk = -3.3;
        line(ax, [xc(k) xc(k+1)]*m2mm, [ymk ymk], 'Color','k','LineWidth',0.8);
        line(ax, [xc(k)   xc(k)  ]*m2mm, [ymk-0.25 ymk+0.25], 'Color','k','LineWidth',0.8);
        line(ax, [xc(k+1) xc(k+1)]*m2mm, [ymk-0.25 ymk+0.25], 'Color','k','LineWidth',0.8);
        text(ax, (xc(k)+xc(k+1))/2*m2mm, ymk-0.7, ...
             sprintf('%.1f mm', (xc(k+1)-xc(k))*m2mm), ...
             'HorizontalAlignment','center','FontSize',8);
    end
    % Electrode width annotation on E1
    ymw = -5.2;
    line(ax, [E_x0(1) E_x1(1)]*m2mm, [ymw ymw], 'Color','k','LineWidth',0.8);
    line(ax, [E_x0(1) E_x0(1)]*m2mm, [ymw-0.25 ymw+0.25], 'Color','k','LineWidth',0.8);
    line(ax, [E_x1(1) E_x1(1)]*m2mm, [ymw-0.25 ymw+0.25], 'Color','k','LineWidth',0.8);
    text(ax, xc(1)*m2mm, ymw-0.7, sprintf('%.2f mm wide', ew*m2mm), ...
         'HorizontalAlignment','center','FontSize',8);
end

%% ── SOLVE FOR SINGLE TISSUE (Figures 1 & 2) ─────────────────────────────
fprintf('Solving 4-probe (E1/E4 drive) for sigma = %.3f S/m ... ', sigma_single);
[res1, ~, ~] = solve_laplace(x_L, x_R, dh, E_x0, E_x1, xc, ew, sp, ...
                              sigma_single, V_drive);
[Z4_s, Vt_s, Ii_s] = compute_Z4(res1, E_x0, E_x1, xc, sigma_single);
fprintf('done.\n');

fprintf('Solving 2-probe (E2/E3 drive) for sigma = %.3f S/m ... ', sigma_single);
[res1_2p, ~, ~] = solve_laplace(x_L, x_R, dh, E_x0, E_x1, xc, ew, sp, ...
                                 sigma_single, V_drive, [2 3]);
[Z2_s, Ii2_s] = compute_Z2(res1_2p, E_x0, E_x1, xc, sigma_single);
fprintf('done.\n');

fprintf('\n════════════════════════════════════════════\n');
fprintf('  sigma          = %.3f S/m\n', sigma_single);
fprintf('  I_inject (4p)  = %.4f mA  (through E1)\n', Ii_s*1e3);
fprintf('  V_transfer     = %.4f V  (E2 − E3)\n', Vt_s);
fprintf('  |Z|_4probe     = %.2f Ω  (E1/E4 drive, E2/E3 sense)\n',  abs(Z4_s));
fprintf('  Phase_4probe   = %.2f°\n',                                 angle(Z4_s)*180/pi);
fprintf('  |Z|_2probe     = %.2f Ω  (E2/E3 drive AND sense — matches 4294A 2-wire)\n', abs(Z2_s));
fprintf('  Phase_2probe   = %.2f°\n',                                 angle(Z2_s)*180/pi);
fprintf('  Z2/Z4 ratio    = %.2fx  (pure geometry effect in this model)\n', abs(Z2_s)/abs(Z4_s));
fprintf('════════════════════════════════════════════\n\n');

[Xg, Yg, Vg, Ex, Ey] = grid_fields(res1, x_L, x_R, dh, 300, 200);

%% ── FIGURE 1: Potential + E-field streamlines ───────────────────────────
fig1 = figure('Name', 'E-field Streamlines', 'Position', [40 60 1000 680]);
ax1  = axes(fig1);

contourf(ax1, Xg*m2mm, Yg*m2mm, Vg, 40, 'LineStyle', 'none');
cbar1 = colorbar(ax1);
cbar1.Label.String = 'Electric potential [V]';
colormap(ax1, 'jet');
hold(ax1, 'on');

% E-field streamlines (white)
h_str = streamslice(Xg*m2mm, Yg*m2mm, Ex, Ey, 2.5);
set(h_str, 'Color', 'w', 'LineWidth', 1.3);

draw_electrodes(ax1, E_x0, E_x1, xc, ew, m2mm);

xlabel(ax1, 'x  [mm]  — longitudinal axis', 'FontSize', 13);
ylabel(ax1, 'y  [mm]  — depth into medium', 'FontSize', 13);
title(ax1, sprintf(['4-electrode array  |  \\sigma = %.2f S/m\n' ...
    'Z_{4-probe} = %.1f \\Omega  |  Z_{2-probe} = %.1f \\Omega'], ...
    sigma_single, Z4_s, Z2_s), 'FontSize', 13);

% Legend patches
col_I = [0.85 0.22 0.10]; col_V = [0.10 0.38 0.85];
hCur = patch(ax1, NaN, NaN, col_I, 'EdgeColor','k', ...
    'DisplayName','Current electrode  E1/E4 (Dirichlet ±V)');
hSen = patch(ax1, NaN, NaN, col_V, 'EdgeColor','k', ...
    'DisplayName','Sense electrode  E2/E3 (floating, no drive BC)');
legend(ax1, [hCur, hSen], 'Location', 'northeast', 'FontSize', 9);

ylim(ax1, [-6.5, dh*m2mm]);
xlim(ax1, [x_L*m2mm, x_R*m2mm]);
set(ax1, 'YDir', 'normal');
grid(ax1, 'on'); grid(ax1, 'minor');

%% ── FIGURE 2: |E| magnitude heat map ────────────────────────────────────
E_mag = sqrt(Ex.^2 + Ey.^2);

fig2 = figure('Name', '|E| Magnitude', 'Position', [1060 60 900 580]);
ax2  = axes(fig2);

imagesc(ax2, Xg(1,:)*m2mm, Yg(:,1)*m2mm, E_mag);
set(ax2, 'YDir', 'normal');
colormap(ax2, 'hot');
cbar2 = colorbar(ax2);
cbar2.Label.String = '|E|  [V/m]';
hold(ax2, 'on');
draw_electrodes(ax2, E_x0, E_x1, xc, ew, m2mm);
xlabel(ax2, 'x  [mm]', 'FontSize', 13);
ylabel(ax2, 'y  [mm]', 'FontSize', 13);
title(ax2, sprintf('|E| Magnitude  |  \\sigma = %.2f S/m', sigma_single), 'FontSize', 13);
ylim(ax2, [-6.5, dh*m2mm]);
xlim(ax2, [x_L*m2mm, x_R*m2mm]);
grid(ax2, 'on');

%% ── SOLVE FOR ALL THREE TISSUES ─────────────────────────────────────────
tissues = {'Blood', 'Clot', 'Wall'};
sigmas  = [sigma_blood, sigma_clot, sigma_wall];
results_all = cell(1,3);
Z4_all = zeros(1,3);
Z2_all = zeros(1,3);

fprintf('Solving for all three tissue types ...\n');
for ti = 1:3
    fprintf('  %s (sigma=%.2f S/m) 4-probe...', tissues{ti}, sigmas(ti));
    [r, ~, ~] = solve_laplace(x_L, x_R, dh, E_x0, E_x1, xc, ew, sp, ...
                               sigmas(ti), V_drive);
    [Z4_all(ti)] = compute_Z4(r, E_x0, E_x1, xc, sigmas(ti));
    Z4_all(ti) = abs(Z4_all(ti));
    [~, ~, Vg_ti, Ex_ti, Ey_ti] = grid_fields(r, x_L, x_R, dh, 250, 160);
    results_all{ti} = struct('Vg', Vg_ti, 'Ex', Ex_ti, 'Ey', Ey_ti);

    fprintf(' 2-probe...');
    [r2, ~, ~] = solve_laplace(x_L, x_R, dh, E_x0, E_x1, xc, ew, sp, ...
                                sigmas(ti), V_drive, [2 3]);
    [Z2_all(ti)] = compute_Z2(r2, E_x0, E_x1, xc, sigmas(ti));
    Z2_all(ti) = abs(Z2_all(ti));
    fprintf(' Z4=%.1f Ω, Z2=%.1f Ω\n', Z4_all(ti), Z2_all(ti));
end

%% ── FIGURE 3: Side-by-side comparison blood / clot / wall ────────────────
[Xg3, Yg3] = meshgrid(linspace(x_L, x_R, 250), linspace(0, dh, 160));

fig3 = figure('Name', 'Tissue Comparison', 'Position', [40 760 1400 560]);
tis_colors = {'Blues', 'Reds', 'Greens'};   % nominal; use jet for all in MATLAB

for ti = 1:3
    ax = subplot(1, 3, ti);
    Vg_ti = results_all{ti}.Vg;
    Ex_ti = results_all{ti}.Ex;
    Ey_ti = results_all{ti}.Ey;

    contourf(ax, Xg3*m2mm, Yg3*m2mm, Vg_ti, 30, 'LineStyle', 'none');
    colormap(ax, 'jet');
    hold(ax, 'on');

    h_s = streamslice(Xg3*m2mm, Yg3*m2mm, Ex_ti, Ey_ti, 2);
    set(h_s, 'Color', 'w', 'LineWidth', 1.1);

    draw_electrodes(ax, E_x0, E_x1, xc, ew, m2mm);

    title(ax, sprintf(['%s  (\\sigma=%.2f S/m)\n' ...
        'Z_4(E1/E4\\rightarrowE2/E3)=%.1f\\Omega\n' ...
        'Z_2(E2/E3 drive+sense)=%.1f\\Omega'], ...
        tissues{ti}, sigmas(ti), Z4_all(ti), Z2_all(ti)), 'FontSize', 10);
    xlabel(ax, 'x [mm]', 'FontSize', 11);
    if ti == 1, ylabel(ax, 'y [mm] (depth)', 'FontSize', 11); end
    ylim(ax, [-6.5, dh*m2mm]);
    xlim(ax, [x_L*m2mm, x_R*m2mm]);
    set(ax, 'YDir', 'normal');
    grid(ax, 'on');
end
sgtitle(fig3, sprintf(['Tetrapolar E-field — Blood vs Clot vs Wall\n' ...
    'Electrode spacing = %.1f mm  |  Electrode %.2f × %.2f mm'], ...
    elec_spacing_mm, elec_width_mm, elec_height_mm), 'FontSize', 13);

%% ── SUMMARY TABLE ────────────────────────────────────────────────────────
fprintf('\n════════ IMPEDANCE SUMMARY ══════════════════════════════════════════\n');
fprintf('  %-8s  %-8s  %-16s  %-16s  %-10s\n', ...
        'Tissue','sigma','Z4 E1/E4→E2/E3','Z2 E2/E3 drive+sense','Z2/Z4');
fprintf('  %s\n', repmat('-',1,70));
for ti = 1:3
    fprintf('  %-8s  %-8.3f  %-16.2f  %-16.2f  %-10.2f\n', ...
        tissues{ti}, sigmas(ti), Z4_all(ti), Z2_all(ti), Z2_all(ti)/Z4_all(ti));
end
fprintf('  %s\n', repmat('-',1,70));
fprintf('  Clot/Blood Z4 ratio : %.2fx\n', Z4_all(2)/Z4_all(1));
fprintf('  Wall/Blood Z4 ratio : %.2fx\n', Z4_all(3)/Z4_all(1));
fprintf('  Clot/Blood Z2 ratio : %.2fx\n', Z2_all(2)/Z2_all(1));
fprintf('  Wall/Blood Z2 ratio : %.2fx\n', Z2_all(3)/Z2_all(1));
fprintf('═══════════════════════════════════════════════════════════════════════\n');
fprintf('\nZ4: tetrapolar — outer pair (E1/E4) injects current, inner pair (E2/E3) senses.\n');
fprintf('Z2: 2-probe    — inner pair (E2/E3) both injects AND senses (matches 4294A 2-wire mode).\n');
fprintf('Z2/Z4 difference in this model is purely geometric (no interface impedance modelled).\n');
fprintf('DC solve only — see frequency sweep below for Cole-Cole complex Z and phase.\n');

%% ═══════════════════════════════════════════════════════════════════════
%  FREQUENCY SWEEP  (Cole-Cole complex sigma, 1 kHz to 100 kHz)
%
%  MATLAB PDE Toolbox solvepde only accepts REAL coefficients — passing
%  complex sigma silently drops the imaginary part → real V/I → phase = 0.
%
%  Correct approach: geometry is frequency-independent for homogeneous media.
%  Compute real cell constants K4 and K2 ONCE from the DC solve, then:
%
%      Z4(f) = K4 / sigma*(f)      Z2(f) = K2 / sigma*(f)
%
%  Phase = angle(Z) = -angle(sigma*(f))  — tissue-specific, physically correct.
%  Same method as monte_carlo_cole_cole.py.
% ═══════════════════════════════════════════════════════════════════════
fprintf('\n── Frequency sweep (cell-constant × Cole-Cole, analytic) ──\n');

tissue_names  = {'blood', 'clot', 'wall'};
tissue_labels = {'Blood', 'Clot', 'Wall'};
nf = numel(freq_list);
nt = 3;

% Cell constants from DC solve (K = |Z_DC| * sigma_DC)
K4 = abs(Z4_s) * real(sigma_single);
K2 = abs(Z2_s) * real(sigma_single);
fprintf('  K4 (tetrapolar) = %.4f m^-1\n', K4);
fprintf('  K2 (2-probe)    = %.4f m^-1\n', K2);

% Pre-allocate complex results
Z4_sweep = zeros(nt, nf);
Z2_sweep = zeros(nt, nf);

% Scale K per tissue so DC reference matches each tissue's 50 kHz conductivity
sigma_50k_blood = real(cole_cole_sigma(50e3, cc_params.blood));
for ti = 1:nt
    p     = cc_params.(tissue_names{ti});
    scale = sigma_50k_blood / real(cole_cole_sigma(50e3, p));
    for fi = 1:nf
        sc = cole_cole_sigma(freq_list(fi), p);
        Z4_sweep(ti, fi) = (K4 * scale) / sc;
        Z2_sweep(ti, fi) = (K2 * scale) / sc;
    end
end

%% ── FIGURE 4: |Z| vs frequency ────────────────────────────────────────
tic_col   = {'b', [0.8 0.3 0], [0.1 0.6 0.1]};     % blue=blood, orange=clot, green=wall
ls4       = '-';                                     % solid = 4-probe
ls2       = '--';                                    % dashed = 2-probe
fhi       = freq_highlight;
fhi_khz   = fhi / 1e3;

% X-axis in kHz
freq_khz = freq_list / 1e3;
fhi_khz  = fhi / 1e3;

fig4 = figure('Name', '|Z| vs Frequency', 'Position', [40 60 1050 520]);
ax4  = axes(fig4);
hold(ax4, 'on');
hL = gobjects(6,1);
for ti = 1:nt
    hL(ti)   = semilogx(ax4, freq_khz, abs(Z4_sweep(ti,:)), ls4, ...
        'Color', tic_col{ti}, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('%s Z4 (tetrapolar)', tissue_labels{ti}));
    hL(ti+3) = semilogx(ax4, freq_khz, abs(Z2_sweep(ti,:)), ls2, ...
        'Color', tic_col{ti}, 'LineWidth', 2.0, ...
        'DisplayName', sprintf('%s Z2 (2-probe)',    tissue_labels{ti}));
end
for hi = 1:numel(fhi_khz)
    [~, idx] = min(abs(freq_khz - fhi_khz(hi)));
    for ti = 1:nt
        plot(ax4, fhi_khz(hi), abs(Z4_sweep(ti,idx)), 'o', 'Color', tic_col{ti}, ...
             'MarkerFaceColor', tic_col{ti}, 'MarkerSize', 8, 'HandleVisibility', 'off');
        plot(ax4, fhi_khz(hi), abs(Z2_sweep(ti,idx)), 's', 'Color', tic_col{ti}, ...
             'MarkerFaceColor', 'w', 'MarkerSize', 8, 'HandleVisibility', 'off');
    end
    xline(ax4, fhi_khz(hi), ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, ...
          'Label', sprintf('%g kHz', fhi_khz(hi)), ...
          'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
end
xlabel(ax4, 'Frequency [kHz]', 'FontSize', 12);
ylabel(ax4, '|Z| [Ω]', 'FontSize', 12);
title(ax4, '|Z| vs Frequency — Blood / Clot / Wall  |  Cole-Cole σ*(f)', 'FontSize', 12);
legend(ax4, hL, 'Location', 'northeast', 'FontSize', 9);
grid(ax4, 'on'); grid(ax4, 'minor');

%% ── FIGURE 5: Phase vs frequency ──────────────────────────────────────
fig5 = figure('Name', 'Phase vs Frequency', 'Position', [1060 60 1050 520]);
ax5  = axes(fig5);
hold(ax5, 'on');
hP = gobjects(6,1);
for ti = 1:nt
    hP(ti)   = semilogx(ax5, freq_khz, angle(Z4_sweep(ti,:))*180/pi, ls4, ...
        'Color', tic_col{ti}, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('%s Z4 (tetrapolar)', tissue_labels{ti}));
    hP(ti+3) = semilogx(ax5, freq_khz, angle(Z2_sweep(ti,:))*180/pi, ls2, ...
        'Color', tic_col{ti}, 'LineWidth', 2.0, ...
        'DisplayName', sprintf('%s Z2 (2-probe)',    tissue_labels{ti}));
end
for hi = 1:numel(fhi_khz)
    [~, idx] = min(abs(freq_khz - fhi_khz(hi)));
    for ti = 1:nt
        plot(ax5, fhi_khz(hi), angle(Z4_sweep(ti,idx))*180/pi, 'o', 'Color', tic_col{ti}, ...
             'MarkerFaceColor', tic_col{ti}, 'MarkerSize', 8, 'HandleVisibility', 'off');
        plot(ax5, fhi_khz(hi), angle(Z2_sweep(ti,idx))*180/pi, 's', 'Color', tic_col{ti}, ...
             'MarkerFaceColor', 'w', 'MarkerSize', 8, 'HandleVisibility', 'off');
    end
    xline(ax5, fhi_khz(hi), ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, ...
          'Label', sprintf('%g kHz', fhi_khz(hi)), ...
          'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
end
xlabel(ax5, 'Frequency [kHz]', 'FontSize', 12);
ylabel(ax5, 'Phase [°]', 'FontSize', 12);
title(ax5, sprintf(['Phase vs Frequency — Blood / Clot / Wall  |  Cole-Cole \\sigma*(f)\n' ...
    'Z4 and Z2 have identical phase here (no electrode interface in model)\n' ...
    'In real measurement Z2 shows extra negative phase from CPE interface']), ...
    'FontSize', 10);
legend(ax5, hP, 'Location', 'southwest', 'FontSize', 9);
grid(ax5, 'on'); grid(ax5, 'minor');
yline(ax5, 0, 'k:', 'HandleVisibility', 'off');

%% ── SPOT-CHECK TABLE ───────────────────────────────────────────────────
fprintf('\n════════ FREQUENCY SPOT-CHECK ══════════════════════════════════════\n');
fprintf('  %-6s  %-8s  %-12s  %-10s  %-12s  %-10s\n', ...
        'Tissue', 'Freq kHz', '|Z4| Ω', 'Phase4 °', '|Z2| Ω', 'Phase2 °');
fprintf('  %s\n', repmat('-',1,68));
for ti = 1:nt
    for hi = 1:numel(fhi)
        [~, idx] = min(abs(freq_khz - fhi_khz(hi)));
        fprintf('  %-6s  %-8.0f  %-12.2f  %-10.2f  %-12.2f  %-10.2f\n', ...
            tissue_labels{ti}, fhi(hi)/1e3, ...
            abs(Z4_sweep(ti,idx)),  angle(Z4_sweep(ti,idx))*180/pi, ...
            abs(Z2_sweep(ti,idx)),  angle(Z2_sweep(ti,idx))*180/pi);
    end
    fprintf('  %s\n', repmat('-',1,68));
end
fprintf('Solid line = Z4 (tetrapolar). Dashed = Z2 (2-probe inner pair).\n');
fprintf('Circles = Z4 markers. Squares = Z2 markers at 5/50/100 kHz.\n');
