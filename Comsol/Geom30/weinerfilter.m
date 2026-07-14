% 1. Setup Time and Signal Parameters
fs = 1000;                  
t = 0:1/fs:1-1/fs;          
f_sig = 5;                  

% 2. Generate Clean Signal, Noise, and Noisy Input
s = sin(2*pi*f_sig*t);      
n = 0.5 * randn(size(t));   
x = s + n;                  

% 3. Estimate One-Sided Power Spectral Densities (PSD)
[S_xx, f] = pwelch(x, [], [], [], fs); 
[S_nn, ~] = pwelch(n, [], [], [], fs);
S_ss = max(S_xx - S_nn, 1e-6); 

% 4. Design the ONE-SIDED Wiener Filter (0 to fs/2)
H_onesided = S_ss ./ (S_ss + S_nn);

% 5. CONSTRUCT THE TWO-SIDED SYMMETRIC FILTER
X = fft(x);                 
N_fft = length(X);
f_fft = (0:N_fft-1)*(fs/N_fft); 

% Map frequencies above fs/2 to their negative counterparts
f_fft_wrapped = f_fft;
f_fft_wrapped(f_fft > fs/2) = fs - f_fft(f_fft > fs/2);

% Interpolate using the wrapped symmetric frequency grid
H_full = interp1(f, H_onesided, f_fft_wrapped, 'linear', 'extrap');

% 6. Apply Filter and Invert
X_filtered = X .* H_full;
s_hat = real(ifft(X_filtered));

% ==========================================
% FIGURE 1: 3-Subplot Time Breakdown
% ==========================================
figure('Name', 'Signal Breakdown');

% Subplot 1: Clean Original
subplot(3,1,1); 
plot(t, s, 'k', 'LineWidth', 1.5); 
title('Original Clean Signal (s)'); 
grid on;

% Subplot 2: Noisy Input
subplot(3,1,2); 
plot(t, x, 'r'); 
title('Noisy Input Signal (x = s + n)'); 
grid on;

% Subplot 3: Cleaned Output
subplot(3,1,3); 
plot(t, s_hat, 'b', 'LineWidth', 1.5); 
title('Wiener Filter Estimate (Full Amplitude)'); 
grid on;
xlabel('Time (seconds)');

% ==========================================
% FIGURE 2: Direct Overlay Comparison Plot
% ==========================================
figure('Name', 'Overlay Comparison');
plot(t, s, 'k-', 'LineWidth', 2);           % Black solid line for original
hold on;
plot(t, s_hat, 'b--', 'LineWidth', 1.5);     % Blue dashed line for estimate
hold off;

% Formatting
title('Signal Comparison: Original vs. Wiener Filter Estimate');
xlabel('Time (seconds)');
ylabel('Amplitude');
legend('Original Clean Signal (s)', 'Wiener Filter Estimate (\hat{s})', 'Location', 'best');
grid on;
