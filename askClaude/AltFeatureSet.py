"""
Biosignal Classifier — Training Code
=====================================
Architecture: GRU (raw signal) + 28 hand-crafted features → 2-layer MLP → 3 classes
Target:        ARM Cortex-M4 (exports Q15 fixed-point weights)
Signal:        150 Hz, 200 ms windows (30 samples), 1 channel
Classes:       0=blood, 1=clot, 2=wall

28 Features per channel
────────────────────────────────────────
Time domain (12):
  01 mean                  07 zero-crossing rate
  02 std                   08 mean absolute value (MAV)
  03 rms                   09 slope sign changes (SSC)
  04 peak-to-peak          10 waveform length
  05 skewness              11 max absolute value
  06 kurtosis              12 interquartile range (IQR)

Frequency domain (8):
  13 spectral centroid     17 power low  (0–15 Hz)
  14 spectral spread       18 power mid  (15–40 Hz)
  15 spectral flatness     19 power high (40–75 Hz)
  16 dominant frequency    20 spectral rolloff (85 %)

Nonlinear / complexity (5):
  21 sample entropy (SampEn)
  22 Hjorth mobility
  23 Hjorth complexity
  24 approximate entropy (ApEn)
  25 signal energy

Shape / morphology (3):
  26 number of local peaks
  27 mean peak interval
  28 crest factor
────────────────────────────────────────
MLP input: GRU_hidden(16) + features(28) = 44
"""

# ─────────────────────────────────────────────
#  Individual feature functions
# ─────────────────────────────────────────────

# ── Time domain ───────────────────────────────
def feat_mean(x):           return float(x.mean())
def feat_std(x):            return float(x.std() + 1e-8)
def feat_rms(x):            return float(np.sqrt((x**2).mean()))
def feat_peak_to_peak(x):   return float(x.max() - x.min())
def feat_max_abs(x):        return float(np.abs(x).max())
def feat_mav(x):            return float(np.abs(x).mean())
def feat_iqr(x):            return float(np.percentile(x, 75) - np.percentile(x, 25))

def feat_skewness(x):
    mu, s = x.mean(), x.std() + 1e-8
    return float(((x - mu)**3).mean() / s**3)

def feat_kurtosis(x):
    mu, s = x.mean(), x.std() + 1e-8
    return float(((x - mu)**4).mean() / s**4)

def feat_zcr(x):
    return float(((x[:-1] * x[1:]) < 0).sum() / len(x))

def feat_ssc(x):
    """Slope sign changes: fraction of sign changes in the first difference."""
    d = np.diff(x)
    return float(((d[:-1] * d[1:]) < 0).sum() / len(x))

def feat_waveform_length(x):
    """Sum of absolute first differences (arc length)."""
    return float(np.abs(np.diff(x)).sum())

# ── Frequency domain ──────────────────────────
def _spectrum(x, fs):
    mag   = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return mag, freqs

def feat_spectral_centroid(x, fs):
    mag, freqs = _spectrum(x, fs)
    return float((freqs * mag).sum() / (mag.sum() + 1e-8))

def feat_spectral_spread(x, fs):
    mag, freqs = _spectrum(x, fs)
    centroid = (freqs * mag).sum() / (mag.sum() + 1e-8)
    return float((mag * (freqs - centroid)**2).sum() / (mag.sum() + 1e-8))

def feat_spectral_flatness(x, fs):
    mag, _ = _spectrum(x, fs)
    mag    = mag + 1e-10
    return float(np.exp(np.log(mag).mean()) / (mag.mean() + 1e-10))

def feat_dominant_freq(x, fs):
    mag, freqs = _spectrum(x, fs)
    return float(freqs[np.argmax(mag)])

def feat_band_power(x, fs, f_low, f_high):
    mag, freqs = _spectrum(x, fs)
    mask = (freqs >= f_low) & (freqs < f_high)
    return float((mag[mask]**2).sum() / (len(x)**2 + 1e-8))

def feat_spectral_rolloff(x, fs, threshold=0.85):
    mag, freqs = _spectrum(x, fs)
    cumsum = np.cumsum(mag)
    idx    = np.searchsorted(cumsum, threshold * cumsum[-1])
    return float(freqs[min(idx, len(freqs) - 1)])

# ── Nonlinear / complexity ────────────────────
def feat_sample_entropy(x, m=2, r_frac=0.2):
    """SampEn — O(N^2), fine for N=30."""
    N = len(x)
    r = r_frac * (x.std() + 1e-8)

    def _count(m):
        c = 0
        for i in range(N - m):
            t = x[i:i + m]
            for j in range(i + 1, N - m):
                if np.max(np.abs(t - x[j:j + m])) < r:
                    c += 1
        return c

    A, B = _count(m + 1), _count(m)
    return float(-np.log((A + 1e-8) / (B + 1e-8)))

def feat_approximate_entropy(x, m=2, r_frac=0.2):
    """ApEn — regularity measure.  Lower = more regular."""
    N = len(x)
    r = r_frac * (x.std() + 1e-8)

    def _phi(m):
        counts = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            t = x[i:i + m]
            for j in range(N - m + 1):
                if np.max(np.abs(t - x[j:j + m])) <= r:
                    counts[i] += 1
        return np.sum(np.log(counts / (N - m + 1) + 1e-8)) / (N - m + 1)

    return float(_phi(m) - _phi(m + 1))

def feat_hjorth_mobility(x):
    dx = np.diff(x)
    return float(np.sqrt(dx.var() / (x.var() + 1e-8)))

def feat_hjorth_complexity(x):
    dx  = np.diff(x)
    ddx = np.diff(dx)
    mob_x  = np.sqrt(dx.var()  / (x.var()  + 1e-8))
    mob_dx = np.sqrt(ddx.var() / (dx.var() + 1e-8))
    return float(mob_dx / (mob_x + 1e-8))

def feat_energy(x):
    return float((x**2).sum())

# ── Shape / morphology ────────────────────────
def feat_n_peaks(x):
    peaks, _ = find_peaks(x)
    return float(len(peaks))

def feat_mean_peak_interval(x):
    peaks, _ = find_peaks(x)
    if len(peaks) < 2:
        return float(len(x))
    return float(np.diff(peaks).mean())

def feat_crest_factor(x):
    return float(np.abs(x).max() / (np.sqrt((x**2).mean()) + 1e-8))
	