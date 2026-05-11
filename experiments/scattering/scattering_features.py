# scattering_features.py
"""
Wavelet Scattering Transform feature extractor for 1D resistance signals.

Uses kymatio to compute a fixed (non-learned) multi-scale representation.
Produces 126 features per 5-second window (J=6, Q=8, averaged over time).

The scattering transform captures translation-invariant texture at multiple
scales — useful for distinguishing signal shapes (sustained plateau vs spike
vs noisy transition) regardless of absolute resistance level.
"""

import numpy as np
from kymatio.numpy import Scattering1D


# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
SAMPLE_RATE = 150
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 750

# Scattering parameters
J = 6   # max scale = 2^J = 64 samples ≈ 0.43s — captures sub-second structure
Q = 8   # wavelets per octave — fine frequency resolution

# ────────────────────────────────────────────────
# Scattering object (created once, reused)
# ────────────────────────────────────────────────
_scattering = Scattering1D(J=J, shape=WINDOW_SAMPLES, Q=Q)

# Determine output dimensionality
_test_out = _scattering(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
SCATTERING_DIM = _test_out.shape[0]  # 126 for J=6, Q=8, T=750


def extract_scattering_features(window: np.ndarray) -> np.ndarray:
    """
    Compute scattering features for a single window of resistance data.
    
    Parameters
    ----------
    window : np.ndarray, shape (750,)
        Raw resistance values for one 5-second window.
    
    Returns
    -------
    features : np.ndarray, shape (SCATTERING_DIM,)
        Time-averaged scattering coefficients (log-scale).
    """
    assert len(window) == WINDOW_SAMPLES, f"Expected {WINDOW_SAMPLES} samples, got {len(window)}"
    
    # Feed raw signal (no z-score normalization) so scattering captures
    # both absolute R level and texture/shape information.
    w = window.astype(np.float32)
    
    # Compute scattering: output shape (n_coeffs, n_time_steps)
    Sx = _scattering(w)
    
    # Average over time dimension → one vector per window
    # Use log(1 + x) for better dynamic range (scattering coefficients are non-negative)
    features = np.log1p(Sx.mean(axis=1))
    
    return features.astype(np.float32)


def extract_scattering_batch(windows: np.ndarray) -> np.ndarray:
    """
    Compute scattering features for a batch of windows.
    
    Parameters
    ----------
    windows : np.ndarray, shape (N, 750)
        Batch of resistance windows.
    
    Returns
    -------
    features : np.ndarray, shape (N, SCATTERING_DIM)
    """
    N = windows.shape[0]
    out = np.empty((N, SCATTERING_DIM), dtype=np.float32)
    for i in range(N):
        out[i] = extract_scattering_features(windows[i])
    return out


if __name__ == "__main__":
    # Quick test
    print(f"Window: {WINDOW_SAMPLES} samples ({WINDOW_SEC}s at {SAMPLE_RATE}Hz)")
    print(f"Scattering params: J={J}, Q={Q}")
    print(f"Output dimension: {SCATTERING_DIM} features per window")
    
    # Test with synthetic signal
    t = np.linspace(0, WINDOW_SEC, WINDOW_SAMPLES)
    # Simulate a clot-like signal: blood baseline + step up
    signal = 800 + 200 * (t > 2.0).astype(float) + 5 * np.random.randn(WINDOW_SAMPLES)
    feats = extract_scattering_features(signal)
    print(f"\nTest signal (step): features shape={feats.shape}, range=[{feats.min():.3f}, {feats.max():.3f}]")
    
    # Flat signal (blood)
    blood = 800 + 5 * np.random.randn(WINDOW_SAMPLES)
    feats_blood = extract_scattering_features(blood)
    print(f"Blood signal:       features shape={feats_blood.shape}, range=[{feats_blood.min():.3f}, {feats_blood.max():.3f}]")
    
    # Difference
    diff = np.abs(feats - feats_blood).mean()
    print(f"Mean |diff| between step and blood: {diff:.4f}")
