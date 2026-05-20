"""
Central configuration for the auto-labeler pipeline.
All "magic numbers" live here — sampling rate, min event duration, chunk size, etc.
"""

# ─── Signal parameters ───────────────────────────────────────────────────────
SAMPLING_RATE_HZ = 150.0          # Impedance sampling rate (Hz)
SAMPLE_DT_SEC = 1.0 / SAMPLING_RATE_HZ  # Time per sample (seconds)
SAMPLE_DT_MS = 1000.0 / SAMPLING_RATE_HZ  # Time per sample (milliseconds)

# ─── Segmentation parameters ─────────────────────────────────────────────────
NUM_CLASSES = 3                   # blood=0, clot=1, wall=2
CLASS_NAMES = ["blood", "clot", "wall"]

# ─── Chunk parameters ────────────────────────────────────────────────────────
CHUNK_SIZE = 4096                 # Samples per training chunk (~27.3s at 150 Hz)

# ─── Post-processing ─────────────────────────────────────────────────────────
MIN_EVENT_DURATION_SEC = 1.0      # Minimum event duration (seconds)
MIN_EVENT_DURATION_SAMPLES = int(MIN_EVENT_DURATION_SEC * SAMPLING_RATE_HZ)  # = 150

# ─── Multi-channel input ─────────────────────────────────────────────────────
NUM_CHANNELS = 5                  # Channels: [z-norm R, dR/dt, d²R/dt², 1s avg, detrended]
