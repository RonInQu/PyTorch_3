"""
Central configuration for the auto-labeler pipeline.
All parameters live here — architecture, training, post-processing, inference.
"""

# ─── Signal parameters ───────────────────────────────────────────────────────
SAMPLING_RATE_HZ = 150.0          # Impedance sampling rate (Hz)
SAMPLE_DT_SEC = 1.0 / SAMPLING_RATE_HZ  # Time per sample (seconds)
SAMPLE_DT_MS = 1000.0 / SAMPLING_RATE_HZ  # Time per sample (milliseconds)

# ─── Segmentation parameters ─────────────────────────────────────────────────
NUM_CLASSES = 3                   # blood=0, clot=1, wall=2
CLASS_NAMES = ["blood", "clot", "wall"]

# ─── Input ────────────────────────────────────────────────────────────────────
NUM_CHANNELS = 1                  # 1=single-channel (z-norm R), 5=multichannel
CHUNK_SIZE = 4096*2                 # Samples per training chunk (~27.3s at 150 Hz)

# ─── Architecture ────────────────────────────────────────────────────────────
BASE_FILTERS = 32                 # Filters in first encoder level (doubles each level)
DEPTH = 5                         # Number of encoder/decoder levels
KERNEL_SIZE = 7                   # Convolution kernel size (odd)
DROPOUT = 0                     # Dropout rate (0=off, 0.2=light regularization)

# ─── Training ────────────────────────────────────────────────────────────────
EPOCHS = 80                       # Max training epochs
BATCH_SIZE = 64                   # Batch size (64 for A100, 16 for smaller GPUs)
LEARNING_RATE = 1e-3              # Peak learning rate for OneCycleLR
WEIGHT_DECAY = 1e-4               # AdamW weight decay
LOSS_FUNCTION = "ce"              # "ce" (CrossEntropy) or "focal" (FocalLoss)
FOCAL_GAMMA = 2.0                 # Focal loss gamma (only used if LOSS_FUNCTION="focal")
PATIENCE = 15                     # Early stopping patience (epochs without improvement)
VAL_FRACTION = 0.15               # Fraction of studies held out for validation
SEED = 42                         # Random seed
NUM_WORKERS = 2                   # DataLoader workers (0 for debugging)
GRAD_CLIP_NORM = 1.0              # Gradient clipping max norm

# ─── Data ────────────────────────────────────────────────────────────────────
TRAIN_STRIDE = None               # Training chunk stride (None = CHUNK_SIZE // 2)
AUGMENT_NOISE_STD = 0.02          # Gaussian noise sigma for augmentation
AUGMENT_SCALE_RANGE = (0.9, 1.1)  # Amplitude scaling range
AUGMENT_OFFSET_RANGE = (-0.1, 0.1)  # DC offset range

# ─── Post-processing ─────────────────────────────────────────────────────────
MIN_EVENT_DURATION_SEC = 5.0      # Minimum event duration (seconds)
MIN_EVENT_DURATION_SAMPLES = int(MIN_EVENT_DURATION_SEC * SAMPLING_RATE_HZ)  # = 150

# ─── Inference ────────────────────────────────────────────────────────────────
INFERENCE_BATCH_SIZE = 32         # Batch size for prediction
INFERENCE_STRIDE = None           # Overlap stride (None = CHUNK_SIZE // 2)
