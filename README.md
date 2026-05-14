# PyTorch_3 — Real-Time Clot/Wall Detection from Single-Frequency Impedance

## Overview

Machine learning pipeline for real-time classification of intravascular catheter impedance signals into three classes: **blood**, **clot**, and **wall** contact. Designed for deployment on the Inquis Medical LiveClot system, using a lightweight GRU network operating on engineered time-domain features extracted from a sliding window.

**Target:** ≥90% clot/wall detection accuracy with minimal false positives that would override device-assisted (DA) decisions.

## Architecture

| Component | Detail |
|-----------|--------|
| Model | GRU(32 hidden) → FC(24) → Softmax(3) |
| Input | 21 features × 8 time steps (sequence) |
| Feature window | 5.0 seconds at 150 Hz (750 samples) |
| Stride | 30 samples (~200 ms inference interval) |
| Post-processing | Temperature scaling (T=1.5) + EMA smoothing + override thresholds |
| Training | Cross-entropy loss, seed 456, AdamW (lr=1e-4, wd=1e-4) |
| Feature set | `clot_wall_focused` — 21 features selected by Cohen's d (clot vs wall) > 0.15 |

## Project Structure

```
PyTorch_3/
├── src/
│   ├── models/
│   │   └── gru_torch_V6.py        # Model + feature extraction + LiveClotDetector
│   ├── training/
│   │   └── train_gru_V6.py        # Training loop with GroupKFold CV
│   └── data/
│       ├── fit_scaler_V6.py       # StandardScaler fitting (parallelized)
│       ├── LabelingWithDuration.py # Parquet labeling (blood/clot/wall)
│       └── Labeling_5Names_V6.py  # 5-class labeling for graphics
├── training_data/                  # Labeled .parquet files (gitignored)
├── test_data/                      # Fixed hold-out test set (8 studies)
├── inference_deploy/Results/       # Inference output (global_summary.txt)
├── versions/                       # Versioned snapshots (manifest.txt per version)
├── experiments/                    # Exploratory work (denoised, scattering, etc.)
├── cache/                          # Precomputed feature arrays (.npz)
└── .gitignore
```

## Pipeline (End-to-End)

### 1. Labeling

```bash
python src/data/LabelingWithDuration.py
```
- Reads raw `.parquet` files from source directories
- Applies duration filter: only sustained events (≥5s) are labeled as clot/wall for training
- Outputs labeled parquets to `training_data/` and `test_data/`

### 2. Fit Scaler

```bash
python src/data/fit_scaler_V6.py
```
- Extracts features from all training parquets (parallelized via joblib)
- Fits a `StandardScaler` on the active feature subset
- Saves scaler as `.pkl` to `src/data/`

### 3. Train Model

```bash
python src/training/train_gru_V6.py
```
- Loads cached features or extracts from parquets
- 5-fold GroupKFold cross-validation (grouped by study)
- Saves best model (by val F1) as `.pt` to `src/training/`
- Outputs training curves and confusion matrices

### 4. Inference / Evaluation

```bash
python src/models/gru_torch_V6.py
```
- Runs `LiveClotDetector` on test set parquets
- Applies temperature scaling → EMA smoothing → override thresholds
- Computes net benefit metric, saves per-study results
- Outputs `global_summary.txt` with aggregate metrics

## Key Design Decisions

- **Feature set is frozen.** The 21-feature `clot_wall_focused` set was selected via Cohen's d analysis. All attempts to add features (texture, shape, spectral) degraded performance.
- **Single seed, single model.** Ensemble averaging (5 seeds) and alternative seeds were tested — all hurt due to triple-softening (ensemble + temperature + EMA).
- **Cross-entropy only.** Focal loss (γ=2) makes the model less decisive and hurts override precision.
- **EMA is asymmetric.** Different blending rates for entering vs. exiting tissue contact, tuned empirically.
- **DA label integration.** Device-assisted labels can be injected as high-confidence priors into the EMA posterior.

## Environment

```
Python:     anaconda3/envs/torch_env (or .venv)
PyTorch:    2.x (CPU or CUDA)
Key deps:   numpy, pandas, scipy, scikit-learn, joblib, matplotlib
OS:         Windows 10/11
```

Set before running:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## Feature Groups (65 total, 21 active)

| Index | Group | Description |
|-------|-------|-------------|
| f0–f9 | Basic stats | mean, std, var, min, max, range, median, windowed |
| f10–f15 | Slopes | Linear regression over 1–6 second sub-windows |
| f16–f21 | Derivative stats | mean, std, var, mean_abs, skew, kurtosis of 1st derivative |
| f22–f27 | EMA | fast/slow EMA, diff, zero crossings, ratio |
| f28–f35 | Detrended | Statistics after linear detrend |
| f36–f39 | Percentiles | p90-mean, IQR, p95-p5, fraction above p95 |
| f40–f42 | Hjorth + 2nd deriv | Mobility, complexity, mean abs 2nd derivative |
| f43–f45 | Pulse | Cardiac pulse amplitude, ratio, rate |
| f46–f50 | Stability | CoV, plateau fraction, settling time, stationarity, R level |
| f51–f56 | Short slopes | Abs linear reg over 0.1–0.6s (fast dynamics) |
| f57–f63 | Rise shape | Amplitude-normalized morphology features |
| f64 | Texture RMS | Bandpass 5–50 Hz RMS (surface roughness) |

**Active set** (`clot_wall_focused`): f39, f21, f4, f19, f41, f9, f5, f23, f0, f34, f28, f29, f3, f38, f17, f32, f42, f27, f1, f20, f40

## Evaluation Metric

**Net Benefit** = Σ per-study (correct_clot × w_clot + correct_wall × w_wall − false_override × penalty)

The model must demonstrate positive net benefit over DA-only baseline to justify deployment. Current baseline: **+63,489**.

## Git Conventions

- Branch: `master` (single branch)
- Remote: `https://github.com/RonInQu/PyTorch_3`
- Large files (`.pt`, `.pkl`, `.parquet`, `.npz`) are gitignored
- Versioned results tracked via `manifest.txt` and `global_summary.txt`
- Commit style: descriptive single-line summaries

## Known Limitations

- Model is sensitive to hyperparameter changes (fragile in development, stable in deployment)
- ~85 training studies — performance improvements require more data
- Single-frequency impedance only (no multi-frequency or phase information)
- Duration filter means short transient events (<5s) are not classified by ML (handled by DA)
