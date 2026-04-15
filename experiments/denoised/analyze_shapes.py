"""Quick analysis: measure shape feature separation between blood/clot/wall on denoised data."""
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

files = sorted(glob.glob("experiments/denoised/train_data/*.parquet"))[:8]
all_r, all_lbl = [], []
for f in files:
    df = pd.read_parquet(f)
    all_r.append(df['magRLoadAdjusted'].values.astype(np.float32))
    all_lbl.append(df['label'].values.astype(int))
r = np.concatenate(all_r)
lbl = np.concatenate(all_lbl)

WIN = 750  # 5s at 150Hz
STRIDE = 150  # 1s stride
shape_data = {0: [], 1: [], 2: []}

for i in range(WIN, len(r) - 1, STRIDE):
    w = r[i - WIN:i]
    l = lbl[i - WIN:i]
    if (l < 0).any():
        continue
    majority = np.bincount(l, minlength=3).argmax()

    d = np.diff(w)
    dd = np.diff(d)
    x = np.arange(WIN, dtype=np.float64)

    # 1. Linearity (R^2 of linear fit)
    p = np.polyfit(x, w, 1)
    resid = w - np.polyval(p, x)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((w - w.mean()) ** 2) + 1e-10
    r2 = max(0, 1 - ss_res / ss_tot)

    # 2. Waveform length (mean |diff|)
    wf_len = np.mean(np.abs(d))

    # 3. Zero crossing rate (detrended)
    trend = np.convolve(w, np.ones(150) / 150, 'same')
    detr = w[75:-75] - trend[75:-75]
    zc = np.sum(np.diff(np.sign(detr)) != 0) / max(len(detr), 1)

    # 4. Peaks per second
    w_std = np.std(w)
    if w_std > 0.01:
        peaks, _ = find_peaks(w, prominence=w_std * 0.3)
        n_peaks = len(peaks) / (WIN / 150)
    else:
        n_peaks = 0

    # 5. Longest monotonic run fraction
    signs = np.sign(d)
    signs[signs == 0] = 1
    changes = np.where(np.diff(signs) != 0)[0]
    runs = np.diff(np.concatenate([[0], changes, [len(signs)]]))
    longest_mono = np.max(runs) / len(d)

    # 6. Rise fraction
    rise_frac = np.sum(d > 0) / len(d)

    # 7. Level change (first quarter vs last quarter, normalized)
    first_q = np.mean(w[:WIN // 4])
    last_q = np.mean(w[-WIN // 4:])
    level_change = (last_q - first_q) / (w_std + 1e-6)

    # 8. Mean |curvature|
    curv = np.abs(dd) / (1 + d[:-1] ** 2) ** 1.5
    curv_mean = np.mean(curv)

    # 9. Crest factor (peak / RMS)
    rms = np.sqrt(np.mean(w ** 2)) + 1e-8
    crest = np.max(np.abs(w)) / rms

    # 10. Steps per second (large jumps > 3*sigma)
    threshold = 3 * np.std(d) + 1e-6
    n_steps = np.sum(np.abs(d) > threshold) / (WIN / 150)

    # 11. Quadratic residual (captures curvature vs linearity)
    p2 = np.polyfit(x, w, 2)
    resid2 = w - np.polyval(p2, x)
    ss_res2 = np.sum(resid2 ** 2)
    r2_quad = max(0, 1 - ss_res2 / (ss_tot))
    quad_improvement = r2_quad - r2  # how much better quadratic fits vs linear

    # 12. Segment complexity: number of direction changes per second
    dir_changes = np.sum(np.diff(signs) != 0) / (WIN / 150)

    # 13. Mean segment slope magnitude (slope of sub-segments between turns)
    if len(runs) > 1:
        slope_mags = []
        pos = 0
        for rl in runs[:50]:  # cap for speed
            seg = w[pos:pos + rl + 1]
            if len(seg) >= 2:
                slope_mags.append(abs(seg[-1] - seg[0]) / len(seg))
            pos += rl
        mean_seg_slope = np.mean(slope_mags) if slope_mags else 0
    else:
        mean_seg_slope = 0

    shape_data[majority].append([
        r2, wf_len, zc, n_peaks, longest_mono,
        rise_frac, level_change, curv_mean, crest, n_steps,
        quad_improvement, dir_changes, mean_seg_slope
    ])

print(f"Windows: blood={len(shape_data[0])}, clot={len(shape_data[1])}, wall={len(shape_data[2])}")

names = [
    'R2_linear', 'waveform_len', 'zero_cross', 'peaks_per_s', 'longest_mono',
    'rise_frac', 'level_change', 'curvature', 'crest_factor', 'steps_per_s',
    'quad_improv', 'dir_chg_per_s', 'mean_seg_slope'
]


def sep(a, b):
    """Cohen's d separation."""
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2) + 1e-8
    return abs(np.mean(a) - np.mean(b)) / pooled


header = f"{'Feature':<16} {'Blood':>10} {'Clot':>10} {'Wall':>10}  {'B-C':>6} {'B-W':>6} {'C-W':>6}  {'max':>6}"
print(f"\n{header}")
print("-" * 85)

for j, name in enumerate(names):
    b = np.array([x[j] for x in shape_data[0]])
    c = np.array([x[j] for x in shape_data[1]])
    w_ = np.array([x[j] for x in shape_data[2]])
    bm, cm, wm = np.median(b), np.median(c), np.median(w_)
    bc = sep(b, c)
    bw = sep(b, w_)
    cw = sep(c, w_)
    mx = max(bc, bw, cw)
    marker = " **" if cw > 0.3 else (" *" if mx > 0.3 else "")
    print(f"{name:<16} {bm:10.4f} {cm:10.4f} {wm:10.4f}  {bc:6.3f} {bw:6.3f} {cw:6.3f}  {mx:6.3f}{marker}")

print("\n** = C-W separation > 0.3 (most clinically important)")
print("*  = any separation > 0.3")
