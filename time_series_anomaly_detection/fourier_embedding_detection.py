"""
Fourier-embedding anomaly detection: adaptive window size + a shape
embedding, instead of a hand-picked fixed window length.

Two limitations of `shape_discord_detection.py` motivated this module (see
the "cons" discussion in README.md):
  1. It needs a fixed subsequence length `m` chosen by hand (we used
     m=100 for every one of the 250 UCR files, which structurally cannot
     fit every domain).
  2. Its output is a bare distance number - no picture of how "normal" and
     "anomalous" windows are actually distributed relative to each other.

This module fixes both by:
  1. **Auto-sizing the window from the series itself**: take the FFT of
     the (detrended) series and use the period of its dominant frequency
     component as the window length. A series' own strongest periodicity
     is a principled, data-driven window size - no manual tuning, and it
     adapts per-series instead of using one constant for every domain.
  2. **Embedding each window as its own Fourier magnitude spectrum**
     (z-normalized first, so embeddings capture shape, not scale) - a
     fixed-length numeric vector per window, regardless of the signal's
     domain. Two windows with the same recurring shape land near each
     other in this embedding space; a structurally different window lands
     far from the rest, visibly and quantitatively.

Every step here is implemented directly on top of numpy's FFT and SVD (no
scikit-learn, no embedding model) - PCA is a few lines of SVD, consistent
with the rest of this repo's "from scratch" approach.
"""

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. Adaptive window size: period of the series' dominant Fourier frequency
# ---------------------------------------------------------------------------
def estimate_period_via_fourier(series, min_period: int = 8, max_period_frac: float = 0.25) -> int:
    """
    Returns a data-driven window size: the period (in samples) of the
    strongest non-trivial frequency component in `series`, clipped to
    [min_period, len(series) * max_period_frac] so it can't collapse to
    the whole series (from a near-DC trend) or to a handful of points
    (from high-frequency noise).
    """
    values = np.asarray(series, dtype=float)
    values = values - values.mean()
    n = len(values)

    spectrum = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(n)  # cycles per sample

    max_period = max(min_period + 1, int(n * max_period_frac))
    f_lo, f_hi = 1.0 / max_period, 1.0 / min_period
    valid = (freqs >= f_lo) & (freqs <= f_hi)
    valid[0] = False  # exclude the DC bin explicitly

    if not valid.any():
        return min_period

    valid_idx = np.where(valid)[0]
    peak_idx = valid_idx[np.argmax(spectrum[valid_idx])]
    period = int(round(1.0 / freqs[peak_idx]))
    return int(np.clip(period, min_period, max_period))


# ---------------------------------------------------------------------------
# 2. Fourier embedding of every window, fully vectorized
# ---------------------------------------------------------------------------
def embed_all_windows(series, m: int, n_coeffs: int = 12):
    """
    Returns (embeddings, flat_mask): embeddings[i] is the Fourier-magnitude
    embedding of series[i:i+m] (z-normalized and Hann-tapered first), for
    every i in one batched FFT call. flat_mask[i] marks windows with ~zero
    variance (undefined shape once z-normalized) so callers can exclude
    them from scoring.
    """
    values = np.asarray(series, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(values, m)  # (n-m+1, m)

    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    flat_mask = (std[:, 0] < 1e-9)
    std_safe = np.where(std > 1e-9, std, 1.0)

    znorm = (windows - mean) / std_safe
    tapered = znorm * np.hanning(m)
    spectrum = np.abs(np.fft.rfft(tapered, axis=1))

    n_coeffs = min(n_coeffs, spectrum.shape[1] - 1)
    embeddings = spectrum[:, 1:n_coeffs + 1]  # drop the DC bin (z-norm already removed the mean)
    return embeddings, flat_mask


# ---------------------------------------------------------------------------
# 3. From-scratch PCA (SVD) for the 2D "diff embedding" visualization
# ---------------------------------------------------------------------------
def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def plot_embedding_diff(positions, embeddings_2d, m, anomaly_start, anomaly_end, save_path, title=""):
    import matplotlib.pyplot as plt

    positions = np.asarray(positions)
    is_anomaly = (positions + m - 1 >= anomaly_start) & (positions <= anomaly_end)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(embeddings_2d[~is_anomaly, 0], embeddings_2d[~is_anomaly, 1],
               s=6, alpha=0.35, color="steelblue", label="normal windows")
    ax.scatter(embeddings_2d[is_anomaly, 0], embeddings_2d[is_anomaly, 1],
               s=35, color="crimson", label="windows overlapping true anomaly")
    ax.set_title(f"Fourier-embedding PCA (window size m={m}, auto-selected){title}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. A detector built on the embedding: distance from each window's
#    embedding to the "normal" reference distribution (fit on the
#    train-only region, never on test-region/anomaly data - no leakage).
# ---------------------------------------------------------------------------
def embedding_distance_detector(series, m: int, train_size: int, n_coeffs: int = 12):
    """
    Returns (positions, scores): for every window position, its
    standardized Euclidean distance to the mean Fourier embedding of the
    TRAIN region only (archive convention: train region is anomaly-free).
    Large distance = shape unlike anything seen in the reference/normal
    period.
    """
    embeddings, flat_mask = embed_all_windows(series, m, n_coeffs)
    positions = np.arange(len(embeddings))

    train_end = max(1, train_size - m + 1)
    train_mask = (positions < train_end) & (~flat_mask)
    if train_mask.sum() < 5:
        train_mask = ~flat_mask  # fallback for tiny train regions

    ref_mean = embeddings[train_mask].mean(axis=0)
    ref_std = embeddings[train_mask].std(axis=0)
    ref_std_safe = np.where(ref_std > 1e-9, ref_std, 1.0)

    z = (embeddings - ref_mean) / ref_std_safe
    scores = np.sqrt((z ** 2).sum(axis=1))
    scores[flat_mask] = 0.0
    return positions, scores


# ---------------------------------------------------------------------------
# 5. Run on 3 UCR files and report: window size chosen, top-1 hit, plot.
# ---------------------------------------------------------------------------
FILES = [
    # (filename, train_size, anomaly_start, anomaly_end) - label, description
    ("001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt", 35000, 52000, 52620,
     "ECG - every point-based detector missed this one; shape discord hit it"),
    ("004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt", 2500, 5400, 5600,
     "ECG-like - every detector so far hit this one"),
    ("008_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature4_4000_5549_5597.txt", 4000, 5549, 5597,
     "Air temperature - shape discord MISSED this one, trend-residual hit it"),
]


def run_on_three_datasets(data_dir: str):
    results = []
    for filename, train_size, anomaly_start, anomaly_end, description in FILES:
        path = os.path.join(data_dir, filename)
        series = pd.Series(np.loadtxt(path))
        n = len(series)

        m = estimate_period_via_fourier(series)
        positions, scores = embedding_distance_detector(series, m, train_size)

        test_mask = positions >= train_size
        top1_pos = int(positions[test_mask][np.argmax(scores[test_mask])])
        hit = anomaly_start <= top1_pos + m - 1 and top1_pos <= anomaly_end

        print(f"\n{filename}")
        print(f"  {description}")
        print(f"  length={n}, auto window size m={m} (vs. fixed m=100 used previously)")
        print(f"  true anomaly=[{anomaly_start},{anomaly_end}], top-1 guess at position {top1_pos} "
              f"(covers [{top1_pos},{top1_pos + m - 1}]) -> {'HIT' if hit else 'MISS'}")

        embeddings, flat_mask = embed_all_windows(series, m)
        embeddings_2d = pca_2d(embeddings)
        plot_path = os.path.join(_HERE, f"fourier_embedding_{filename.split('_UCR_')[0]}.png")
        plot_embedding_diff(positions, embeddings_2d, m, anomaly_start, anomaly_end, plot_path,
                             title=f"\n{filename}")
        print(f"  saved embedding plot to {plot_path}")

        results.append({
            "file": filename, "length": n, "window_size_m": m,
            "anomaly_start": anomaly_start, "anomaly_end": anomaly_end,
            "top1_position": top1_pos, "hit": hit,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "ucr_data")
    df = run_on_three_datasets(data_dir)
    print("\n=== Summary (3 datasets) ===")
    print(df.to_string(index=False))
