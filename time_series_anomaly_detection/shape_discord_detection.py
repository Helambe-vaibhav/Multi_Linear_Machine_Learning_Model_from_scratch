"""
Shape-based ("discord") anomaly detection via point extension.

`anomaly_detection.py` compares single VALUES against a local rolling
mean/std. That works for spikes and level shifts, but the real-world
validation in `evaluate_ucr.py` shows it MISSES most of the UCR Anomaly
Archive (10.8% top-1 hit rate at best) - because many of that archive's
anomalies are not value spikes at all, they're a subtle change in local
*shape* (e.g. one distorted heartbeat that has a totally ordinary mean/std).

The fix is "point extension": instead of asking "is this one value far
from its neighbours?", extend every point into a subsequence (a window of
length `m` starting at that point) and ask "does this subsequence's SHAPE
have a good match anywhere else in the series?". A subsequence with no good
match anywhere else - i.e. an unusually large nearest-neighbor distance -
is a discord (Keogh et al.'s term). This is the actual reference technique
(Matrix Profile / MASS) the UCR Anomaly Archive itself was validated
against, so it should recover exactly the shape anomalies the point-value
detectors miss.

Core primitive: MASS (Mueen's Algorithm for Similarity Search) computes the
z-normalized Euclidean distance from one query subsequence to EVERY
subsequence of a series in O(n log n) via FFT convolution (here via
scipy.signal.fftconvolve for the sliding dot product), instead of the
O(n*m) cost of comparing the query against each window one at a time.
"""

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. MASS: z-normalized Euclidean distance profile of one query subsequence
#    against every subsequence of a series.
# ---------------------------------------------------------------------------
def _window_start_aligned_stats(series: np.ndarray, m: int):
    """
    Rolling mean/std of every length-m window, aligned to the window's
    START index (position i covers series[i:i+m]) rather than pandas'
    default end-aligned convention.
    """
    s = pd.Series(series)
    end_aligned_mean = s.rolling(m).mean().to_numpy()
    end_aligned_std = s.rolling(m).std(ddof=0).to_numpy()
    # value currently at index i+m-1 describes window [i, i+m) -> move it to i
    mean = end_aligned_mean[m - 1:]
    std = end_aligned_std[m - 1:]
    return mean, std


def mass_distance_profile(query: np.ndarray, series: np.ndarray) -> np.ndarray:
    """
    Returns the z-normalized Euclidean distance from `query` (length m) to
    every length-m window of `series` (length n): an array of length
    n - m + 1, where entry i is the distance to series[i:i+m].
    """
    m = len(query)
    n = len(series)
    query_mean, query_std = np.mean(query), np.std(query)
    query_std = query_std if query_std > 1e-9 else 1e-9

    window_mean, window_std = _window_start_aligned_stats(series, m)
    window_std = np.where(window_std > 1e-9, window_std, 1e-9)

    # Sliding dot product of query against every window of series, via FFT
    # convolution (scipy) instead of an O(n*m) direct loop: convolving with
    # a reversed kernel in 'valid' mode gives exactly the aligned dot
    # products, dot[i] = sum(query * series[i:i+m]).
    dot = fftconvolve(series, query[::-1], mode="valid")

    # z-normalized squared Euclidean distance, expanded in terms of the raw
    # dot product (avoids actually z-normalizing and re-convolving):
    #   D^2 = 2m * (1 - (dot - m*mu_q*mu_w) / (m*sigma_q*sigma_w))
    dist_sq = 2 * m * (1 - (dot - m * query_mean * window_mean) / (m * query_std * window_std))
    return np.sqrt(np.clip(dist_sq, 0, None))


# ---------------------------------------------------------------------------
# 2. Discord scoring: nearest-neighbor distance per (sampled) query position,
#    with a trivial-match exclusion zone around the query itself.
# ---------------------------------------------------------------------------
@dataclass
class ShapeDiscord:
    position: int
    discord_score: float
    neighbor_position: int
    explanation: str


def compute_discord_scores(
    series: np.ndarray,
    m: int,
    query_positions,
    exclusion_factor: float = 0.5,
) -> tuple:
    """
    For each position in `query_positions`, extends it into a length-m
    subsequence and finds its nearest-neighbor distance elsewhere in the
    series (excluding a trivial-match zone of +/- m*exclusion_factor around
    itself). Returns (scores, neighbor_positions) as dicts keyed by position.
    """
    n = len(series)
    exclusion = max(1, int(m * exclusion_factor))
    scores, neighbors = {}, {}

    for i in query_positions:
        if i < 0 or i + m > n:
            continue
        query = series[i:i + m]
        if np.std(query) < 1e-9:
            continue  # flat/constant subsequence - not a meaningful shape query
        profile = mass_distance_profile(query, series)

        lo, hi = max(0, i - exclusion), min(len(profile), i + exclusion + 1)
        profile[lo:hi] = np.inf

        neighbor_pos = int(np.argmin(profile))
        scores[i] = float(profile[neighbor_pos])
        neighbors[i] = neighbor_pos

    return scores, neighbors


def shape_discord_detector(
    series: pd.Series,
    m: int = 50,
    max_queries: int = 3000,
    exclusion_factor: float = 0.5,
    candidate_positions=None,
) -> list:
    """
    Ranks positions in `series` by how poorly their extended subsequence
    (length m) matches anywhere else in the series. `candidate_positions`
    restricts the search (e.g. to a "test region"); if the number of valid
    positions exceeds `max_queries`, they're subsampled evenly so runtime
    stays bounded on long real-world series - each MASS call is O(n log n),
    so this keeps total cost to roughly O(max_queries * n log n) regardless
    of how long the series is.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    valid_range = range(0, n - m + 1) if candidate_positions is None else candidate_positions
    valid_range = [i for i in valid_range if 0 <= i <= n - m]

    if len(valid_range) > max_queries:
        idx = np.linspace(0, len(valid_range) - 1, max_queries).astype(int)
        query_positions = [valid_range[i] for i in idx]
    else:
        query_positions = valid_range

    scores, neighbors = compute_discord_scores(values, m, query_positions, exclusion_factor)

    results = []
    for pos, score in scores.items():
        neighbor_pos = neighbors[pos]
        results.append(ShapeDiscord(
            position=pos, discord_score=score, neighbor_position=neighbor_pos,
            explanation=(
                f"The subsequence starting at position {pos} (length {m}) has no good shape match "
                f"anywhere else in the series - its closest look-alike is at position {neighbor_pos}, "
                f"but even that match is z-normalized distance {score:.2f} away. A normal, recurring "
                f"pattern would have a much closer match; this one's *shape* is structurally unusual, "
                f"even if its raw value/variance look ordinary."
            ),
        ))
    results.sort(key=lambda r: r.discord_score, reverse=True)
    return results


def scores_as_series(results: list, index) -> pd.Series:
    """Sparse pd.Series of discord scores (NaN where not evaluated), useful
    for reusing the same top-1 evaluation protocol as evaluate_ucr.py."""
    out = pd.Series(np.nan, index=index)
    for r in results:
        out.iloc[r.position] = r.discord_score
    return out


# ---------------------------------------------------------------------------
# 3. "Diff patterns": plot the anomalous subsequence against its nearest
#    normal-looking match, z-normalized so shape (not scale) is compared.
# ---------------------------------------------------------------------------
def plot_pattern_comparison(series: pd.Series, discord: ShapeDiscord, m: int, save_path: str) -> None:
    import matplotlib.pyplot as plt

    values = series.to_numpy(dtype=float)
    anomalous = values[discord.position:discord.position + m]
    normal = values[discord.neighbor_position:discord.neighbor_position + m]

    def znorm(x):
        std = np.std(x)
        return (x - np.mean(x)) / (std if std > 1e-9 else 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(anomalous, color="crimson", label=f"anomalous pattern (pos {discord.position})")
    axes[0].plot(normal, color="steelblue", alpha=0.7, label=f"nearest match (pos {discord.neighbor_position})")
    axes[0].set_title("Raw values")
    axes[0].legend(fontsize=8)

    axes[1].plot(znorm(anomalous), color="crimson", label="anomalous (z-normalized)")
    axes[1].plot(znorm(normal), color="steelblue", alpha=0.7, label="nearest match (z-normalized)")
    axes[1].set_title(f"Z-normalized SHAPE comparison (distance={discord.discord_score:.2f})")
    axes[1].legend(fontsize=8)

    fig.suptitle("Diff pattern: normal vs. anomalous subsequence shape")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Demo: a SHAPE-only anomaly that the point-based detectors in
#    anomaly_detection.py cannot see (same mean/std as the rest of the
#    series - only the local waveform pattern differs).
# ---------------------------------------------------------------------------
def generate_shape_anomaly_demo(n_points: int = 2000, period: int = 40, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    series = np.sin(2 * np.pi * t / period) + rng.normal(0, 0.05, n_points)

    # Distort one cycle's SHAPE (double frequency for one period) without
    # changing its local mean/std noticeably - a pure discord, no spike.
    start = 1000
    distorted = np.sin(2 * np.pi * t[start:start + period] / (period / 2))
    series[start:start + period] = distorted + rng.normal(0, 0.05, period)

    return pd.Series(series), start


if __name__ == "__main__":
    from anomaly_detection import rolling_zscore_detector

    print("=== Demo: shape-only anomaly (point detectors vs. shape discord) ===")
    series, true_start = generate_shape_anomaly_demo()

    point_result = rolling_zscore_detector(series, window=40, threshold=3.0)
    print(f"Rolling Z-Score flagged {len(point_result.anomalies)} point(s) "
          f"(true shape anomaly starts at {true_start}) - expected to miss it, "
          "since mean/std at the distorted cycle look ordinary.")

    m = 40
    results = shape_discord_detector(series, m=m, max_queries=2000)
    top = results[0]
    print(f"\nTop shape discord: position={top.position} (true anomaly at {true_start}), "
          f"score={top.discord_score:.2f}")
    print(top.explanation)

    out_path = os.path.join(_HERE, "shape_discord_demo_pattern.png")
    plot_pattern_comparison(series, top, m, out_path)
    print(f"\nSaved pattern comparison plot to {out_path}")
