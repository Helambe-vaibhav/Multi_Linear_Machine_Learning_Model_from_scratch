"""
Time Series Anomaly Detection - from scratch (no sklearn/statsmodels required).

Implements three classic detectors on top of plain numpy/pandas:
  1. Rolling Z-Score        - flags points far from a local rolling mean/std.
  2. Rolling IQR            - flags points outside a local Tukey fence.
  3. Trend-Residual Z-Score - removes the trend (via a rolling mean "seasonal/
                               trend" baseline) first, then flags large
                               residuals. Better for series with trend/season.

Each detector doesn't just say "this point is an anomaly" - it also produces
a plain-English *explanation* of why the point looks anomalous, classifying
it as one of:
  - "spike"        : one-off point shooting far above/below its neighbours.
  - "level_shift"   : the local baseline itself moved (mean before/after the
                      point differs a lot) -> a regime change, not a blip.
  - "volatility_change" : the local variance changed sharply -> the series
                      became noisier/calmer, not necessarily off-level.

Run this file directly for a demo on a synthetic series with injected
anomalies (see `generate_synthetic_series`).
"""

import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Synthetic data with known, injected anomalies (for demos/testing)
# ---------------------------------------------------------------------------
def generate_synthetic_series(n_points: int = 365, seed: int = 42) -> pd.Series:
    """Trend + weekly seasonality + noise, with 3 kinds of injected anomalies."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)

    trend = 0.05 * t
    seasonality = 5 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 1, n_points)
    values = 20 + trend + seasonality + noise

    # Point anomaly (spike): one-off extreme value
    values[100] += 25

    # Contextual anomaly (dip): value is normal globally, unusual for its context
    values[200] -= 15

    # Level shift: the baseline jumps and stays there
    values[260:] += 12

    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    return pd.Series(values, index=dates, name="value")


# ---------------------------------------------------------------------------
# 2. Detector output
# ---------------------------------------------------------------------------
@dataclass
class Anomaly:
    index: object          # timestamp / label of the anomalous point
    value: float
    score: float            # magnitude of deviation (higher = more anomalous)
    kind: str                # "spike" | "level_shift" | "volatility_change"
    explanation: str


@dataclass
class DetectionResult:
    scores: pd.Series
    is_anomaly: pd.Series
    anomalies: list = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "timestamp": a.index,
            "value": round(a.value, 3),
            "score": round(a.score, 3),
            "kind": a.kind,
            "explanation": a.explanation,
        } for a in self.anomalies])


# ---------------------------------------------------------------------------
# 3. Shared helper: classify *why* a flagged point is anomalous
# ---------------------------------------------------------------------------
def _classify_reason(series: pd.Series, idx_pos: int, window: int) -> tuple[str, str]:
    """Compare the local window before/after idx_pos to decide the anomaly type."""
    n = len(series)
    before_lo, before_hi = max(0, idx_pos - window), idx_pos
    after_lo, after_hi = idx_pos + 1, min(n, idx_pos + 1 + window)

    before = series.iloc[before_lo:before_hi]
    after = series.iloc[after_lo:after_hi]
    value = series.iloc[idx_pos]

    mean_before = before.mean() if len(before) else value
    mean_after = after.mean() if len(after) else value
    std_before = before.std(ddof=0) if len(before) > 1 else 0.0
    std_after = after.std(ddof=0) if len(after) > 1 else 0.0

    baseline_shift = abs(mean_after - mean_before)
    local_std = max(std_before, 1e-9)
    is_persistent_shift = len(after) >= max(3, window // 2) and baseline_shift > 2 * local_std

    vol_ratio = (std_after + 1e-9) / (std_before + 1e-9)
    is_volatility_change = vol_ratio > 2.5 or vol_ratio < 0.4

    if is_persistent_shift:
        direction = "up" if mean_after > mean_before else "down"
        return (
            "level_shift",
            f"The series baseline shifted {direction} around this point "
            f"(mean before={mean_before:.2f} -> mean after={mean_after:.2f}); "
            "this looks like a regime change / structural break rather than a one-off blip. "
            "Likely cause: a real change in the underlying process (e.g. new pricing, "
            "sensor recalibration, policy change) rather than noise."
        )

    if is_volatility_change:
        return (
            "volatility_change",
            f"Local variance changed sharply around this point (std before={std_before:.2f} "
            f"-> std after={std_after:.2f}). The level may look normal, but the series became "
            "noisier or calmer. Likely cause: instability in the data source (e.g. system load, "
            "intermittent faults) rather than a single bad reading."
        )

    direction = "above" if value > mean_before else "below"
    return (
        "spike",
        f"Value {value:.2f} is a one-off point far {direction} its local neighbourhood "
        f"(local mean={mean_before:.2f}, local std={local_std:.2f}), and the series returns "
        "to its previous baseline right after. Likely cause: a transient event "
        "(e.g. a sensor glitch, a one-time outlier order, a logging error) rather than "
        "a lasting change."
    )


# ---------------------------------------------------------------------------
# 4. Detector 1: Rolling Z-Score
# ---------------------------------------------------------------------------
def rolling_zscore_detector(
    series: pd.Series, window: int = 14, threshold: float = 3.0
) -> DetectionResult:
    """
    Flags a point if it's more than `threshold` standard deviations away from
    the mean of the `window` points *before* it (so detection stays causal /
    usable in real time, unlike a centred rolling window).
    """
    rolling_mean = series.rolling(window, min_periods=window // 2).mean().shift(1)
    rolling_std = series.rolling(window, min_periods=window // 2).std(ddof=0).shift(1)
    rolling_std = rolling_std.replace(0, np.nan)

    z = (series - rolling_mean) / rolling_std
    scores = z.abs().fillna(0)
    flagged = scores > threshold

    return _build_result(series, scores, flagged, window)


# ---------------------------------------------------------------------------
# 5. Detector 2: Rolling IQR (Tukey fence), robust to outliers skewing mean/std
# ---------------------------------------------------------------------------
def rolling_iqr_detector(
    series: pd.Series, window: int = 14, k: float = 1.5
) -> DetectionResult:
    """
    Vectorized via pandas' rolling quantile (no per-point Python loop), shifted
    by 1 to stay causal like the other detectors - same idea as
    `rolling_zscore_detector`, just with quartiles instead of mean/std.
    """
    min_periods = max(4, window // 2)
    q1 = series.rolling(window, min_periods=min_periods).quantile(0.25).shift(1)
    q3 = series.rolling(window, min_periods=min_periods).quantile(0.75).shift(1)
    iqr = q3 - q1
    lower = (q1 - k * iqr).fillna(-np.inf)
    upper = (q3 + k * iqr).fillna(np.inf)

    dist_below = (lower - series).clip(lower=0)
    dist_above = (series - upper).clip(lower=0)
    span = (upper - lower).replace([np.inf, -np.inf], np.nan)
    span = span.where(span > 0, np.nan)
    scores = ((dist_below + dist_above) / span).fillna(0)

    flagged = (series < lower) | (series > upper)
    return _build_result(series, scores, flagged, window)


# ---------------------------------------------------------------------------
# 6. Detector 3: Trend-Residual Z-Score (de-trend first, then z-score)
# ---------------------------------------------------------------------------
def trend_residual_detector(
    series: pd.Series, trend_window: int = 14, z_window: int = 14, threshold: float = 3.0
) -> DetectionResult:
    """
    Removes a slow-moving trend/seasonal baseline (centred rolling mean) and
    z-scores the *residual*. This avoids false positives on series that have
    a genuine trend or seasonality, which a plain rolling z-score can confuse
    with anomalies.
    """
    baseline = series.rolling(trend_window, center=True, min_periods=1).mean()
    residual = series - baseline
    residual_result = rolling_zscore_detector(residual, window=z_window, threshold=threshold)

    # Re-derive explanations against the ORIGINAL series (not the residual)
    # so the reported values/means make sense to a reader, while scores stay
    # based on the de-trended residual.
    anomalies = []
    for a in residual_result.anomalies:
        pos = series.index.get_loc(a.index)
        kind, explanation = _classify_reason(series, pos, z_window)
        anomalies.append(Anomaly(
            index=a.index, value=float(series.loc[a.index]), score=a.score,
            kind=kind, explanation=explanation,
        ))
    return DetectionResult(scores=residual_result.scores, is_anomaly=residual_result.is_anomaly, anomalies=anomalies)


def _build_result(series, scores, flagged, window) -> DetectionResult:
    anomalies = []
    for pos, (idx, is_flag) in enumerate(flagged.items()):
        if not is_flag:
            continue
        kind, explanation = _classify_reason(series, pos, window)
        anomalies.append(Anomaly(
            index=idx, value=float(series.iloc[pos]), score=float(scores.iloc[pos]),
            kind=kind, explanation=explanation,
        ))
    return DetectionResult(scores=scores, is_anomaly=flagged, anomalies=anomalies)


# ---------------------------------------------------------------------------
# 7. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    series = generate_synthetic_series()

    print("=== Rolling Z-Score detector ===")
    result_z = rolling_zscore_detector(series, window=14, threshold=3.0)
    print(result_z.summary().to_string(index=False))

    print("\n=== Rolling IQR detector ===")
    result_iqr = rolling_iqr_detector(series, window=14, k=1.5)
    print(result_iqr.summary().to_string(index=False))

    print("\n=== Trend-Residual Z-Score detector ===")
    result_tr = trend_residual_detector(series, trend_window=14, z_window=14, threshold=3.0)
    print(result_tr.summary().to_string(index=False))

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(series.index, series.values, label="series", color="steelblue")
        for res, color, marker, label in [
            (result_z, "red", "o", "z-score"),
            (result_iqr, "orange", "x", "IQR"),
            (result_tr, "green", "^", "trend-residual"),
        ]:
            if res.anomalies:
                xs = [a.index for a in res.anomalies]
                ys = [a.value for a in res.anomalies]
                ax.scatter(xs, ys, color=color, marker=marker, s=80, label=f"{label} anomaly", zorder=5)
        ax.set_title("Time Series Anomaly Detection - Demo")
        ax.set_xlabel("date")
        ax.set_ylabel("value")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output.png")
        fig.savefig(out_path, dpi=120)
        print(f"\nSaved plot to {out_path}")
    except ImportError:
        print("\nmatplotlib not installed - skipping plot.")
