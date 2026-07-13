# Time Series Anomaly Detection (from scratch) + Explanations

This project detects anomalies in time series data using simple statistical
methods implemented from scratch (numpy/pandas only), and — just as
importantly — **explains why each flagged point looks anomalous**, instead of
only returning a yes/no flag.

It follows the same "from scratch" spirit as the rest of this repository:
no black-box anomaly-detection libraries, just the underlying math made
explicit and readable.

## Files

| File | Purpose |
|---|---|
| `anomaly_detection.py` | Detectors + explanation logic + a runnable demo |
| `demo_output.png` | Example plot produced by running the demo |

Run the demo:

```bash
python3 time_series_anomaly_detection/anomaly_detection.py
```

## What is a time series anomaly?

A **time series** is a sequence of values ordered in time (e.g. daily sales,
server CPU load, sensor readings). An **anomaly** (or outlier) is a data
point, or short sequence of points, that deviates from the pattern the rest
of the series establishes — trend, seasonality, and typical noise level.

Anomalies are usually grouped into three types:

1. **Point anomaly (spike)** — a single value that is far outside the normal
   range for a single instant, then the series returns to normal.
   *Example: one sensor reading suddenly jumps because of a glitch.*

2. **Contextual anomaly** — a value that isn't extreme in the *global*
   sense, but is unusual *given its context* (e.g. time of day, day of week,
   season). *Example: 2 AM traffic volume equal to rush-hour volume.*

3. **Collective anomaly / level shift** — a sequence of points, or a change
   in the baseline itself, that together looks abnormal even though no
   individual point is extreme. *Example: the series' mean permanently
   jumps to a new level after an event.*

## Why do anomalies occur? (root causes)

Anomalies are symptoms — the interesting question is *what produced them*.
Common real-world causes:

- **Data quality issues**: sensor glitches, missing/duplicated readings,
  logging bugs, unit-conversion errors, clock skew.
- **One-off external events**: a flash sale, a viral post, a power outage,
  a holiday — genuine but rare occurrences, not errors.
- **Regime changes / structural breaks**: a pricing change, a new feature
  launch, a policy change, a system upgrade — the underlying process itself
  changed, so the "new normal" is different from the old one.
- **Increasing/decreasing volatility**: the process becomes less stable
  (e.g. system under increasing load, market uncertainty) even if the
  average level doesn't move much.
- **Fraud / malicious activity**: unusual transaction patterns, intrusion
  attempts — anomalies that are the actual signal of interest.
- **Seasonality mismatch**: a value that's normal for one season/context but
  anomalous in another (a contextual anomaly).

Because the *cause* differs so much by anomaly *shape*, this project doesn't
stop at "is this point anomalous?" — it also classifies *how* the point
deviates (spike vs. level shift vs. volatility change), which is a strong
hint toward the likely cause. See "Explanations" below.

## Detection methods implemented

All three methods work on a rolling/local window so they adapt to trend and
scale, rather than assuming the whole series is stationary.

### 1. Rolling Z-Score (`rolling_zscore_detector`)
For each point, compute the mean and standard deviation of the previous
`window` points, then score how many standard deviations away the current
point is:

```
z = (x_t - rolling_mean) / rolling_std
```

Flag if `|z| > threshold` (default 3, i.e. ~99.7% confidence under a normal
assumption). Simple and fast; sensitive to values that are far from a
*locally stable* mean/std.

### 2. Rolling IQR / Tukey fence (`rolling_iqr_detector`)
For each point, compute Q1 and Q3 of the previous `window` points and flag
values outside `[Q1 - k*IQR, Q3 + k*IQR]` (default `k=1.5`, the classic
boxplot rule). More robust than z-score when the local window itself
already contains outliers, since the median/quartiles aren't dragged around
by extreme values the way mean/std are.

### 3. Trend-Residual Z-Score (`trend_residual_detector`)
First removes a slow-moving baseline (a centred rolling mean, which acts as
a simple trend/seasonality estimate), then applies the rolling z-score to
the **residual**. This avoids false positives on series that have a genuine
trend or seasonal cycle, which a plain z-score can mistake for anomalies.

## Explanations: classifying *why* a point is anomalous

For every flagged point, `_classify_reason()` compares the local window
**before** and **after** the point to decide which shape of anomaly it is:

- If the mean level *before* vs. *after* the point differs by more than
  ~2 local standard deviations and persists → **`level_shift`**
  ("the baseline moved — likely a real regime change").
- Else if the local variance before vs. after changes sharply (more than
  ~2.5x or less than ~0.4x) → **`volatility_change`**
  ("the series got noisier/calmer — likely instability in the source").
- Otherwise → **`spike`**
  ("a one-off point far from its neighbours that reverts right after —
  likely a transient event or bad reading").

Each `Anomaly` returned by a detector carries a human-readable
`explanation` string built from these comparisons (with the actual before/
after means, stds, and the point's value), e.g.:

```
Value 54.50 is a one-off point far above its local neighbourhood
(local mean=24.24, local std=3.50), and the series returns to its
previous baseline right after. Likely cause: a transient event
(e.g. a sensor glitch, a one-time outlier order, a logging error)
rather than a lasting change.
```

This turns a raw "anomaly at index 100" into something a human can act on.

## Example output

Running the demo on a synthetic series (trend + weekly seasonality + 3
injected anomalies: a spike, a dip, and a level shift) correctly recovers
all three and classifies them by type:

![demo output](demo_output.png)

## Usage on your own data

```python
import pandas as pd
from anomaly_detection import rolling_zscore_detector, trend_residual_detector

series = pd.read_csv("your_data.csv", index_col="date", parse_dates=True)["value"]

result = trend_residual_detector(series, trend_window=14, z_window=14, threshold=3.0)
print(result.summary())          # DataFrame: timestamp, value, score, kind, explanation
```

## Choosing a method

- Series is roughly flat, no trend/season → **Rolling Z-Score** is simplest.
- Local windows may already contain outliers, or data isn't normally
  distributed → **Rolling IQR** is more robust.
- Series has a clear trend or seasonal cycle (sales, traffic, weather) →
  **Trend-Residual Z-Score** avoids flagging normal seasonal peaks/troughs.

## Limitations

- These are univariate, unsupervised, threshold-based methods — good
  baselines, not a replacement for domain-tuned or ML-based detectors
  (e.g. Isolation Forest, Prophet, LSTM autoencoders) on complex, noisy,
  multi-seasonal data.
- The `level_shift` / `volatility_change` / `spike` classification is a
  heuristic based on before/after window statistics, not a causal
  inference — it narrows down *what kind* of deviation occurred, and the
  README's "why anomalies occur" list above should guide the human
  investigation into the actual root cause.
