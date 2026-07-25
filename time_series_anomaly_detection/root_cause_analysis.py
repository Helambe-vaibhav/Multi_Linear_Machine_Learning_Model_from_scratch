"""
Multivariate Root-Cause Analysis for Time Series Anomalies - from scratch.

`anomaly_detection.py` answers "is this point anomalous, and what shape of
deviation is it (spike / level shift / volatility change)?". This module
answers the next question an on-call engineer actually asks: "given several
related upstream metrics, which one most plausibly *caused* it?".

Three from-scratch signals are combined per candidate series, no black-box
causal-discovery library required:

  1. Lagged cross-correlation - does the candidate move together with the
     target, and at what lag? A candidate that *leads* the target by a
     positive lag is a much stronger causal hint than one that moves in
     lockstep (lag 0, could just be shared seasonality) or that lags behind
     the target (rules it out - effects don't precede their causes).

  2. Granger-causality F-test - built directly on ordinary least squares,
     the same linear-regression machinery the rest of this repo builds from
     scratch (see "machine learning multi linear from scratch.ipynb"): does
     adding the candidate's lagged values to a linear model of the target
     significantly reduce residual variance versus using the target's own
     lagged values alone? This is literally comparing two multi-linear
     regressions.

  3. Anomaly co-occurrence - did the candidate *also* show its own
     anomalous deviation (via the detectors in anomaly_detection.py)
     shortly before the target's anomaly?

These combine into a ranked list of root-cause candidates, each with a
plain-English explanation.

IMPORTANT CAVEAT: correlation and Granger-causality are evidence, not proof
of causation - confounders (a shared unobserved driver) can produce the same
signature. See README.md "Limitations".
"""

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from anomaly_detection import rolling_zscore_detector

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. Synthetic multivariate demo data: one true upstream cause + three decoys
# ---------------------------------------------------------------------------
def generate_multivariate_demo_data(n_points: int = 200, seed: int = 7):
    """
    Builds a target series driven by a lagged upstream 'cause' series, plus
    three decoys designed to fool naive (non-lagged) correlation checks:
      - seasonal_decoy: correlated with the target via shared seasonality,
        but not the trigger of the anomaly.
      - unrelated_spike_decoy: has its own anomaly, but at an unrelated time.
      - pure_noise_decoy: no relationship at all.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")

    cause = 10 + 2 * np.sin(2 * np.pi * t / 14) + rng.normal(0, 0.5, n_points)
    trigger_at = 120
    cause[trigger_at] += 8  # the actual triggering event

    lag = 3
    propagated = np.roll(cause, lag)
    propagated[:lag] = cause[0]
    target = 20 + 0.9 * (propagated - propagated.mean()) + rng.normal(0, 0.6, n_points)

    seasonal_decoy = 15 + 2.1 * np.sin(2 * np.pi * t / 14) + rng.normal(0, 0.5, n_points)

    unrelated_spike_decoy = 5 + rng.normal(0, 1, n_points)
    unrelated_spike_decoy[60] += 6

    pure_noise_decoy = rng.normal(0, 1, n_points)

    target_series = pd.Series(target, index=dates, name="target")
    candidates = {
        "upstream_cause": pd.Series(cause, index=dates, name="upstream_cause"),
        "seasonal_decoy": pd.Series(seasonal_decoy, index=dates, name="seasonal_decoy"),
        "unrelated_spike_decoy": pd.Series(unrelated_spike_decoy, index=dates, name="unrelated_spike_decoy"),
        "pure_noise_decoy": pd.Series(pure_noise_decoy, index=dates, name="pure_noise_decoy"),
    }
    return target_series, candidates


# ---------------------------------------------------------------------------
# 2. Lagged cross-correlation
# ---------------------------------------------------------------------------
def best_lagged_correlation(candidate: pd.Series, target: pd.Series, max_lag: int) -> tuple[int, float]:
    """
    Tests candidate[t - lag] against target[t] for lag in [0, max_lag] and
    returns the (lag, correlation) pair with the largest |correlation|.
    A positive lag means the candidate leads the target by that many steps.
    """
    best_lag, best_corr = 0, 0.0
    for lag in range(0, max_lag + 1):
        shifted = candidate.shift(lag) if lag else candidate
        aligned = pd.concat([shifted, target], axis=1).dropna()
        if len(aligned) < 5:
            continue
        corr = np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        if abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, corr
    return best_lag, float(best_corr)


# ---------------------------------------------------------------------------
# 3. Granger-causality F-test via plain OLS (normal equations, no statsmodels)
# ---------------------------------------------------------------------------
def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    return float(np.sum(residuals ** 2))


def granger_causality_fstat(
    candidate: pd.Series, target: pd.Series, max_lag: int = 5
) -> tuple[float, float | None]:
    """
    Restricted model:   y_t = c + sum_i a_i * y_{t-i}
    Unrestricted model: y_t = c + sum_i a_i * y_{t-i} + sum_i b_i * x_{t-i}
    Returns (F-statistic, p-value). Larger F => stronger evidence that
    `candidate` Granger-causes `target`. p-value is None if scipy isn't
    installed (the F-statistic alone is enough to *rank* candidates).
    """
    lags = list(range(1, max_lag + 1))
    frame = pd.DataFrame({"y": target})
    for lag in lags:
        frame[f"y_lag{lag}"] = target.shift(lag)
        frame[f"x_lag{lag}"] = candidate.shift(lag)
    frame = frame.dropna()

    n = len(frame)
    p = len(lags)
    if n < 4 * p + 5:
        return 0.0, None

    y = frame["y"].to_numpy()
    y_cols = [f"y_lag{lag}" for lag in lags]
    x_cols = [f"x_lag{lag}" for lag in lags]

    X_restricted = np.column_stack([np.ones(n), frame[y_cols].to_numpy()])
    X_full = np.column_stack([X_restricted, frame[x_cols].to_numpy()])

    rss_restricted = _ols_rss(X_restricted, y)
    rss_full = _ols_rss(X_full, y)

    df1, df2 = p, n - (2 * p + 1)
    if df2 <= 0 or rss_full <= 0:
        return 0.0, None
    f_stat = max(((rss_restricted - rss_full) / df1) / (rss_full / df2), 0.0)

    p_value = None
    try:
        from scipy.stats import f as f_dist
        p_value = float(f_dist.sf(f_stat, df1, df2))
    except ImportError:
        pass
    return float(f_stat), p_value


# ---------------------------------------------------------------------------
# 4. Combine signals into a ranked root-cause report
# ---------------------------------------------------------------------------
# Fixed weights for the three signals. Kept as named constants (rather than
# buried literals) because the whole point of this scoring scheme is that
# every one of these numbers is inspectable - see EXPLAINABILITY.md.
CORRELATION_WEIGHT = 0.4
GRANGER_WEIGHT = 0.4
CO_OCCURRENCE_BONUS = 0.2


@dataclass
class RootCauseCandidate:
    name: str
    lag: int
    cross_correlation: float
    granger_f_stat: float
    granger_p_value: float | None
    co_occurring_anomaly: bool
    correlation_contribution: float   # exact share of `score` from signal 1
    granger_contribution: float       # exact share of `score` from signal 2
    co_occurrence_contribution: float  # exact share of `score` from signal 3
    score: float
    explanation: str


def analyze_root_causes(
    target: pd.Series,
    anomaly_timestamp,
    candidates: dict,
    max_lag: int = 10,
    detector_window: int = 14,
) -> list:
    """
    Ranks `candidates` (name -> pd.Series, same index as `target`) by how
    plausibly each one explains the anomaly at `anomaly_timestamp`.

    The score is a plain weighted sum of three named, bounded terms - unlike
    a learned/black-box score, each candidate's `*_contribution` fields are
    the EXACT numbers that were added together to produce `score` (no
    post-hoc approximation, no SHAP/LIME-style surrogate model needed).
    """
    results = []
    for name, series in candidates.items():
        lag, corr = best_lagged_correlation(series, target, max_lag)
        f_stat, p_value = granger_causality_fstat(series, target, max_lag)

        window_start = anomaly_timestamp - pd.Timedelta(days=max_lag)
        co_occurred = False
        if series.loc[window_start:anomaly_timestamp].shape[0] >= max(4, detector_window // 2):
            local_result = rolling_zscore_detector(series, window=detector_window, threshold=2.5)
            co_occurred = bool(local_result.is_anomaly.loc[window_start:anomaly_timestamp].any())

        # log1p compresses the unbounded F-statistic onto a comparable [0,1]
        # scale before weighting; this is the only nonlinear step, and it's
        # applied identically to every candidate so rankings stay comparable.
        corr_contribution = abs(corr) * CORRELATION_WEIGHT
        granger_contribution = min(np.log1p(f_stat) / 5, 1.0) * GRANGER_WEIGHT
        co_occurrence_contribution = CO_OCCURRENCE_BONUS if co_occurred else 0.0
        score = corr_contribution + granger_contribution + co_occurrence_contribution

        results.append(RootCauseCandidate(
            name=name, lag=lag, cross_correlation=corr, granger_f_stat=f_stat,
            granger_p_value=p_value, co_occurring_anomaly=co_occurred,
            correlation_contribution=corr_contribution, granger_contribution=granger_contribution,
            co_occurrence_contribution=co_occurrence_contribution, score=score,
            explanation=_explain(name, lag, corr, f_stat, p_value, co_occurred,
                                  corr_contribution, granger_contribution, co_occurrence_contribution, score),
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _explain(name, lag, corr, f_stat, p_value, co_occurred,
             corr_contribution, granger_contribution, co_occurrence_contribution, score) -> str:
    if lag > 0:
        timing = f"leads the target by {lag} step(s)"
    elif lag == 0:
        timing = "moves in lockstep with the target (lag 0 - could be shared seasonality rather than causal)"
    else:
        timing = "lags behind the target (unlikely to be a cause - effects don't precede their causes)"

    p_text = f", p={p_value:.4f}" if p_value is not None else " (install scipy for a p-value)"
    co_text = (
        "it also showed its own detected anomaly shortly before the target's, which is consistent with it "
        "triggering the downstream effect"
        if co_occurred else
        "it did not show its own detected anomaly in that window, so this may be an ambient leading indicator "
        "rather than the triggering event"
    )
    return (
        f"'{name}' {timing} (correlation={corr:.2f} at lag={lag}). "
        f"Granger-causality F-stat={f_stat:.2f}{p_text} - its past values help linearly predict the target "
        f"beyond the target's own history. Additionally, {co_text}. "
        f"Score breakdown: {corr_contribution:.2f} from correlation (of {CORRELATION_WEIGHT:.1f} max) + "
        f"{granger_contribution:.2f} from Granger evidence (of {GRANGER_WEIGHT:.1f} max) + "
        f"{co_occurrence_contribution:.2f} co-occurrence bonus (of {CO_OCCURRENCE_BONUS:.1f} max) "
        f"= {score:.2f} total."
    )


def explain_ranking(results: list) -> str:
    """
    A comparative, narrative explanation of *why* the top-ranked candidate
    outranked the runner-up - the question a plain per-candidate score can't
    answer on its own. Every number quoted here is read directly off the
    two candidates' contribution fields, not re-derived or approximated.
    """
    if len(results) < 2:
        return results[0].explanation if results else "No candidates to rank."

    top, runner_up = results[0], results[1]
    deltas = {
        "correlation": top.correlation_contribution - runner_up.correlation_contribution,
        "Granger evidence": top.granger_contribution - runner_up.granger_contribution,
        "co-occurrence": top.co_occurrence_contribution - runner_up.co_occurrence_contribution,
    }
    driving_signal = max(deltas, key=lambda k: deltas[k])

    lines = [
        f"Top-ranked root cause: '{top.name}' (score={top.score:.2f}), ahead of "
        f"'{runner_up.name}' (score={runner_up.score:.2f}) by {top.score - runner_up.score:.2f}.",
        f"The gap is driven mainly by {driving_signal}: '{top.name}' beat "
        f"'{runner_up.name}' by {deltas[driving_signal]:.2f} on that term alone.",
    ]
    if not top.co_occurring_anomaly and runner_up.co_occurring_anomaly:
        lines.append(
            f"Caveat: '{runner_up.name}' had its own co-occurring anomaly and '{top.name}' did not - "
            "worth double-checking the top pick isn't just a stronger correlation with a less direct link."
        )
    return " ".join(lines)


def summarize(results: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "candidate": r.name,
        "lag": r.lag,
        "correlation": round(r.cross_correlation, 3),
        "granger_f_stat": round(r.granger_f_stat, 3),
        "p_value": None if r.granger_p_value is None else round(r.granger_p_value, 4),
        "co_occurring_anomaly": r.co_occurring_anomaly,
        "correlation_contribution": round(r.correlation_contribution, 3),
        "granger_contribution": round(r.granger_contribution, 3),
        "co_occurrence_contribution": round(r.co_occurrence_contribution, 3),
        "score": round(r.score, 3),
        "explanation": r.explanation,
    } for r in results])


def plot_score_decomposition(results: list, save_path: str) -> None:
    """
    Stacked horizontal bar chart: each candidate's total score broken into
    its exact three contributing terms. This is the explainability payoff
    of scoring via a transparent weighted sum instead of a learned model -
    the "attribution" is read directly off the numbers used to rank, not
    approximated after the fact the way SHAP/LIME approximate a black box.
    """
    import matplotlib.pyplot as plt

    names = [r.name for r in results][::-1]
    corr_vals = [r.correlation_contribution for r in results][::-1]
    granger_vals = [r.granger_contribution for r in results][::-1]
    co_vals = [r.co_occurrence_contribution for r in results][::-1]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(results) + 1.5))
    ax.barh(names, corr_vals, label=f"correlation (max {CORRELATION_WEIGHT})", color="#4C72B0")
    ax.barh(names, granger_vals, left=corr_vals, label=f"Granger evidence (max {GRANGER_WEIGHT})", color="#DD8452")
    left2 = [c + g for c, g in zip(corr_vals, granger_vals)]
    ax.barh(names, co_vals, left=left2, label=f"co-occurrence bonus (max {CO_OCCURRENCE_BONUS})", color="#55A868")

    for i, r in enumerate(reversed(results)):
        ax.text(r.score + 0.01, i, f"{r.score:.2f}", va="center", fontsize=9)

    ax.set_xlabel("root-cause score (exact sum of the three bars)")
    ax.set_title("Root-Cause Score Decomposition (fully attributable, no post-hoc approximation)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    target, candidates = generate_multivariate_demo_data()

    detection = rolling_zscore_detector(target, window=14, threshold=3.0)
    if not detection.anomalies:
        raise SystemExit("No anomaly detected in the demo target series - unexpected.")
    top_anomaly = max(detection.anomalies, key=lambda a: a.score)
    print(f"Target anomaly detected at {top_anomaly.index} (value={top_anomaly.value:.2f}, "
          f"kind={top_anomaly.kind})\n")

    results = analyze_root_causes(target, top_anomaly.index, candidates, max_lag=7)
    print("=== Root-cause ranking ===")
    print(summarize(results).to_string(index=False))

    print("\n=== Comparative explanation (top vs. runner-up) ===")
    print(explain_ranking(results))

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(target.index, target.values, label="target", color="black", linewidth=2)
        for name, series in candidates.items():
            ax.plot(series.index, series.values, label=name, alpha=0.6)
        ax.axvline(top_anomaly.index, color="red", linestyle="--", label="target anomaly")
        ax.set_title(f"Root-Cause Analysis Demo - top cause: {results[0].name}")
        ax.set_xlabel("date")
        ax.set_ylabel("value")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(_HERE, "root_cause_demo_output.png")
        fig.savefig(out_path, dpi=120)
        print(f"\nSaved plot to {out_path}")

        decomposition_path = os.path.join(_HERE, "root_cause_score_decomposition.png")
        plot_score_decomposition(results, decomposition_path)
        print(f"Saved plot to {decomposition_path}")
    except ImportError:
        print("\nmatplotlib not installed - skipping plots.")
