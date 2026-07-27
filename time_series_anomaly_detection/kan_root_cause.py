"""
KAN-based nonlinear root-cause detection - from scratch.

`root_cause_analysis.py`'s Granger-causality test is built on plain OLS,
which can only detect LINEAR relationships between a candidate's lagged
values and the target. That structurally misses real causal relationships
like:
  - a threshold/saturating effect ("the target only reacts once the
    candidate exceeds some level"),
  - a magnitude-only effect ("the target reacts to how far the candidate
    is from its own baseline, regardless of direction") - which has near-
    ZERO linear correlation even when the causal link is strong and real.

This module adds a **Kolmogorov-Arnold Network (KAN) edge** - the same
idea behind KAN-AD (see LITERATURE_REVIEW.md §1): instead of a fixed
weight per input (an ordinary linear regression coefficient), each input
gets its OWN learned univariate function. KAN-AD's key design choice was
replacing the original KAN paper's B-spline edges with a Fourier basis
(smoother, more globally aware, less sensitive to local noise) - the exact
same basis this project's Fourier-embedding work already relies on.

The crucial simplification used here: a single-layer KAN with a *fixed*
Fourier basis is linear IN ITS COEFFICIENTS (only nonlinear in the raw
input, via the basis expansion) - which means the whole layer can be fit
with plain OLS on an expanded feature matrix. No gradient descent, no
PyTorch, no training loop: this is a Generalized Additive Model, fit with
the exact same `np.linalg.lstsq` machinery as the rest of this repo's
Granger-causality test and the notebook's original linear regression.

Two things fall out of this "for free":
  1. **Nonlinear Granger causality**: compare a restricted KAN (target's
     own lag only) against an unrestricted KAN (+ the candidate's lag) via
     the same nested-model F-test as the linear version - but now able to
     detect nonlinear relationships the linear test misses entirely.
  2. **Interpretability**: each edge's learned function can be evaluated
     and plotted directly over the candidate's value range - literally
     showing HOW the candidate influences the target (a threshold? a
     saturating curve? magnitude-only?), not just a single correlation
     number or F-statistic.
"""

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. A single KAN edge: one input -> one learned univariate function,
#    represented as a small Fourier series (KAN-AD's basis choice).
# ---------------------------------------------------------------------------
def _fit_znorm(x: np.ndarray):
    mean = np.mean(x)
    std = np.std(x)
    return mean, (std if std > 1e-9 else 1.0)


def _edge_features(x_norm: np.ndarray, n_harmonics: int, include_bias: bool) -> np.ndarray:
    """
    Expands a (z-normalized) 1D input into a fixed Fourier basis. A plain
    OLS fit on these features can represent any smooth periodic-ish
    function of the input - this IS the learned "edge function."

    `tanh`-squashes the z-normalized input into (-pi, pi) first: a Fourier
    basis is periodic, and raw z-scores are unbounded, so without this an
    outlier could alias/wrap around into a completely different part of
    the basis. tanh is smooth and order-preserving, so it just compresses
    extreme values rather than distorting the shape near the bulk of the
    data.
    """
    x_scaled = np.pi * np.tanh(x_norm / 2)
    feats = [np.ones_like(x_scaled)] if include_bias else []
    for k in range(1, n_harmonics + 1):
        feats.append(np.cos(k * x_scaled))
        feats.append(np.sin(k * x_scaled))
    return np.column_stack(feats)


def _ols_fit(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    rss = float(np.sum(residuals ** 2))
    return beta, rss


# ---------------------------------------------------------------------------
# 2. Nonlinear Granger-causality F-test built on the KAN edge
# ---------------------------------------------------------------------------
@dataclass
class KANEdgeResult:
    f_stat: float
    p_value: float | None
    edge_coeffs: np.ndarray
    norm_mean: float
    norm_std: float
    n_harmonics: int
    n_obs: int


def kan_nonlinear_granger_test(
    candidate: pd.Series, target: pd.Series, lag: int, n_harmonics: int = 5
) -> KANEdgeResult:
    """
    Restricted model:   y_t = f(y_{t-1})                     (KAN edge on target's own lag)
    Unrestricted model:  y_t = f(y_{t-1}) + g(x_{t-lag})       (+ a KAN edge on the candidate)

    Both fit via OLS on Fourier-expanded features (a Generalized Additive
    Model). Returns an F-test comparing the two, PLUS the fitted candidate
    edge's coefficients so its learned function can be plotted - unlike
    the linear Granger test, a rejection here also comes with a picture of
    *what shape* of nonlinear relationship was found.
    """
    frame = pd.DataFrame({
        "y": target,
        "y_lag1": target.shift(1),
        "x_lag": candidate.shift(lag),
    }).dropna()
    n = len(frame)
    y = frame["y"].to_numpy()

    y_mean, y_std = _fit_znorm(frame["y_lag1"].to_numpy())
    x_mean, x_std = _fit_znorm(frame["x_lag"].to_numpy())

    y_lag_norm = (frame["y_lag1"].to_numpy() - y_mean) / y_std
    x_lag_norm = (frame["x_lag"].to_numpy() - x_mean) / x_std

    y_lag_feats = _edge_features(y_lag_norm, n_harmonics, include_bias=True)
    x_lag_feats = _edge_features(x_lag_norm, n_harmonics, include_bias=False)

    X_restricted = y_lag_feats
    X_unrestricted = np.column_stack([y_lag_feats, x_lag_feats])

    _, rss_restricted = _ols_fit(X_restricted, y)
    beta_unrestricted, rss_unrestricted = _ols_fit(X_unrestricted, y)

    p = x_lag_feats.shape[1]  # extra parameters contributed by the candidate's edge
    df1 = p
    df2 = n - X_unrestricted.shape[1]
    if df2 <= 0 or rss_unrestricted <= 0:
        f_stat = 0.0
    else:
        f_stat = max(((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2), 0.0)

    p_value = None
    if df2 > 0:
        try:
            from scipy.stats import f as f_dist
            p_value = float(f_dist.sf(f_stat, df1, df2))
        except ImportError:
            pass

    candidate_edge_coeffs = beta_unrestricted[-p:]
    return KANEdgeResult(
        f_stat=float(f_stat), p_value=p_value, edge_coeffs=candidate_edge_coeffs,
        norm_mean=x_mean, norm_std=x_std, n_harmonics=n_harmonics, n_obs=n,
    )


def evaluate_edge_function(edge: KANEdgeResult, x_values: np.ndarray) -> np.ndarray:
    """Evaluates the candidate's learned edge function at raw (un-normalized) x_values."""
    x_norm = (x_values - edge.norm_mean) / edge.norm_std
    feats = _edge_features(x_norm, edge.n_harmonics, include_bias=False)
    return feats @ edge.edge_coeffs


def plot_kan_edge_function(edge: KANEdgeResult, x_min: float, x_max: float, save_path: str,
                            xlabel: str = "candidate value (at its Granger-tested lag)") -> None:
    import matplotlib.pyplot as plt

    x_grid = np.linspace(x_min, x_max, 300)
    y_grid = evaluate_edge_function(edge, x_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_grid, y_grid, color="crimson", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("learned contribution to target")
    p_text = f", p={edge.p_value:.4f}" if edge.p_value is not None else ""
    ax.set_title(f"KAN edge function - the candidate's learned nonlinear effect\n"
                 f"(F-stat={edge.f_stat:.2f}{p_text}, n={edge.n_obs})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Demo: a purely nonlinear (magnitude-only) causal relationship that
#    linear Granger causality is structurally blind to.
# ---------------------------------------------------------------------------
def generate_nonlinear_causal_demo(n_points: int = 400, seed: int = 11):
    """
    target_t = 10 + 4 * cause_{t-lag}^2 + noise.

    Squaring a zero-mean, symmetric input produces an effect that depends
    only on MAGNITUDE, not sign - a textbook case where Pearson
    correlation (and therefore linear Granger causality) is close to zero
    even though the causal relationship is strong, real, and deterministic
    up to noise.
    """
    rng = np.random.default_rng(seed)
    cause = rng.normal(0, 1, n_points)

    lag = 3
    shifted = np.roll(cause, lag)
    shifted[:lag] = 0.0
    target = 10 + 4 * shifted ** 2 + rng.normal(0, 0.5, n_points)

    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    return (
        pd.Series(target, index=dates, name="target"),
        pd.Series(cause, index=dates, name="nonlinear_cause"),
        lag,
    )


# ---------------------------------------------------------------------------
# 4. Integration point: a nonlinear "second opinion" alongside the existing
#    linear root-cause ranking, without touching its (already-validated)
#    scoring logic.
# ---------------------------------------------------------------------------
def kan_diagnostics_for_candidates(results: list, target: pd.Series, candidates: dict,
                                    n_harmonics: int = 5, flag_threshold: float = 10.0) -> pd.DataFrame:
    """
    Takes the ranked output of `root_cause_analysis.analyze_root_causes`
    and runs the KAN nonlinear Granger test for each candidate at its own
    best lag (already found by the linear analysis). Returns a DataFrame
    rather than mutating the input, so the original (validated) linear
    ranking is left untouched - this is meant to be read ALONGSIDE it, to
    catch candidates whose linear evidence looked weak but that still have
    a real nonlinear relationship (the exact case the linear-only test in
    `root_cause_analysis.py` structurally cannot detect).
    """
    rows = []
    for r in results:
        lag = max(r.lag, 1)  # the nonlinear test needs a positive lag to shift by
        edge = kan_nonlinear_granger_test(candidates[r.name], target, lag=lag, n_harmonics=n_harmonics)
        linear_weak = r.granger_f_stat < flag_threshold
        kan_strong = edge.f_stat > flag_threshold
        rows.append({
            "candidate": r.name,
            "linear_granger_f_stat": round(r.granger_f_stat, 3),
            "kan_granger_f_stat": round(edge.f_stat, 3),
            "kan_p_value": None if edge.p_value is None else round(edge.p_value, 4),
            "possible_nonlinear_relationship_missed_by_linear_test": bool(linear_weak and kan_strong),
        })
    return pd.DataFrame(rows).sort_values("kan_granger_f_stat", ascending=False)


if __name__ == "__main__":
    from root_cause_analysis import granger_causality_fstat

    target, cause, true_lag = generate_nonlinear_causal_demo()

    print("=== Linear vs. KAN (nonlinear) Granger causality on a magnitude-only effect ===")
    print(f"True relationship: target = 10 + 4*cause[t-{true_lag}]^2 + noise (sign-independent)\n")

    lin_f, lin_p = granger_causality_fstat(cause, target, max_lag=5)
    p_text = f", p={lin_p:.4f}" if lin_p is not None else " (install scipy for a p-value)"
    print(f"Linear Granger F-test:  F={lin_f:.2f}{p_text}")
    print("  -> expected to be weak/insignificant: linear regression can't see a sign-independent effect.\n")

    kan_result = kan_nonlinear_granger_test(cause, target, lag=true_lag, n_harmonics=5)
    p_text = f", p={kan_result.p_value:.4f}" if kan_result.p_value is not None else " (install scipy for a p-value)"
    print(f"KAN nonlinear Granger test: F={kan_result.f_stat:.2f}{p_text}")
    print("  -> expected to be strong: the Fourier-basis edge can represent the U-shaped (quadratic-like) effect.\n")

    out_path = os.path.join(_HERE, "kan_edge_function_demo.png")
    plot_kan_edge_function(kan_result, float(cause.min()), float(cause.max()), out_path,
                            xlabel=f"cause value (lag {true_lag})")
    print(f"Saved edge function plot to {out_path}")
    print("(the plot should show a clear U-shape / symmetric-around-zero curve, "
          "recovering the true squared relationship directly from data)")
