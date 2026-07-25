"""
CUSUM changepoint detection - from scratch.

`anomaly_detection.py`'s `level_shift` classification used to be a bare
heuristic: "does the mean of the window after the point differ from the
mean before by more than 2 local standard deviations?". That's a single
threshold check with no notion of statistical significance, no confidence
level, and no attempt to locate exactly *where* within the window the
shift actually began (it just assumes the shift starts at the flagged
point).

CUSUM (cumulative sum control chart) is the standard, well-understood tool
for this: a sequential test that accumulates evidence of a sustained shift
over time, rather than comparing two single window-average snapshots. It
gives three things the heuristic didn't:
  1. A real detection **statistic** (how many standard deviations of
     *sustained* evidence there is - not just a mean difference).
  2. A principled **decision threshold** `h`, whose false-alarm rate is a
     known, standard property of the CUSUM test (not an arbitrary "2x").
  3. The CUSUM's own estimate of **where the shift actually started**
     (the point since the last reset of the running sum), which can
     differ from the originally-flagged index.
"""

import numpy as np
import pandas as pd


def cusum_statistic(before: pd.Series, after: pd.Series, k: float = 0.5):
    """
    Two-sided CUSUM: treats `before` as the normal/reference period (used
    only to estimate baseline mean and std) and scans forward through
    `after` looking for a sustained shift away from that baseline.

    `k` is the CUSUM "allowance" in standard-deviation units (the amount of
    drift the test tolerates before it starts accumulating evidence - the
    textbook default is half the shift size you want to be sensitive to,
    so k=0.5 targets detecting roughly a 1-sigma sustained shift).

    Returns (max_statistic, direction, changepoint_offset):
      - max_statistic: the peak cumulative-sum value reached (in std units)
      - direction: "up" or "down"
      - changepoint_offset: index INTO `after` where the run that produced
        the peak began (the CUSUM's own changepoint location estimate)
    """
    baseline_mean = before.mean()
    baseline_std = before.std(ddof=0)
    baseline_std = baseline_std if baseline_std > 1e-9 else 1e-9

    z = (after.to_numpy() - baseline_mean) / baseline_std

    s_pos = s_neg = 0.0
    start_pos = start_neg = 0
    max_pos = max_neg = 0.0
    argmax_pos = argmax_neg = 0

    for i, zi in enumerate(z):
        if s_pos == 0.0:
            start_pos = i
        s_pos = max(0.0, s_pos + zi - k)
        if s_pos > max_pos:
            max_pos, argmax_pos = s_pos, start_pos

        if s_neg == 0.0:
            start_neg = i
        s_neg = max(0.0, s_neg - zi - k)
        if s_neg > max_neg:
            max_neg, argmax_neg = s_neg, start_neg

    if max_pos >= max_neg:
        return float(max_pos), "up", int(argmax_pos)
    return float(max_neg), "down", int(argmax_neg)


def is_confirmed_level_shift(before: pd.Series, after: pd.Series, h: float = 5.0, k: float = 0.5):
    """
    Returns (confirmed, statistic, direction, changepoint_offset). `h=5.0`
    is a standard CUSUM threshold (analogous to a ~3-4 sigma single-point
    test, but for *sustained* evidence rather than one sample) - the
    textbook average-run-length tables (e.g. Montgomery, "Introduction to
    Statistical Quality Control") put the false-alarm rate for k=0.5,h=5
    at roughly 1 in several hundred windows under a stable process, versus
    the previous heuristic's untuned "2 local std" cutoff.
    """
    if len(before) < 2 or len(after) < 2:
        return False, 0.0, "up", 0
    stat, direction, offset = cusum_statistic(before, after, k=k)
    return stat > h, stat, direction, offset
