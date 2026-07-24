"""
Real-world validation on the UCR Anomaly Archive (Keogh et al.; see Wu &
Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed", 2021)
- the "UCR" benchmark this project's LITERATURE_REVIEW.md cites as one of
the standard TSAD datasets used across the surveyed literature.

The archive is NOT bundled in this repo (330MB+ across 250 real sensor/ECG/
temperature/etc. files) - download it yourself:

    curl -o ucr_anomaly.zip https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
    unzip -j ucr_anomaly.zip "*/UCR_Anomaly_FullData/*.txt" -d ucr_data/

Each file is one column of values, named:
    <index>_UCR_Anomaly_<name>_<train_size>_<anomaly_start>_<anomaly_end>.txt
The first `train_size` points are anomaly-free by construction (the
archive's convention - meant to let a detector calibrate on normal data
before the labeled anomaly interval [anomaly_start, anomaly_end] arrives
later in the series).

Evaluation protocol: rather than point-adjusted F1 (shown to be gameable -
see LITERATURE_REVIEW.md §3), we use the stricter "top-1" protocol common
in this specific archive's own literature (e.g. MERLIN, Matrix Profile
discord papers): each detector's single highest-scoring point in the test
region (everything after `train_size`) is its "best guess" at the anomaly
location, checked for whether it lands inside the labeled interval. This
can't be gamed by flagging lots of points the way point-adjustment can.
"""

import os
import re
import sys
import time
import numpy as np
import pandas as pd

from anomaly_detection import rolling_zscore_detector, rolling_iqr_detector, trend_residual_detector
from shape_discord_detection import shape_discord_detector
from fourier_embedding_detection import estimate_period_via_fourier, embedding_distance_detector

_HERE = os.path.dirname(os.path.abspath(__file__))
FILENAME_RE = re.compile(r"^(\d+)_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$")

DETECTORS = {
    "zscore": lambda s: rolling_zscore_detector(s, window=100, threshold=4.0),
    "iqr": lambda s: rolling_iqr_detector(s, window=100, k=3.0),
    "trend_residual": lambda s: trend_residual_detector(s, trend_window=100, z_window=100, threshold=4.0),
}

SHAPE_M = 100
SHAPE_MAX_QUERIES = 1000
FOURIER_MAX_WINDOWS = 30000


def parse_filename(filename: str) -> dict:
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Unrecognized UCR anomaly filename format: {filename}")
    idx, name, train_size, anomaly_start, anomaly_end = m.groups()
    return {
        "index": int(idx), "name": name, "train_size": int(train_size),
        "anomaly_start": int(anomaly_start), "anomaly_end": int(anomaly_end),
    }


def load_series(path: str) -> pd.Series:
    return pd.Series(np.loadtxt(path))


def evaluate_file(path: str, filename: str, include_shape: bool = True, include_fourier: bool = True) -> dict:
    meta = parse_filename(filename)
    series = load_series(path)
    test_index = series.index[meta["train_size"]:]

    row = {
        "file": filename, "name": meta["name"], "length": len(series),
        "train_size": meta["train_size"],
        "anomaly_start": meta["anomaly_start"], "anomaly_end": meta["anomaly_end"],
    }

    for det_name, det_fn in DETECTORS.items():
        result = det_fn(series)
        test_scores = result.scores.loc[test_index]
        if test_scores.max(skipna=True) in (0, None) or test_scores.isna().all():
            row[f"{det_name}_hit"] = False
            row[f"{det_name}_top1_index"] = None
            continue
        top1_idx = int(test_scores.idxmax())
        hit = meta["anomaly_start"] <= top1_idx <= meta["anomaly_end"]
        row[f"{det_name}_hit"] = bool(hit)
        row[f"{det_name}_top1_index"] = top1_idx

    if include_shape:
        candidate_positions = range(meta["train_size"], len(series) - SHAPE_M)
        discords = shape_discord_detector(
            series, m=SHAPE_M, max_queries=SHAPE_MAX_QUERIES, candidate_positions=candidate_positions
        )
        if discords:
            top = discords[0]
            hit = meta["anomaly_start"] <= top.position <= meta["anomaly_end"]
            row["shape_discord_hit"] = bool(hit)
            row["shape_discord_top1_index"] = top.position
        else:
            row["shape_discord_hit"] = False
            row["shape_discord_top1_index"] = None

    if include_fourier:
        m = estimate_period_via_fourier(series)
        positions, scores = embedding_distance_detector(
            series, m, train_size=meta["train_size"], max_windows=FOURIER_MAX_WINDOWS
        )
        test_mask = positions >= meta["train_size"]
        if test_mask.any() and np.nanmax(scores[test_mask]) > 0:
            top1_pos = int(positions[test_mask][np.argmax(scores[test_mask])])
            hit = meta["anomaly_start"] <= top1_pos + m - 1 and top1_pos <= meta["anomaly_end"]
            row["fourier_embedding_hit"] = bool(hit)
            row["fourier_embedding_top1_index"] = top1_pos
        else:
            row["fourier_embedding_hit"] = False
            row["fourier_embedding_top1_index"] = None
        row["fourier_embedding_m"] = m

    return row


def run_evaluation(data_dir: str, limit: int = None) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if limit:
        files = files[:limit]

    rows = []
    for i, filename in enumerate(files):
        t0 = time.time()
        try:
            row = evaluate_file(os.path.join(data_dir, filename), filename)
            row["error"] = None
        except Exception as exc:
            row = {"file": filename, "error": str(exc)}
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"[{i + 1}/{len(files)}] {filename[:65]:65s} {row['seconds']:>6.2f}s")

    return pd.DataFrame(rows)


def plot_example(data_dir: str, filename: str, detector_name: str, save_path: str) -> None:
    """Plots one UCR file: full series, shaded true anomaly interval, and the
    detector's top-1 guess in the test region - for the README's hit/miss
    illustrations."""
    import matplotlib.pyplot as plt

    meta = parse_filename(filename)
    series = load_series(os.path.join(data_dir, filename))
    result = DETECTORS[detector_name](series)
    test_index = series.index[meta["train_size"]:]
    top1_idx = int(result.scores.loc[test_index].idxmax())
    hit = meta["anomaly_start"] <= top1_idx <= meta["anomaly_end"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, color="steelblue", linewidth=0.8)
    ax.axvspan(meta["anomaly_start"], meta["anomaly_end"], color="red", alpha=0.25, label="labeled anomaly")
    ax.axvline(meta["train_size"], color="gray", linestyle=":", label="train/test split")
    ax.axvline(top1_idx, color="green" if hit else "black", linestyle="--",
               label=f"{detector_name} top-1 guess ({'HIT' if hit else 'MISS'})")
    ax.set_title(f"{meta['name']} ({'hit' if hit else 'miss'}) - {filename}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Top-1 hit rate on the UCR Anomaly Archive (real-world data) ===")
    n_total = len(df)
    n_errors = df["error"].notna().sum() if "error" in df.columns else 0
    print(f"Files evaluated: {n_total} ({n_errors} failed to parse/load)")
    for det_name in list(DETECTORS) + ["shape_discord", "fourier_embedding"]:
        col = f"{det_name}_hit"
        if col in df.columns:
            valid = df[col].notna()
            hit_rate = df.loc[valid, col].mean()
            print(f"  {det_name:16s}: {hit_rate:.1%} ({int(df.loc[valid, col].sum())}/{valid.sum()} files)")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "ucr_data")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not os.path.isdir(data_dir):
        raise SystemExit(
            f"'{data_dir}' not found. Download the UCR Anomaly Archive first - see the "
            "module docstring at the top of this file for the download/unzip commands."
        )

    df = run_evaluation(data_dir, limit=limit)

    out_csv = os.path.join(_HERE, "ucr_evaluation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed per-file results to {out_csv}")

    print_summary(df)
