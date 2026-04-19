"""Naive baselines for next-candle direction."""

from __future__ import annotations

import numpy as np


def persistence_baseline(y_hist: np.ndarray) -> float:
    """
    Predict same direction as last realized move (requires previous label).

    For first sample, return 0.5 (uncertain).
    """
    if len(y_hist) < 2:
        return 0.5
    # last label is direction at last index; "persistence" = repeat last y
    return float(y_hist[-1])


def moving_average_baseline(closes: np.ndarray, window: int = 5) -> float:
    """
    Probability-like score: 1 if close[-1] > SMA(prev), else 0 (deterministic).

    Used for comparison; not calibrated probabilities.
    """
    if len(closes) < window + 1:
        return 0.5
    sma = closes[-window - 1 : -1].mean()
    return 1.0 if closes[-1] > sma else 0.0


def momentum_rank_scores(
    close: np.ndarray,
    *,
    end_rows: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Rank baseline: score = trailing simple return over ``lookback`` bars (exclusive end).

    ``close`` is ``[n_panel, n_symbols]``; ``end_rows`` are row indices ``t``.
    Returns ``[n_samples, n_symbols]`` scores (higher = stronger momentum).
    """
    close = np.asarray(close, dtype=np.float64)
    end_rows = np.asarray(end_rows, dtype=np.int64)
    n_s = close.shape[1]
    out = np.full((len(end_rows), n_s), np.nan, dtype=np.float64)
    for i, t in enumerate(end_rows):
        a = t - lookback
        if a < 0:
            continue
        c0 = close[a]
        c1 = close[t]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = np.where(np.isfinite(c0) & np.isfinite(c1) & (c0 > 0), c1 / c0 - 1.0, np.nan)
    return out


def equal_score_baseline(n_samples: int, n_symbols: int) -> np.ndarray:
    """Flat scores (no cross-sectional edge)."""
    return np.zeros((n_samples, n_symbols), dtype=np.float64)


def persistence_probs_on_test(y_val: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Persistence baseline on the test slice: predict previous realized direction.

    First test step uses the last validation label; later steps use prior test labels.
    """
    out = np.zeros(len(y_test), dtype=np.float64)
    for j in range(len(y_test)):
        if j == 0:
            out[j] = float(y_val[-1]) if len(y_val) else 0.5
        else:
            out[j] = float(y_test[j - 1])
    return out
