"""Cross-sectional return labels at each timestamp (leakage-safe: uses t and t+1 only)."""

from __future__ import annotations

import numpy as np


def raw_returns_forward(close: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Per-symbol forward simple return from row i to i+1.

    Parameters
    ----------
    close
        ``[n_rows, n_symbols]``, NaN where missing.

    Returns
    -------
    r
        ``[n_rows, n_symbols]`` with ``r[i,s] = close[i+1,s]/close[i,s] - 1``,
        NaN where undefined or non-finite inputs.
    """
    close = np.asarray(close, dtype=np.float64)
    n, s = close.shape
    out = np.full((n, s), np.nan, dtype=np.float64)
    if n < 2:
        return out
    a = close[:-1]
    b = close[1:]
    valid = np.isfinite(a) & np.isfinite(b) & (a > eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = b / a - 1.0
    rr = np.where(valid, rr, np.nan)
    out[:-1] = rr
    return out


def cross_sectional_targets(
    raw: np.ndarray,
    *,
    mode: str = "cross_sectional_return",
) -> np.ndarray:
    """Demean raw forward returns across the live cross-section at each row.

    ``mode``:
      - ``cross_sectional_return`` — subtract nanmedian across symbols.
      - ``raw_return`` — return ``raw`` unchanged.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if mode == "raw_return":
        return raw.copy()
    if mode != "cross_sectional_return":
        raise ValueError(f"Unknown label mode: {mode}")
    out = np.full_like(raw, np.nan, dtype=np.float64)
    for i in range(raw.shape[0]):
        row = raw[i]
        if not np.any(np.isfinite(row)):
            continue
        m = np.nanmedian(row)
        if not np.isfinite(m):
            continue
        out[i] = row - m
    return out


def bucket_labels_by_quantile(
    values: np.ndarray,
    *,
    q: float = 0.33,
) -> np.ndarray:
    """Per-row top / middle / bottom bucket (0,1,2) from cross-sectional ``values``.

    NaN entries stay NaN. Uses nan-friendly ranks; ties split arbitrarily.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.full_like(values, np.nan, dtype=np.float64)
    q = float(q)
    if not (0 < q < 0.5):
        raise ValueError("q must be between 0 and 0.5")
    n_s = values.shape[1]
    k = max(1, int(np.floor(n_s * q)))
    for i in range(values.shape[0]):
        row = values[i]
        valid = np.isfinite(row)
        if valid.sum() < 3:
            continue
        x = row[valid]
        order = np.argsort(x)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(x))
        bucket = np.full(x.shape[0],1.0)
        bucket[ranks < k] = 2.0
        bucket[ranks >= len(x) - k] = 0.0
        br = np.full(n_s, np.nan)
        br[np.where(valid)[0]] = bucket
        out[i] = br
    return out
