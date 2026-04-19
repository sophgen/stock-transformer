"""Cross-sectional return labels at each timestamp (leakage-safe: uses t and t+1 only)."""

from __future__ import annotations

import numpy as np


def raw_returns_forward(close: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Per-symbol forward simple return from row i to i+1."""
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
    sectors: np.ndarray | None = None,
) -> np.ndarray:
    """Demean raw forward returns per ``mode``."""
    raw = np.asarray(raw, dtype=np.float64)
    if mode == "raw_return":
        return raw.copy()
    if mode == "cross_sectional_return":
        return _demean_by_func(raw, lambda row, m: float(np.nanmedian(row[m])))
    if mode == "equal_weighted_return":
        return _demean_by_func(raw, lambda row, m: float(np.nanmean(row[m])))
    if mode == "sector_neutral_return":
        if sectors is None:
            raise ValueError("sector_neutral_return requires sectors[S]")
        return _sector_neutral_demean(raw, sectors)
    raise ValueError(f"Unknown label mode: {mode}")


def _demean_by_func(
    raw: np.ndarray,
    center_fn: callable,
) -> np.ndarray:
    out = np.full_like(raw, np.nan, dtype=np.float64)
    for i in range(raw.shape[0]):
        row = raw[i]
        m = np.isfinite(row)
        if not np.any(m):
            continue
        c = center_fn(row, m)
        if not np.isfinite(c):
            continue
        out[i] = np.where(m, row - c, np.nan)
    return out


def _sector_neutral_demean(raw: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    out = np.full_like(raw, np.nan, dtype=np.float64)
    n, s = raw.shape
    if len(sectors) != s:
        raise ValueError("sectors must have length n_symbols")
    for i in range(n):
        row = raw[i]
        for j in range(s):
            if not np.isfinite(row[j]):
                continue
            sec = sectors[j]
            peer = np.isfinite(row) & (sectors == sec)
            if not np.any(peer):
                continue
            med = float(np.nanmedian(row[peer]))
            if not np.isfinite(med):
                continue
            out[i, j] = row[j] - med
    return out


def bucket_labels_by_quantile(
    values: np.ndarray,
    *,
    q: float = 0.33,
) -> np.ndarray:
    """Per-row top / middle / bottom bucket (0,1,2) from cross-sectional ``values``."""
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
        bucket = np.full(x.shape[0], 1.0)
        bucket[ranks < k] = 0.0
        bucket[ranks >= len(x) - k] = 2.0
        br = np.full(n_s, np.nan)
        br[np.where(valid)[0]] = bucket
        out[i] = br
    return out
