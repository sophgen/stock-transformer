"""Cross-sectional panel features (numeric columns in the ``[L,S,F]`` tensor)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def trailing_simple_returns(close: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """``[n, s]`` simple return ``close[t]/close[t-1]-1``; row 0 all NaN."""
    close = np.asarray(close, dtype=np.float64)
    n, s = close.shape
    out = np.full((n, s), np.nan, dtype=np.float64)
    if n < 2:
        return out
    a, b = close[:-1], close[1:]
    valid = np.isfinite(a) & np.isfinite(b) & (a > eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = b / a - 1.0
    rr = np.where(valid, rr, np.nan)
    out[1:] = rr
    return out


def rolling_volatility_logret(close: np.ndarray, *, window: int = 5) -> np.ndarray:
    """Rolling std of log(close_t/close_{t-1}) with ``min_periods=1``."""
    close = np.asarray(close, dtype=np.float64)
    n, s = close.shape
    lr = np.full((n, s), np.nan, dtype=np.float64)
    if n < 2:
        return lr
    a, b = close[:-1], close[1:]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr[1:] = np.where(ok, np.log(b / a), np.nan)
    out = np.full((n, s), np.nan, dtype=np.float64)
    w = int(window)
    for j in range(s):
        col = lr[:, j]
        for i in range(n):
            lo = max(0, i - w + 1)
            seg = col[lo : i + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                continue
            out[i, j] = float(np.std(seg, ddof=0))
    return out


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Per-row percentile rank in ``(0, 1]`` using average ranks; NaN preserved."""
    values = np.asarray(values, dtype=np.float64)
    n, s = values.shape
    out = np.full((n, s), np.nan, dtype=np.float64)
    for i in range(n):
        row = values[i]
        m = np.isfinite(row)
        if int(m.sum()) < 1:
            continue
        sub = row[m]
        r = rankdata(sub, method="average")
        pct = r / float(len(sub))
        br = np.full(s, np.nan)
        br[np.where(m)[0]] = pct
        out[i] = br
    return out


def zscore_cross_section(values: np.ndarray) -> np.ndarray:
    """Per-row z-score; NaN where scale is degenerate."""
    values = np.asarray(values, dtype=np.float64)
    n, s = values.shape
    out = np.full((n, s), np.nan, dtype=np.float64)
    for i in range(n):
        row = values[i]
        m = np.isfinite(row)
        if int(m.sum()) < 2:
            continue
        sub = row[m]
        mu = float(np.mean(sub))
        sigma = float(np.std(sub, ddof=0))
        if sigma < 1e-12:
            continue
        z = (sub - mu) / sigma
        br = np.full(s, np.nan)
        br[np.where(m)[0]] = z
        out[i] = br
    return out


def relative_strength_vs_ew(trailing_ret: np.ndarray) -> np.ndarray:
    """``ret[s] - nanmean(ret)`` per row."""
    trailing_ret = np.asarray(trailing_ret, dtype=np.float64)
    n, s = trailing_ret.shape
    out = np.full((n, s), np.nan, dtype=np.float64)
    for i in range(n):
        row = trailing_ret[i]
        m = np.isfinite(row)
        if not np.any(m):
            continue
        mu = float(np.nanmean(row))
        out[i] = np.where(m, row - mu, np.nan)
    return out


def relative_volume_vs_median(volume: np.ndarray) -> np.ndarray:
    """``vol[s] / nanmedian(vol)`` per row."""
    volume = np.asarray(volume, dtype=np.float64)
    n, s = volume.shape
    out = np.full((n, s), np.nan, dtype=np.float64)
    for i in range(n):
        row = volume[i]
        m = np.isfinite(row) & (row >= 0)
        if not np.any(m):
            continue
        med = float(np.nanmedian(row[m]))
        if med < 1e-12:
            continue
        out[i] = np.where(m, row / med, np.nan)
    return out
