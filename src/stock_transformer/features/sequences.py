"""Autoregressive windows and next-candle direction labels (no lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Per-candle features using information available at end of that candle only.

    Returns (aligned_frame, X) where X[i] corresponds to df row i after alignment.
    """
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    low = df["low"].astype(float)
    v = df["volume"].astype(float).clip(lower=0)

    log_ret = np.log(c / c.shift(1))
    hl_range = (h - low) / c.replace(0, np.nan)
    oc_ratio = (c - o) / c.replace(0, np.nan)

    feat = pd.DataFrame(
        {
            "log_ret": log_ret,
            "hl_range": hl_range,
            "oc_ratio": oc_ratio,
            "log_vol": np.log1p(v),
        }
    )
    valid = feat.notna().all(axis=1)
    aligned = df.loc[valid].reset_index(drop=True)
    X = feat.loc[valid].to_numpy(dtype=np.float64)
    return aligned, X


def build_direction_labels(closes: np.ndarray) -> np.ndarray:
    """Binary label: 1 if next close > current close, else 0. Last index is NaN (no future)."""
    n = len(closes)
    y = np.full(n, np.nan)
    for t in range(n - 1):
        y[t] = 1.0 if closes[t + 1] > closes[t] else 0.0
    return y


def build_windows(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.Series,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """
    Build 3D windows ending at t, label = direction t -> t+1.

    Returns (X_win, y_win, end_indices, end_timestamps).
    """
    n = X.shape[0]
    xs: list[np.ndarray] = []
    ys: list[float] = []
    ends: list[int] = []
    ts_list: list[pd.Timestamp] = []

    for t in range(lookback - 1, n - 1):
        if np.isnan(y[t]):
            continue
        xs.append(X[t - lookback + 1 : t + 1])
        ys.append(float(y[t]))
        ends.append(t)
        ts_list.append(pd.Timestamp(timestamps.iloc[t]))

    if not xs:
        return (
            np.empty((0, lookback, X.shape[1])),
            np.empty((0,)),
            np.empty((0,), dtype=int),
            pd.Series(dtype="datetime64[ns]"),
        )

    X_win = np.stack(xs, axis=0)
    y_win = np.asarray(ys, dtype=np.float64)
    end_idx = np.asarray(ends, dtype=int)
    ts_end = pd.Series(ts_list)
    return X_win, y_win, end_idx, ts_end


def validate_no_lookahead(
    df: pd.DataFrame,
    end_indices: np.ndarray,
    lookback: int,
) -> None:
    """Assert each window uses only rows up to end index (inclusive)."""
    ts = pd.to_datetime(df["timestamp"])
    for t_end in end_indices:
        start = int(t_end) - lookback + 1
        if start < 0:
            raise AssertionError(f"Invalid window start {start} for end {t_end}")
        w_ts_max = ts.iloc[int(t_end)]
        if ts.iloc[start : int(t_end) + 1].max() != w_ts_max:
            raise AssertionError("Window max timestamp mismatch")
        # label uses t_end+1 — features must not include it
        if int(t_end) >= len(df) - 1:
            raise AssertionError("end index has no future candle")
