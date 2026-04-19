"""Build ``[N, L, S, F]`` universe tensors and masks from an aligned OHLCV panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_UNIVERSE_FEATURES = 5  # open, high, low, close log-rel to prev close, log1p(vol)


def _col(panel: pd.DataFrame, field: str, sym: str) -> np.ndarray:
    return panel[f"{field}__{sym}"].to_numpy(dtype=np.float64)


def universe_features_from_panel(
    panel: pd.DataFrame,
    symbols: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row, per-symbol candle features (same spirit as ``candle_log_returns``).

    For row ``i`` and symbol ``s``, uses previous row ``i-1`` close as reference.
    Returns NaN features where any OHLCV is missing or ``i==0``.

    Returns
    -------
    feats
        ``[n_rows, n_symbols, 5]``
    valid        ``[n_rows, n_symbols]`` bool, True where feature is usable.
    """
    n = len(panel)
    s = len(symbols)
    feats = np.full((n, s, N_UNIVERSE_FEATURES), np.nan, dtype=np.float64)
    valid = np.zeros((n, s), dtype=bool)
    eps = 1e-10
    prev_close = np.full((n, s), np.nan, dtype=np.float64)
    for j, sym in enumerate(symbols):
        prev_close[1:, j] = _col(panel, "close", sym)[:-1]
    o = np.stack([_col(panel, "open", sym) for sym in symbols], axis=1)
    h = np.stack([_col(panel, "high", sym) for sym in symbols], axis=1)
    lo = np.stack([_col(panel, "low", sym) for sym in symbols], axis=1)
    c = np.stack([_col(panel, "close", sym) for sym in symbols], axis=1)
    v = np.clip(
        np.stack([_col(panel, "volume", sym) for sym in symbols], axis=1),
        0,
        None,
    )
    pc = prev_close
    ok = (
        np.isfinite(o)
        & np.isfinite(h)
        & np.isfinite(lo)
        & np.isfinite(c)
        & np.isfinite(v)
        & np.isfinite(pc)
        & (pc > 0)
    )
    ok[0, :] = False
    safe = np.where(pc == 0, eps, pc)
    with np.errstate(divide="ignore", invalid="ignore"):
        f0 = np.log(o / safe + eps)
        f1 = np.log(h / safe + eps)
        f2 = np.log(lo / safe + eps)
        f3 = np.log(c / safe + eps)
    f4 = np.log1p(v)
    stack = np.stack([f0, f1, f2, f3, f4], axis=-1)
    feats = np.where(ok[..., None], stack, np.nan)
    valid = ok
    return feats, valid


def build_universe_samples(
    panel: pd.DataFrame,
    symbols: tuple[str, ...],
    close: np.ndarray,
    *,
    lookback: int,
    min_coverage_symbols: int,
    label_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    """Construct leakage-safe windows: features use rows ``<= t``, labels ``t→t+1``.

    Returns
    -------
    X
        ``[N, L, S, F]``
    mask
        ``[N, L, S]`` — True where padded or missing OHLCV.
    y
        ``[N, S]`` cross-sectional (or raw) targets; NaN where no forward return.
    raw_ret
        ``[N, S]`` simple forward returns (for diagnostics).
    ts        Prediction timestamps (row ``t``).
    end_row
        ``[N]`` int row indices ``t`` into ``panel`` (same order as ``ts``).
    """
    from stock_transformer.labels.cross_sectional import cross_sectional_targets, raw_returns_forward

    lookback = int(lookback)
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    row_feats, row_valid = universe_features_from_panel(panel, symbols)
    n = len(panel)
    raw = raw_returns_forward(close)
    y_full = cross_sectional_targets(raw, mode=label_mode)

    xs: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rrs: list[np.ndarray] = []
    ts_list: list[pd.Timestamp] = []
    ends: list[int] = []

    ts_all = pd.to_datetime(panel["timestamp"])

    for t in range(lookback - 1, n - 1):
        win = slice(t - lookback + 1, t + 1)
        block = row_feats[win]
        vblock = row_valid[win]
        mask = ~vblock
        fut = raw[t]
        live = np.isfinite(fut)
        if int(live.sum()) < int(min_coverage_symbols):
            continue
        y_row = y_full[t].copy()
        xs.append(block)
        ms.append(mask)
        ys.append(y_row)
        rrs.append(fut.copy())
        ts_list.append(pd.Timestamp(ts_all.iloc[t]))
        ends.append(t)

    if not xs:
        s_n = len(symbols)
        empty_x = np.empty((0, lookback, s_n, N_UNIVERSE_FEATURES), dtype=np.float32)
        empty_m = np.empty((0, lookback, s_n), dtype=bool)
        empty_y = np.empty((0, s_n), dtype=np.float32)
        empty_r = np.empty((0, s_n), dtype=np.float32)
        return empty_x, empty_m, empty_y, empty_r, pd.Series(dtype="datetime64[ns]"), np.empty((0,), dtype=np.int64)

    X = np.stack(xs).astype(np.float32)
    mask = np.stack(ms)
    y = np.stack(ys).astype(np.float32)
    raw_stack = np.stack(rrs).astype(np.float32)
    end_row = np.asarray(ends, dtype=np.int64)
    return X, mask, y, raw_stack, pd.Series(ts_list), end_row
