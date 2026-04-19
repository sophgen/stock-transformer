"""Build ``[N, L, S, F]`` universe tensors and masks from an aligned OHLCV panel."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stock_transformer.features import cross_sectional as cs

DEFAULT_UNIVERSE_FEATURE_NAMES: tuple[str, ...] = (
    "open_log_prev_close",
    "high_log_prev_close",
    "low_log_prev_close",
    "close_log_prev_close",
    "log1p_volume",
)

OPTIONAL_CROSS_SECTIONAL_FEATURES: tuple[str, ...] = (
    "cs_trailing_return_pct_rank",
    "cs_volume_pct_rank",
    "cs_trailing_vol_pct_rank",
    "cs_trailing_return_zscore",
    "relative_strength_ew",
    "relative_volume_median",
)

ALL_KNOWN_FEATURES: frozenset[str] = frozenset(DEFAULT_UNIVERSE_FEATURE_NAMES + OPTIONAL_CROSS_SECTIONAL_FEATURES)

# Back-compat with pre-M9a imports
N_UNIVERSE_FEATURES = len(DEFAULT_UNIVERSE_FEATURE_NAMES)


def feature_schema(feature_names: Sequence[str] | None = None) -> dict[str, Any]:
    names = list(feature_names or DEFAULT_UNIVERSE_FEATURE_NAMES)
    h = hashlib.sha256(json.dumps(names, sort_keys=True).encode()).hexdigest()[:16]
    return {"features": names, "n": len(names), "hash": h}


def _col(panel: pd.DataFrame, field: str, sym: str) -> np.ndarray:
    return panel[f"{field}__{sym}"].to_numpy(dtype=np.float64)


def _base_candle_features(panel: pd.DataFrame, symbols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """OHLC log vs prev close + log1p(volume); same layout as legacy v1."""
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


def _volume_matrix(panel: pd.DataFrame, symbols: tuple[str, ...]) -> np.ndarray:
    return np.stack([_col(panel, "volume", sym) for sym in symbols], axis=1)


def build_row_feature_tensor(
    panel: pd.DataFrame,
    symbols: tuple[str, ...],
    close: np.ndarray,
    feature_names: Sequence[str],
    *,
    vol_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Full ``[n_rows, n_symbols, F]`` feature stack and per-(row,symbol) validity."""
    names = list(feature_names)
    unknown = [n for n in names if n not in ALL_KNOWN_FEATURES]
    if unknown:
        raise ValueError(f"Unknown feature names: {unknown}")

    base_stack, base_ok = _base_candle_features(panel, symbols)
    base_map = {DEFAULT_UNIVERSE_FEATURE_NAMES[i]: base_stack[..., i] for i in range(len(DEFAULT_UNIVERSE_FEATURE_NAMES))}

    vol = _volume_matrix(panel, symbols)
    tr = cs.trailing_simple_returns(close)
    volat = cs.rolling_volatility_logret(close, window=vol_window)

    derived: dict[str, np.ndarray] = {
        "cs_trailing_return_pct_rank": cs.percentile_rank(tr),
        "cs_volume_pct_rank": cs.percentile_rank(vol),
        "cs_trailing_vol_pct_rank": cs.percentile_rank(volat),
        "cs_trailing_return_zscore": cs.zscore_cross_section(tr),
        "relative_strength_ew": cs.relative_strength_vs_ew(tr),
        "relative_volume_median": cs.relative_volume_vs_median(vol),
    }

    planes: list[np.ndarray] = []
    for name in names:
        if name in base_map:
            planes.append(base_map[name])
        else:
            planes.append(derived[name])
    stack = np.stack(planes, axis=-1)
    finite = np.isfinite(stack).all(axis=-1)
    return stack, finite


def build_universe_samples(
    panel: pd.DataFrame,
    symbols: tuple[str, ...],
    close: np.ndarray,
    *,
    lookback: int,
    min_coverage_symbols: int,
    label_mode: str,
    feature_names: Sequence[str] | None = None,
    vol_window: int = 5,
    sectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    """Construct leakage-safe windows: features use rows ``<= t``, labels ``t→t+1``."""
    from stock_transformer.labels.cross_sectional import cross_sectional_targets, raw_returns_forward

    lookback = int(lookback)
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    fnames = tuple(feature_names) if feature_names is not None else DEFAULT_UNIVERSE_FEATURE_NAMES
    row_feats, row_valid = build_row_feature_tensor(panel, symbols, close, fnames, vol_window=vol_window)
    n = len(panel)
    raw = raw_returns_forward(close)
    y_full = cross_sectional_targets(raw, mode=label_mode, sectors=sectors)

    xs: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rrs: list[np.ndarray] = []
    ts_list: list[pd.Timestamp] = []
    ends: list[int] = []

    ts_all = pd.to_datetime(panel["timestamp"])
    f_dim = len(fnames)

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
        empty_x = np.empty((0, lookback, s_n, f_dim), dtype=np.float32)
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
