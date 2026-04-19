"""Synthetic OHLCV for tests and offline demos (no API)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_random_walk_candles(
    n: int = 800,
    *,
    symbol: str = "TEST",
    timeframe: str = "daily",
    seed: int = 0,
) -> pd.DataFrame:
    """Single-timeframe synthetic candles (backward compat)."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(log_ret))
    noise = rng.normal(0, 0.002, size=n)
    open_ = np.r_[close[0], close[:-1]] * (1 + noise)
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, size=n))
    vol = rng.integers(1_000, 10_000, size=n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


_TF_SPEC: list[tuple[str, float, str]] = [
    # (timeframe, fraction-of-daily, pandas freq)
    ("monthly", 1 / 21, "BME"),
    ("weekly", 1 / 5, "W-FRI"),
    ("daily", 1.0, "B"),
    ("60min", 7.0, "h"),
]


def synthetic_multitimeframe_candles(
    n_daily: int = 800,
    *,
    symbol: str = "TEST",
    seed: int = 0,
    timeframes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate correlated synthetic candles for multiple timeframes.

    Each timeframe is an independent random walk whose length is derived from
    ``n_daily``.  All series start from the same base date so that their
    timestamps overlap realistically.
    """
    rng = np.random.default_rng(seed)
    specs = _TF_SPEC if timeframes is None else [s for s in _TF_SPEC if s[0] in set(timeframes)]

    result: dict[str, pd.DataFrame] = {}
    base_date = "2018-01-01"

    for tf, ratio, freq in specs:
        if timeframes is not None and tf not in timeframes:
            continue
        n = max(int(n_daily * ratio), 30)
        log_ret = rng.normal(0, 0.008, size=n)
        close = 100 * np.exp(np.cumsum(log_ret))
        noise = rng.normal(0, 0.002, size=n)
        open_ = np.r_[close[0], close[:-1]] * (1 + noise)
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, size=n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, size=n))
        vol = rng.integers(500, 5_000, size=n).astype(float)
        idx = pd.date_range(base_date, periods=n, freq=freq)
        result[tf] = pd.DataFrame(
            {
                "timestamp": idx,
                "symbol": symbol.upper(),
                "timeframe": tf,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )

    return result


def synthetic_universe_candles(
    n_bars: int,
    symbols: list[str] | tuple[str, ...],
    *,
    timeframe: str = "daily",
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Shared business-day calendar; correlated random walks for multi-ticker tests."""
    rng = np.random.default_rng(seed)
    n = int(n_bars)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    base_ret = rng.normal(0, 0.008, size=n)
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(str(s).upper() for s in symbols):
        noise = rng.normal(0, 0.004, size=n) if i > 0 else 0.0
        log_ret = base_ret * (0.85 + 0.05 * i) + (noise if isinstance(noise, np.ndarray) else 0.0)
        close = 50 * np.exp(np.cumsum(log_ret))
        o_noise = rng.normal(0, 0.002, size=n)
        open_ = np.r_[close[0], close[:-1]] * (1 + o_noise)
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, size=n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, size=n))
        vol = rng.integers(500, 8_000, size=n).astype(float)
        out[sym] = pd.DataFrame(
            {
                "timestamp": idx,
                "symbol": sym,
                "timeframe": timeframe,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )
    return out
