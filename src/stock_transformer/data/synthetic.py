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
    allowed = {tf for tf, *_ in _TF_SPEC}
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
