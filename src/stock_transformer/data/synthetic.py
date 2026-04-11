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
