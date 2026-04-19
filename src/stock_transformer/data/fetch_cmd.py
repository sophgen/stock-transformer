"""Fetch OHLCV into local cache (used by ``stx fetch`` and optional scripts)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from stock_transformer.data.alphavantage import AlphaVantageClient, fetch_candles_for_universe

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE: tuple[str, ...] = ("MSTR", "IBIT", "COIN", "QQQ")


def fetch_universe_sample_data(
    cache_dir: str,
    symbols: Sequence[str],
    *,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download daily-adjusted bars for ``symbols`` into ``cache_dir`` (raw + canonical)."""
    use_cache = not refresh
    client = AlphaVantageClient(cache_dir=cache_dir)
    candles = fetch_candles_for_universe(
        client,
        list(symbols),
        "daily",
        use_adjusted_daily=True,
        daily_outputsize="full",
        use_cache=use_cache,
        force_refresh_canonical=refresh,
        store="csv",
        data_source="rest",
    )
    for sym in sorted(candles.keys()):
        df = candles[sym]
        ts = df["timestamp"]
        logger.info("%s: %s rows, %s → %s", sym, len(df), ts.min().date(), ts.max().date())
    logger.info("Cache root: %s", cache_dir)
    return candles
