"""
Alpha Vantage OHLCV ingestion.

Mirrors MCP tools `TIME_SERIES_INTRADAY`, `TIME_SERIES_DAILY` / `_ADJUSTED`,
`TIME_SERIES_MONTHLY` / `_ADJUSTED` via the public REST API (same data as Cursor MCP).

Set ``ALPHAVANTAGE_API_KEY`` in the environment (or pass ``api_key``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from stock_transformer.data.cache_paths import canonical_candles_path, raw_response_path
from stock_transformer.data.canonicalize import (
    av_error_message,
    canonicalize_intraday,
    canonicalize_series,
)

BASE_URL = "https://www.alphavantage.co/query"

# Minimal delay between HTTP calls to respect free-tier rate limits
DEFAULT_MIN_INTERVAL_SEC = 12.0


class AlphaVantageClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: str | Path = "data",
        min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC,
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self.cache_root = Path(cache_dir)
        self.min_interval_sec = min_interval_sec
        self._last_call = 0.0

    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        self._last_call = time.monotonic()

    def query(
        self,
        function: str,
        params: dict[str, Any],
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "Missing API key: set ALPHAVANTAGE_API_KEY or pass api_key= to AlphaVantageClient."
            )
        full = {"function": function, "apikey": self.api_key, **params}
        raw_path = raw_response_path(self.cache_root, function, full)
        if use_cache and raw_path.exists():
            return json.loads(raw_path.read_text())

        self._throttle()
        resp = requests.get(BASE_URL, params=full, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        err = av_error_message(payload)
        if err:
            raise RuntimeError(f"Alpha Vantage API: {err}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload))
        return payload


def fetch_candles_for_timeframe(
    client: AlphaVantageClient,
    symbol: str,
    timeframe: str,
    *,
    use_adjusted_daily: bool = True,
    use_adjusted_monthly: bool = True,
    intraday_month: str | None = None,
    intraday_extended_hours: bool = False,
    intraday_outputsize: str = "full",
    daily_outputsize: str = "full",
    use_cache: bool = True,
    force_refresh_canonical: bool = False,
) -> pd.DataFrame:
    """
    Fetch and return canonical candles for one timeframe string.

    ``timeframe`` values: ``1min``, ``5min``, ``15min``, ``30min``, ``60min``,
    ``daily``, ``monthly``.
    """
    symbol = symbol.upper()
    tf = timeframe.lower()

    if tf in {"1min", "5min", "15min", "30min", "60min"}:
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": tf,
            "adjusted": "true",
            "extended_hours": "true" if intraday_extended_hours else "false",
            "outputsize": intraday_outputsize,
            "datatype": "json",
        }
        if intraday_month:
            params["month"] = intraday_month
        payload = client.query("TIME_SERIES_INTRADAY", params, use_cache=use_cache)
        df = canonicalize_intraday(payload, symbol=symbol, timeframe=tf)
        tag = f"intraday_{tf}_{intraday_month or 'latest'}"

    elif tf == "daily":
        fn = "TIME_SERIES_DAILY_ADJUSTED" if use_adjusted_daily else "TIME_SERIES_DAILY"
        params = {"symbol": symbol, "outputsize": daily_outputsize, "datatype": "json"}
        payload = client.query(fn, params, use_cache=use_cache)
        df = canonicalize_series(
            payload, symbol=symbol, timeframe="daily", adjusted=use_adjusted_daily
        )
        tag = "daily_adj" if use_adjusted_daily else "daily_raw"

    elif tf == "monthly":
        fn = "TIME_SERIES_MONTHLY_ADJUSTED" if use_adjusted_monthly else "TIME_SERIES_MONTHLY"
        params = {"symbol": symbol, "datatype": "json"}
        payload = client.query(fn, params, use_cache=use_cache)
        df = canonicalize_series(
            payload, symbol=symbol, timeframe="monthly", adjusted=use_adjusted_monthly
        )
        tag = "monthly_adj" if use_adjusted_monthly else "monthly_raw"

    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    out_path = canonical_candles_path(client.cache_root, symbol, tf, tag)
    if force_refresh_canonical or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df
