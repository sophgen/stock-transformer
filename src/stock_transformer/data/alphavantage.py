"""
Alpha Vantage OHLCV ingestion.

Mirrors MCP tools ``TIME_SERIES_INTRADAY``, ``TIME_SERIES_DAILY`` / ``_ADJUSTED``,
``TIME_SERIES_WEEKLY`` / ``_ADJUSTED``,
``TIME_SERIES_MONTHLY`` / ``_ADJUSTED`` via the public REST API.

Set ``ALPHAVANTAGE_API_KEY`` in the environment (or pass ``api_key``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import requests

from stock_transformer.data.cache_paths import canonical_candles_path, raw_response_path
from stock_transformer.data.canonicalize import (
    av_error_message,
    canonicalize_intraday,
    canonicalize_series,
)
from stock_transformer.data.mcp_canonicalize import unwrap_mcp_alphavantage_payload
from stock_transformer.data.store import CandleStore

BASE_URL = "https://www.alphavantage.co/query"

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
            raise RuntimeError("Missing API key: set ALPHAVANTAGE_API_KEY or pass api_key= to AlphaVantageClient.")
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
    use_adjusted_weekly: bool = True,
    use_adjusted_monthly: bool = True,
    intraday_month: str | None = None,
    intraday_extended_hours: bool = False,
    intraday_outputsize: str = "full",
    daily_outputsize: str = "full",
    use_cache: bool = True,
    force_refresh_canonical: bool = False,
    store: str | None = None,
    data_source: str = "rest",
) -> pd.DataFrame:
    """Fetch and return canonical candles for one timeframe.

    Supported ``timeframe`` values: ``1min``, ``5min``, ``15min``, ``30min``,
    ``60min``, ``daily``, ``weekly``, ``monthly``.

    ``store``: ``\"csv\"`` or ``\"parquet\"`` for partitioned canonical layout under
    ``cache_dir/canonical/``; ``None`` keeps legacy ``processed/candles`` CSV.
    ``data_source``: ``\"rest\"`` (default) or ``\"mcp\"`` (unwrap nested tool JSON).
    """
    symbol = symbol.upper()
    tf = timeframe.lower()
    st = store
    if st in ("csv", "parquet"):
        bk = cast(Literal["csv", "parquet"], st)
        cstore = CandleStore(client.cache_root, backend=bk)
        if use_cache and not force_refresh_canonical:
            cached = cstore.read(symbol, tf)
            if cached is not None:
                return cached

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
        if str(data_source).lower() == "mcp":
            payload = unwrap_mcp_alphavantage_payload(payload)
        df = canonicalize_intraday(payload, symbol=symbol, timeframe=tf)
        tag = f"intraday_{tf}_{intraday_month or 'latest'}"

    elif tf == "daily":
        fn = "TIME_SERIES_DAILY_ADJUSTED" if use_adjusted_daily else "TIME_SERIES_DAILY"
        params = {"symbol": symbol, "outputsize": daily_outputsize, "datatype": "json"}
        payload = client.query(fn, params, use_cache=use_cache)
        if str(data_source).lower() == "mcp":
            payload = unwrap_mcp_alphavantage_payload(payload)
        df = canonicalize_series(payload, symbol=symbol, timeframe="daily", adjusted=use_adjusted_daily)
        tag = "daily_adj" if use_adjusted_daily else "daily_raw"

    elif tf == "weekly":
        fn = "TIME_SERIES_WEEKLY_ADJUSTED" if use_adjusted_weekly else "TIME_SERIES_WEEKLY"
        params = {"symbol": symbol, "datatype": "json"}
        payload = client.query(fn, params, use_cache=use_cache)
        if str(data_source).lower() == "mcp":
            payload = unwrap_mcp_alphavantage_payload(payload)
        df = canonicalize_series(payload, symbol=symbol, timeframe="weekly", adjusted=use_adjusted_weekly)
        tag = "weekly_adj" if use_adjusted_weekly else "weekly_raw"

    elif tf == "monthly":
        fn = "TIME_SERIES_MONTHLY_ADJUSTED" if use_adjusted_monthly else "TIME_SERIES_MONTHLY"
        params = {"symbol": symbol, "datatype": "json"}
        payload = client.query(fn, params, use_cache=use_cache)
        if str(data_source).lower() == "mcp":
            payload = unwrap_mcp_alphavantage_payload(payload)
        df = canonicalize_series(payload, symbol=symbol, timeframe="monthly", adjusted=use_adjusted_monthly)
        tag = "monthly_adj" if use_adjusted_monthly else "monthly_raw"

    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if st in ("csv", "parquet"):
        bk = cast(Literal["csv", "parquet"], st)
        CandleStore(client.cache_root, backend=bk).write(symbol, tf, df)
    else:
        out_path = canonical_candles_path(client.cache_root, symbol, tf, tag)
        if force_refresh_canonical or not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
    return df


def fetch_candles_for_universe(
    client: AlphaVantageClient,
    symbols: list[str] | tuple[str, ...],
    timeframe: str,
    *,
    use_adjusted_daily: bool = True,
    use_adjusted_weekly: bool = True,
    use_adjusted_monthly: bool = True,
    intraday_month: str | None = None,
    intraday_extended_hours: bool = False,
    intraday_outputsize: str = "full",
    daily_outputsize: str = "full",
    use_cache: bool = True,
    force_refresh_canonical: bool = False,
    store: str | None = None,
    data_source: str = "rest",
) -> dict[str, pd.DataFrame]:
    """Fetch canonical candles for every symbol, respecting client throttling."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym.upper()] = fetch_candles_for_timeframe(
            client,
            sym,
            timeframe,
            use_adjusted_daily=use_adjusted_daily,
            use_adjusted_weekly=use_adjusted_weekly,
            use_adjusted_monthly=use_adjusted_monthly,
            intraday_month=intraday_month,
            intraday_extended_hours=intraday_extended_hours,
            intraday_outputsize=intraday_outputsize,
            daily_outputsize=daily_outputsize,
            use_cache=use_cache,
            force_refresh_canonical=force_refresh_canonical,
            store=store,
            data_source=data_source,
        )
    return out
