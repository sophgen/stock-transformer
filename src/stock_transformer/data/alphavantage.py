"""
Alpha Vantage OHLCV ingestion.

Mirrors MCP tools ``TIME_SERIES_INTRADAY``, ``TIME_SERIES_DAILY`` / ``_ADJUSTED``,
``TIME_SERIES_WEEKLY`` / ``_ADJUSTED``,
``TIME_SERIES_MONTHLY`` / ``_ADJUSTED`` via the public REST API.

Set ``ALPHAVANTAGE_API_KEY`` in the environment (or pass ``api_key``).

Intraday history beyond the latest window requires one API call per calendar month
(``month=YYYY-MM``). Pass ``intraday_months`` or a single ``intraday_month``; results
are merged and deduplicated by timestamp.

Premium / higher throughput: set ``ALPHAVANTAGE_MIN_INTERVAL_SEC`` (default ``12``)
or pass ``min_interval_sec`` to :class:`AlphaVantageClient`. Rate-limit ``Note``
responses trigger automatic retries (``ALPHAVANTAGE_QUERY_RETRIES``, default ``5``).
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
DEFAULT_QUERY_RETRIES = 5
_ENV_MIN_INTERVAL = "ALPHAVANTAGE_MIN_INTERVAL_SEC"
_ENV_QUERY_RETRIES = "ALPHAVANTAGE_QUERY_RETRIES"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _is_rate_limit_note(payload: dict[str, Any]) -> bool:
    note = payload.get("Note") or payload.get("Information")
    if not isinstance(note, str):
        return False
    lower = note.lower()
    return (
        "thank you for using alpha vantage" in lower
        or "api call frequency" in lower
        or "5 calls per minute" in lower
        or "premium endpoint" in lower
    )


def merge_intraday_canonical_frames(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate intraday canonical dataframes and dedupe by ``timestamp`` (last wins)."""
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


def _resolved_intraday_months(
    intraday_month: str | None,
    intraday_months: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    """Return ``None`` for the default single call (latest window), else YYYY-MM list."""
    if intraday_months:
        out = [str(m).strip() for m in intraday_months if str(m).strip()]
        return out or None
    if intraday_month and str(intraday_month).strip():
        return [str(intraday_month).strip()]
    return None


class AlphaVantageClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: str | Path = "data",
        min_interval_sec: float | None = None,
        query_retries: int | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self.cache_root = Path(cache_dir)
        self.min_interval_sec = (
            float(min_interval_sec) if min_interval_sec is not None else _env_float(_ENV_MIN_INTERVAL, DEFAULT_MIN_INTERVAL_SEC)
        )
        self.query_retries = (
            int(query_retries) if query_retries is not None else _env_int(_ENV_QUERY_RETRIES, DEFAULT_QUERY_RETRIES)
        )
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

        for attempt in range(max(1, self.query_retries)):
            self._throttle()
            resp = requests.get(BASE_URL, params=full, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            err = av_error_message(payload)
            if not err:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(json.dumps(payload))
                return payload
            if attempt + 1 < self.query_retries and _is_rate_limit_note(payload):
                backoff = self.min_interval_sec * (2**attempt)
                time.sleep(backoff)
                continue
            raise RuntimeError(f"Alpha Vantage API: {err}")


def fetch_candles_for_timeframe(
    client: AlphaVantageClient,
    symbol: str,
    timeframe: str,
    *,
    use_adjusted_daily: bool = True,
    use_adjusted_weekly: bool = True,
    use_adjusted_monthly: bool = True,
    intraday_month: str | None = None,
    intraday_months: list[str] | tuple[str, ...] | None = None,
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

    For intraday, ``intraday_months`` (if non-empty) takes precedence over
    ``intraday_month``. Each month issues a separate API request; months are merged
    and deduplicated by timestamp.

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
        months = _resolved_intraday_months(intraday_month, intraday_months)
        base_params: dict[str, Any] = {
            "symbol": symbol,
            "interval": tf,
            "adjusted": "true",
            "extended_hours": "true" if intraday_extended_hours else "false",
            "outputsize": intraday_outputsize,
            "datatype": "json",
        }
        if months:
            frames: list[pd.DataFrame] = []
            for m in months:
                params = {**base_params, "month": m}
                payload = client.query("TIME_SERIES_INTRADAY", params, use_cache=use_cache)
                if str(data_source).lower() == "mcp":
                    payload = unwrap_mcp_alphavantage_payload(payload)
                frames.append(canonicalize_intraday(payload, symbol=symbol, timeframe=tf))
            df = merge_intraday_canonical_frames(frames)
            tag = f"intraday_{tf}_{'_'.join(months)}"
        else:
            payload = client.query("TIME_SERIES_INTRADAY", base_params, use_cache=use_cache)
            if str(data_source).lower() == "mcp":
                payload = unwrap_mcp_alphavantage_payload(payload)
            df = canonicalize_intraday(payload, symbol=symbol, timeframe=tf)
            tag = f"intraday_{tf}_latest"

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
    intraday_months: list[str] | tuple[str, ...] | None = None,
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
            intraday_months=intraday_months,
            intraday_extended_hours=intraday_extended_hours,
            intraday_outputsize=intraday_outputsize,
            daily_outputsize=daily_outputsize,
            use_cache=use_cache,
            force_refresh_canonical=force_refresh_canonical,
            store=store,
            data_source=data_source,
        )
    return out
