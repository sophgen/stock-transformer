"""Fetch daily OHLCV candles from Alpha Vantage and align into a multi-ticker panel."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_MIN_INTERVAL_SEC = 12.0
DEFAULT_QUERY_RETRIES = 5


def _slug(params: dict) -> str:
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _raw_path(cache_root: Path, function: str, params: dict) -> Path:
    d = cache_root / "raw" / function.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_slug(params)}.json"


def _is_error(payload: dict[str, Any]) -> str | None:
    for key in ("Error Message", "Note", "Information"):
        if key in payload:
            return str(payload[key])
    return None


def _is_rate_limit(payload: dict[str, Any]) -> bool:
    note = payload.get("Note") or payload.get("Information")
    if not isinstance(note, str):
        return False
    lower = note.lower()
    return "api call frequency" in lower or "5 calls per minute" in lower


class AlphaVantageClient:
    def __init__(self, cache_dir: str | Path = "data") -> None:
        self.api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self.cache_root = Path(cache_dir)
        self.min_interval = float(
            os.environ.get("ALPHAVANTAGE_MIN_INTERVAL_SEC", DEFAULT_MIN_INTERVAL_SEC)
        )
        self.retries = int(
            os.environ.get("ALPHAVANTAGE_QUERY_RETRIES", DEFAULT_QUERY_RETRIES)
        )
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def query(self, function: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "Missing ALPHAVANTAGE_API_KEY. "
                "Copy .env.example to .env and set your key."
            )
        full = {"function": function, "apikey": self.api_key, **params}
        path = _raw_path(self.cache_root, function, full)
        if path.exists():
            return json.loads(path.read_text())

        import requests

        for attempt in range(max(1, self.retries)):
            self._throttle()
            resp = requests.get(BASE_URL, params=full, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            err = _is_error(payload)
            if not err:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload))
                return payload
            if attempt + 1 < self.retries and _is_rate_limit(payload):
                time.sleep(self.min_interval * (2 ** attempt))
                continue
            raise RuntimeError(f"Alpha Vantage API error: {err}")
        raise RuntimeError("Exhausted retries")


def _parse_daily(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse AV daily adjusted JSON into a DataFrame."""
    series_key = None
    for k in payload:
        if isinstance(k, str) and k.startswith("Time Series"):
            series_key = k
            break
    if not series_key:
        raise ValueError(f"No time series in response: {list(payload.keys())}")

    rows = []
    for date_str, fields in payload[series_key].items():
        rows.append({
            "timestamp": pd.Timestamp(date_str),
            "open": float(fields["1. open"]),
            "high": float(fields["2. high"]),
            "low": float(fields["3. low"]),
            "close": float(fields["5. adjusted close"]),
            "volume": float(fields.get("6. volume", fields.get("5. volume", 0)) or 0),
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_universe(
    symbols: list[str],
    cache_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    """Fetch daily adjusted candles for each symbol, return {symbol: DataFrame}."""
    client = AlphaVantageClient(cache_dir=cache_dir)
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym = sym.upper()
        print(f"  Fetching {sym}...")
        params = {"symbol": sym, "outputsize": "full", "datatype": "json"}
        payload = client.query("TIME_SERIES_DAILY_ADJUSTED", params)
        result[sym] = _parse_daily(payload)
    return result


def align_universe(candles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Inner-join all symbols on date, return aligned DataFrames sharing the same index."""
    date_sets = [set(df["timestamp"]) for df in candles.values()]
    common = sorted(set.intersection(*date_sets))
    if not common:
        raise ValueError("No overlapping dates across symbols")

    common_set = set(common)
    aligned: dict[str, pd.DataFrame] = {}
    for sym, df in candles.items():
        mask = df["timestamp"].isin(common_set)
        aligned[sym] = df[mask].sort_values("timestamp").reset_index(drop=True)

    print(f"  Aligned {len(candles)} symbols on {len(common)} common trading days")
    print(f"  Date range: {common[0].date()} to {common[-1].date()}")
    return aligned
