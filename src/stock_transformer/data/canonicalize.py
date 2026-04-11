"""Normalize Alpha Vantage JSON into canonical OHLCV candles."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _find_series_key(payload: dict[str, Any]) -> str | None:
    for k in payload:
        if isinstance(k, str) and k.startswith("Time Series"):
            return k
    return None


def canonicalize_intraday(
    payload: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Parse intraday JSON into canonical columns."""
    key = _find_series_key(payload)
    if not key:
        raise ValueError(f"No time series in payload: keys={list(payload.keys())}")
    series = payload[key]
    rows = []
    for ts, ohlcv in series.items():
        rows.append(
            {
                "timestamp": pd.Timestamp(ts),
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "open": float(ohlcv["1. open"]),
                "high": float(ohlcv["2. high"]),
                "low": float(ohlcv["3. low"]),
                "close": float(ohlcv["4. close"]),
                "volume": float(ohlcv.get("5. volume", 0) or 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def canonicalize_series(
    payload: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
    adjusted: bool = False,
) -> pd.DataFrame:
    """Parse daily/weekly/monthly JSON (raw or adjusted) into canonical columns."""
    key = _find_series_key(payload)
    if not key:
        raise ValueError(f"No time series in payload: keys={list(payload.keys())}")
    series = payload[key]
    rows = []
    for date_str, fields in series.items():
        # Daily keys are dates; monthly may be 'YYYY-MM-DD'
        ts = pd.Timestamp(date_str)
        if adjusted:
            close = float(fields["5. adjusted close"])
        else:
            close = float(fields["4. close"])
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "open": float(fields["1. open"]),
                "high": float(fields["2. high"]),
                "low": float(fields["3. low"]),
                "close": close,
                "volume": float(fields.get("6. volume", fields.get("5. volume", 0)) or 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def av_error_message(payload: dict[str, Any]) -> str | None:
    if "Error Message" in payload:
        return str(payload["Error Message"])
    if "Note" in payload:
        return str(payload["Note"])
    if "Information" in payload:
        return str(payload["Information"])
    return None
