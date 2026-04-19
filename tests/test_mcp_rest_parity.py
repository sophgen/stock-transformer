"""M12 MCP unwrap + canonicalizer parity (no live network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_transformer.data.canonicalize import canonicalize_series
from stock_transformer.data.mcp_canonicalize import unwrap_mcp_alphavantage_payload


def test_daily_wrapped_fixture_matches_flat_series():
    path = Path(__file__).resolve().parent / "fixtures" / "av_raw" / "daily_wrapped.json"
    wrapped = json.loads(path.read_text())
    inner = unwrap_mcp_alphavantage_payload(wrapped)
    df = canonicalize_series(inner, symbol="ZZZ", timeframe="daily", adjusted=True)
    assert len(df) == 2
    assert df["symbol"].iloc[0] == "ZZZ"
    assert df["timeframe"].iloc[0] == "daily"


def test_unwrap_idempotent_on_rest_shaped_payload():
    path = Path(__file__).resolve().parent / "fixtures" / "av_raw" / "daily_wrapped.json"
    wrapped = json.loads(path.read_text())
    a = unwrap_mcp_alphavantage_payload(wrapped)
    b = unwrap_mcp_alphavantage_payload(a)
    df_a = canonicalize_series(a, symbol="X", timeframe="daily", adjusted=True)
    df_b = canonicalize_series(b, symbol="X", timeframe="daily", adjusted=True)
    pd.testing.assert_frame_equal(df_a, df_b)
