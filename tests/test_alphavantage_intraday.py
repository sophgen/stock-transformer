"""Tests for intraday merge and month resolution (Alpha Vantage multi-month fetch)."""

from __future__ import annotations

import pandas as pd

from stock_transformer.data.alphavantage import (
    _resolved_intraday_months,
    merge_intraday_canonical_frames,
)


def test_merge_intraday_dedupes_timestamp() -> None:
    a = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-15 10:00:00", "2026-01-15 11:00:00"]),
            "symbol": ["X", "X"],
            "timeframe": ["60min", "60min"],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100.0, 200.0],
        }
    )
    b = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-15 11:00:00", "2026-01-16 10:00:00"]),
            "symbol": ["X", "X"],
            "timeframe": ["60min", "60min"],
            "open": [3.0, 4.0],
            "high": [3.0, 4.0],
            "low": [3.0, 4.0],
            "close": [3.0, 4.0],
            "volume": [300.0, 400.0],
        }
    )
    out = merge_intraday_canonical_frames([a, b])
    assert len(out) == 3
    row_11 = out.loc[out["timestamp"] == pd.Timestamp("2026-01-15 11:00:00")].iloc[0]
    assert float(row_11["close"]) == 3.0


def test_resolved_intraday_months_precedence() -> None:
    assert _resolved_intraday_months(None, ["2026-01", "2026-02"]) == ["2026-01", "2026-02"]
    assert _resolved_intraday_months("2026-03", ["2026-01", "2026-02"]) == ["2026-01", "2026-02"]
    assert _resolved_intraday_months("2026-03", None) == ["2026-03"]
    assert _resolved_intraday_months(None, None) is None
    assert _resolved_intraday_months(None, []) is None
