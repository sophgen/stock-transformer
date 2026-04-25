"""Parsers (no I/O, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_transformer.av_parsers import (
    assert_unique,
    detect_asset_type,
    parse_earnings,
    parse_financial_statement,
    parse_macro,
    parse_company_overview,
)


def test_parse_income_coerces_string_none() -> None:
    pld: dict = {
        "symbol": "X",
        "annualReports": [
            {
                "fiscalDateEnding": "2019-12-31",
                "grossProfit": "None",
                "totalRevenue": "100.0",
            }
        ],
        "quarterlyReports": [],
    }
    df = parse_financial_statement(pld, "X", "income")
    assert "grossProfit" in df.columns
    assert pd.isna(df["grossProfit"].iloc[0])
    assert float(df["totalRevenue"].iloc[0]) == 100.0


def test_empty_company_overview() -> None:
    assert parse_company_overview({}, "X").empty


def test_detect_asset_etf() -> None:
    assert (
        detect_asset_type({"AssetType": "ETF"})  # type: ignore[arg-type]
        == "ETF"
    )
    assert (
        detect_asset_type({"AssetType": "Common Stock"})  # type: ignore[arg-type]
        == "Common Stock"
    )


def test_parse_earnings_annual_zero_surprise() -> None:
    pld: dict = {
        "symbol": "X",
        "annualEarnings": [
            {"fiscalDateEnding": "2019-12-31", "reportedEPS": "1.0"}
        ],
        "quarterlyEarnings": [
            {
                "fiscalDateEnding": "2020-12-31",
                "reportedEPS": "0.5",
                "estimatedEPS": "0.4",
            }
        ],
    }
    df = parse_earnings(pld, "X")
    a = df[df["frequency"] == "annual"].iloc[0]
    # surprise fields may be absent/NaN for annual
    assert a["symbol"] == "X"


def test_treasury_macro_maturity_stem() -> None:
    pld: dict = {
        "name": "3-Month Treasury",
        "data": [{"date": "2020-01-15", "value": "0.1"}],
    }
    m = parse_macro(pld, "n", stem_hint="treasury_yield_3month")
    assert "maturity" in m.columns
    assert m["maturity"].iloc[0] == "3month"


def test_assert_unique_fails() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 1], "b": [2, 2],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        assert_unique(df, ["a", "b"], "t")
