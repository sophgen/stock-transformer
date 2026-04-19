"""M8 richer membership rows."""

from __future__ import annotations

import pandas as pd

from stock_transformer.data.universe import membership_table_from_panel


def test_membership_listings_and_sectors():
    ts = pd.date_range("2024-01-01", periods=5, freq="D")
    symbols = ("A", "B")
    listings = {"A": (pd.Timestamp("2024-01-02"), None), "B": (None, pd.Timestamp("2024-01-04"))}
    sec = {"A": "Tech", "B": "Fin"}
    cap = {"A": "large", "B": "mid"}
    rows = membership_table_from_panel(
        ts,
        symbols,
        listings=listings,
        sector_by_symbol=sec,
        market_cap_bucket_by_symbol=cap,
    )
    assert len(rows) == 2
    assert rows[0]["symbol"] == "A"
    assert rows[0]["sector"] == "Tech"
    assert rows[0]["market_cap_bucket"] == "large"
    assert rows[1]["symbol"] == "B"
