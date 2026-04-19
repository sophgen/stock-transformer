"""M8 partitioned candle store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stock_transformer.data.store import CandleStore


def _sample_df():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["MSTR", "MSTR"],
            "timeframe": ["daily", "daily"],
            "open": [1.0, 1.1],
            "high": [1.2, 1.2],
            "low": [0.9, 1.0],
            "close": [1.1, 1.15],
            "volume": [100.0, 110.0],
        }
    )


@pytest.mark.parametrize("backend", ["csv", "parquet"])
def test_candle_store_roundtrip(tmp_path, backend):
    df0 = _sample_df()
    st = CandleStore(tmp_path, backend=backend)
    st.write("MSTR", "daily", df0)
    df1 = st.read("MSTR", "daily")
    assert df1 is not None
    pd.testing.assert_frame_equal(
        df0.sort_values("timestamp").reset_index(drop=True),
        df1.sort_values("timestamp").reset_index(drop=True),
        check_dtype=False,
    )
