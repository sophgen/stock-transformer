"""Canonical data key uniqueness."""

from stock_transformer.data.synthetic import synthetic_random_walk_candles


def test_no_duplicate_timestamps():
    df = synthetic_random_walk_candles(n=200, seed=99)
    keys = df["symbol"] + "|" + df["timeframe"] + "|" + df["timestamp"].astype(str)
    assert keys.nunique() == len(df)
    assert df["timestamp"].is_monotonic_increasing
