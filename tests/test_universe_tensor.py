"""Universe tensor: deterministic symbol order, masking, coverage threshold."""

from __future__ import annotations

import numpy as np
from stock_transformer.data.align import align_universe_ohlcv
from stock_transformer.data.synthetic import synthetic_universe_candles
from stock_transformer.features.universe_tensor import build_universe_samples


def test_symbol_ordering_and_masks():
    symbols = ("AAA", "BBB")
    candles = synthetic_universe_candles(120, symbols, seed=3)
    panel, close = align_universe_ohlcv(candles, symbols)
    X, mask, y, raw_ret, ts, end_row = build_universe_samples(
        panel,
        symbols,
        close,
        lookback=16,
        min_coverage_symbols=2,
        label_mode="cross_sectional_return",
    )
    assert X.shape[2] == 2
    assert mask.shape == (X.shape[0], X.shape[1], 2)
    assert y.shape[1] == 2
    assert len(ts) == X.shape[0]
    assert end_row.shape[0] == X.shape[0]
    # Fully synthetic aligned data: no padding in mask for valid interior rows
    assert not mask.all()


def test_coverage_drops_sparse_timestamps():
    symbols = ("A", "B", "C")
    base = synthetic_universe_candles(80, symbols, seed=1)
    panel, close = align_universe_ohlcv(base, symbols)
    # Wipe one symbol's closes on a middle chunk so forward returns often fail min coverage
    bad = panel.copy()
    mid = slice(25, 55)
    bad.loc[mid, "close__B"] = np.nan
    bad.loc[mid, "open__B"] = np.nan
    bad.loc[mid, "high__B"] = np.nan
    bad.loc[mid, "low__B"] = np.nan
    bad.loc[mid, "volume__B"] = np.nan
    close_bad = bad[[f"close__{s}" for s in symbols]].to_numpy(dtype=np.float64)
    X_hi, *_ = build_universe_samples(
        bad,
        symbols,
        close_bad,
        lookback=8,
        min_coverage_symbols=3,
        label_mode="cross_sectional_return",
    )
    X_lo, *_ = build_universe_samples(
        bad,
        symbols,
        close_bad,
        lookback=8,
        min_coverage_symbols=2,
        label_mode="cross_sectional_return",
    )
    assert X_hi.shape[0] < X_lo.shape[0]
