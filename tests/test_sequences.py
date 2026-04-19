"""Sequence builder and no-lookahead guarantees."""

import numpy as np

from stock_transformer.data.synthetic import synthetic_random_walk_candles
from stock_transformer.features.sequences import (
    build_direction_labels,
    build_feature_matrix,
    build_windows,
    validate_no_lookahead,
)


def test_direction_labels():
    closes = np.array([10.0, 11.0, 10.5, 12.0])
    y = build_direction_labels(closes)
    assert y[0] == 1.0
    assert y[1] == 0.0
    assert y[2] == 1.0
    assert np.isnan(y[3])


def test_windows_no_future_features_in_window():
    df = synthetic_random_walk_candles(n=100, seed=1)
    lookback = 10
    aligned, X = build_feature_matrix(df)
    y = build_direction_labels(aligned["close"].to_numpy())
    X_win, y_win, end_idx, _ = build_windows(X, y, aligned["timestamp"], lookback)
    validate_no_lookahead(aligned, end_idx, lookback)
    assert X_win.shape[1] == lookback
    assert len(y_win) == X_win.shape[0]
    # Last window end index must leave room for t+1
    assert end_idx.max() < len(aligned) - 1


def test_window_uses_only_past_candles():
    df = synthetic_random_walk_candles(n=50, seed=2)
    lookback = 5
    aligned, X = build_feature_matrix(df)
    y = build_direction_labels(aligned["close"].to_numpy())
    X_win, _, end_idx, _ = build_windows(X, y, aligned["timestamp"], lookback)
    t_end = int(end_idx[10])
    w = X_win[10]
    # Feature row at last position must match X row t_end
    np.testing.assert_allclose(w[-1], X[t_end])
