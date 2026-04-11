"""Walk-forward fold chronology."""

import pandas as pd

from stock_transformer.backtest.walkforward import (
    WalkForwardConfig,
    assert_fold_chronology,
    generate_folds,
)


def test_generate_folds_and_chronology():
    cfg = WalkForwardConfig(train_bars=20, val_bars=10, test_bars=10, step_bars=15)
    n = 100
    ts = pd.date_range("2020-01-01", periods=n, freq="h")
    folds = generate_folds(n, cfg)
    assert len(folds) >= 1
    for f in folds:
        assert f.train.stop == f.val.start
        assert f.val.stop == f.test.start
        assert_fold_chronology(ts, f)


def test_insufficient_windows_no_folds():
    cfg = WalkForwardConfig(train_bars=50, val_bars=50, test_bars=50, step_bars=10)
    assert generate_folds(100, cfg) == []
