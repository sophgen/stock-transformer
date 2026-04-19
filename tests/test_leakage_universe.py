"""M7a leakage-safe universe dataset checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_transformer.backtest.walkforward import WalkForwardConfig, assert_fold_chronology, generate_folds
from stock_transformer.data.align import align_universe_ohlcv
from stock_transformer.data.synthetic import synthetic_universe_candles
from stock_transformer.features.scaling import TrainOnlyScaler
from stock_transformer.features.universe_tensor import build_universe_samples


@pytest.fixture
def universe_panel():
    symbols = ("MSTR", "IBIT", "COIN", "QQQ")
    candles = synthetic_universe_candles(n_bars=400, symbols=symbols, timeframe="daily", seed=7)
    panel, close = align_universe_ohlcv(candles, symbols)
    return panel, close, symbols


def test_features_do_not_reference_future(universe_panel):
    panel, close, symbols = universe_panel
    X0, mask0, _, _, _, end_row = build_universe_samples(
        panel,
        symbols,
        close,
        lookback=32,
        min_coverage_symbols=3,
        label_mode="cross_sectional_return",
    )
    idx = X0.shape[0] // 2
    t_row = int(end_row[idx])
    rng = np.random.default_rng(0)
    panel2 = panel.copy()
    for r in range(t_row + 1, len(panel2)):
        for c in panel2.columns:
            if c == "timestamp":
                continue
            panel2.iloc[r, panel2.columns.get_loc(c)] = float(rng.standard_normal() * 10 + 50)
    close2 = panel2[[f"close__{s}" for s in symbols]].to_numpy(dtype=np.float64)
    X1, mask1, *_ = build_universe_samples(
        panel2,
        symbols,
        close2,
        lookback=32,
        min_coverage_symbols=3,
        label_mode="cross_sectional_return",
    )
    np.testing.assert_array_equal(X0[idx], X1[idx])
    np.testing.assert_array_equal(mask0[idx], mask1[idx])


def test_label_uses_only_t_to_tplus1(universe_panel):
    panel, close, symbols = universe_panel
    X0, _, y0, _, _, end_row = build_universe_samples(
        panel,
        symbols,
        close,
        lookback=32,
        min_coverage_symbols=3,
        label_mode="cross_sectional_return",
    )
    idx = min(10, X0.shape[0] - 1)
    t_row = int(end_row[idx])
    panel2 = panel.copy()
    fut_row = t_row + 2
    if fut_row < len(panel2):
        for sym in symbols:
            panel2.loc[fut_row, f"close__{sym}"] = 999.0
    close2 = panel2[[f"close__{s}" for s in symbols]].to_numpy(dtype=np.float64)
    _, _, y1, *_ = build_universe_samples(
        panel2,
        symbols,
        close2,
        lookback=32,
        min_coverage_symbols=3,
        label_mode="cross_sectional_return",
    )
    np.testing.assert_array_equal(y0[idx], y1[idx])


def test_fold_boundaries_monotonic(universe_panel):
    panel, close, symbols = universe_panel
    _, _, _, _, ts, _ = build_universe_samples(
        panel,
        symbols,
        close,
        lookback=32,
        min_coverage_symbols=3,
        label_mode="cross_sectional_return",
    )
    n = len(ts)
    wf = WalkForwardConfig(train_bars=80, val_bars=30, test_bars=30, step_bars=30)
    folds = generate_folds(n, wf)
    for f in folds:
        assert_fold_chronology(ts, f)
        assert ts.iloc[f.train.stop - 1] < ts.iloc[f.val.start]
        assert ts.iloc[f.val.stop - 1] < ts.iloc[f.test.start]


def test_pit_universe_membership(universe_panel):
    panel, close, symbols = universe_panel
    panel2 = panel.copy()
    panel2.loc[0:99, ["open__IBIT", "high__IBIT", "low__IBIT", "close__IBIT", "volume__IBIT"]] = np.nan
    close2 = panel2[[f"close__{s}" for s in symbols]].to_numpy(dtype=np.float64)
    _, mask, _, _, _, end_row_pit = build_universe_samples(
        panel2,
        symbols,
        close2,
        lookback=32,
        min_coverage_symbols=3,
        label_mode="cross_sectional_return",
    )
    sym_i = symbols.index("IBIT")
    for i in range(mask.shape[0]):
        if int(end_row_pit[i]) < 100:
            assert mask[i, :, sym_i].all()


def test_coverage_drop(universe_panel):
    panel, close, symbols = universe_panel
    panel2 = panel.copy()
    mid = slice(len(panel2) // 3, 2 * len(panel2) // 3)
    for sym in symbols[2:]:
        panel2.loc[mid, [f"open__{sym}", f"high__{sym}", f"low__{sym}", f"close__{sym}", f"volume__{sym}"]] = np.nan
    close2 = panel2[[f"close__{s}" for s in symbols]].to_numpy(dtype=np.float64)
    X_hi, *_ = build_universe_samples(
        panel2,
        symbols,
        close2,
        lookback=16,
        min_coverage_symbols=3,
        label_mode="cross_sectional_return",
    )
    X_lo, *_ = build_universe_samples(
        panel2,
        symbols,
        close2,
        lookback=16,
        min_coverage_symbols=2,
        label_mode="cross_sectional_return",
    )
    assert X_hi.shape[0] < X_lo.shape[0]


def test_deterministic_symbol_order():
    syms_a = ("MSTR", "IBIT", "COIN", "QQQ")
    syms_b = ("QQQ", "COIN", "IBIT", "MSTR")
    c1 = synthetic_universe_candles(200, syms_a, seed=3)
    c2 = {s: c1[s] for s in syms_b}
    p1, cl1 = align_universe_ohlcv(c1, syms_a)
    p2, cl2 = align_universe_ohlcv(c2, syms_b)
    perm = [syms_b.index(s) for s in syms_a]
    X1, m1, y1, r1, t1, e1 = build_universe_samples(
        p1, syms_a, cl1, lookback=16, min_coverage_symbols=3, label_mode="cross_sectional_return"
    )
    X2, m2, y2, r2, t2, e2 = build_universe_samples(
        p2, syms_b, cl2, lookback=16, min_coverage_symbols=3, label_mode="cross_sectional_return"
    )
    X2r = X2[..., perm, :]
    m2r = m2[..., perm]
    y2r = y2[:, perm]
    np.testing.assert_allclose(X1, X2r, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(m1, m2r)
    np.testing.assert_allclose(y1, y2r, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(e1, e2)
    pd.testing.assert_series_equal(t1, t2)


def test_train_scaling_fit_on_train_only(universe_panel):
    panel, close, symbols = universe_panel
    X, mask, *_ = build_universe_samples(
        panel,
        symbols,
        close,
        lookback=24,
        min_coverage_symbols=3,
        label_mode="cross_sectional_return",
    )
    n = X.shape[0]
    tr = slice(0, n // 2)
    va = slice(n // 2, n)
    s1 = TrainOnlyScaler()
    s1.fit(X[tr], mask[tr])
    s2 = TrainOnlyScaler()
    s2.fit(X, mask)
    out1 = s1.transform(X[va])
    out2 = s2.transform(X[va])
    assert not np.allclose(out1, out2)


def test_target_symbol_not_required_live(universe_panel):
    panel, close, symbols = universe_panel
    panel2 = panel.copy()
    panel2.loc[200:250, ["open__MSTR", "high__MSTR", "low__MSTR", "close__MSTR", "volume__MSTR"]] = np.nan
    close2 = panel2[[f"close__{s}" for s in symbols]].to_numpy(dtype=np.float64)
    _, _, y, *_ = build_universe_samples(
        panel2,
        symbols,
        close2,
        lookback=16,
        min_coverage_symbols=2,
        label_mode="cross_sectional_return",
    )
    assert y.shape[0] > 0
    m_ix = symbols.index("MSTR")
    assert np.any(np.isnan(y[:, m_ix]))
