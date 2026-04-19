"""M9a cross-sectional numeric helpers."""

import numpy as np

from stock_transformer.features import cross_sectional as cs


def test_percentile_rank_shape_and_bounds():
    x = np.array([[1.0, 2.0, np.nan], [np.nan, 5.0, 5.0]], dtype=np.float64)
    p = cs.percentile_rank(x)
    assert p.shape == x.shape
    assert 0.0 < float(p[0, 0]) <= 1.0
    assert np.isnan(p[0, 2])


def test_zscore_row_degenerate():
    x = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    z = cs.zscore_cross_section(x)
    assert np.isnan(z).all()


def test_relative_strength_and_volume():
    ret = np.array([[0.1, -0.1, 0.2]], dtype=np.float64)
    rs = cs.relative_strength_vs_ew(ret)
    assert np.isclose(float(np.nanmean(rs[0, np.isfinite(rs[0])])), 0.0, atol=1e-10)
    vol = np.array([[1.0, 2.0, 4.0]], dtype=np.float64)
    rv = cs.relative_volume_vs_median(vol)
    assert np.isclose(float(rv[0, 1]), 1.0, atol=1e-10)


def test_trailing_simple_returns_shape_and_first_row_nan():
    close = np.array([[100.0, 50.0], [110.0, 55.0], [105.0, 60.0]], dtype=np.float64)
    r = cs.trailing_simple_returns(close)
    assert r.shape == close.shape
    assert np.isnan(r[0]).all()
    assert np.isclose(r[1, 0], 0.1)
    assert np.isclose(r[1, 1], 0.1)


def test_trailing_simple_returns_short_panel():
    close = np.array([[1.0]], dtype=np.float64).reshape(1, 1)
    r = cs.trailing_simple_returns(close)
    assert r.shape == (1, 1)
    assert np.isnan(r).all()


def test_rolling_volatility_logret_non_degenerate():
    # Monotone positive closes -> varying log returns -> positive std by row 3 with window 2
    close = np.array(
        [[1.0, 10.0], [2.0, 12.0], [1.5, 9.0], [3.0, 15.0], [2.0, 14.0]],
        dtype=np.float64,
    )
    v = cs.rolling_volatility_logret(close, window=3)
    assert v.shape == close.shape
    assert np.isnan(v[0]).all()
    assert np.any(np.isfinite(v[3]) & (v[3] > 1e-12))


def test_rolling_volatility_respects_invalid_closes():
    close = np.array([[1.0, 1.0], [np.nan, 2.0], [2.0, np.nan]], dtype=np.float64)
    v = cs.rolling_volatility_logret(close, window=5)
    assert v.shape == close.shape
    assert np.isnan(v[0]).all()
