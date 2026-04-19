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
