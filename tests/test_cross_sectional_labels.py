"""Cross-sectional labels use only t and t+1; median demeaning is contemporaneous."""

from __future__ import annotations

import numpy as np

from stock_transformer.labels.cross_sectional import cross_sectional_targets, raw_returns_forward


def test_raw_returns_forward_no_future_leak_shape():
    close = np.array(
        [
            [100.0, 50.0],
            [102.0, 49.0],
            [101.0, 51.0],
        ],
        dtype=np.float64,
    )
    r = raw_returns_forward(close)
    assert r.shape == close.shape
    exp0 = 102.0 / 100.0 - 1.0
    exp1 = 49.0 / 50.0 - 1.0
    assert np.isclose(r[0, 0], exp0)
    assert np.isclose(r[0, 1], exp1)
    assert np.isnan(r[-1]).all()


def test_cross_sectional_median_demeaning():
    raw = np.array(
        [
            [0.10, -0.10, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    y = cross_sectional_targets(raw, mode="cross_sectional_return")
    med = np.nanmedian(raw[0])
    assert np.isclose(med, 0.0)
    assert np.isclose(y[0, 0], 0.10)
    assert np.isclose(y[0, 1], -0.10)
    assert np.isnan(y[0, 2])


def test_raw_mode_passthrough():
    raw = np.ones((2, 2), dtype=np.float64) * 0.05
    y = cross_sectional_targets(raw, mode="raw_return")
    assert np.allclose(y, raw)
