"""Cross-sectional labels use only t and t+1; median demeaning is contemporaneous."""

from __future__ import annotations

import numpy as np
import pytest

from stock_transformer.labels.cross_sectional import (
    bucket_labels_by_quantile,
    cross_sectional_targets,
    raw_returns_forward,
)


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


def test_bucket_labels_encoding_2_is_top():
    """2 = top bucket_q fraction (best return), 0 = bottom, 1 = middle."""
    # 6 symbols; q=0.33 → k=1 top and k=1 bottom
    values = np.array([[0.01, 0.02, 0.03, -0.01, -0.02, -0.03]], dtype=np.float64)
    out = bucket_labels_by_quantile(values, q=0.33)
    assert out.shape == values.shape
    # highest value (0.03, col 2) → bucket 2
    assert float(out[0, 2]) == 2.0, f"top return should be bucket 2, got {out[0, 2]}"
    # lowest value (-0.03, col 5) → bucket 0
    assert float(out[0, 5]) == 0.0, f"bottom return should be bucket 0, got {out[0, 5]}"
    # middle values → bucket 1
    for col in (1, 3, 4):
        assert float(out[0, col]) == 1.0, f"col {col} should be middle bucket 1, got {out[0, col]}"


def test_bucket_labels_nan_for_fewer_than_3_peers():
    """Rows with fewer than 3 finite values produce all NaN."""
    values = np.array([[0.1, np.nan, np.nan, np.nan]], dtype=np.float64)
    out = bucket_labels_by_quantile(values, q=0.33)
    assert np.isnan(out).all()


def test_bucket_labels_nan_preserved():
    """NaN in input → NaN in output even when other entries are bucketed."""
    values = np.array([[0.3, 0.2, np.nan, 0.1, -0.1, -0.2]], dtype=np.float64)
    out = bucket_labels_by_quantile(values, q=0.33)
    assert np.isnan(out[0, 2])
    # highest finite value (0.3) → bucket 2
    assert float(out[0, 0]) == 2.0


def test_bucket_labels_invalid_q():
    with pytest.raises(ValueError):
        bucket_labels_by_quantile(np.ones((1, 4)), q=0.6)
