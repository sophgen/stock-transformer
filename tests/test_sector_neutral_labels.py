"""M9b sector-neutral demeaning."""

from __future__ import annotations

import numpy as np

from stock_transformer.labels.cross_sectional import cross_sectional_targets


def test_sector_neutral_zeros_sector_mean():
    raw = np.array(
        [
            [0.1, 0.2, -0.1, -0.2],
            [0.05, 0.05, -0.05, -0.05],
        ],
        dtype=np.float64,
    )
    sectors = np.array(["A", "A", "B", "B"], dtype=object)
    y = cross_sectional_targets(raw, mode="sector_neutral_return", sectors=sectors)
    for i in range(raw.shape[0]):
        for sec in ("A", "B"):
            m = sectors == sec
            sub = y[i, m]
            sub = sub[np.isfinite(sub)]
            assert abs(float(np.mean(sub))) < 1e-10


def test_equal_weighted_is_nanmean_demean():
    raw = np.array([[0.0, 0.2, 0.4]], dtype=np.float64)
    y = cross_sectional_targets(raw, mode="equal_weighted_return")
    assert np.allclose(np.nanmean(y[0, np.isfinite(y[0])]), 0.0, atol=1e-10)
