"""Unit tests for per-timestamp ranking metrics (M7b Kendall + Spearman helpers)."""

from __future__ import annotations

import numpy as np

from stock_transformer.backtest.metrics import kendall_per_timestamp, spearman_per_timestamp


def test_kendall_per_timestamp_perfect_agreement():
    scores = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    y = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    tau = kendall_per_timestamp(scores, y, min_valid=3)
    assert tau.shape == (1,)
    assert np.isclose(float(tau[0]), 1.0, atol=1e-9)


def test_kendall_per_timestamp_reverse_order():
    scores = np.array([[3.0, 2.0, 1.0]], dtype=np.float64)
    y = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    tau = kendall_per_timestamp(scores, y, min_valid=3)
    assert np.isclose(float(tau[0]), -1.0, atol=1e-9)


def test_kendall_per_timestamp_insufficient_valid():
    scores = np.array([[1.0, np.nan, 3.0]], dtype=np.float64)
    y = np.array([[0.1, np.nan, 0.3]], dtype=np.float64)
    tau = kendall_per_timestamp(scores, y, min_valid=3)
    assert np.isnan(tau[0])


def test_kendall_per_timestamp_zero_variance_scores():
    scores = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
    y = np.array([[0.1, 0.5, 0.2]], dtype=np.float64)
    tau = kendall_per_timestamp(scores, y, min_valid=3)
    assert np.isnan(tau[0])


def test_kendall_matches_spearman_sign_on_monotonic():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((4, 6))
    y = scores + 0.1 * rng.standard_normal((4, 6))
    k = kendall_per_timestamp(scores, y, min_valid=4)
    sp = spearman_per_timestamp(scores, y, min_valid=4)
    for i in range(4):
        if np.isfinite(k[i]) and np.isfinite(sp[i]):
            assert (k[i] > 0) == (sp[i] > 0)
