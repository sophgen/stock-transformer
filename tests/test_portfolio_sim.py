"""M11: top-k portfolio simulation (turnover, costs, books)."""

from __future__ import annotations

import numpy as np
import pytest

from stock_transformer.backtest.portfolio_sim import (
    aggregate_portfolio_sim_folds,
    simulate_topk_portfolio,
)


def test_stable_scores_zero_turnover_after_first_rebalance():
    s = 4
    n = 5
    scores = np.tile(np.array([4.0, 3.0, 2.0, 1.0]), (n, 1))
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, size=(n, s))
    pad = np.zeros((n, s), dtype=bool)
    out = simulate_topk_portfolio(
        scores,
        r,
        pad_last=pad,
        book="long_only",
        top_k=2,
        transaction_cost_one_way_bps=0.0,
    )
    to = np.asarray(out["turnover"], dtype=float)
    m = np.isfinite(to)
    assert m.all()
    assert to[0] == pytest.approx(1.0)
    assert np.all(to[1:] == pytest.approx(0.0))


def test_transaction_cost_reduces_mean_net_vs_gross():
    n, s = 8, 4
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(n, s))
    r = rng.normal(0, 0.02, size=(n, s))
    pad = np.zeros((n, s), dtype=bool)
    out0 = simulate_topk_portfolio(
        scores,
        r,
        pad_last=pad,
        book="long_only",
        top_k=2,
        transaction_cost_one_way_bps=0.0,
        record_series=False,
    )
    out1 = simulate_topk_portfolio(
        scores,
        r,
        pad_last=pad,
        book="long_only",
        top_k=2,
        transaction_cost_one_way_bps=50.0,
        record_series=False,
    )
    assert out1["mean_cost"] > 0
    assert out1["mean_net_return"] < out0["mean_net_return"]


def test_long_short_top_bottom_k():
    # k=1: long highest score, short lowest; dollar-neutral weights ±0.5.
    scores = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=float)
    r = np.array([[0.02, 0.01, 0.01, 0.04]], dtype=float)
    pad = np.zeros((1, 4), dtype=bool)
    out = simulate_topk_portfolio(
        scores,
        r,
        pad_last=pad,
        book="long_short",
        top_k=1,
        transaction_cost_one_way_bps=0.0,
    )
    assert out["n_periods"] == 1
    # 0.5 * r[3] - 0.5 * r[0] = 0.5 * 0.04 - 0.5 * 0.02
    assert out["mean_gross_return"] == pytest.approx(0.01)


def test_padding_excludes_symbol():
    n, s = 2, 3
    scores = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    r = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    pad = np.array([[False, False, True], [False, False, True]])
    out = simulate_topk_portfolio(
        scores,
        r,
        pad_last=pad,
        book="long_only",
        top_k=2,
        transaction_cost_one_way_bps=0.0,
    )
    # Only first two names tradable; weights 0.5 each; return 0.5 * 0.1 + 0.5 * 0.2
    assert out["mean_gross_return"] == pytest.approx(0.15)


def test_aggregate_portfolio_sim_folds():
    rows = [
        {"fold_id": 0, "mean_net_return": 0.01, "n_periods": 10, "mean_turnover": 0.5},
        {"fold_id": 1, "mean_net_return": 0.02, "n_periods": 12, "mean_turnover": 0.6},
    ]
    agg = aggregate_portfolio_sim_folds(rows)
    assert agg["mean_net_return_mean"] == pytest.approx(0.015)
    assert agg["n_periods_mean"] == pytest.approx(11.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share shape"):
        simulate_topk_portfolio(
            np.zeros((2, 3)),
            np.zeros((2, 4)),
            pad_last=np.zeros((2, 3), dtype=bool),
            book="long_only",
            top_k=1,
            transaction_cost_one_way_bps=0.0,
            record_series=False,
        )
