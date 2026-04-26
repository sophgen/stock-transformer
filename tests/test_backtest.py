"""Tests for the backtesting engine (no torch / no network dependencies)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_transformer.backtest import (
    BacktestConfig,
    OHLC_CLOSE,
    OHLC_HIGH,
    OHLC_LOW,
    OHLC_OPEN,
    backtest,
    backtest_bracket,
    buy_and_hold,
    compare,
    compute_metrics,
    confidence_weighted,
    long_only,
    long_short,
    simulate_close_to_close,
    simulate_ohlc_bracket,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _make_logret(rng: np.random.Generator, n: int = 252) -> np.ndarray:
    """Random OHLC log-returns that satisfy low <= open/close <= high."""
    close = rng.normal(0.0005, 0.01, size=n)
    open_ = close + rng.normal(0.0, 0.003, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.005, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.005, size=n))
    return np.stack([open_, high, low, close], axis=1)


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


def test_long_short_signs_predictions() -> None:
    pred = np.array([
        [0, 0, 0,  0.01],
        [0, 0, 0, -0.02],
        [0, 0, 0,  0.0],
    ])
    pos = long_short()(pred)
    assert pos.tolist() == [1.0, -1.0, 0.0]


def test_long_short_threshold_excludes_small_predictions() -> None:
    pred = np.zeros((4, 4))
    pred[:, OHLC_CLOSE] = [0.001, -0.001, 0.005, -0.005]
    pos = long_short(threshold=0.002)(pred)
    assert pos.tolist() == [0.0, 0.0, 1.0, -1.0]


def test_long_only_never_shorts() -> None:
    pred = np.zeros((3, 4))
    pred[:, OHLC_CLOSE] = [0.01, -0.01, 0.0]
    pos = long_only()(pred)
    assert pos.tolist() == [1.0, 0.0, 0.0]


def test_confidence_weighted_clips_to_cap() -> None:
    pred = np.zeros((3, 4))
    pred[:, OHLC_CLOSE] = [0.001, 0.05, -0.10]   # 0.001*50=0.05, 0.05*50=2.5, -5
    pos = confidence_weighted(scale=50.0, cap=1.0)(pred)
    assert pos.tolist() == [0.05, 1.0, -1.0]


def test_buy_and_hold_is_always_long() -> None:
    pred = np.zeros((10, 4))
    assert buy_and_hold()(pred).tolist() == [1.0] * 10


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_empty_input_returns_nans() -> None:
    m = compute_metrics(np.array([]))
    assert m["n_days"] == 0.0
    assert np.isnan(m["sharpe"])
    assert np.isnan(m["max_drawdown"])
    assert m["ending_equity"] == 1.0


def test_metrics_zero_pnl_zero_volatility() -> None:
    m = compute_metrics(np.zeros(10))
    assert m["total_return"] == 0.0
    assert m["annualized_volatility"] == 0.0
    assert m["sharpe"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert m["hit_rate"] == 0.0


def test_metrics_total_return_matches_expected_growth() -> None:
    # Constant 1% daily log-return for 252 days → total ~ exp(2.52) - 1.
    pnl = np.full(252, 0.01)
    m = compute_metrics(pnl, starting_capital=1.0, annualization_factor=252)
    np.testing.assert_allclose(m["total_log_return"], 2.52)
    np.testing.assert_allclose(m["total_return"], np.expm1(2.52))
    np.testing.assert_allclose(m["cagr"], np.exp(0.01 * 252) - 1)
    # No volatility → Sharpe = 0 by convention (avoids divide-by-zero).
    assert m["sharpe"] == 0.0


def test_metrics_sharpe_matches_manual_formula(rng: np.random.Generator) -> None:
    pnl = rng.normal(0.001, 0.01, size=500)
    m = compute_metrics(pnl, annualization_factor=252)
    expected = pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(m["sharpe"], expected)


def test_metrics_max_drawdown_simple_case() -> None:
    # Equity multiplies by exp of cumsum.
    # log-returns 0, ln(0.5), ln(1.2) → equity 1.0, 0.5, 0.6.
    # Running peak is 1.0; max drawdown trough is 0.5 → DD = -0.5.
    pnl = np.array([0.0, np.log(0.5), np.log(1.2)])
    m = compute_metrics(pnl)
    np.testing.assert_allclose(m["max_drawdown"], -0.5)


def test_metrics_profit_factor_only_winners_returns_inf() -> None:
    pnl = np.array([0.01, 0.02, 0.005])
    m = compute_metrics(pnl)
    assert np.isinf(m["profit_factor"])


def test_metrics_with_positions_reports_exposure_and_turnover() -> None:
    pnl = np.array([0.01, -0.01, 0.0, 0.005])
    pos = np.array([1.0, -1.0, 0.0, 1.0])
    m = compute_metrics(pnl, positions=pos)
    # Active days: 1, -1, 1 → 3/4
    assert m["exposure"] == 0.75
    # Turnover: |1-0| + |-1-1| + |0-(-1)| + |1-0| = 1 + 2 + 1 + 1 = 5
    assert m["total_turnover"] == 5.0
    # hit_rate_trades: among the 3 active days, pnl values are
    #   [0.01, -0.01, 0.005] → 2 positive out of 3 = 2/3.
    np.testing.assert_allclose(m["hit_rate_trades"], 2.0 / 3.0)


# ---------------------------------------------------------------------------
# Close-to-close simulator
# ---------------------------------------------------------------------------


def test_close_to_close_matches_buy_and_hold_for_unit_long_no_costs() -> None:
    r = np.array([0.01, -0.005, 0.02, -0.001])
    cfg = BacktestConfig(cost_bps=0.0, slippage_bps=0.0)
    res = simulate_close_to_close(
        positions=np.ones_like(r),
        actual_close_logret=r,
        config=cfg,
    )
    np.testing.assert_allclose(res.pnl_log, r)
    np.testing.assert_allclose(res.equity[-1], cfg.starting_capital * np.exp(r.sum()))
    np.testing.assert_allclose(res.equity, res.benchmark_equity)


def test_close_to_close_short_position_inverts_pnl() -> None:
    r = np.array([0.01, -0.02, 0.03])
    cfg = BacktestConfig(cost_bps=0.0, slippage_bps=0.0)
    res = simulate_close_to_close(
        positions=-np.ones_like(r),
        actual_close_logret=r,
        config=cfg,
    )
    np.testing.assert_allclose(res.pnl_log, -r)


def test_close_to_close_costs_subtract_proportional_to_turnover() -> None:
    # Flip every day: +1, -1, +1 → turnover 1, 2, 2.
    pos = np.array([1.0, -1.0, 1.0])
    r = np.zeros_like(pos)  # zero gross so we measure pure cost
    cfg = BacktestConfig(cost_bps=10.0, slippage_bps=0.0)  # 10 bps = 0.001
    res = simulate_close_to_close(
        positions=pos, actual_close_logret=r, config=cfg,
    )
    # Cost per turnover = 10 bps = 0.001. Daily costs = -0.001 * [1,2,2].
    np.testing.assert_allclose(res.cost_log, [-0.001, -0.002, -0.002])
    np.testing.assert_allclose(res.turnover, [1.0, 2.0, 2.0])
    np.testing.assert_allclose(res.pnl_log, res.cost_log)


def test_backtest_uses_strategy_to_derive_positions() -> None:
    pred = np.zeros((4, 4))
    pred[:, OHLC_CLOSE] = [0.01, -0.01, 0.0, 0.02]
    actual = np.zeros_like(pred)
    actual[:, OHLC_CLOSE] = [0.005, -0.01, 0.0, 0.01]
    res = backtest(
        pred_logret=pred,
        actual_logret=actual,
        strategy=long_short(),
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
    )
    np.testing.assert_allclose(res.positions, [1.0, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(res.gross_pnl_log, [0.005, 0.01, 0.0, 0.01])


def test_backtest_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="must be"):
        backtest(
            pred_logret=np.zeros((10, 3)),
            actual_logret=np.zeros((10, 3)),
            strategy=long_short(),
        )


def test_backtest_dates_length_must_match() -> None:
    pred = np.zeros((3, 4))
    actual = np.zeros((3, 4))
    with pytest.raises(ValueError):
        backtest(
            pred_logret=pred,
            actual_logret=actual,
            strategy=long_short(),
            dates=["a", "b"],
        )


# ---------------------------------------------------------------------------
# OHLC bracket simulator
# ---------------------------------------------------------------------------


def _bracket_sample(open_lr, high_lr, low_lr, close_lr,
                     pred_high_lr, pred_low_lr) -> tuple[np.ndarray, np.ndarray]:
    """Build a single-sample (pred, actual) pair for bracket tests."""
    actual = np.array([[open_lr, high_lr, low_lr, close_lr]])
    pred = np.array([[0.0, pred_high_lr, pred_low_lr, 0.0]])
    return pred, actual


def test_bracket_long_take_profit_hit() -> None:
    # Predicted high 0.01, predicted low -0.01. Actual high reaches 0.02 → TP at 0.01.
    pred, actual = _bracket_sample(
        open_lr=0.0, high_lr=0.02, low_lr=-0.005, close_lr=0.015,
        pred_high_lr=0.01, pred_low_lr=-0.01,
    )
    res = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[1.0],
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
    )
    # Realized = exit - entry = 0.01 - 0.0 = 0.01.
    np.testing.assert_allclose(res.gross_pnl_log[0], 0.01)


def test_bracket_long_stop_loss_hit() -> None:
    pred, actual = _bracket_sample(
        open_lr=0.0, high_lr=0.005, low_lr=-0.02, close_lr=-0.01,
        pred_high_lr=0.01, pred_low_lr=-0.01,
    )
    res = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[1.0],
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
    )
    # Realized = -0.01 - 0.0 = -0.01.
    np.testing.assert_allclose(res.gross_pnl_log[0], -0.01)


def test_bracket_long_neither_hit_uses_actual_close() -> None:
    pred, actual = _bracket_sample(
        open_lr=0.0, high_lr=0.005, low_lr=-0.005, close_lr=0.003,
        pred_high_lr=0.01, pred_low_lr=-0.01,
    )
    res = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[1.0],
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
    )
    np.testing.assert_allclose(res.gross_pnl_log[0], 0.003)


def test_bracket_long_invalid_falls_through_to_close() -> None:
    # Predicted high BELOW entry (model is bearish). Bracket invalid → fall through to close.
    pred, actual = _bracket_sample(
        open_lr=0.0, high_lr=0.02, low_lr=-0.02, close_lr=0.01,
        pred_high_lr=-0.005, pred_low_lr=-0.02,
    )
    res = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[1.0],
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
    )
    np.testing.assert_allclose(res.gross_pnl_log[0], 0.01)


def test_bracket_long_both_hit_pessimistic_gives_stop_loss() -> None:
    pred, actual = _bracket_sample(
        open_lr=0.0, high_lr=0.02, low_lr=-0.02, close_lr=0.005,
        pred_high_lr=0.01, pred_low_lr=-0.01,
    )
    res_pess = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[1.0],
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
        tie_break="pessimistic",
    )
    res_opt = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[1.0],
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
        tie_break="optimistic",
    )
    res_mid = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[1.0],
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
        tie_break="midpoint",
    )
    np.testing.assert_allclose(res_pess.gross_pnl_log[0], -0.01)
    np.testing.assert_allclose(res_opt.gross_pnl_log[0],   0.01)
    np.testing.assert_allclose(res_mid.gross_pnl_log[0],   0.0)


def test_bracket_short_take_profit_hit() -> None:
    # Short: TP at predicted low (-0.01), SL at predicted high (0.01).
    pred, actual = _bracket_sample(
        open_lr=0.0, high_lr=0.005, low_lr=-0.02, close_lr=-0.015,
        pred_high_lr=0.01, pred_low_lr=-0.01,
    )
    res = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[-1.0],
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
    )
    # Short realized = entry - exit = 0 - (-0.01) = 0.01.
    np.testing.assert_allclose(res.gross_pnl_log[0], 0.01)


def test_bracket_flat_intent_yields_zero_pnl() -> None:
    pred, actual = _bracket_sample(0.0, 0.02, -0.02, 0.015, 0.01, -0.01)
    res = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[0.0],
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
    )
    assert res.gross_pnl_log[0] == 0.0
    assert res.cost_log[0] == 0.0


def test_bracket_costs_double_count_entry_and_exit() -> None:
    pred, actual = _bracket_sample(0.0, 0.005, -0.005, 0.003, 0.01, -0.01)
    cfg = BacktestConfig(cost_bps=10.0, slippage_bps=0.0)  # 10 bps -> 0.001 per side
    res = simulate_ohlc_bracket(
        pred_logret=pred, actual_logret=actual, intent=[1.0], config=cfg,
    )
    # Expected turnover = 2 (open + close).
    assert res.turnover[0] == 2.0
    np.testing.assert_allclose(res.cost_log[0], -0.002)


def test_bracket_invalid_tie_break_raises() -> None:
    pred = np.zeros((1, 4))
    actual = np.zeros((1, 4))
    with pytest.raises(ValueError, match="tie_break"):
        simulate_ohlc_bracket(
            pred_logret=pred, actual_logret=actual,
            intent=[1.0], tie_break="bogus",
        )


def test_backtest_bracket_uses_strategy_sign() -> None:
    # Three crafted samples:
    #   day 0: long  → predicted high (0.01) hit; predicted low (-0.01) NOT hit → TP exit at 0.01.
    #   day 1: short → predicted low (-0.01) hit; predicted high (0.01) NOT hit → TP exit at -0.01.
    #   day 2: flat  → no position, zero pnl.
    pred = np.zeros((3, 4))
    pred[:, OHLC_HIGH] = [0.01, 0.01, 0.01]
    pred[:, OHLC_LOW] = [-0.01, -0.01, -0.01]
    pred[:, OHLC_CLOSE] = [0.005, -0.005, 0.0]
    actual = np.zeros_like(pred)
    actual[:, OHLC_HIGH] = [0.02, 0.005, 0.0]   # day 1's high doesn't reach pred_high
    actual[:, OHLC_LOW] = [-0.005, -0.02, 0.0]  # day 0's low doesn't reach pred_low
    actual[:, OHLC_CLOSE] = [0.005, -0.015, 0.0]
    res = backtest_bracket(
        pred_logret=pred,
        actual_logret=actual,
        strategy=long_short(),
        config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
    )
    np.testing.assert_allclose(res.positions, [1.0, -1.0, 0.0])
    # Long TP: exit_lr - entry_lr = 0.01 - 0 = 0.01.
    # Short TP: entry_lr - exit_lr = 0 - (-0.01) = 0.01.
    np.testing.assert_allclose(res.gross_pnl_log, [0.01, 0.01, 0.0])


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_compare_returns_dataframe_indexed_by_name(rng: np.random.Generator) -> None:
    actual = _make_logret(rng, n=60)
    pred = actual + rng.normal(0, 0.002, actual.shape)  # noisy oracle
    cfg = BacktestConfig(cost_bps=1.0, slippage_bps=0.5)

    a = backtest(pred_logret=pred, actual_logret=actual, strategy=long_short(),
                 config=cfg, name="long_short")
    b = backtest(pred_logret=pred, actual_logret=actual, strategy=long_only(),
                 config=cfg, name="long_only")
    df = compare([a, b])
    assert isinstance(df, pd.DataFrame)
    assert set(df.index) == {"long_short", "long_only"}
    assert "sharpe" in df.columns
    assert "max_drawdown" in df.columns


def test_summary_runs_and_includes_strategy_name(rng: np.random.Generator) -> None:
    actual = _make_logret(rng, n=20)
    pred = actual.copy()
    res = backtest(pred_logret=pred, actual_logret=actual, strategy=long_short(),
                   config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0),
                   name="oracle")
    text = res.summary()
    assert "oracle" in text
    assert "Sharpe" in text
    assert "max drawdown" in text


def test_oracle_strategy_outperforms_buy_and_hold_when_costs_zero(rng: np.random.Generator) -> None:
    """If predictions equal actuals, long_short() must beat buy & hold."""
    actual = _make_logret(rng, n=400)
    pred = actual.copy()
    res = backtest(pred_logret=pred, actual_logret=actual, strategy=long_short(),
                   config=BacktestConfig(cost_bps=0.0, slippage_bps=0.0))
    assert res.metrics["total_return"] > res.benchmark_metrics["total_return"]
    assert res.metrics["sharpe"] > res.benchmark_metrics["sharpe"]
    # Oracle: every active day is profitable.
    assert res.metrics["hit_rate_trades"] == pytest.approx(1.0)
