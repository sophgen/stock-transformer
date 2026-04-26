"""Backtesting engine for the CandleTransformer.

Two simulators sit alongside each other:

* :func:`simulate_close_to_close` — the standard close-to-close engine. A
  strategy maps each prediction into a position in ``[-1, +1]`` (fraction of
  equity, sign = direction). Daily realized log-return is
  ``position_t * actual_close_logret_t`` minus a turnover-proportional cost.
* :func:`simulate_ohlc_bracket` — uses the model's full predicted candle as a
  bracket order. Enter at the next session's actual open, set the take-profit
  at the predicted high (for longs) or low (for shorts), set the stop-loss
  at the opposite predicted extreme, exit at actual close if neither is
  touched. When both are touched the same day, ``tie_break`` decides
  (default ``"pessimistic"`` — stop-loss wins).

Both simulators return the same :class:`BacktestResult`, which carries the
equity curve, daily P&L, a buy-and-hold benchmark on the same window, and a
bag of headline metrics (Sharpe, Sortino, max drawdown, CAGR, hit rate,
profit factor, exposure, turnover).

The math operates entirely in **log-return space**:

* Position ``p`` applied to a daily log-return ``r`` contributes ``p * r`` to
  the strategy's realized log-return. This is the standard small-r
  approximation; for ``|p| <= 1`` and daily moves it is accurate to a few
  basis points.
* Costs are charged in log space too: ``log(1 - cost) ≈ -cost`` for small
  ``cost``, so a turnover of ``Δp`` and a per-side fee+slippage of ``c``
  subtracts ``c * |Δp|`` from the day's log-return.

Anything more sophisticated (tax lots, margin, intraday execution,
overnight gaps, capacity) is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

# `pandas` and `matplotlib` are imported lazily inside methods that need them so
# importing this module stays cheap and side-effect-free.

ArrayLike = np.ndarray | Sequence[float]

OHLC_OPEN = 0
OHLC_HIGH = 1
OHLC_LOW = 2
OHLC_CLOSE = 3

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """Knobs that don't depend on the strategy itself.

    Attributes:
        cost_bps: Round-trip-equivalent commission per unit of turnover, in
            basis points. ``1.0`` = 1 bp = 0.01 %. Charged proportionally to
            ``|Δposition|`` each day, so flipping from +1 → -1 costs ``2 *
            cost_bps``.
        slippage_bps: Same model as ``cost_bps`` but represents adverse
            execution (mid-to-fill) instead of fees. Combined with
            ``cost_bps`` into the per-bp turnover charge.
        starting_capital: Notional starting equity. Only affects the dollar
            equity curve; metrics are scale-invariant.
        annualization_factor: Trading days per year used for Sharpe / Sortino
            / CAGR. ``252`` = standard US equities convention.
        risk_free_log: Daily log risk-free rate to subtract from the strategy
            and benchmark before computing Sharpe/Sortino. Defaults to 0.
            Pass e.g. ``np.log(1 + 0.04) / 252`` for a 4 % annual rfr.
    """

    cost_bps: float = 1.0
    slippage_bps: float = 0.5
    starting_capital: float = 10_000.0
    annualization_factor: int = TRADING_DAYS_PER_YEAR
    risk_free_log: float = 0.0

    @property
    def cost_per_turnover(self) -> float:
        """Combined cost+slippage as a decimal (e.g. 1.5 bps → 0.00015)."""
        return (self.cost_bps + self.slippage_bps) * 1e-4


@dataclass
class BacktestResult:
    """Output of a backtest run.

    All daily arrays are aligned to ``dates``: ``arr[i]`` is the value on the
    i-th sample (i.e. position decided at end of day ``i-1``, P&L realized
    over day ``i``).
    """

    dates: np.ndarray            # [N] (datetime64 / object); strategy-specific x-axis
    positions: np.ndarray        # [N]   position held during day i
    pnl_log: np.ndarray          # [N]   net realized log-return on day i
    gross_pnl_log: np.ndarray    # [N]   pre-cost log-return on day i
    cost_log: np.ndarray         # [N]   negative log-cost charged on day i
    turnover: np.ndarray         # [N]   |position[i] - position[i-1]|, position[-1] := 0
    equity: np.ndarray           # [N]   $ equity at end of day i (after costs)
    benchmark_pnl_log: np.ndarray  # [N] buy-and-hold daily log-return
    benchmark_equity: np.ndarray   # [N] $ equity for buy-and-hold
    config: BacktestConfig
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    name: str = "strategy"

    # ---- text views ------------------------------------------------------

    def summary(self) -> str:
        """Two-column table: strategy vs. buy-and-hold benchmark."""
        cfg = self.config
        rows = [
            ("n_days",            f"{int(self.metrics['n_days']):d}",                f"{int(self.benchmark_metrics['n_days']):d}"),
            ("total return",      f"{self.metrics['total_return']:.2%}",             f"{self.benchmark_metrics['total_return']:.2%}"),
            ("CAGR",              f"{self.metrics['cagr']:.2%}",                     f"{self.benchmark_metrics['cagr']:.2%}"),
            ("ann. volatility",   f"{self.metrics['annualized_volatility']:.2%}",    f"{self.benchmark_metrics['annualized_volatility']:.2%}"),
            ("Sharpe",            f"{self.metrics['sharpe']:.2f}",                   f"{self.benchmark_metrics['sharpe']:.2f}"),
            ("Sortino",           f"{self.metrics['sortino']:.2f}",                  f"{self.benchmark_metrics['sortino']:.2f}"),
            ("max drawdown",      f"{self.metrics['max_drawdown']:.2%}",             f"{self.benchmark_metrics['max_drawdown']:.2%}"),
            ("hit rate (all)",    f"{self.metrics['hit_rate']:.1%}",                 f"{self.benchmark_metrics['hit_rate']:.1%}"),
            ("hit rate (trades)", f"{self.metrics['hit_rate_trades']:.1%}",          "-"),
            ("profit factor",     _fmt_pf(self.metrics['profit_factor']),            _fmt_pf(self.benchmark_metrics['profit_factor'])),
            ("exposure",          f"{self.metrics['exposure']:.1%}",                 f"{self.benchmark_metrics['exposure']:.1%}"),
            ("turnover (Σ|Δp|)",  f"{self.metrics['total_turnover']:.2f}",           "0.00"),
            ("ending equity ($)", f"${self.metrics['ending_equity']:,.2f}",          f"${self.benchmark_metrics['ending_equity']:,.2f}"),
        ]
        header = (
            f"Backtest: {self.name}\n"
            f"  costs={cfg.cost_bps:g}bps  slippage={cfg.slippage_bps:g}bps  "
            f"start=${cfg.starting_capital:,.0f}  ann={cfg.annualization_factor}d\n"
        )
        col1 = max(len(r[0]) for r in rows) + 2
        col2 = max(max(len(r[1]) for r in rows), len(self.name)) + 2
        col3 = max(len(r[2]) for r in rows) + 2
        sep = "  " + "-" * (col1 + col2 + col3 + 2) + "\n"
        out = [header, sep]
        out.append(f"  {'metric':<{col1}}{self.name:<{col2}}{'buy & hold':<{col3}}\n")
        out.append(sep)
        for label, a, b in rows:
            out.append(f"  {label:<{col1}}{a:<{col2}}{b:<{col3}}\n")
        return "".join(out)

    def __str__(self) -> str:  # pragma: no cover — convenience
        return self.summary()

    # ---- plot helpers ----------------------------------------------------

    def plot_equity(self, ax=None, *, log_scale: bool = True):
        """Equity curve vs buy-and-hold (matplotlib axis)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(self.dates, self.equity, color="tab:red", lw=1.2, label=self.name)
        ax.plot(self.dates, self.benchmark_equity, color="tab:blue", lw=1.0, label="buy & hold")
        ax.axhline(self.config.starting_capital, color="gray", lw=0.5)
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel("equity ($)")
        ax.set_title(f"Equity curve — {self.name}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left")
        return ax

    def plot_drawdown(self, ax=None):
        """Underwater (drawdown from peak) plot."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(11, 3))
        dd = _drawdown(self.equity)
        bdd = _drawdown(self.benchmark_equity)
        ax.fill_between(self.dates, dd, 0, color="tab:red", alpha=0.4, label=self.name)
        ax.plot(self.dates, bdd, color="tab:blue", lw=0.9, label="buy & hold")
        ax.set_ylabel("drawdown")
        ax.set_title("Drawdown from running peak")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
        return ax

    def plot_overview(self, fig=None):
        """Three-panel overview: equity, drawdown, daily-return histogram."""
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure(figsize=(13, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 2])
        ax_eq = fig.add_subplot(gs[0])
        ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)
        ax_hist = fig.add_subplot(gs[2])
        self.plot_equity(ax_eq)
        self.plot_drawdown(ax_dd)
        bins = max(20, min(80, len(self.pnl_log) // 25))
        ax_hist.hist(self.pnl_log, bins=bins, color="tab:red", alpha=0.6, label=self.name)
        ax_hist.hist(self.benchmark_pnl_log, bins=bins, color="tab:blue", alpha=0.4, label="buy & hold")
        ax_hist.axvline(0, color="k", lw=0.5)
        ax_hist.set_title("Daily log-return distribution")
        ax_hist.set_xlabel("log-return")
        ax_hist.grid(alpha=0.3)
        ax_hist.legend(loc="upper left")
        fig.tight_layout()
        return fig


def _fmt_pf(x: float) -> str:
    if not np.isfinite(x):
        return "inf"
    return f"{x:.2f}"


# ---------------------------------------------------------------------------
# Strategy factories — pred_logret -> position
# ---------------------------------------------------------------------------


Strategy = Callable[[np.ndarray], np.ndarray]
"""A strategy maps ``pred_logret`` (shape ``[N, 4]``) to positions in ``[-1, +1]``."""


def long_short(threshold: float = 0.0) -> Strategy:
    """Go long if predicted close return > threshold; short if < -threshold; flat otherwise.

    Default ``threshold=0`` matches the naive "trade the sign of the prediction" baseline.
    Raise the threshold to require stronger conviction before opening a position.
    """

    def fn(pred: np.ndarray) -> np.ndarray:
        c = np.asarray(pred)[:, OHLC_CLOSE]
        out = np.zeros_like(c, dtype=np.float64)
        out[c > threshold] = 1.0
        out[c < -threshold] = -1.0
        return out

    fn.__name__ = f"long_short(t={threshold:g})"
    return fn


def long_only(threshold: float = 0.0) -> Strategy:
    """Go long when predicted close return > threshold, flat otherwise (no shorting)."""

    def fn(pred: np.ndarray) -> np.ndarray:
        c = np.asarray(pred)[:, OHLC_CLOSE]
        out = np.zeros_like(c, dtype=np.float64)
        out[c > threshold] = 1.0
        return out

    fn.__name__ = f"long_only(t={threshold:g})"
    return fn


def confidence_weighted(scale: float = 50.0, cap: float = 1.0) -> Strategy:
    """Position size ∝ predicted close magnitude, clipped to ``[-cap, +cap]``.

    A predicted log-return of 0.01 (≈1 %) at ``scale=50`` gives position 0.5.
    Higher ``scale`` means more aggressive sizing per unit of predicted move.
    """

    def fn(pred: np.ndarray) -> np.ndarray:
        c = np.asarray(pred)[:, OHLC_CLOSE]
        return np.clip(c * scale, -cap, cap)

    fn.__name__ = f"confidence_weighted(scale={scale:g}, cap={cap:g})"
    return fn


def buy_and_hold() -> Strategy:
    """Always-long unit-position strategy. Useful as a sanity strategy."""

    def fn(pred: np.ndarray) -> np.ndarray:
        return np.ones(len(pred), dtype=np.float64)

    fn.__name__ = "buy_and_hold"
    return fn


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    pnl_log: np.ndarray,
    *,
    positions: np.ndarray | None = None,
    starting_capital: float = 1.0,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
    risk_free_log: float = 0.0,
) -> dict[str, float]:
    """Headline performance metrics for a daily log-return series.

    All metrics are scale-invariant (don't depend on ``starting_capital``)
    except ``ending_equity``. ``positions`` is optional but enables
    ``hit_rate_trades`` (only days where the strategy was non-flat) and
    ``exposure``.

    Returns a flat ``dict[str, float]`` keyed by:

    * ``n_days``                 — number of days in the series
    * ``total_log_return``       — sum of daily log-returns (post-costs)
    * ``total_return``           — ``exp(total_log_return) - 1`` (simple total return)
    * ``cagr``                   — annualized compounded growth rate
    * ``annualized_volatility``  — std(daily log-returns) × √annualization
    * ``sharpe``                 — (mean - rf) / std × √annualization
    * ``sortino``                — (mean - rf) / downside_std × √annualization
    * ``max_drawdown``           — most-negative peak-to-trough drawdown
    * ``hit_rate``               — fraction of all days with positive P&L
    * ``hit_rate_trades``        — fraction of *active* days (|position|>0) with positive P&L
    * ``avg_win`` / ``avg_loss`` — mean daily log-return on winning / losing days
    * ``profit_factor``          — Σ winners / |Σ losers|; ``+inf`` if no losers
    * ``exposure``               — fraction of days with |position|>0 (NaN if positions absent)
    * ``total_turnover``         — Σ |Δposition|; absent when positions=None
    * ``ending_equity``          — ``starting_capital × exp(total_log_return)``
    """
    pnl = np.asarray(pnl_log, dtype=np.float64).reshape(-1)
    n = int(len(pnl))
    out: dict[str, float] = {"n_days": float(n)}
    if n == 0:
        # Empty input → all metrics NaN, ending equity = starting capital.
        nan_keys = [
            "total_log_return", "total_return", "cagr", "annualized_volatility",
            "sharpe", "sortino", "max_drawdown", "hit_rate", "hit_rate_trades",
            "avg_win", "avg_loss", "profit_factor", "exposure", "total_turnover",
        ]
        for k in nan_keys:
            out[k] = float("nan")
        out["ending_equity"] = float(starting_capital)
        return out

    total_log = float(pnl.sum())
    out["total_log_return"] = total_log
    out["total_return"] = float(np.expm1(total_log))

    # Annualized growth rate. Use exponential compounding so it matches
    # the equity curve's geometric growth.
    years = n / annualization_factor
    out["cagr"] = float(np.exp(total_log / years) - 1.0) if years > 0 else 0.0

    daily_mean = float(pnl.mean())
    daily_std = float(pnl.std(ddof=1)) if n > 1 else 0.0
    out["annualized_volatility"] = float(daily_std * np.sqrt(annualization_factor))

    excess = daily_mean - risk_free_log
    out["sharpe"] = (
        float(excess / daily_std * np.sqrt(annualization_factor))
        if daily_std > 0 else 0.0
    )

    downside = pnl[pnl < risk_free_log] - risk_free_log
    if len(downside) > 1:
        downside_std = float(np.sqrt((downside ** 2).mean()))
        out["sortino"] = (
            float(excess / downside_std * np.sqrt(annualization_factor))
            if downside_std > 0 else 0.0
        )
    else:
        out["sortino"] = 0.0

    equity_curve = np.exp(np.cumsum(pnl))
    out["max_drawdown"] = float(_drawdown(equity_curve).min())

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    out["hit_rate"] = float((pnl > 0).mean())
    out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    sum_losses = float(-losses.sum())
    out["profit_factor"] = (
        float(wins.sum() / sum_losses) if sum_losses > 0 else float("inf")
    )

    if positions is not None:
        pos = np.asarray(positions, dtype=np.float64).reshape(-1)
        active = np.abs(pos) > 0
        out["exposure"] = float(active.mean())
        if active.any():
            out["hit_rate_trades"] = float((pnl[active] > 0).mean())
        else:
            out["hit_rate_trades"] = 0.0
        # turnover[t] = |pos[t] - pos[t-1]|, with pos[-1] := 0
        prev = np.concatenate(([0.0], pos[:-1]))
        out["total_turnover"] = float(np.abs(pos - prev).sum())
    else:
        out["exposure"] = float("nan")
        out["hit_rate_trades"] = float("nan")
        out["total_turnover"] = float("nan")

    out["ending_equity"] = float(starting_capital * np.exp(total_log))
    return out


def _drawdown(equity: np.ndarray) -> np.ndarray:
    """Per-day drawdown from running peak. Always ≤ 0."""
    eq = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    return eq / peak - 1.0


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------


def _coerce_dates(dates: object, n: int) -> np.ndarray:
    """Best-effort convert dates iterable → 1D array of length n."""
    if dates is None:
        return np.arange(n)
    arr = np.asarray(list(dates) if not hasattr(dates, "__len__") else dates)
    if len(arr) != n:
        raise ValueError(f"len(dates)={len(arr)} does not match number of samples {n}")
    return arr


def _benchmark(actual_close_logret: np.ndarray, config: BacktestConfig) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Buy-and-hold benchmark: full unit-position long every day, no costs."""
    pnl = np.asarray(actual_close_logret, dtype=np.float64).reshape(-1)
    equity = config.starting_capital * np.exp(np.cumsum(pnl))
    metrics = compute_metrics(
        pnl,
        positions=np.ones_like(pnl),
        starting_capital=config.starting_capital,
        annualization_factor=config.annualization_factor,
        risk_free_log=config.risk_free_log,
    )
    return pnl, equity, metrics


def simulate_close_to_close(
    *,
    positions: ArrayLike,
    actual_close_logret: ArrayLike,
    dates: object = None,
    config: BacktestConfig | None = None,
    name: str = "close-to-close",
) -> BacktestResult:
    """Close-to-close backtest given a position series and the realized close log-returns.

    The position is taken at the close of day ``t-1`` (i.e. decided by the
    model after observing day ``t-1``'s candle) and held overnight; P&L is
    realized at the close of day ``t``. Costs are charged proportionally to
    ``|Δposition|`` between consecutive days.
    """
    cfg = config or BacktestConfig()
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    r = np.asarray(actual_close_logret, dtype=np.float64).reshape(-1)
    if pos.shape != r.shape:
        raise ValueError(f"positions {pos.shape} vs returns {r.shape} shape mismatch")
    n = len(pos)
    dates_arr = _coerce_dates(dates, n)

    # Turnover: assume we start flat. Day 0 incurs |pos[0]| of turnover.
    prev = np.concatenate(([0.0], pos[:-1]))
    turnover = np.abs(pos - prev)

    cost = cfg.cost_per_turnover
    cost_log = -cost * turnover               # log-space cost charged on day t
    gross = pos * r                            # gross daily log-return
    net = gross + cost_log                     # net daily log-return

    equity = cfg.starting_capital * np.exp(np.cumsum(net))
    bench_pnl, bench_eq, bench_metrics = _benchmark(r, cfg)

    metrics = compute_metrics(
        net,
        positions=pos,
        starting_capital=cfg.starting_capital,
        annualization_factor=cfg.annualization_factor,
        risk_free_log=cfg.risk_free_log,
    )

    return BacktestResult(
        dates=dates_arr,
        positions=pos,
        pnl_log=net,
        gross_pnl_log=gross,
        cost_log=cost_log,
        turnover=turnover,
        equity=equity,
        benchmark_pnl_log=bench_pnl,
        benchmark_equity=bench_eq,
        config=cfg,
        metrics=metrics,
        benchmark_metrics=bench_metrics,
        name=name,
    )


def backtest(
    *,
    pred_logret: np.ndarray,
    actual_logret: np.ndarray,
    strategy: Strategy,
    dates: object = None,
    config: BacktestConfig | None = None,
    name: str | None = None,
) -> BacktestResult:
    """High-level convenience wrapper for close-to-close backtesting.

    ``pred_logret`` and ``actual_logret`` must both be ``[N, 4]`` arrays of
    next-day OHLC log-returns from today's close (the same shape
    ``CandleTransformer`` predicts and ``build_features`` produces).

    The backtester reads only the close channel (``[:, 3]``) for actual
    realized P&L; the open/high/low channels are still passed through the
    strategy so confidence-weighted strategies can use them too.
    """
    pred = np.asarray(pred_logret, dtype=np.float64)
    actual = np.asarray(actual_logret, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError(f"pred_logret must be [N, 4]; got {pred.shape}")
    if actual.shape != pred.shape:
        raise ValueError(f"actual_logret {actual.shape} vs pred {pred.shape} mismatch")

    pos = strategy(pred)
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    if len(pos) != len(pred):
        raise ValueError(f"strategy returned {len(pos)} positions for {len(pred)} samples")

    return simulate_close_to_close(
        positions=pos,
        actual_close_logret=actual[:, OHLC_CLOSE],
        dates=dates,
        config=config,
        name=name or getattr(strategy, "__name__", "strategy"),
    )


# ---------------------------------------------------------------------------
# OHLC bracket simulator
# ---------------------------------------------------------------------------


def _bracket_exit_logret(
    intent: float,
    entry_lr: float,
    pred_high_lr: float,
    pred_low_lr: float,
    act_high_lr: float,
    act_low_lr: float,
    act_close_lr: float,
    *,
    tie_break: str,
) -> tuple[float, str]:
    """Decide the exit log-return for a single bracket trade.

    Returns ``(exit_lr, reason)`` where ``reason`` is one of:
    ``"tp"``, ``"sl"``, ``"close"``, ``"invalid"``, or ``"flat"``.
    All log-returns are measured against today's close (same convention as
    ``build_features`` targets).
    """
    if intent == 0:
        return 0.0, "flat"

    if intent > 0:
        # Long: TP at predicted high, SL at predicted low.
        tp_lr = pred_high_lr
        sl_lr = pred_low_lr
        # Bracket must straddle the entry; otherwise it would trigger immediately.
        if tp_lr <= entry_lr or sl_lr >= entry_lr:
            return act_close_lr, "invalid"
        hit_tp = act_high_lr >= tp_lr
        hit_sl = act_low_lr <= sl_lr
        if hit_tp and hit_sl:
            if tie_break == "pessimistic":
                return sl_lr, "sl"
            elif tie_break == "optimistic":
                return tp_lr, "tp"
            else:
                # 50/50 split is the only well-defined unbiased tie-break.
                return 0.5 * (tp_lr + sl_lr), "tie"
        if hit_tp:
            return tp_lr, "tp"
        if hit_sl:
            return sl_lr, "sl"
        return act_close_lr, "close"
    else:
        # Short: TP at predicted low (price falls), SL at predicted high.
        tp_lr = pred_low_lr
        sl_lr = pred_high_lr
        if tp_lr >= entry_lr or sl_lr <= entry_lr:
            return act_close_lr, "invalid"
        hit_tp = act_low_lr <= tp_lr
        hit_sl = act_high_lr >= sl_lr
        if hit_tp and hit_sl:
            if tie_break == "pessimistic":
                return sl_lr, "sl"
            elif tie_break == "optimistic":
                return tp_lr, "tp"
            else:
                return 0.5 * (tp_lr + sl_lr), "tie"
        if hit_tp:
            return tp_lr, "tp"
        if hit_sl:
            return sl_lr, "sl"
        return act_close_lr, "close"


def simulate_ohlc_bracket(
    *,
    pred_logret: np.ndarray,
    actual_logret: np.ndarray,
    intent: ArrayLike,
    dates: object = None,
    config: BacktestConfig | None = None,
    tie_break: str = "pessimistic",
    name: str = "OHLC bracket",
) -> BacktestResult:
    """Bracket-order simulation that uses the model's full predicted candle.

    Trade-flow per active sample (``intent[i] != 0``):

    * Enter at the **actual** open (``actual_logret[i, OPEN]``). Using the
      actual open keeps the entry realistic — the predicted open is only
      used to gate intent.
    * Take-profit at predicted high (long) / predicted low (short).
    * Stop-loss at predicted low (long) / predicted high (short).
    * If the bracket is "invalid" (TP below entry / SL above entry for a
      long, or vice versa for a short) the trade falls through to actual
      close — the bracket would otherwise trigger immediately.
    * If neither bracket level is hit, exit at actual close.
    * If both are hit on the same day, ``tie_break`` decides:
        - ``"pessimistic"`` (default) — stop-loss wins (worst case).
        - ``"optimistic"``           — take-profit wins (best case).
        - ``"midpoint"``             — exit halfway between TP and SL.

    Position sizing is ``intent[i] ∈ {-1, 0, +1}``; partial sizing is not
    supported here (bracket math is only well-defined per-trade).
    Costs and slippage are charged proportionally to ``|Δintent|`` (open and
    close of a flip both count) just like the close-to-close engine.

    The benchmark series is buy-and-hold of the close, identical to the
    close-to-close benchmark, so results are directly comparable.
    """
    cfg = config or BacktestConfig()
    pred = np.asarray(pred_logret, dtype=np.float64)
    actual = np.asarray(actual_logret, dtype=np.float64)
    intent_arr = np.asarray(intent, dtype=np.float64).reshape(-1)
    if pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError(f"pred_logret must be [N, 4]; got {pred.shape}")
    if actual.shape != pred.shape:
        raise ValueError(f"actual_logret {actual.shape} vs pred {pred.shape} mismatch")
    if len(intent_arr) != len(pred):
        raise ValueError("intent length must match pred_logret length")
    if tie_break not in ("pessimistic", "optimistic", "midpoint"):
        raise ValueError(f"tie_break must be pessimistic|optimistic|midpoint; got {tie_break!r}")

    n = len(pred)
    dates_arr = _coerce_dates(dates, n)

    gross = np.zeros(n, dtype=np.float64)
    for i in range(n):
        intent_i = float(intent_arr[i])
        if intent_i == 0.0:
            continue
        entry_lr = float(actual[i, OHLC_OPEN])
        exit_lr, _reason = _bracket_exit_logret(
            intent_i,
            entry_lr,
            float(pred[i, OHLC_HIGH]),
            float(pred[i, OHLC_LOW]),
            float(actual[i, OHLC_HIGH]),
            float(actual[i, OHLC_LOW]),
            float(actual[i, OHLC_CLOSE]),
            tie_break=tie_break,
        )
        # Long: realized = exit - entry. Short: realized = entry - exit.
        gross[i] = (exit_lr - entry_lr) if intent_i > 0 else (entry_lr - exit_lr)

    # Treat each bracket trade as 100 % notional in/out: turnover is 2 *
    # |intent[i]| if there's a trade that day (open + close), plus the cost
    # of flipping any residual position to flat between days. Since intent is
    # in {-1, 0, +1} and we close out at end of day, residual is always 0.
    turnover = 2.0 * np.abs(intent_arr)
    cost = cfg.cost_per_turnover
    cost_log = -cost * turnover
    net = gross + cost_log

    equity = cfg.starting_capital * np.exp(np.cumsum(net))
    bench_pnl, bench_eq, bench_metrics = _benchmark(actual[:, OHLC_CLOSE], cfg)

    metrics = compute_metrics(
        net,
        positions=intent_arr,
        starting_capital=cfg.starting_capital,
        annualization_factor=cfg.annualization_factor,
        risk_free_log=cfg.risk_free_log,
    )

    return BacktestResult(
        dates=dates_arr,
        positions=intent_arr,
        pnl_log=net,
        gross_pnl_log=gross,
        cost_log=cost_log,
        turnover=turnover,
        equity=equity,
        benchmark_pnl_log=bench_pnl,
        benchmark_equity=bench_eq,
        config=cfg,
        metrics=metrics,
        benchmark_metrics=bench_metrics,
        name=name,
    )


def backtest_bracket(
    *,
    pred_logret: np.ndarray,
    actual_logret: np.ndarray,
    strategy: Strategy = None,
    dates: object = None,
    config: BacktestConfig | None = None,
    tie_break: str = "pessimistic",
    name: str | None = None,
) -> BacktestResult:
    """High-level OHLC bracket backtest.

    The strategy here decides only **direction** (intent in ``{-1, 0, +1}``);
    its output is sign-quantized before being passed to the simulator.
    Defaults to :func:`long_short` if no strategy is provided.
    """
    pred = np.asarray(pred_logret, dtype=np.float64)
    actual = np.asarray(actual_logret, dtype=np.float64)
    strat = strategy or long_short()
    raw = strat(pred)
    intent = np.sign(np.asarray(raw, dtype=np.float64))
    return simulate_ohlc_bracket(
        pred_logret=pred,
        actual_logret=actual,
        intent=intent,
        dates=dates,
        config=config,
        tie_break=tie_break,
        name=name or f"bracket: {getattr(strat, '__name__', 'strategy')}",
    )


# ---------------------------------------------------------------------------
# Sweep / comparison helpers
# ---------------------------------------------------------------------------


def compare(results: Iterable[BacktestResult]) -> "pd.DataFrame":  # noqa: F821 (forward ref)
    """Stack the headline metrics from many runs into a single DataFrame.

    Useful for hyper-parameter sweeps (``threshold``, ``cost_bps``, etc.) and
    for visual comparison of close-to-close vs. bracket simulations.
    """
    import pandas as pd

    rows: list[dict[str, float | str]] = []
    for r in results:
        m = dict(r.metrics)
        m["name"] = r.name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("name")
    keep = [
        "total_return", "cagr", "annualized_volatility", "sharpe", "sortino",
        "max_drawdown", "hit_rate_trades", "profit_factor", "exposure",
        "total_turnover", "ending_equity",
    ]
    return df[[c for c in keep if c in df.columns]]
