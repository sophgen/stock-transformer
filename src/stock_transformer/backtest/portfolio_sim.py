"""Top-k portfolio simulation from cross-sectional scores (M11).

Uses realized forward raw returns per row. Turnover is one-way
``sum_s |w_t(s) - w_{t-1}(s)|``. Transaction costs apply to that turnover
(one-way bps). This is a deliberate stub — no borrow costs, slippage model,
or capacity.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

Book = Literal["long_only", "long_short"]


def _weights_long_topk(
    scores: np.ndarray,
    tradable: np.ndarray,
    top_k: int,
) -> np.ndarray | None:
    idx = np.where(tradable)[0]
    if idx.size < top_k:
        return None
    order = idx[np.argsort(-scores[idx])]
    picked = order[:top_k]
    w = np.zeros(scores.shape[0], dtype=np.float64)
    w[picked] = 1.0 / top_k
    return w


def _weights_long_short_topk(
    scores: np.ndarray,
    tradable: np.ndarray,
    top_k: int,
) -> np.ndarray | None:
    idx = np.where(tradable)[0]
    if idx.size < 2 * top_k:
        return None
    order = idx[np.argsort(scores[idx])]
    short_p = order[:top_k]
    long_p = order[-top_k:]
    w = np.zeros(scores.shape[0], dtype=np.float64)
    w[long_p] = 0.5 / top_k
    w[short_p] = -0.5 / top_k
    return w


def simulate_topk_portfolio(
    scores: np.ndarray,
    raw_returns: np.ndarray,
    *,
    pad_last: np.ndarray,
    book: Book,
    top_k: int,
    transaction_cost_one_way_bps: float,
    record_series: bool = True,
) -> dict[str, Any]:
    """Simulate period-by-period rebalancing on test predictions.

    Parameters
    ----------
    scores:
        Model scores ``[N, S]`` (higher is better).
    raw_returns:
        Realized forward simple returns ``[N, S]`` (same alignment as scores).
    pad_last:
        Padding mask at the last timestep ``[N, S]``, **True** = invalid /
        padded symbol (same convention as feature tensors).
    book:
        ``long_only`` (equal-weight long top-k) or ``long_short`` (top k long,
        bottom k short; dollar-neutral, gross 100%).
    top_k:
        Names in the long leg; short leg uses the same k in ``long_short``.
    transaction_cost_one_way_bps:
        Fee in basis points applied once to one-way turnover each period.
    """
    scores = np.asarray(scores, dtype=np.float64)
    raw_returns = np.asarray(raw_returns, dtype=np.float64)
    pad_last = np.asarray(pad_last, dtype=bool)
    if scores.shape != raw_returns.shape or scores.shape != pad_last.shape:
        raise ValueError("scores, raw_returns, pad_last must share shape [N, S]")
    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    rate = float(transaction_cost_one_way_bps) * 1e-4

    n = scores.shape[0]
    gross = np.full(n, np.nan, dtype=np.float64)
    net = np.full(n, np.nan, dtype=np.float64)
    turnover = np.full(n, np.nan, dtype=np.float64)
    cost = np.full(n, np.nan, dtype=np.float64)

    w_prev = np.zeros(scores.shape[1], dtype=np.float64)

    for i in range(n):
        tradable = (~pad_last[i]) & np.isfinite(scores[i]) & np.isfinite(raw_returns[i])
        if book == "long_only":
            w = _weights_long_topk(scores[i], tradable, top_k)
        else:
            w = _weights_long_short_topk(scores[i], tradable, top_k)
        if w is None:
            continue

        to = float(np.sum(np.abs(w - w_prev)))
        c = rate * to
        g = float(np.dot(w, raw_returns[i]))
        gross[i] = g
        cost[i] = c
        net[i] = g - c
        turnover[i] = to
        w_prev = w

    fin = np.isfinite(net)
    if not np.any(fin):
        out_empty: dict[str, Any] = {
            "n_periods": 0,
            "mean_gross_return": float("nan"),
            "mean_net_return": float("nan"),
            "mean_turnover": float("nan"),
            "mean_cost": float("nan"),
            "cumulative_gross_return": float("nan"),
            "cumulative_net_return": float("nan"),
            "sharpe_net": float("nan"),
        }
        if record_series:
            out_empty["gross_period_returns"] = gross.tolist()
            out_empty["net_period_returns"] = net.tolist()
            out_empty["turnover"] = turnover.tolist()
        return out_empty

    net_s = net[fin]
    gross_s = gross[fin]
    to_s = turnover[fin]
    cost_s = cost[fin]

    cum_net = float(np.prod(1.0 + net_s) - 1.0)
    cum_gross = float(np.prod(1.0 + gross_s) - 1.0)
    std = float(np.nanstd(net_s, ddof=1)) if net_s.size > 1 else float("nan")
    sharpe = float(np.nanmean(net_s) / std) if std > 1e-12 else float("nan")

    out: dict[str, Any] = {
        "n_periods": int(fin.sum()),
        "mean_gross_return": float(np.nanmean(gross_s)),
        "mean_net_return": float(np.nanmean(net_s)),
        "mean_turnover": float(np.nanmean(to_s)),
        "mean_cost": float(np.nanmean(cost_s)),
        "cumulative_gross_return": cum_gross,
        "cumulative_net_return": cum_net,
        "sharpe_net": sharpe,
    }
    if record_series:
        out["gross_period_returns"] = gross.tolist()
        out["net_period_returns"] = net.tolist()
        out["turnover"] = turnover.tolist()
    return out


def aggregate_portfolio_sim_folds(by_fold: list[dict[str, Any]]) -> dict[str, float]:
    """Mean/std across folds for numeric portfolio summary keys."""
    if not by_fold:
        return {}
    keys = [
        "n_periods",
        "mean_gross_return",
        "mean_net_return",
        "mean_turnover",
        "mean_cost",
        "cumulative_gross_return",
        "cumulative_net_return",
        "sharpe_net",
    ]
    agg: dict[str, float] = {}
    for k in keys:
        vals: list[float] = []
        for row in by_fold:
            if k not in row:
                continue
            v = float(row[k])
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            agg[f"{k}_mean"] = float("nan")
            continue
        arr = np.array(vals, dtype=np.float64)
        agg[f"{k}_mean"] = float(arr.mean())
        agg[f"{k}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return agg
