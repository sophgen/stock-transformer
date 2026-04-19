"""M6b tabular baselines."""

from __future__ import annotations

import numpy as np

from stock_transformer.features.tabular import flatten_universe_sample
from stock_transformer.model.baselines import mean_reversion_rank_scores, momentum_rank_scores
from stock_transformer.model.baselines_tabular import fit_gbt_ranker, fit_linear_cs_ranker


def test_flatten_round_trip_shape():
    rng = np.random.default_rng(0)
    N, L, S, F = 5, 4, 3, 2
    X = rng.standard_normal((N, L, S, F)).astype(np.float32)
    mask = rng.random((N, L, S)) > 0.3
    y = rng.standard_normal((N, S)).astype(np.float32)
    last = ~mask[:, -1, :]
    y[~last | (rng.random((N, S)) > 0.5)] = np.nan
    Xf, yf, gid, sid = flatten_universe_sample(X, mask, y)
    expected = int((last & np.isfinite(y)).sum())
    assert Xf.shape == (expected, L * F)
    assert yf.shape == (expected,)
    assert gid.shape == (expected,)
    assert sid.shape == (expected,)


def test_linear_ranker_beats_equal_on_synthetic():
    rng = np.random.default_rng(1)
    N, L, S, F = 80, 6, 4, 3
    X = rng.standard_normal((N, L, S, F)).astype(np.float32)
    mask = np.zeros((N, L, S), dtype=bool)
    signal = X[:, -1, :, 0]
    y = (0.5 * signal + 0.1 * rng.standard_normal((N, S))).astype(np.float32)
    Xf, yf, gid, _sid = flatten_universe_sample(X, mask, y)
    assert Xf.shape[0] > 20
    n_tr = max(20, Xf.shape[0] * 2 // 3)
    lin = fit_linear_cs_ranker(Xf[:n_tr], yf[:n_tr], gid[:n_tr])
    pred = lin.predict(Xf[n_tr:].astype(np.float64))
    rho = np.corrcoef(pred, yf[n_tr:])[0, 1]
    assert rho > 0.05


def test_gbt_ranker_runs_on_synthetic():
    rng = np.random.default_rng(2)
    N, L, S, F = 40, 5, 3, 2
    X = rng.standard_normal((N, L, S, F)).astype(np.float32)
    mask = np.zeros((N, L, S), dtype=bool)
    y = rng.standard_normal((N, S)).astype(np.float32)
    Xf, yf, gid, _ = flatten_universe_sample(X, mask, y)
    gbt = fit_gbt_ranker(Xf, yf, gid)
    pred = gbt.predict(Xf)
    assert np.all(np.isfinite(pred))
    assert pred.shape[0] == Xf.shape[0]


def test_mean_reversion_baseline_is_negative_of_momentum():
    close = np.array(
        [
            [100.0, 200.0],
            [101.0, 198.0],
            [102.0, 196.0],
            [103.0, 195.0],
        ],
        dtype=np.float64,
    )
    end_rows = np.array([3], dtype=np.int64)
    mom = momentum_rank_scores(close, end_rows=end_rows, lookback=2)
    mr = mean_reversion_rank_scores(close, end_rows=end_rows, lookback=2)
    assert np.allclose(mom + mr, 0.0, equal_nan=True)
