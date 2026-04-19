"""Cross-sectional tabular baselines: ridge regression and gradient boosting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


@dataclass
class LinearCSRanker:
    """Ridge regressor on flattened ``[L*F]`` features."""

    model: Ridge

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
        return np.asarray(self.model.predict(X), dtype=np.float64)


def fit_linear_cs_ranker(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    *,
    alpha: float = 1.0,
) -> LinearCSRanker:
    del groups_tr  # unused; kept for API symmetry with GBT ranker
    X_tr = np.nan_to_num(np.asarray(X_tr, dtype=np.float64), nan=0.0)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    model = Ridge(alpha=float(alpha))
    model.fit(X_tr, y_tr)
    return LinearCSRanker(model=model)


@dataclass
class GBTRanker:
    """HistGradientBoosting regressor on flattened windows (portable vs. native GBDT libs)."""

    model: HistGradientBoostingRegressor

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self.model.predict(X), dtype=np.float64)


def fit_gbt_ranker(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
) -> GBTRanker:
    del groups_tr
    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    p: dict[str, Any] = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "max_iter": 120,
        "random_state": 42,
    }
    if params:
        p.update(params)
    model = HistGradientBoostingRegressor(**p)
    model.fit(X_tr, y_tr)
    return GBTRanker(model=model)


def scatter_predictions(
    pred_flat: np.ndarray,
    group_ids: np.ndarray,
    sym_ids: np.ndarray,
    *,
    n_samples: int,
    n_symbols: int,
) -> np.ndarray:
    """Rebuild ``[N, S]`` scores from flattened predictions (NaN where missing)."""
    out = np.full((n_samples, n_symbols), np.nan, dtype=np.float64)
    for pr, g, s in zip(pred_flat, group_ids, sym_ids, strict=True):
        out[int(g), int(s)] = float(pr)
    return out
