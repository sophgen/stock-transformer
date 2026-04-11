"""Classification + regression metrics and fold aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
    return out


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """MAE, RMSE, and directional accuracy from regression predictions.

    Directional accuracy: fraction of samples where the predicted close-return
    sign matches the actual close-return sign (column index 3 = close_ret).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    out: dict[str, float] = {"mae": mae, "rmse": rmse}

    if y_true.ndim == 2 and y_true.shape[1] > 3:
        true_dir = (y_true[:, 3] > 0).astype(int)
        pred_dir = (y_pred[:, 3] > 0).astype(int)
        out["directional_accuracy"] = float(accuracy_score(true_dir, pred_dir))

    return out


def aggregate_fold_metrics(per_fold: list[dict[str, Any]]) -> dict[str, float]:
    """Mean and std across folds for numeric metric keys."""
    if not per_fold:
        return {}
    keys = [k for k in per_fold[0].keys() if k != "fold_id"]
    agg: dict[str, float] = {}
    for k in keys:
        vals = [float(f[k]) for f in per_fold if not np.isnan(f.get(k, np.nan))]
        if not vals:
            continue
        arr = np.array(vals)
        agg[f"{k}_mean"] = float(arr.mean())
        agg[f"{k}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return agg
