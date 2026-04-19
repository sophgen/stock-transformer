"""Classification + regression metrics and fold aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kendalltau, rankdata
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


def safe_nanmean(a: np.ndarray) -> float:
    """Mean of finite values; NaN if all values are non-finite."""
    a = np.asarray(a, dtype=np.float64)
    if not np.any(np.isfinite(a)):
        return float("nan")
    return float(np.nanmean(a))


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
    """MAE, RMSE, and directional accuracy from regression predictions."""
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


def kendall_per_timestamp(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    min_valid: int = 3,
) -> np.ndarray:
    """Kendall tau per row; NaN where insufficient finite pairs or zero variance."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    out = np.full(scores.shape[0], np.nan, dtype=np.float64)
    for i in range(scores.shape[0]):
        s = scores[i]
        y = y_true[i]
        m = np.isfinite(s) & np.isfinite(y)
        if int(m.sum()) < min_valid:
            continue
        ss, yy = s[m], y[m]
        if np.nanstd(ss) < 1e-12 or np.nanstd(yy) < 1e-12:
            continue
        tau, _ = kendalltau(ss, yy)
        if np.isfinite(tau):
            out[i] = float(tau)
    return out


def spearman_per_timestamp(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    min_valid: int = 3,
) -> np.ndarray:
    """Spearman correlation per row; NaN where insufficient finite pairs."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    out = np.full(scores.shape[0], np.nan, dtype=np.float64)
    for i in range(scores.shape[0]):
        s = scores[i]
        y = y_true[i]
        m = np.isfinite(s) & np.isfinite(y)
        if int(m.sum()) < min_valid:
            continue
        ss, yy = s[m], y[m]
        if np.nanstd(ss) < 1e-12 or np.nanstd(yy) < 1e-12:
            continue
        rs = rankdata(ss, method="average")
        ry = rankdata(yy, method="average")
        rs = (rs - rs.mean()) / (rs.std(ddof=0) + 1e-30)
        ry = (ry - ry.mean()) / (ry.std(ddof=0) + 1e-30)
        out[i] = float((rs * ry).mean())
    return out


def top_k_hit_rate(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    k: int = 2,
    min_valid: int = 3,
) -> float:
    """Fraction of rows where the top-k by ``scores`` intersects top-k by ``y_true``."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    k = int(k)
    hits = 0
    total = 0
    n_s = scores.shape[1]
    kk = min(k, n_s)
    for i in range(scores.shape[0]):
        s = scores[i]
        y = y_true[i]
        m = np.isfinite(s) & np.isfinite(y)
        if int(m.sum()) < min_valid:
            continue
        idx = np.where(m)[0]
        s_sub = s[m]
        y_sub = y[m]
        top_pred = set(idx[np.argsort(-s_sub)[:kk]])
        top_true = set(idx[np.argsort(-y_sub)[:kk]])
        hits += int(len(top_pred & top_true) > 0)
        total += 1
    return float(hits / total) if total else float("nan")


def ndcg_at_k_per_timestamp(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    k: int = 3,
    min_valid: int = 3,
) -> np.ndarray:
    """Simple NDCG per row using ``y_true`` (shifted to non-negative) as gains."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    k = int(k)
    out = np.full(scores.shape[0], np.nan, dtype=np.float64)
    for i in range(scores.shape[0]):
        s = scores[i]
        y = y_true[i]
        m = np.isfinite(s) & np.isfinite(y)
        if int(m.sum()) < min_valid:
            continue
        s_sub = s[m]
        y_sub = y[m]
        kk = min(k, len(s_sub))
        order = np.argsort(-s_sub)
        rel = y_sub - np.nanmin(y_sub)
        rel = np.maximum(rel, 0.0)
        gains = rel[order[:kk]]
        discounts = np.log2(np.arange(2, kk + 2))
        dcg = float(np.sum(gains / discounts))
        ideal_order = np.argsort(-rel)
        ig = rel[ideal_order[:kk]]
        idcg = float(np.sum(ig / discounts))
        out[i] = dcg / idcg if idcg > 0 else np.nan
    return out


def masked_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """MAE / RMSE on finite entries (optional boolean mask)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    if not m.any():
        return {"mae": float("nan"), "rmse": float("nan")}
    yt = y_true[m]
    yp = y_pred[m]
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {"mae": mae, "rmse": rmse}


def per_sector_metric_summary(
    scores: np.ndarray,
    y_true: np.ndarray,
    symbols: tuple[str, ...],
    sectors: np.ndarray,
    *,
    min_valid: int = 2,
) -> dict[str, dict[str, float]]:
    """Spearman summary per sector label."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    uniq = np.unique(sectors)
    for sec in uniq:
        idx = [i for i in range(len(symbols)) if sectors[i] == sec]
        if len(idx) < 2:
            continue
        sub_s = scores[:, idx]
        sub_y = y_true[:, idx]
        mv = min(min_valid, len(idx))
        rho = spearman_per_timestamp(sub_s, sub_y, min_valid=max(2, mv))
        out[str(sec)] = {
            "spearman_mean": float(np.nanmean(rho)),
            "n_symbols": float(len(idx)),
        }
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
