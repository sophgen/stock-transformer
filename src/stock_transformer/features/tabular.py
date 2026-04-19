"""Flatten universe tensors for tabular baselines (LightGBM / ridge)."""

from __future__ import annotations

import numpy as np


def flatten_universe_sample(
    X: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_flat, y_flat, group_ids, sym_ids).

    One row per (n, s) where ``mask[n, -1, s]`` is False AND ``isfinite(y[n, s])``.

    Parameters
    ----------
    X
        ``[N, L, S, F]`` float32
    mask
        ``[N, L, S]`` bool, True = padded / invalid
    y
        ``[N, S]`` float32, NaN = no label
    """
    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(y, dtype=np.float32)
    n, ell, s, f = X.shape
    last_ok = ~mask[:, -1, :]
    y_ok = np.isfinite(y)
    take = last_ok & y_ok
    if not take.any():
        return (
            np.empty((0, ell * f), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    gn, gs = np.where(take)
    X_flat = X[gn, :, gs, :].reshape(-1, ell * f).astype(np.float32, copy=False)
    y_flat = y[gn, gs].astype(np.float64, copy=False)
    group_ids = gn.astype(np.int64, copy=False)
    sym_ids = gs.astype(np.int64, copy=False)
    return X_flat, y_flat, group_ids, sym_ids
