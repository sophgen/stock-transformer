"""Train-fold-only normalization for universe tensors."""

from __future__ import annotations

import numpy as np


class TrainOnlyScaler:
    """Per-feature mean/std over all positions where ``mask`` is False (valid)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, mask: np.ndarray) -> TrainOnlyScaler:
        X = np.asarray(X, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if X.ndim != 4:
            raise ValueError(f"Expected X [N,L,S,F], got shape {X.shape}")
        n, ell, s, f = X.shape
        if mask.shape != (n, ell, s):
            raise ValueError("mask must match X[...,0].shape")
        valid = ~mask
        valid = np.broadcast_to(valid[..., None], (n, ell, s, f))
        sum_w = np.zeros(f, dtype=np.float64)
        sum_x = np.zeros(f, dtype=np.float64)
        sum_x2 = np.zeros(f, dtype=np.float64)
        for fi in range(f):
            m = valid[..., fi]
            if not np.any(m):
                sum_w[fi] = 0
                continue
            xv = X[..., fi][m]
            sum_w[fi] = float(xv.size)
            sum_x[fi] = float(xv.sum())
            sum_x2[fi] = float(np.dot(xv, xv))
        mean = np.zeros(f, dtype=np.float64)
        std = np.ones(f, dtype=np.float64)
        for fi in range(f):
            w = sum_w[fi]
            if w < 1:
                mean[fi] = 0.0
                std[fi] = 1.0
                continue
            mean[fi] = sum_x[fi] / w
            var = max(sum_x2[fi] / w - mean[fi] ** 2, 0.0)
            std[fi] = float(np.sqrt(var)) if var > 1e-20 else 1.0
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        X = np.asarray(X, dtype=np.float32)
        m = self.mean_.astype(np.float32, copy=False)
        s = self.std_.astype(np.float32, copy=False)
        return (X - m) / s
