"""Shared training loops for single-symbol and universe ranker models."""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn as nn

from stock_transformer.backtest.progress import NullProgress, ProgressCallback
from stock_transformer.model.losses import training_loss
from stock_transformer.model.transformer_classifier import CandleTransformer
from stock_transformer.model.transformer_ranker import TransformerRanker

logger = logging.getLogger(__name__)

TModule = TypeVar("TModule", bound=nn.Module)


def _append_training_log_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def _run_supervised_epochs(
    model: TModule,
    cfg: dict[str, Any],
    opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    *,
    epochs: int,
    train_epoch: Callable[[], float],
    val_loss: Callable[[], float],
    log_path: Path | None = None,
    progress: ProgressCallback | None = None,
    progress_fold_id: int = 0,
) -> TModule:
    """Optimizer-driven epoch loop with plateau scheduling, early stopping, and CSV logging."""
    prog = progress if progress is not None else NullProgress()
    patience = int(cfg.get("early_stopping_patience", 0))
    best_state = None
    best_val = float("inf")
    stalled = 0

    for epoch in range(epochs):
        train_loss_mean = train_epoch()
        model.eval()
        with torch.no_grad():
            vloss = val_loss()

        lr_now = float(opt.param_groups[0]["lr"])
        if log_path is not None:
            _append_training_log_row(
                log_path,
                {
                    "epoch": epoch,
                    "train_loss": train_loss_mean,
                    "val_loss": vloss,
                    "lr": lr_now,
                },
            )

        if scheduler is not None:
            scheduler.step(vloss)

        improved = vloss < best_val - 1e-12
        if improved:
            best_val = vloss
            best_state = deepcopy(model.state_dict())
            stalled = 0
        else:
            stalled += 1

        if patience > 0 and stalled >= patience:
            break

        logger.debug("epoch %s train=%.6f val=%.6f", epoch, train_loss_mean, vloss)
        prog.on_epoch_end(
            progress_fold_id,
            epoch,
            epochs,
            {"train_loss": train_loss_mean, "val_loss": vloss, "lr": lr_now},
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_candle_transformer(
    X_feat: np.ndarray,
    X_tf: np.ndarray,
    X_mask: np.ndarray,
    y_reg: np.ndarray,
    y_dir: np.ndarray,
    X_feat_va: np.ndarray,
    X_tf_va: np.ndarray,
    X_mask_va: np.ndarray,
    y_reg_va: np.ndarray,
    y_dir_va: np.ndarray,
    model: CandleTransformer,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    log_path: Path | None = None,
    progress: ProgressCallback | None = None,
    progress_fold_id: int = 0,
) -> CandleTransformer:
    """Train with optional early stopping, LR reduction on plateau, and CSV logging."""
    torch.manual_seed(int(cfg.get("seed", 42)))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)
    reg_loss_fn = nn.MSELoss()
    dir_loss_fn = nn.BCEWithLogitsLoss()
    alpha = float(cfg.get("loss_alpha", 0.5))

    use_plateau = bool(cfg.get("lr_reduce_on_plateau", False))
    sched_patience = int(cfg.get("lr_scheduler_patience", 3))
    sched_factor = float(cfg.get("lr_scheduler_factor", 0.5))
    sched_min_lr = float(cfg.get("lr_scheduler_min_lr", 1e-7))
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=sched_factor,
            patience=sched_patience,
            min_lr=sched_min_lr,
        )
        if use_plateau
        else None
    )

    def _to(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.from_numpy(arr).to(dtype).to(device)

    xf_tr, xtf_tr, xm_tr = _to(X_feat), _to(X_tf, torch.long), _to(X_mask, torch.bool)
    yr_tr, yd_tr = _to(y_reg), _to(y_dir)

    xf_va, xtf_va, xm_va = _to(X_feat_va), _to(X_tf_va, torch.long), _to(X_mask_va, torch.bool)
    yr_va, yd_va = _to(y_reg_va), _to(y_dir_va)

    bs = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    n = xf_tr.size(0)

    def train_epoch() -> float:
        model.train()
        perm = torch.randperm(n, device=device)
        train_loss_acc = 0.0
        n_batches = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            c_pred, d_logit = model(xf_tr[idx], xtf_tr[idx], xm_tr[idx])
            loss = alpha * reg_loss_fn(c_pred, yr_tr[idx]) + (1 - alpha) * dir_loss_fn(d_logit, yd_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_acc += float(loss.detach())
            n_batches += 1
        return train_loss_acc / max(n_batches, 1)

    def val_loss_fn() -> float:
        c_pred_v, d_logit_v = model(xf_va, xtf_va, xm_va)
        return alpha * float(reg_loss_fn(c_pred_v, yr_va)) + (1 - alpha) * float(dir_loss_fn(d_logit_v, yd_va))

    return _run_supervised_epochs(
        model,
        cfg,
        opt,
        scheduler,
        epochs=epochs,
        train_epoch=train_epoch,
        val_loss=val_loss_fn,
        log_path=log_path,
        progress=progress,
        progress_fold_id=progress_fold_id,
    )


def train_transformer_ranker(
    X: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    X_va: np.ndarray,
    mask_va: np.ndarray,
    y_va: np.ndarray,
    model: TransformerRanker,
    cfg: dict[str, Any],
    device: torch.device,
    loss_name: str,
    *,
    log_path: Path | None = None,
    progress: ProgressCallback | None = None,
    progress_fold_id: int = 0,
) -> TransformerRanker:
    """Train the universe ranker with the chosen ranking loss."""
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)
    bs = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    gen = torch.Generator()
    gen.manual_seed(int(cfg.get("seed", 42)))

    use_plateau = bool(cfg.get("lr_reduce_on_plateau", False))
    sched_patience = int(cfg.get("lr_scheduler_patience", 3))
    sched_factor = float(cfg.get("lr_scheduler_factor", 0.5))
    sched_min_lr = float(cfg.get("lr_scheduler_min_lr", 1e-7))
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=sched_factor,
            patience=sched_patience,
            min_lr=sched_min_lr,
        )
        if use_plateau
        else None
    )

    def to_t(a: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.from_numpy(a).to(dtype).to(device)

    xm_tr, mk_tr, yt_tr = to_t(X), to_t(mask, torch.bool), to_t(y)
    xm_va, mk_va, yt_va = to_t(X_va), to_t(mask_va, torch.bool), to_t(y_va)

    n = xm_tr.size(0)
    pad_last_tr = mk_tr[:, -1, :]
    pad_last_va = mk_va[:, -1, :]

    def train_epoch() -> float:
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        train_loss_acc = 0.0
        n_batches = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            pred = model(xm_tr[idx], mk_tr[idx])
            label_ok = torch.isfinite(yt_tr[idx]) & (~pad_last_tr[idx])
            loss = training_loss(loss_name, pred, yt_tr[idx], mask_valid=label_ok)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_acc += float(loss.detach())
            n_batches += 1
        return train_loss_acc / max(n_batches, 1)

    def val_loss_fn() -> float:
        pv = model(xm_va, mk_va)
        label_ok_va = torch.isfinite(yt_va) & (~pad_last_va)
        return float(training_loss(loss_name, pv, yt_va, mask_valid=label_ok_va))

    return _run_supervised_epochs(
        model,
        cfg,
        opt,
        scheduler,
        epochs=epochs,
        train_epoch=train_epoch,
        val_loss=val_loss_fn,
        log_path=log_path,
        progress=progress,
        progress_fold_id=progress_fold_id,
    )
