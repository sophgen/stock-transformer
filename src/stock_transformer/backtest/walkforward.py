"""Rolling-origin walk-forward splits on time-ordered window samples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardConfig:
    """Lengths are in **window samples** (one per prediction time), not raw bars."""

    train_bars: int
    val_bars: int
    test_bars: int
    step_bars: int


@dataclass(frozen=True)
class FoldSlices:
    fold_id: int
    train: slice
    val: slice
    test: slice


def generate_folds(n_windows: int, cfg: WalkForwardConfig) -> list[FoldSlices]:
    """Non-overlapping train/val/test blocks; advance start by ``step_bars`` each fold."""
    folds: list[FoldSlices] = []
    pos = 0
    fid = 0
    while True:
        tr_end = pos + cfg.train_bars
        va_end = tr_end + cfg.val_bars
        te_end = va_end + cfg.test_bars
        if te_end > n_windows:
            break
        folds.append(
            FoldSlices(
                fold_id=fid,
                train=slice(pos, tr_end),
                val=slice(tr_end, va_end),
                test=slice(va_end, te_end),
            )
        )
        pos += cfg.step_bars
        fid += 1
    return folds


def assert_fold_chronology(end_timestamps, fold: FoldSlices) -> None:
    """Enforce max(train) < min(val) < min(test) on window end times."""
    import pandas as pd

    ts = pd.Series(pd.to_datetime(end_timestamps))
    tr_max = ts.iloc[fold.train].max()
    va_min = ts.iloc[fold.val].min()
    te_min = ts.iloc[fold.test].min()
    if not (tr_max < va_min < te_min):
        raise AssertionError(
            f"Fold {fold.fold_id} chronology violated: train_max={tr_max}, val_min={va_min}, test_min={te_min}"
        )
