"""Optional progress hooks for long-running walk-forward training (CLI can wire stderr or Rich)."""

from __future__ import annotations

from typing import Any, Protocol


class ProgressCallback(Protocol):
    """Called from runners and training loops (per fold / per epoch)."""

    def on_fold_start(self, fold_id: int, total_folds: int) -> None: ...

    def on_epoch_end(
        self,
        fold_id: int,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, Any],
    ) -> None: ...

    def on_fold_end(self, fold_id: int, summary: dict[str, Any]) -> None: ...


class NullProgress:
    """No-op progress (default when ``progress`` is omitted)."""

    def on_fold_start(self, fold_id: int, total_folds: int) -> None:
        return

    def on_epoch_end(
        self,
        fold_id: int,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        return

    def on_fold_end(self, fold_id: int, summary: dict[str, Any]) -> None:
        return
