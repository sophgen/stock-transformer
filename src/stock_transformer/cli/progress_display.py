"""Optional Rich-backed stderr lines for fold/epoch progress (training stays in ``backtest/``)."""

from __future__ import annotations

from typing import Any

import click

from stock_transformer.backtest.progress import ProgressCallback


class StxCliProgress:
    """Implements ``ProgressCallback`` using plain stderr or Rich when available.

    Training code only sees the protocol in :mod:`stock_transformer.backtest.progress`;
    this class is the optional human-facing adapter so CI and headless runs stay quiet
    when Rich is off or missing.
    """

    def __init__(self, *, use_rich: bool) -> None:
        self._use_rich = use_rich
        self._console: Any = None
        if use_rich:
            try:
                from rich.console import Console

                self._console = Console(stderr=True, highlight=False)
            except ImportError:
                self._use_rich = False

    def on_fold_start(self, fold_id: int, total_folds: int) -> None:
        msg = f"[stx] fold {fold_id + 1}/{total_folds}"
        if self._console is not None:
            self._console.print(f"[bold cyan]{msg}[/]")
        else:
            click.echo(msg, err=True)

    def on_epoch_end(
        self,
        fold_id: int,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        tl = float(metrics.get("train_loss", float("nan")))
        vl = float(metrics.get("val_loss", float("nan")))
        line = f"[stx] fold {fold_id} epoch {epoch + 1}/{total_epochs} train_loss={tl:.6f} val_loss={vl:.6f}"
        if self._console is not None:
            self._console.print(line)
        else:
            click.echo(line, err=True)

    def on_fold_end(self, fold_id: int, summary: dict[str, Any]) -> None:
        line = f"[stx] fold {fold_id} training complete"
        if self._console is not None:
            self._console.print(f"[green]{line}[/]")
        else:
            click.echo(line, err=True)


def backtest_progress_callback(ctx: click.Context, *, dry_run: bool) -> ProgressCallback | None:
    """Return a callback unless the user asked for quiet or dry-run (no training noise)."""
    root = ctx.find_root()
    root.ensure_object(dict)
    if dry_run or root.obj.get("quiet"):
        return None
    return StxCliProgress(use_rich=bool(root.obj.get("use_rich")))
