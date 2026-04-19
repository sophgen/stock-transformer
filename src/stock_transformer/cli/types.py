"""Small CLI-facing types shared between Click handlers and tooling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StxResult:
    """Structured exit payload for programmatic callers (exit code + optional stderr line).

    Keeps embedders from parsing Click or logging output when they invoke
    :func:`stock_transformer.cli.services.run_backtest` directly.
    """

    code: int
    message: str | None = None
