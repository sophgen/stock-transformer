"""Shared run metadata for walk-forward experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from stock_transformer.backtest.artifacts import save_summary
from stock_transformer.backtest.run_helpers import git_head_short


@dataclass
class RunContext:
    """Holds mutable run state and writes ``summary.json`` in one place."""

    run_dir: Path
    device: torch.device
    config: dict[str, Any]
    git_sha: str
    summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        run_dir: Path,
        device: torch.device,
        config: dict[str, Any],
    ) -> RunContext:
        """Build a context with a fresh summary dict and current ``git_sha`` (if any)."""
        return cls(
            run_dir=run_dir,
            device=device,
            config=config,
            git_sha=git_head_short(),
            summary={},
        )

    def finalize(self) -> None:
        """Persist ``summary.json`` under ``run_dir``."""
        save_summary(self.run_dir, self.summary)
