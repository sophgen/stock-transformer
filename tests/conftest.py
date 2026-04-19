"""Shared pytest fixtures for CLI and integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def stx_runner() -> CliRunner:
    """CliRunner for invoking ``stx`` in tests (propagate exceptions like real CLI)."""
    # Click 8.3+ CliRunner keeps stdout/stderr split; ``mix_stderr`` was removed.
    return CliRunner(catch_exceptions=False)


@pytest.fixture
def quick_universe_yaml(tmp_path: Path) -> Path:
    """Minimal universe config for fast synthetic CLI runs."""
    cfg = yaml.safe_load((REPO / "configs" / "universe.yaml").read_text())
    cfg["artifacts_dir"] = str(tmp_path / "a")
    cfg["device"] = "cpu"
    cfg["train_bars"] = 50
    cfg["val_bars"] = 15
    cfg["test_bars"] = 15
    cfg["step_bars"] = 15
    cfg["epochs"] = 1
    cfg["synthetic_n_bars"] = 200
    p = tmp_path / "u.yaml"
    p.write_text(yaml.dump(cfg))
    return p
