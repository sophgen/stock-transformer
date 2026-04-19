"""Subprocess helper: mocked partial failure → exit 2 (no real training)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from click.testing import CliRunner

from stock_transformer.cli import cli


def main() -> int:
    def boom(*_a, **_k):
        return {"run_dir": str(REPO / "artifacts"), "fold_errors": [{"fold_id": 0, "error": "x"}]}

    with patch("stock_transformer.cli.run_experiment", boom):
        r = CliRunner().invoke(cli, ["backtest", "-c", str(REPO / "configs" / "default.yaml")])
        return r.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
