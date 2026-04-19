"""Subprocess helper: KeyboardInterrupt from runner → exit 130 (SIGINT path)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from click.testing import CliRunner

from stock_transformer.cli import cli


def main() -> int:
    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    with patch("stock_transformer.cli.run_experiment", interrupt):
        r = CliRunner().invoke(cli, ["backtest", "-c", str(REPO / "configs" / "default.yaml")])
        return r.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
