"""Thin wrapper: ``python scripts/sweep_loss.py`` → ``stx sweep``."""

from __future__ import annotations

import sys

from stock_transformer.cli import main


def main_script() -> int:
    return main(["sweep", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main_script())
