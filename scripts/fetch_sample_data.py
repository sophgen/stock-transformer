"""Thin wrapper: ``python scripts/fetch_sample_data.py`` → ``stx fetch``."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from stock_transformer.cli import main


def main_script() -> int:
    load_dotenv()
    return main(["fetch", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main_script())
