#!/usr/bin/env python3
"""CLI for bulk AlphaVantage download (see docs/data_download.md)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from stock_transformer.av_download import run_download  # noqa: E402


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Bulk download from AlphaVantage")
    p.add_argument(
        "-c", "--config", default="configs/download.yaml",
        help="Path to download manifest YAML",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print planned call count only (no network)",
    )
    p.add_argument(
        "--retry-errors", action="store_true",
        help="Only retry (symbol, function) pairs from _errors_latest.json",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Bypass disk cache / TTL (force refetch)",
    )
    p.add_argument(
        "--symbols", default=None,
        help="Comma-separated symbols (overrides symbols_file in config)",
    )
    args = p.parse_args()
    override = (
        [x.strip() for x in args.symbols.split(",") if x.strip()]
        if args.symbols
        else None
    )
    summary = run_download(
        args.config,
        dry_run=args.dry_run,
        retry_errors=args.retry_errors,
        no_cache=args.no_cache,
        symbols_override=override,
    )
    print(summary)
    if summary.interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
