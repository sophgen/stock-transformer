"""Fetch Alpha Vantage daily-adjusted candles for the pilot universe into local cache.

One-shot developer helper: populates ``data/raw/`` (raw JSON) and ``data/canonical/``
(partitioned CSV) for offline runs with ``stx-backtest -c configs/sample.yaml``.

Requires ``ALPHAVANTAGE_API_KEY`` in the environment (or pass ``api_key`` if extended).
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from stock_transformer.data.alphavantage import AlphaVantageClient, fetch_candles_for_universe

UNIVERSE: tuple[str, ...] = ("MSTR", "IBIT", "COIN", "QQQ")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Fetch daily-adjusted OHLCV for the universe into cache_dir.",
        epilog="GPU/MPS applies to training only: use `stx-backtest --device mps` (this script has no --device).",
    )
    p.add_argument("--cache-dir", default="data", help="Root for raw/ and canonical/ (default: data)")
    p.add_argument("--symbols", nargs="+", default=list(UNIVERSE), help=f"Symbols (default: {' '.join(UNIVERSE)})")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download from API and overwrite canonical CSV (ignores raw JSON cache)",
    )
    args = p.parse_args(argv)

    use_cache = not args.refresh
    client = AlphaVantageClient(cache_dir=args.cache_dir)
    candles = fetch_candles_for_universe(
        client,
        args.symbols,
        "daily",
        use_adjusted_daily=True,
        daily_outputsize="full",
        use_cache=use_cache,
        force_refresh_canonical=args.refresh,
        store="csv",
        data_source="rest",
    )
    for sym in sorted(candles.keys()):
        df = candles[sym]
        ts = df["timestamp"]
        print(f"{sym}: {len(df)} rows, {ts.min().date()} → {ts.max().date()}")
    print(f"Cache root: {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
