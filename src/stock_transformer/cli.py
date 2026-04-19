"""CLI: walk-forward backtest from YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from stock_transformer.backtest.runner import run_from_config_path


def main(argv: list[str] | None = None) -> int:
    """Dispatch is implemented in ``run_from_config_path`` → ``run_experiment_dispatch`` (universe vs single-symbol)."""
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Stock transformer walk-forward forecast evaluation "
        "(single-symbol multi-timeframe or universe mode via experiment_mode in YAML)"
    )
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to experiment YAML",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random-walk candles (no Alpha Vantage API calls)",
    )
    args = p.parse_args(argv)
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1
    summary = run_from_config_path(args.config, synthetic=args.synthetic)
    print("Run complete. Artifacts:", summary.get("run_dir"))
    if summary.get("fold_errors"):
        return 2
    if summary.get("error") in ("partial_failure", "no_folds"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
