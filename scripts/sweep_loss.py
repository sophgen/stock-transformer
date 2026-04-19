#!/usr/bin/env python3
"""Run universe experiment for each loss and merge ``by_loss`` into one summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from stock_transformer.backtest.loss_sweep import run_loss_sweep


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep loss= mse, listnet, approx_ndcg")
    p.add_argument("-c", "--config", type=Path, default=Path("configs/universe.yaml"))
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    merged = run_loss_sweep(base, config_path=args.config, use_synthetic=args.synthetic)
    print(json.dumps(merged, indent=2, default=str))


if __name__ == "__main__":
    main()
