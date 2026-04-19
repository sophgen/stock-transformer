#!/usr/bin/env python3
"""Run universe experiment for each loss and merge ``by_loss`` into one summary."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

from stock_transformer.backtest.universe_runner import run_universe_experiment


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep loss= mse, listnet, approx_ndcg")
    p.add_argument("-c", "--config", type=Path, default=Path("configs/universe.yaml"))
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    by_loss: dict[str, object] = {}
    for loss in ("mse", "listnet", "approx_ndcg"):
        cfg = deepcopy(base)
        cfg["loss"] = loss
        out = run_universe_experiment(cfg, use_synthetic=args.synthetic)
        by_loss[loss] = {
            "aggregate": out.get("aggregate"),
            "folds": out.get("folds"),
            "run_dir": out.get("run_dir"),
        }
    merged = {"by_loss": by_loss, "config": str(args.config)}
    print(json.dumps(merged, indent=2, default=str))


if __name__ == "__main__":
    main()
