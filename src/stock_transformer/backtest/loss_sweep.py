"""Run multiple training losses and merge results into one ``summary.json`` (M10)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stock_transformer.backtest.universe_runner import run_universe_experiment


def run_loss_sweep(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    use_synthetic: bool = False,
    sweep_dir: Path | None = None,
) -> dict[str, Any]:
    """Run ``mse``, ``listnet``, ``approx_ndcg``; write ``summary.json`` with top-level ``by_loss``."""
    art = Path(config.get("artifacts_dir", "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    if sweep_dir is None:
        sweep_dir = art / f"universe_sweep_loss_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "config_snapshot.yaml").write_text(yaml.dump(config, sort_keys=True))

    by_loss: dict[str, object] = {}
    for loss in ("mse", "listnet", "approx_ndcg"):
        cfg = deepcopy(config)
        cfg["loss"] = loss
        out = run_universe_experiment(cfg, use_synthetic=use_synthetic)
        by_loss[loss] = {
            "aggregate": out.get("aggregate"),
            "folds": out.get("folds"),
            "run_dir": out.get("run_dir"),
        }

    merged: dict[str, Any] = {
        "experiment": "sweep_loss",
        "by_loss": by_loss,
        "config": str(config_path.resolve()) if config_path else None,
        "sweep_dir": str(sweep_dir),
    }
    summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps(merged, indent=2, default=str))
    merged["summary_path"] = str(summary_path)
    return merged
