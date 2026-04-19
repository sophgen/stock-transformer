"""M9b aggregate.per_sector in summary."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from stock_transformer.backtest.universe_runner import run_universe_experiment


def test_summary_has_per_sector_block(tmp_path):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "universe.yaml").read_text())
    cfg["artifacts_dir"] = str(tmp_path / "art")
    cfg["device"] = "cpu"
    cfg["train_bars"] = 50
    cfg["val_bars"] = 15
    cfg["test_bars"] = 15
    cfg["step_bars"] = 15
    cfg["epochs"] = 1
    cfg["synthetic_n_bars"] = 220
    out = run_universe_experiment(cfg, use_synthetic=True)
    run_dir = Path(out["run_dir"])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "aggregate" in summary
    assert "per_sector" in summary["aggregate"]
