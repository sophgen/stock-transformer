"""M7b required artifact files per run."""

from __future__ import annotations

from pathlib import Path

import yaml

from stock_transformer.backtest.universe_runner import run_universe_experiment


def test_universe_run_writes_all_artifacts(tmp_path):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "universe.yaml").read_text())
    cfg["artifacts_dir"] = str(tmp_path / "art")
    cfg["device"] = "cpu"
    cfg["train_bars"] = 50
    cfg["val_bars"] = 15
    cfg["test_bars"] = 15
    cfg["step_bars"] = 15
    cfg["epochs"] = 1
    cfg["synthetic_n_bars"] = 220
    summary = run_universe_experiment(cfg, use_synthetic=True)
    run_dir = Path(summary["run_dir"])
    for name in (
        "config_snapshot.yaml",
        "universe_membership.json",
        "feature_schema.json",
        "folds.json",
        "summary.json",
        "predictions_universe.csv",
    ):
        assert (run_dir / name).exists(), name
