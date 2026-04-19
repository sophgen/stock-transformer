"""End-to-end universe experiment on synthetic data (no API)."""

from __future__ import annotations

from pathlib import Path

import yaml

from stock_transformer.backtest.universe_runner import run_universe_experiment


def test_universe_synthetic_smoke(tmp_path: Path):
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "universe.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["artifacts_dir"] = str(tmp_path / "art")
    cfg["device"] = "cpu"
    cfg["train_bars"] = 80
    cfg["val_bars"] = 20
    cfg["test_bars"] = 20
    cfg["step_bars"] = 20
    cfg["epochs"] = 1
    cfg["synthetic_n_bars"] = 400
    summary = run_universe_experiment(cfg, use_synthetic=True)
    assert summary.get("error") != "no_folds"
    assert summary["n_folds"] >= 1
    assert "aggregate" in summary
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "predictions_universe.csv").exists()
