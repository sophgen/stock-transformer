"""M10: loss sweep writes merged ``summary.json`` with ``by_loss``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from stock_transformer.backtest.loss_sweep import run_loss_sweep


@pytest.fixture
def minimal_cfg(tmp_path: Path) -> dict:
    return {
        "artifacts_dir": str(tmp_path),
        "experiment_mode": "universe",
        "symbols": ["MSTR", "IBIT", "COIN", "QQQ"],
        "target_symbol": "MSTR",
        "timeframe": "daily",
        "lookback": 16,
        "min_coverage_symbols": 3,
        "label_mode": "cross_sectional_return",
        "train_bars": 80,
        "val_bars": 20,
        "test_bars": 20,
        "step_bars": 20,
        "d_model": 32,
        "nhead": 4,
        "num_temporal_layers": 1,
        "num_cross_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.1,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "seed": 0,
        "loss": "mse",
    }


def test_sweep_writes_summary_with_by_loss(tmp_path: Path, minimal_cfg: dict) -> None:
    sweep_root = tmp_path / "sweep_runs"

    def _fake_run(cfg: dict, *, use_synthetic: bool = False) -> dict:
        loss = str(cfg.get("loss", "mse"))
        rd = tmp_path / f"run_{loss}"
        rd.mkdir(parents=True, exist_ok=True)
        return {
            "aggregate": {"spearman_mean": 0.1 if loss == "mse" else 0.2},
            "folds": [{"fold_id": 0, "loss": loss}],
            "run_dir": str(rd),
        }

    with patch("stock_transformer.backtest.loss_sweep.run_universe_experiment", side_effect=_fake_run):
        out = run_loss_sweep(minimal_cfg, config_path=Path("configs/universe.yaml"), sweep_dir=sweep_root)

    assert out["experiment"] == "sweep_loss"
    assert set(out["by_loss"].keys()) == {"mse", "listnet", "approx_ndcg"}
    for loss, block in out["by_loss"].items():
        assert block["aggregate"]["spearman_mean"] == (0.1 if loss == "mse" else 0.2)
        assert "run_dir" in block

    sp = Path(out["summary_path"])
    assert sp.exists()
    assert sp.parent == sweep_root
    text = sp.read_text(encoding="utf-8")
    assert '"by_loss"' in text
    assert '"listnet"' in text
