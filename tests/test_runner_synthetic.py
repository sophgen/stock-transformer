"""End-to-end runner on synthetic data (no API)."""

import pytest

from stock_transformer.backtest.runner import run_experiment


def test_run_experiment_synthetic_smoke():
    cfg = {
        "symbol": "TST",
        "timeframes": ["daily"],
        "lookback": 12,
        "train_bars": 120,
        "val_bars": 30,
        "test_bars": 30,
        "step_bars": 40,
        "use_adjusted_daily": True,
        "use_adjusted_monthly": True,
        "intraday_extended_hours": False,
        "intraday_outputsize": "compact",
        "daily_outputsize": "compact",
        "cache_dir": "data",
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 0,
        "default_threshold": 0.5,
        "artifacts_dir": "artifacts",
    }
    out = run_experiment(cfg, use_synthetic=True)
    assert "run_dir" in out
    assert "daily" in out["timeframes"]
    tf = out["timeframes"]["daily"]
    assert "error" not in tf
    assert tf["n_folds"] >= 1
