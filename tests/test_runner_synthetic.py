"""End-to-end runner on synthetic multi-timeframe data (no API)."""

from stock_transformer.backtest.runner import run_experiment


def test_run_experiment_synthetic_smoke():
    cfg = {
        "symbol": "TST",
        "timeframes": ["monthly", "weekly", "daily"],
        "prediction_timeframe": "daily",
        "lookbacks": {"monthly": 6, "weekly": 8, "daily": 16},
        "max_seq_len": 64,
        "train_bars": 100,
        "val_bars": 30,
        "test_bars": 30,
        "step_bars": 40,
        "use_adjusted_daily": True,
        "use_adjusted_weekly": True,
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
        "epochs": 2,
        "batch_size": 32,
        "learning_rate": 0.01,
        "loss_alpha": 0.5,
        "device": "cpu",
        "seed": 0,
        "default_threshold": 0.5,
        "artifacts_dir": "artifacts",
        "synthetic_n_daily": 800,
    }
    out = run_experiment(cfg, use_synthetic=True)
    assert "run_dir" in out
    assert out["n_folds"] >= 1
    assert "aggregate" in out
    assert "folds" in out
    assert len(out["folds"]) >= 1

    fold0 = out["folds"][0]
    assert "test_cls_accuracy" in fold0
    assert "test_reg_mae" in fold0
