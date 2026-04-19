"""End-to-end walk-forward experiment: multi-timeframe autoregressive candle prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from stock_transformer.backtest.artifacts import save_config_snapshot, save_predictions_csv
from stock_transformer.backtest.context import RunContext
from stock_transformer.backtest.env_config import apply_stx_env_overrides
from stock_transformer.backtest.metrics import (
    aggregate_fold_metrics,
    classification_metrics,
    regression_metrics,
)
from stock_transformer.backtest.run_helpers import allocate_run_dir, append_fold_error_log, fold_error_record
from stock_transformer.backtest.training import train_candle_transformer
from stock_transformer.backtest.walkforward import (
    WalkForwardConfig,
    assert_fold_chronology,
    generate_folds,
)
from stock_transformer.config_models import coerce_experiment_config, coerce_single_symbol_config
from stock_transformer.data.alphavantage import AlphaVantageClient, fetch_candles_for_timeframe
from stock_transformer.data.synthetic import synthetic_multitimeframe_candles
from stock_transformer.device import resolve_device
from stock_transformer.features.sequences import (
    N_CANDLE_FEATURES,
    TIMEFRAME_IDS,
    build_multitimeframe_samples,
)
from stock_transformer.model.transformer_classifier import CandleTransformer, batch_predict

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment file into a plain dict (no validation yet)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Grid-search a probability threshold on validation labels to maximize F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        m = classification_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    return best_t


def run_experiment_dispatch(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
) -> dict[str, Any]:
    """Load-free dispatch: universe vs single-symbol experiment."""
    if str(config.get("experiment_mode", "single_symbol")).lower() == "universe":
        from stock_transformer.backtest.universe_runner import run_universe_experiment

        return run_universe_experiment(config, use_synthetic=use_synthetic)
    return run_experiment(config, use_synthetic=use_synthetic)


def run_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    skip_fetch: bool = False,
) -> dict[str, Any]:
    """Run multi-timeframe walk-forward experiment; write artifacts (single-symbol mode only)."""
    config = coerce_single_symbol_config(config)

    device = resolve_device(str(config.get("device", "auto")))
    cache_dir = Path(str(config.get("cache_dir", "data")))
    art = Path(str(config.get("artifacts_dir", "artifacts")))
    run_dir = allocate_run_dir(art, "run")
    save_config_snapshot(run_dir, config)
    infer_bs = int(config.get("inference_batch_size", 256))

    symbol = str(config["symbol"])
    timeframes: list[str] = list(config["timeframes"])
    prediction_tf = str(config.get("prediction_timeframe", "daily"))

    lookbacks: dict[str, int] = config.get("lookbacks", {})
    default_lb = int(config.get("lookback", 32))
    for tf in timeframes:
        lookbacks.setdefault(tf, default_lb)

    max_seq_len = int(config.get("max_seq_len", 256))

    wf = WalkForwardConfig(
        train_bars=int(config["train_bars"]),
        val_bars=int(config["val_bars"]),
        test_bars=int(config["test_bars"]),
        step_bars=int(config["step_bars"]),
    )

    if use_synthetic:
        logger.info("Using synthetic multi-timeframe candles (symbol=%s)", symbol)
        candles_by_tf = synthetic_multitimeframe_candles(
            n_daily=int(config.get("synthetic_n_daily", 1200)),
            symbol=symbol,
            seed=int(config.get("seed", 42)),
            timeframes=timeframes,
        )
    else:
        logger.info("Fetching candles from cache/API (symbol=%s, timeframes=%s)", symbol, timeframes)
        client = AlphaVantageClient(cache_dir=cache_dir)
        candles_by_tf = {}
        for tf in timeframes:
            candles_by_tf[tf] = fetch_candles_for_timeframe(
                client,
                symbol,
                tf,
                use_adjusted_daily=bool(config.get("use_adjusted_daily", True)),
                use_adjusted_weekly=bool(config.get("use_adjusted_weekly", True)),
                use_adjusted_monthly=bool(config.get("use_adjusted_monthly", True)),
                intraday_month=config.get("intraday_month"),
                intraday_extended_hours=bool(config.get("intraday_extended_hours", False)),
                intraday_outputsize=str(config.get("intraday_outputsize", "full")),
                daily_outputsize=str(config.get("daily_outputsize", "full")),
            )

    if prediction_tf not in candles_by_tf:
        raise ValueError(f"prediction_timeframe '{prediction_tf}' not in fetched timeframes {list(candles_by_tf)}")

    X_feat, X_tf_ids, X_mask, y_reg, y_dir, ts_pred = build_multitimeframe_samples(
        candles_by_tf,
        prediction_tf=prediction_tf,
        lookbacks=lookbacks,
        max_seq_len=max_seq_len,
    )

    n_samples = X_feat.shape[0]
    folds = generate_folds(n_samples, wf)
    logger.info("Built %s samples, %s folds (device=%s)", n_samples, len(folds), device)

    ctx = RunContext.create(run_dir, device, config)
    ctx.summary.update(
        {
            "symbol": symbol,
            "prediction_timeframe": prediction_tf,
            "timeframes": timeframes,
            "n_samples": n_samples,
            "n_folds": len(folds),
            "device": str(device),
            "folds": [],
            "run_dir": str(run_dir),
            "git_sha": ctx.git_sha,
        }
    )

    if not folds:
        ctx.summary["error"] = "no_folds"
        ctx.finalize()
        return ctx.summary

    fold_rows: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []
    fold_errors: list[dict[str, Any]] = []

    try:
        for fold in folds:
            try:
                assert_fold_chronology(ts_pred, fold)
                logger.info("Fold %s: train/val/test slices", fold.fold_id)

                sl_tr, sl_va, sl_te = fold.train, fold.val, fold.test

                n_tf = len(TIMEFRAME_IDS)
                model = CandleTransformer(
                    n_candle_features=N_CANDLE_FEATURES,
                    n_timeframes=n_tf,
                    d_model=int(config["d_model"]),
                    nhead=int(config["nhead"]),
                    num_layers=int(config["num_layers"]),
                    dim_feedforward=int(config["dim_feedforward"]),
                    dropout=float(config["dropout"]),
                    max_seq_len=int(config.get("max_seq_len", 256)),
                ).to(device)
                log_path = run_dir / f"training_log_fold_{fold.fold_id}.csv"
                model = train_candle_transformer(
                    X_feat[sl_tr],
                    X_tf_ids[sl_tr],
                    X_mask[sl_tr],
                    y_reg[sl_tr],
                    y_dir[sl_tr],
                    X_feat[sl_va],
                    X_tf_ids[sl_va],
                    X_mask[sl_va],
                    y_reg[sl_va],
                    y_dir[sl_va],
                    model,
                    config,
                    device,
                    log_path=log_path,
                )

                candle_va, prob_va = batch_predict(
                    model, X_feat[sl_va], X_tf_ids[sl_va], X_mask[sl_va], device, batch_size=infer_bs
                )
                thr = tune_threshold(y_dir[sl_va], prob_va)

                candle_te, prob_te = batch_predict(
                    model, X_feat[sl_te], X_tf_ids[sl_te], X_mask[sl_te], device, batch_size=infer_bs
                )
                m_cls = classification_metrics(y_dir[sl_te], prob_te, threshold=thr)
                m_reg = regression_metrics(y_reg[sl_te], candle_te)

                row: dict[str, Any] = {"fold_id": fold.fold_id, "threshold": thr}
                row.update({f"test_cls_{k}": v for k, v in m_cls.items()})
                row.update({f"test_reg_{k}": v for k, v in m_reg.items()})
                fold_rows.append(row)

                pred_df = pd.DataFrame(
                    {
                        "timestamp": ts_pred.iloc[sl_te].values
                        if hasattr(ts_pred, "iloc")
                        else ts_pred[sl_te.start : sl_te.stop],
                        "symbol": symbol,
                        "y_dir_true": y_dir[sl_te],
                        "y_dir_prob": prob_te,
                        "y_dir_pred": (prob_te >= thr).astype(int),
                        "y_close_ret_true": y_reg[sl_te, 3] if y_reg.ndim == 2 else y_reg[sl_te],
                        "y_close_ret_pred": candle_te[:, 3] if candle_te.ndim == 2 else candle_te,
                        "fold_id": fold.fold_id,
                    }
                )
                all_preds.append(pred_df)
            except Exception as exc:  # noqa: BLE001 — isolate fold failures; log and continue
                rec = fold_error_record(fold.fold_id, exc)
                append_fold_error_log(run_dir, rec)
                fold_errors.append(rec)
                logger.exception("Fold %s failed", fold.fold_id)
                continue
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving partial single-symbol results")
        ctx.summary["error"] = "interrupted"
        ctx.summary["folds"] = fold_rows
        if fold_errors:
            ctx.summary["fold_errors"] = fold_errors
        pred_path = run_dir / f"predictions__{symbol}.csv"
        _pred_cols = [
            "timestamp",
            "symbol",
            "y_dir_true",
            "y_dir_prob",
            "y_dir_pred",
            "y_close_ret_true",
            "y_close_ret_pred",
            "fold_id",
        ]
        save_predictions_csv(
            pred_path, pd.concat(all_preds, ignore_index=True) if all_preds else None, columns=_pred_cols
        )
        ctx.finalize()
        raise

    ctx.summary["folds"] = fold_rows
    if fold_errors:
        ctx.summary["error"] = "partial_failure"
        ctx.summary["fold_errors"] = fold_errors
    if fold_rows:
        ctx.summary["aggregate"] = aggregate_fold_metrics(fold_rows)

    pred_path = run_dir / f"predictions__{symbol}.csv"
    _pred_cols = [
        "timestamp",
        "symbol",
        "y_dir_true",
        "y_dir_prob",
        "y_dir_pred",
        "y_close_ret_true",
        "y_close_ret_pred",
        "fold_id",
    ]
    save_predictions_csv(pred_path, pd.concat(all_preds, ignore_index=True) if all_preds else None, columns=_pred_cols)

    ctx.finalize()
    logger.info("Run finished: %s", run_dir)
    return ctx.summary


def run_from_config_path(
    path: str | Path,
    *,
    synthetic: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Load YAML, apply env + device overrides (CLI > env > file), validate, run."""
    cfg = load_config(path)
    apply_stx_env_overrides(cfg)
    if device is not None and str(device).strip():
        cfg["device"] = str(device).strip()
    cfg = coerce_experiment_config(cfg)
    return run_experiment_dispatch(cfg, use_synthetic=synthetic)
