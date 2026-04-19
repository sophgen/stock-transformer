"""Walk-forward cross-sectional ranking experiment (multi-ticker universe)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from stock_transformer.backtest.artifacts import save_config_snapshot, save_fold_payload, save_summary
from stock_transformer.backtest.metrics import (
    aggregate_fold_metrics,
    kendall_per_timestamp,
    masked_regression_metrics,
    ndcg_at_k_per_timestamp,
    per_sector_metric_summary,
    safe_nanmean,
    spearman_per_timestamp,
    top_k_hit_rate,
)
from stock_transformer.backtest.portfolio_sim import (
    Book,
    aggregate_portfolio_sim_folds,
    simulate_topk_portfolio,
)
from stock_transformer.backtest.progress import ProgressCallback
from stock_transformer.backtest.run_helpers import (
    allocate_run_dir,
    append_fold_error_log,
    fold_error_record,
    git_head_short,
)
from stock_transformer.backtest.training import train_transformer_ranker
from stock_transformer.backtest.walkforward import WalkForwardConfig, assert_fold_chronology, generate_folds
from stock_transformer.config_models import coerce_universe_config
from stock_transformer.data.align import align_universe_ohlcv
from stock_transformer.data.alphavantage import AlphaVantageClient, fetch_candles_for_universe
from stock_transformer.data.synthetic import synthetic_universe_candles
from stock_transformer.data.universe import (
    UniverseConfig,
    load_sector_map,
    membership_table_from_panel,
    sectors_for_symbols,
)
from stock_transformer.device import resolve_device
from stock_transformer.features.scaling import TrainOnlyScaler
from stock_transformer.features.tabular import flatten_universe_sample
from stock_transformer.features.universe_tensor import (
    DEFAULT_UNIVERSE_FEATURE_NAMES,
    build_universe_samples,
    feature_schema,
)
from stock_transformer.model.baselines import (
    equal_score_baseline,
    mean_reversion_rank_scores,
    momentum_rank_scores,
)
from stock_transformer.model.baselines_tabular import (
    fit_gbt_ranker,
    fit_linear_cs_ranker,
    scatter_predictions,
)
from stock_transformer.model.transformer_ranker import TransformerRanker, batch_predict

logger = logging.getLogger(__name__)


def _ranks_descending_matrix(vals: np.ndarray) -> np.ndarray:
    """Per-row descending ranks; NaN where the value is non-finite (same as legacy ``_ranks_descending`` per row)."""
    vals = np.asarray(vals, dtype=np.float64)
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    for i in range(vals.shape[0]):
        row = vals[i]
        m = np.isfinite(row)
        if m.any():
            out[i, m] = rankdata(-row[m], method="average")
    return out


def run_universe_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = coerce_universe_config(config)
    u = load_universe_config_from_dict(config)
    device = resolve_device(str(config.get("device", "auto")))
    infer_bs = int(config.get("inference_batch_size", 128))
    cache_dir = Path(str(config.get("cache_dir", "data")))
    art = Path(str(config.get("artifacts_dir", "artifacts")))
    run_dir = allocate_run_dir(art, "universe_run")
    save_config_snapshot(run_dir, config)

    feature_names: tuple[str, ...] = tuple(config.get("features") or list(DEFAULT_UNIVERSE_FEATURE_NAMES))
    fs = feature_schema(feature_names)
    fs_out = {**fs, "git_sha": git_head_short()}
    (run_dir / "feature_schema.json").write_text(json.dumps(fs_out, indent=2))

    symbols = u.symbols
    tf = u.timeframe
    lookback = u.lookback
    loss_name = str(config.get("loss", "mse")).lower()
    store = config.get("store")
    data_source = str(config.get("data_source", "rest"))

    psim_raw = config.get("portfolio_sim")
    psim_enabled = bool(isinstance(psim_raw, dict) and psim_raw.get("enabled", False))
    psim_book: Book = "long_only"
    psim_k = 2
    psim_bps = 0.0
    if psim_enabled and isinstance(psim_raw, dict):
        book_s = str(psim_raw.get("book", "long_only")).lower().replace("-", "_")
        psim_book = "long_short" if book_s in ("long_short", "ls") else "long_only"
        psim_k = int(psim_raw.get("top_k", 2))
        psim_bps = float(psim_raw.get("transaction_cost_one_way_bps", 0.0))

    wf = WalkForwardConfig(
        train_bars=int(config["train_bars"]),
        val_bars=int(config["val_bars"]),
        test_bars=int(config["test_bars"]),
        step_bars=int(config["step_bars"]),
    )

    sector_map: dict[str, str] = {}
    default_sector = "Unknown"
    sm_path = config.get("sector_map_path")
    if sm_path:
        sector_map, default_sector = load_sector_map(Path(sm_path))
    sectors_arr = sectors_for_symbols(symbols, sector_map, default_sector)

    if u.label_mode == "sector_neutral_return" and not sm_path:
        err_out: dict[str, Any] = {
            "error": "sector_map_path required for sector_neutral_return",
            "run_dir": str(run_dir),
        }
        save_summary(run_dir, err_out)
        return err_out

    sectors_for_labels: np.ndarray | None = sectors_arr if u.label_mode == "sector_neutral_return" else None

    if use_synthetic:
        logger.info("Synthetic universe candles (%s symbols, tf=%s)", len(symbols), tf)
        candles = synthetic_universe_candles(
            int(config.get("synthetic_n_bars", 600)),
            symbols,
            timeframe=tf,
            seed=int(config.get("seed", 42)),
        )
    else:
        logger.info("Loading universe candles (%s symbols, tf=%s)", len(symbols), tf)
        client = AlphaVantageClient(cache_dir=cache_dir)
        candles = fetch_candles_for_universe(
            client,
            symbols,
            tf,
            use_adjusted_daily=bool(config.get("use_adjusted_daily", True)),
            use_adjusted_weekly=bool(config.get("use_adjusted_weekly", True)),
            use_adjusted_monthly=bool(config.get("use_adjusted_monthly", True)),
            intraday_month=config.get("intraday_month"),
            intraday_extended_hours=bool(config.get("intraday_extended_hours", False)),
            intraday_outputsize=str(config.get("intraday_outputsize", "full")),
            daily_outputsize=str(config.get("daily_outputsize", "full")),
            store=store if store in ("csv", "parquet") else None,
            data_source=data_source,
        )

    panel, close = align_universe_ohlcv(candles, symbols)
    sec_by_sym = {s: str(sectors_arr[i]) for i, s in enumerate(symbols)}
    mem = membership_table_from_panel(
        panel["timestamp"].values,
        symbols,
        sector_by_symbol=sec_by_sym,
    )
    (run_dir / "universe_membership.json").write_text(json.dumps(mem, indent=2, default=str))

    X, mask, y, raw_ret, ts_pred, end_row = build_universe_samples(
        panel,
        symbols,
        close,
        lookback=lookback,
        min_coverage_symbols=u.min_coverage_symbols,
        label_mode=u.label_mode,
        feature_names=feature_names,
        sectors=sectors_for_labels,
    )

    n_samples = X.shape[0]
    folds = generate_folds(n_samples, wf)
    tgt_ix = u.target_index()

    folds_payload: dict[str, Any] = {}
    for fold in folds:
        folds_payload[str(fold.fold_id)] = {
            "train": {
                "i_start": fold.train.start,
                "i_end": fold.train.stop - 1,
                "timestamp_start": str(ts_pred.iloc[fold.train.start]),
                "timestamp_end": str(ts_pred.iloc[fold.train.stop - 1]),
            },
            "val": {
                "i_start": fold.val.start,
                "i_end": fold.val.stop - 1,
                "timestamp_start": str(ts_pred.iloc[fold.val.start]),
                "timestamp_end": str(ts_pred.iloc[fold.val.stop - 1]),
            },
            "test": {
                "i_start": fold.test.start,
                "i_end": fold.test.stop - 1,
                "timestamp_start": str(ts_pred.iloc[fold.test.start]),
                "timestamp_end": str(ts_pred.iloc[fold.test.stop - 1]),
            },
        }
    save_fold_payload(run_dir, folds_payload)

    summary: dict[str, Any] = {
        "experiment": "universe",
        "symbols": list(symbols),
        "target_symbol": u.target_symbol,
        "timeframe": tf,
        "n_samples": n_samples,
        "n_folds": len(folds),
        "device": str(device),
        "folds": [],
        "run_dir": str(run_dir),
        "loss": loss_name,
        "feature_schema_hash": fs["hash"],
        "git_sha": git_head_short(),
    }

    if not folds:
        summary["error"] = "no_folds"
        save_summary(run_dir, summary)
        pred_path = run_dir / "predictions_universe.csv"
        pred_path.write_text(
            "timestamp,symbol,timeframe,y_true_raw_return,y_true_relative_return,"
            "y_score,y_rank_pred,y_rank_true,fold_id\n"
        )
        return summary

    if dry_run:
        summary["dry_run"] = True
        summary["fold_plan"] = folds_payload
        save_summary(run_dir, summary)
        pred_path = run_dir / "predictions_universe.csv"
        pred_path.write_text(
            "timestamp,symbol,timeframe,y_true_raw_return,y_true_relative_return,"
            "y_score,y_rank_pred,y_rank_true,fold_id\n"
        )
        logger.info("Dry run: wrote fold plan to %s", run_dir / "folds.json")
        return summary

    fold_rows: list[dict[str, Any]] = []
    pred_records: list[dict[str, Any]] = []
    fold_errors: list[dict[str, Any]] = []
    per_fold_sector: list[dict[str, dict[str, float]]] = []
    portfolio_fold_summaries: list[dict[str, Any]] = []

    logger.info("Universe walk-forward: %s samples, %s folds", n_samples, len(folds))

    try:
        for fold in folds:
            try:
                assert_fold_chronology(ts_pred, fold)
                logger.info("Universe fold %s", fold.fold_id)
                if progress is not None:
                    progress.on_fold_start(fold.fold_id, len(folds))
                sl_tr, sl_va, sl_te = fold.train, fold.val, fold.test

                scaler = TrainOnlyScaler()
                scaler.fit(X[sl_tr], mask[sl_tr])
                X_tr = scaler.transform(X[sl_tr])
                X_va = scaler.transform(X[sl_va])
                X_te = scaler.transform(X[sl_te])

                model = TransformerRanker(
                    n_features=int(X.shape[-1]),
                    n_symbols=len(symbols),
                    d_model=int(config["d_model"]),
                    nhead=int(config["nhead"]),
                    num_temporal_layers=int(config.get("num_temporal_layers", 2)),
                    num_cross_layers=int(config.get("num_cross_layers", 1)),
                    dim_feedforward=int(config["dim_feedforward"]),
                    dropout=float(config["dropout"]),
                    max_seq_len=int(X_tr.shape[1]) + 8,
                ).to(device)
                log_path = run_dir / f"training_log_fold_{fold.fold_id}.csv"
                model = train_transformer_ranker(
                    X_tr,
                    mask[sl_tr],
                    y[sl_tr],
                    X_va,
                    mask[sl_va],
                    y[sl_va],
                    model,
                    config,
                    device,
                    loss_name,
                    log_path=log_path,
                    progress=progress,
                    progress_fold_id=fold.fold_id,
                )

                if bool(config.get("save_models", False)):
                    torch.save(
                        model.state_dict(),
                        run_dir / f"model_state_fold_{fold.fold_id}.pt",
                    )

                pred_te = batch_predict(model, X_te, mask[sl_te], device, batch_size=infer_bs)
                y_te = y[sl_te]
                raw_te = raw_ret[sl_te]

                rho = spearman_per_timestamp(pred_te, y_te, min_valid=u.min_coverage_symbols)
                tau = kendall_per_timestamp(pred_te, y_te, min_valid=u.min_coverage_symbols)
                ndcg = ndcg_at_k_per_timestamp(pred_te, y_te, k=min(3, len(symbols)))
                row: dict[str, Any] = {
                    "fold_id": fold.fold_id,
                    "spearman_mean": safe_nanmean(rho),
                    "kendall_mean": safe_nanmean(tau),
                    "ndcg3_mean": safe_nanmean(ndcg),
                    "top2_hit": top_k_hit_rate(pred_te, y_te, k=2, min_valid=u.min_coverage_symbols),
                }
                m_tgt = masked_regression_metrics(y_te[:, tgt_ix], pred_te[:, tgt_ix])
                row.update({f"target_{k}": v for k, v in m_tgt.items()})

                mom = momentum_rank_scores(close, end_rows=end_row[sl_te], lookback=lookback)
                mrev = mean_reversion_rank_scores(close, end_rows=end_row[sl_te], lookback=lookback)
                eq = equal_score_baseline(len(y_te), len(symbols))
                row["baseline_momentum_spearman_mean"] = safe_nanmean(spearman_per_timestamp(mom, y_te))
                row["baseline_mean_reversion_spearman_mean"] = safe_nanmean(spearman_per_timestamp(mrev, y_te))
                row["baseline_equal_spearman_mean"] = safe_nanmean(spearman_per_timestamp(eq, y_te))

                Xf_tr, yf_tr, gid_tr, _sid_tr = flatten_universe_sample(X[sl_tr], mask[sl_tr], y[sl_tr])
                Xf_te, _, gid_te, sid_te = flatten_universe_sample(X[sl_te], mask[sl_te], y[sl_te])
                if Xf_tr.shape[0] >= 8 and Xf_te.shape[0] >= 1:
                    lin = fit_linear_cs_ranker(Xf_tr, yf_tr, gid_tr, alpha=1.0)
                    gbt = fit_gbt_ranker(Xf_tr, yf_tr, gid_tr)
                    plin = lin.predict(Xf_te.astype(np.float64, copy=False))
                    pgbt = gbt.predict(Xf_te)
                    s_lin = scatter_predictions(plin, gid_te, sid_te, n_samples=len(y_te), n_symbols=len(symbols))
                    s_gbt = scatter_predictions(pgbt, gid_te, sid_te, n_samples=len(y_te), n_symbols=len(symbols))
                    row["baseline_linear_spearman_mean"] = safe_nanmean(spearman_per_timestamp(s_lin, y_te))
                    row["baseline_gbt_spearman_mean"] = safe_nanmean(spearman_per_timestamp(s_gbt, y_te))
                else:
                    row["baseline_linear_spearman_mean"] = float("nan")
                    row["baseline_gbt_spearman_mean"] = float("nan")

                fold_rows.append(row)
                if progress is not None:
                    progress.on_fold_end(fold.fold_id, row)
                per_fold_sector.append(per_sector_metric_summary(pred_te, y_te, symbols, sectors_arr, min_valid=2))

                pad_te = mask[sl_te][:, -1, :]
                if psim_enabled:
                    psim_res = simulate_topk_portfolio(
                        pred_te,
                        raw_te,
                        pad_last=pad_te,
                        book=psim_book,
                        top_k=psim_k,
                        transaction_cost_one_way_bps=psim_bps,
                        record_series=False,
                    )
                    portfolio_fold_summaries.append(
                        {
                            "fold_id": fold.fold_id,
                            "n_periods": psim_res["n_periods"],
                            "mean_gross_return": psim_res["mean_gross_return"],
                            "mean_net_return": psim_res["mean_net_return"],
                            "mean_turnover": psim_res["mean_turnover"],
                            "mean_cost": psim_res["mean_cost"],
                            "cumulative_gross_return": psim_res["cumulative_gross_return"],
                            "cumulative_net_return": psim_res["cumulative_net_return"],
                            "sharpe_net": psim_res["sharpe_net"],
                        }
                    )

                ranks_p = _ranks_descending_matrix(pred_te)
                ranks_t = _ranks_descending_matrix(raw_te)
                y_rel_all = raw_te - np.nanmedian(raw_te, axis=1, keepdims=True)
                for i in range(pred_te.shape[0]):
                    ts = ts_pred.iloc[sl_te.start + i]
                    ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    fold_id = fold.fold_id
                    for si, sym in enumerate(symbols):
                        pad = pad_te[i, si]
                        score = np.nan if pad else float(pred_te[i, si])
                        pred_records.append(
                            {
                                "timestamp": ts_iso,
                                "symbol": sym,
                                "timeframe": tf,
                                "y_true_raw_return": raw_te[i, si],
                                "y_true_relative_return": y_rel_all[i, si],
                                "y_score": score,
                                "y_rank_pred": ranks_p[i, si],
                                "y_rank_true": ranks_t[i, si],
                                "fold_id": fold_id,
                            }
                        )
            except Exception as exc:  # noqa: BLE001 — isolate fold failures; log and continue
                rec = fold_error_record(fold.fold_id, exc)
                append_fold_error_log(run_dir, rec)
                fold_errors.append(rec)
                logger.exception("Universe fold %s failed", fold.fold_id)
                continue
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving partial universe results")
        summary["error"] = "interrupted"
        summary["folds"] = fold_rows
        if fold_errors:
            summary["fold_errors"] = fold_errors
        pred_path = run_dir / "predictions_universe.csv"
        if pred_records:
            pd.DataFrame(pred_records).to_csv(pred_path, index=False)
        else:
            pred_path.write_text(
                "timestamp,symbol,timeframe,y_true_raw_return,y_true_relative_return,"
                "y_score,y_rank_pred,y_rank_true,fold_id\n"
            )
        save_summary(run_dir, summary)
        raise

    summary["folds"] = fold_rows
    if fold_errors:
        summary["error"] = "partial_failure"
        summary["fold_errors"] = fold_errors
    if fold_rows:
        summary["aggregate"] = aggregate_fold_metrics(fold_rows)
        if per_fold_sector:
            sec_keys: set[str] = set()
            for d in per_fold_sector:
                sec_keys.update(d.keys())
            agg_sec: dict[str, dict[str, float]] = {}
            for sk in sec_keys:
                vals = [
                    float(d[sk]["spearman_mean"])
                    for d in per_fold_sector
                    if sk in d and np.isfinite(d[sk]["spearman_mean"])
                ]
                if vals:
                    agg_sec[sk] = {
                        "spearman_mean": float(np.mean(vals)),
                        "n_folds": float(len(vals)),
                    }
            summary["aggregate"]["per_sector"] = agg_sec

    if psim_enabled:
        if portfolio_fold_summaries:
            summary["portfolio_sim"] = {
                "enabled": True,
                "book": psim_book,
                "top_k": psim_k,
                "transaction_cost_one_way_bps": psim_bps,
                "by_fold": portfolio_fold_summaries,
                "aggregate": aggregate_portfolio_sim_folds(portfolio_fold_summaries),
            }
        else:
            summary["portfolio_sim"] = {
                "enabled": True,
                "book": psim_book,
                "top_k": psim_k,
                "transaction_cost_one_way_bps": psim_bps,
                "error": "no_fold_results",
            }

    pred_path = run_dir / "predictions_universe.csv"
    if pred_records:
        pd.DataFrame(pred_records).to_csv(pred_path, index=False)
    else:
        pred_path.write_text(
            "timestamp,symbol,timeframe,y_true_raw_return,y_true_relative_return,"
            "y_score,y_rank_pred,y_rank_true,fold_id\n"
        )

    save_summary(run_dir, summary)
    logger.info("Universe run finished: %s", run_dir)
    return summary


def run_universe_from_config_path(
    path: str | Path,
    *,
    synthetic: bool = False,
    device: str | None = None,
    seed: int | None = None,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Load YAML, validate as universe experiment, and run (or dry-run) the universe pipeline."""
    from stock_transformer.backtest.runner import prepare_backtest_config

    cfg = prepare_backtest_config(path, device=device, seed=seed)
    if str(cfg.get("experiment_mode") or "").lower() != "universe":
        raise ValueError(
            f"experiment_mode must be 'universe' for run_universe_from_config_path (got {cfg.get('experiment_mode')!r})"
        )
    return run_universe_experiment(cfg, use_synthetic=synthetic, dry_run=dry_run, progress=progress)


def load_universe_config_from_dict(config: dict[str, Any]) -> UniverseConfig:
    """Build :class:`UniverseConfig` from a merged experiment dict (no extra file)."""
    syms = config.get("symbols") or []
    symbols = tuple(str(s).upper() for s in syms)
    tgt = str(config.get("target_symbol", symbols[0] if symbols else "")).upper()
    lm = str(config.get("label_mode", "cross_sectional_return"))
    return UniverseConfig(
        symbols=symbols,
        target_symbol=tgt,
        timeframe=str(config.get("timeframe", "daily")).lower(),
        lookback=int(config.get("lookback", 32)),
        min_coverage_symbols=int(config.get("min_coverage_symbols", max(2, len(symbols) - 1) if symbols else 2)),
        label_mode=lm,
        raw=config,
    )
