"""Walk-forward cross-sectional ranking experiment (multi-ticker universe)."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import rankdata

from stock_transformer.backtest.metrics import (
    aggregate_fold_metrics,
    kendall_per_timestamp,
    masked_regression_metrics,
    ndcg_at_k_per_timestamp,
    per_sector_metric_summary,
    spearman_per_timestamp,
    top_k_hit_rate,
)
from stock_transformer.backtest.walkforward import WalkForwardConfig, assert_fold_chronology, generate_folds
from stock_transformer.data.align import align_universe_ohlcv
from stock_transformer.data.alphavantage import AlphaVantageClient, fetch_candles_for_universe
from stock_transformer.data.synthetic import synthetic_universe_candles
from stock_transformer.data.universe import (
    UniverseConfig,
    load_sector_map,
    membership_table_from_panel,
    sectors_for_symbols,
)
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
from stock_transformer.model.losses import training_loss
from stock_transformer.model.transformer_classifier import resolve_device
from stock_transformer.model.transformer_ranker import TransformerRanker


def _safe_nanmean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    if not np.any(np.isfinite(a)):
        return float("nan")
    return float(np.nanmean(a))


def _git_head_short(cwd: Path | None = None) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd or Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()[:40]
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _ranks_descending(vals: np.ndarray) -> np.ndarray:
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    m = np.isfinite(vals)
    if not np.any(m):
        return out
    out[m] = rankdata(-vals[m], method="average")
    return out


def _peer_relative_raw(raw_row: np.ndarray) -> np.ndarray:
    m = np.isfinite(raw_row)
    out = np.full_like(raw_row, np.nan, dtype=np.float64)
    if not np.any(m):
        return out
    med = float(np.nanmedian(raw_row[m]))
    out[m] = raw_row[m] - med
    return out


def _train_ranker(
    X: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    X_va: np.ndarray,
    mask_va: np.ndarray,
    y_va: np.ndarray,
    cfg: dict[str, Any],
    n_symbols: int,
    device: torch.device,
    loss_name: str,
    n_features: int,
) -> TransformerRanker:
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))
    model = TransformerRanker(
        n_features=n_features,
        n_symbols=n_symbols,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_temporal_layers=int(cfg.get("num_temporal_layers", 2)),
        num_cross_layers=int(cfg.get("num_cross_layers", 1)),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        max_seq_len=int(X.shape[1]) + 8,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)
    bs = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    gen = torch.Generator()
    gen.manual_seed(int(cfg.get("seed", 42)))

    def to_t(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.from_numpy(a).to(dtype).to(device)

    xm_tr, mk_tr, yt_tr = to_t(X), to_t(mask, torch.bool), to_t(y)
    xm_va, mk_va, yt_va = to_t(X_va), to_t(mask_va, torch.bool), to_t(y_va)

    n = xm_tr.size(0)
    best_state = None
    best_val = float("inf")
    pad_last_tr = mk_tr[:, -1, :]
    pad_last_va = mk_va[:, -1, :]

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            pred = model(xm_tr[idx], mk_tr[idx])
            label_ok = torch.isfinite(yt_tr[idx]) & (~pad_last_tr[idx])
            loss = training_loss(loss_name, pred, yt_tr[idx], mask_valid=label_ok)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(xm_va, mk_va)
            label_ok_va = torch.isfinite(yt_va) & (~pad_last_va)
            vloss = float(training_loss(loss_name, pv, yt_va, mask_valid=label_ok_va))
        if vloss < best_val:
            best_val = vloss
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_ranker(
    model: TransformerRanker,
    X: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    bs = 128
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xf = torch.from_numpy(X[i : i + bs]).float().to(device)
            mk = torch.from_numpy(mask[i : i + bs]).bool().to(device)
            parts.append(model(xf, mk).cpu().numpy())
    return np.concatenate(parts, axis=0)


def run_universe_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
) -> dict[str, Any]:
    u = load_universe_config_from_dict(config)
    device = resolve_device(config.get("device", "auto"))
    cache_dir = Path(config.get("cache_dir", "data"))
    art = Path(config.get("artifacts_dir", "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    run_dir = art / f"universe_run_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(yaml.dump(config, sort_keys=True))

    feature_names: tuple[str, ...] = tuple(
        config.get("features") or list(DEFAULT_UNIVERSE_FEATURE_NAMES)
    )
    fs = feature_schema(feature_names)
    fs_out = {**fs, "git_sha": _git_head_short()}
    (run_dir / "feature_schema.json").write_text(json.dumps(fs_out, indent=2))

    symbols = u.symbols
    tf = u.timeframe
    lookback = u.lookback
    loss_name = str(config.get("loss", "mse")).lower()
    store = config.get("store")
    data_source = str(config.get("data_source", "rest"))

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
        summary = {"error": "sector_map_path required for sector_neutral_return", "run_dir": str(run_dir)}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    sectors_for_labels: np.ndarray | None = (
        sectors_arr if u.label_mode == "sector_neutral_return" else None
    )

    if use_synthetic:
        candles = synthetic_universe_candles(
            int(config.get("synthetic_n_bars", 600)),
            symbols,
            timeframe=tf,
            seed=int(config.get("seed", 42)),
        )
    else:
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
    (run_dir / "folds.json").write_text(json.dumps(folds_payload, indent=2, default=str))

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
    }

    if not folds:
        summary["error"] = "no_folds"
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (run_dir / "predictions_universe.csv").write_text(
            "timestamp,symbol,timeframe,y_true_raw_return,y_true_relative_return,"
            "y_score,y_rank_pred,y_rank_true,fold_id\n"
        )
        return summary

    fold_rows: list[dict[str, Any]] = []
    all_pred_rows: list[pd.DataFrame] = []
    fold_errors: list[dict[str, Any]] = []
    per_fold_sector: list[dict[str, dict[str, float]]] = []

    for fold in folds:
        try:
            assert_fold_chronology(ts_pred, fold)
            sl_tr, sl_va, sl_te = fold.train, fold.val, fold.test

            scaler = TrainOnlyScaler()
            scaler.fit(X[sl_tr], mask[sl_tr])
            X_tr = scaler.transform(X[sl_tr])
            X_va = scaler.transform(X[sl_va])
            X_te = scaler.transform(X[sl_te])

            model = _train_ranker(
                X_tr,
                mask[sl_tr],
                y[sl_tr],
                X_va,
                mask[sl_va],
                y[sl_va],
                config,
                len(symbols),
                device,
                loss_name,
                n_features=X.shape[-1],
            )

            if bool(config.get("save_models", False)):
                torch.save(
                    model.state_dict(),
                    run_dir / f"model_state_fold_{fold.fold_id}.pt",
                )

            pred_te = _predict_ranker(model, X_te, mask[sl_te], device)
            y_te = y[sl_te]
            raw_te = raw_ret[sl_te]

            rho = spearman_per_timestamp(pred_te, y_te, min_valid=u.min_coverage_symbols)
            tau = kendall_per_timestamp(pred_te, y_te, min_valid=u.min_coverage_symbols)
            ndcg = ndcg_at_k_per_timestamp(pred_te, y_te, k=min(3, len(symbols)))
            row: dict[str, Any] = {
                "fold_id": fold.fold_id,
                "spearman_mean": _safe_nanmean(rho),
                "kendall_mean": _safe_nanmean(tau),
                "ndcg3_mean": _safe_nanmean(ndcg),
                "top2_hit": top_k_hit_rate(pred_te, y_te, k=2, min_valid=u.min_coverage_symbols),
            }
            m_tgt = masked_regression_metrics(y_te[:, tgt_ix], pred_te[:, tgt_ix])
            row.update({f"target_{k}": v for k, v in m_tgt.items()})

            mom = momentum_rank_scores(close, end_rows=end_row[sl_te], lookback=lookback)
            mrev = mean_reversion_rank_scores(close, end_rows=end_row[sl_te], lookback=lookback)
            eq = equal_score_baseline(len(y_te), len(symbols))
            row["baseline_momentum_spearman_mean"] = _safe_nanmean(spearman_per_timestamp(mom, y_te))
            row["baseline_mean_reversion_spearman_mean"] = _safe_nanmean(
                spearman_per_timestamp(mrev, y_te)
            )
            row["baseline_equal_spearman_mean"] = _safe_nanmean(spearman_per_timestamp(eq, y_te))

            Xf_tr, yf_tr, gid_tr, _sid_tr = flatten_universe_sample(X[sl_tr], mask[sl_tr], y[sl_tr])
            Xf_te, _, gid_te, sid_te = flatten_universe_sample(X[sl_te], mask[sl_te], y[sl_te])
            if Xf_tr.shape[0] >= 8 and Xf_te.shape[0] >= 1:
                lin = fit_linear_cs_ranker(Xf_tr, yf_tr, gid_tr, alpha=1.0)
                gbt = fit_gbt_ranker(Xf_tr, yf_tr, gid_tr)
                plin = lin.predict(Xf_te.astype(np.float64, copy=False))
                pgbt = gbt.predict(Xf_te)
                s_lin = scatter_predictions(
                    plin, gid_te, sid_te, n_samples=len(y_te), n_symbols=len(symbols)
                )
                s_gbt = scatter_predictions(
                    pgbt, gid_te, sid_te, n_samples=len(y_te), n_symbols=len(symbols)
                )
                row["baseline_linear_spearman_mean"] = _safe_nanmean(
                    spearman_per_timestamp(s_lin, y_te)
                )
                row["baseline_gbt_spearman_mean"] = _safe_nanmean(
                    spearman_per_timestamp(s_gbt, y_te)
                )
            else:
                row["baseline_linear_spearman_mean"] = float("nan")
                row["baseline_gbt_spearman_mean"] = float("nan")

            fold_rows.append(row)
            per_fold_sector.append(
                per_sector_metric_summary(pred_te, y_te, symbols, sectors_arr, min_valid=2)
            )

            for i, j in enumerate(range(sl_te.start, sl_te.stop)):
                ts = ts_pred.iloc[j]
                fold_id = fold.fold_id
                ranks_p = _ranks_descending(pred_te[i])
                ranks_t = _ranks_descending(raw_te[i])
                y_rel_row = _peer_relative_raw(raw_te[i])
                for si, sym in enumerate(symbols):
                    pad = mask[sl_te][i, -1, si]
                    score = np.nan if pad else float(pred_te[i, si])
                    rec = {
                        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                        "symbol": sym,
                        "timeframe": tf,
                        "y_true_raw_return": raw_te[i, si],
                        "y_true_relative_return": y_rel_row[si],
                        "y_score": score,
                        "y_rank_pred": ranks_p[si],
                        "y_rank_true": ranks_t[si],
                        "fold_id": fold_id,
                    }
                    all_pred_rows.append(pd.DataFrame([rec]))
        except Exception as exc:  # noqa: BLE001
            fold_errors.append({"fold_id": fold.fold_id, "error": str(exc)})
            continue

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

    pred_path = run_dir / "predictions_universe.csv"
    if all_pred_rows:
        pd.concat(all_pred_rows, ignore_index=True).to_csv(pred_path, index=False)
    else:
        pred_path.write_text(
            "timestamp,symbol,timeframe,y_true_raw_return,y_true_relative_return,"
            "y_score,y_rank_pred,y_rank_true,fold_id\n"
        )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


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
        min_coverage_symbols=int(
            config.get("min_coverage_symbols", max(2, len(symbols) - 1) if symbols else 2)
        ),
        label_mode=lm,
        raw=config,
    )


def run_universe_from_config_path(path: str | Path, *, synthetic: bool = False) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return run_universe_experiment(cfg, use_synthetic=synthetic)
