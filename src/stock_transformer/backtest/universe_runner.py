"""Walk-forward cross-sectional ranking experiment (multi-ticker universe)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from stock_transformer.backtest.metrics import (
    aggregate_fold_metrics,
    masked_regression_metrics,
    ndcg_at_k_per_timestamp,
    spearman_per_timestamp,
    top_k_hit_rate,
)
from stock_transformer.backtest.walkforward import WalkForwardConfig, assert_fold_chronology, generate_folds
from stock_transformer.data.align import align_universe_ohlcv
from stock_transformer.data.alphavantage import AlphaVantageClient, fetch_candles_for_universe
from stock_transformer.data.synthetic import synthetic_universe_candles
from stock_transformer.data.universe import UniverseConfig, membership_table_from_panel
from stock_transformer.features.universe_tensor import N_UNIVERSE_FEATURES, build_universe_samples
from stock_transformer.model.baselines import equal_score_baseline, momentum_rank_scores
from stock_transformer.model.transformer_classifier import resolve_device
from stock_transformer.model.transformer_ranker import TransformerRanker


def _safe_nanmean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    if not np.any(np.isfinite(a)):
        return float("nan")
    return float(np.nanmean(a))


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(target)
    if not m.any():
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    err = (pred - torch.nan_to_num(target, nan=0.0)) **2
    return err[m].mean()


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
) -> TransformerRanker:
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = TransformerRanker(
        n_features=N_UNIVERSE_FEATURES,
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

    def to_t(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.from_numpy(a).to(dtype).to(device)

    xm_tr, mk_tr, yt_tr = to_t(X), to_t(mask, torch.bool), to_t(y)
    xm_va, mk_va, yt_va = to_t(X_va), to_t(mask_va, torch.bool), to_t(y_va)

    n = xm_tr.size(0)
    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            pred = model(xm_tr[idx], mk_tr[idx])
            loss = _masked_mse(pred, yt_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(xm_va, mk_va)
            vloss = float(_masked_mse(pv, yt_va))
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

    symbols = u.symbols
    tf = u.timeframe
    lookback = u.lookback

    wf = WalkForwardConfig(
        train_bars=int(config["train_bars"]),
        val_bars=int(config["val_bars"]),
        test_bars=int(config["test_bars"]),
        step_bars=int(config["step_bars"]),
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
        )

    panel, close = align_universe_ohlcv(candles, symbols)
    mem = membership_table_from_panel(panel["timestamp"].values, symbols)
    (run_dir / "universe_membership.json").write_text(json.dumps(mem, indent=2, default=str))

    X, mask, y, raw_ret, ts_pred, end_row = build_universe_samples(
        panel,
        symbols,
        close,
        lookback=lookback,
        min_coverage_symbols=u.min_coverage_symbols,
        label_mode=u.label_mode,
    )

    n_samples = X.shape[0]
    folds = generate_folds(n_samples, wf)
    tgt_ix = u.target_index()

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
    }

    if not folds:
        summary["error"] = "no_folds"
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    fold_rows: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []

    for fold in folds:
        assert_fold_chronology(ts_pred, fold)
        sl_tr, sl_va, sl_te = fold.train, fold.val, fold.test

        model = _train_ranker(
            X[sl_tr], mask[sl_tr], y[sl_tr],
            X[sl_va], mask[sl_va], y[sl_va],
            config,
            len(symbols),
            device,
        )

        pred_te = _predict_ranker(model, X[sl_te], mask[sl_te], device)
        y_te = y[sl_te]

        rho = spearman_per_timestamp(pred_te, y_te)
        ndcg = ndcg_at_k_per_timestamp(pred_te, y_te, k=min(3, len(symbols)))
        row: dict[str, Any] = {
            "fold_id": fold.fold_id,
            "spearman_mean": _safe_nanmean(rho),
            "ndcg3_mean": _safe_nanmean(ndcg),
            "top2_hit": top_k_hit_rate(pred_te, y_te, k=2, min_valid=u.min_coverage_symbols),
        }
        m_tgt = masked_regression_metrics(y_te[:, tgt_ix], pred_te[:, tgt_ix])
        row.update({f"target_{k}": v for k, v in m_tgt.items()})

        mom = momentum_rank_scores(close, end_rows=end_row[sl_te], lookback=lookback)
        eq = equal_score_baseline(len(y_te), len(symbols))
        row["baseline_momentum_spearman_mean"] = _safe_nanmean(spearman_per_timestamp(mom, y_te))
        row["baseline_equal_spearman_mean"] = _safe_nanmean(spearman_per_timestamp(eq, y_te))
        fold_rows.append(row)

        raw_te = raw_ret[sl_te]
        for i, j in enumerate(range(sl_te.start, sl_te.stop)):
            rec = {
                "timestamp": ts_pred.iloc[j],
                "timeframe": tf,
                "fold_id": fold.fold_id,
            }
            for si, sym in enumerate(symbols):
                rec[f"y_true_relative_{sym}"] = y_te[i, si]
                rec[f"y_true_raw_{sym}"] = raw_te[i, si]
                rec[f"y_score_{sym}"] = pred_te[i, si]
            rec["target_symbol"] = u.target_symbol
            all_preds.append(pd.DataFrame([rec]))

    summary["aggregate"] = aggregate_fold_metrics(fold_rows)
    summary["folds"] = fold_rows
    pred_path = run_dir / "predictions_universe.csv"
    pd.concat(all_preds, ignore_index=True).to_csv(pred_path, index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def load_universe_config_from_dict(config: dict[str, Any]) -> UniverseConfig:
    """Build :class:`UniverseConfig` from a merged experiment dict (no extra file)."""
    syms = config.get("symbols") or []
    symbols = tuple(str(s).upper() for s in syms)
    tgt = str(config.get("target_symbol", symbols[0] if symbols else "")).upper()
    return UniverseConfig(
        symbols=symbols,
        target_symbol=tgt,
        timeframe=str(config.get("timeframe", "daily")).lower(),
        lookback=int(config.get("lookback", 32)),
        min_coverage_symbols=int(
            config.get("min_coverage_symbols", max(2, len(symbols) - 1) if symbols else 2)
        ),
        label_mode=str(config.get("label_mode", "cross_sectional_return")),
        raw=config,
    )


def run_universe_from_config_path(path: str | Path, *, synthetic: bool = False) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return run_universe_experiment(cfg, use_synthetic=synthetic)
