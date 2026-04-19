"""End-to-end walk-forward experiment: multi-timeframe autoregressive candle prediction."""

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
    classification_metrics,
    regression_metrics,
)
from stock_transformer.backtest.walkforward import (
    WalkForwardConfig,
    assert_fold_chronology,
    generate_folds,
)
from stock_transformer.data.alphavantage import AlphaVantageClient, fetch_candles_for_timeframe
from stock_transformer.data.synthetic import synthetic_multitimeframe_candles
from stock_transformer.features.sequences import (
    N_CANDLE_FEATURES,
    TIMEFRAME_IDS,
    build_multitimeframe_samples,
)
from stock_transformer.model.transformer_classifier import (
    CandleTransformer,
    predict_direction_proba,
    resolve_device,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── training ───────────────────────────────────────────────────────────────


def _train_model(
    X_feat: np.ndarray,
    X_tf: np.ndarray,
    X_mask: np.ndarray,
    y_reg: np.ndarray,
    y_dir: np.ndarray,
    X_feat_va: np.ndarray,
    X_tf_va: np.ndarray,
    X_mask_va: np.ndarray,
    y_reg_va: np.ndarray,
    y_dir_va: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
) -> CandleTransformer:
    torch.manual_seed(int(cfg.get("seed", 42)))

    n_tf = len(TIMEFRAME_IDS)
    model = CandleTransformer(
        n_candle_features=N_CANDLE_FEATURES,
        n_timeframes=n_tf,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        max_seq_len=int(cfg.get("max_seq_len", 256)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)
    reg_loss_fn = nn.MSELoss()
    dir_loss_fn = nn.BCEWithLogitsLoss()
    alpha = float(cfg.get("loss_alpha", 0.5))

    def _to(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.from_numpy(arr).to(dtype).to(device)

    xf_tr, xtf_tr, xm_tr = _to(X_feat), _to(X_tf, torch.long), _to(X_mask, torch.bool)
    yr_tr, yd_tr = _to(y_reg), _to(y_dir)

    xf_va, xtf_va, xm_va = _to(X_feat_va), _to(X_tf_va, torch.long), _to(X_mask_va, torch.bool)
    yr_va, yd_va = _to(y_reg_va), _to(y_dir_va)

    bs = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    n = xf_tr.size(0)
    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            c_pred, d_logit = model(xf_tr[idx], xtf_tr[idx], xm_tr[idx])
            loss = alpha * reg_loss_fn(c_pred, yr_tr[idx]) + (1 - alpha) * dir_loss_fn(d_logit, yd_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            c_pred_v, d_logit_v = model(xf_va, xtf_va, xm_va)
            vloss = alpha * float(reg_loss_fn(c_pred_v, yr_va)) + (1 - alpha) * float(dir_loss_fn(d_logit_v, yd_va))
        if vloss < best_val:
            best_val = vloss
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── inference ──────────────────────────────────────────────────────────────


def _predict(
    model: CandleTransformer,
    X_feat: np.ndarray,
    X_tf: np.ndarray,
    X_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (candle_preds, direction_probs) as numpy arrays."""
    model.eval()
    candle_parts: list[np.ndarray] = []
    dir_parts: list[np.ndarray] = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(X_feat), bs):
            xf = torch.from_numpy(X_feat[i : i + bs]).float().to(device)
            xt = torch.from_numpy(X_tf[i : i + bs]).long().to(device)
            xm = torch.from_numpy(X_mask[i : i + bs]).bool().to(device)
            c_pred, d_logit = model(xf, xt, xm)
            candle_parts.append(c_pred.cpu().numpy())
            dir_parts.append(predict_direction_proba(d_logit).cpu().numpy())
    return np.concatenate(candle_parts), np.concatenate(dir_parts)


# ── threshold tuning ───────────────────────────────────────────────────────


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        m = classification_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    return best_t


# ── main experiment loop ───────────────────────────────────────────────────


def run_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    skip_fetch: bool = False,
) -> dict[str, Any]:
    """Run multi-timeframe walk-forward experiment; write artifacts."""

    if str(config.get("experiment_mode", "single")).lower() == "universe":
        from stock_transformer.backtest.universe_runner import run_universe_experiment

        return run_universe_experiment(config, use_synthetic=use_synthetic)

    device = resolve_device(config.get("device", "auto"))
    cache_dir = Path(config.get("cache_dir", "data"))
    art = Path(config.get("artifacts_dir", "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    run_dir = art / f"run_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(yaml.dump(config, sort_keys=True))

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

    # ── 1. data ingestion ──────────────────────────────────────────────
    if use_synthetic:
        candles_by_tf = synthetic_multitimeframe_candles(
            n_daily=int(config.get("synthetic_n_daily", 1200)),
            symbol=symbol,
            seed=int(config.get("seed", 42)),
            timeframes=timeframes,
        )
    else:
        client = AlphaVantageClient(cache_dir=cache_dir)
        candles_by_tf: dict[str, pd.DataFrame] = {}
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
        raise ValueError(
            f"prediction_timeframe '{prediction_tf}' not in fetched timeframes {list(candles_by_tf)}"
        )

    # ── 2. multi-timeframe tokenization ────────────────────────────────
    X_feat, X_tf_ids, X_mask, y_reg, y_dir, ts_pred = build_multitimeframe_samples(
        candles_by_tf,
        prediction_tf=prediction_tf,
        lookbacks=lookbacks,
        max_seq_len=max_seq_len,
    )

    n_samples = X_feat.shape[0]
    folds = generate_folds(n_samples, wf)

    summary: dict[str, Any] = {
        "symbol": symbol,
        "prediction_timeframe": prediction_tf,
        "timeframes": timeframes,
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

    # ── 3. walk-forward loop ───────────────────────────────────────────
    fold_rows: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []

    for fold in folds:
        assert_fold_chronology(ts_pred, fold)

        sl_tr, sl_va, sl_te = fold.train, fold.val, fold.test

        model = _train_model(
            X_feat[sl_tr], X_tf_ids[sl_tr], X_mask[sl_tr], y_reg[sl_tr], y_dir[sl_tr],
            X_feat[sl_va], X_tf_ids[sl_va], X_mask[sl_va], y_reg[sl_va], y_dir[sl_va],
            config, device,
        )

        # Validation — tune threshold
        candle_va, prob_va = _predict(model, X_feat[sl_va], X_tf_ids[sl_va], X_mask[sl_va], device)
        thr = tune_threshold(y_dir[sl_va], prob_va)

        # Test
        candle_te, prob_te = _predict(model, X_feat[sl_te], X_tf_ids[sl_te], X_mask[sl_te], device)
        m_cls = classification_metrics(y_dir[sl_te], prob_te, threshold=thr)
        m_reg = regression_metrics(y_reg[sl_te], candle_te)

        row: dict[str, Any] = {"fold_id": fold.fold_id, "threshold": thr}
        row.update({f"test_cls_{k}": v for k, v in m_cls.items()})
        row.update({f"test_reg_{k}": v for k, v in m_reg.items()})
        fold_rows.append(row)

        pred_df = pd.DataFrame(
            {
                "timestamp": ts_pred.iloc[sl_te].values if hasattr(ts_pred, "iloc") else ts_pred[sl_te.start:sl_te.stop],
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

    agg = aggregate_fold_metrics(fold_rows)
    summary["aggregate"] = agg
    summary["folds"] = fold_rows

    pred_path = run_dir / f"predictions__{symbol}.csv"
    pd.concat(all_preds, ignore_index=True).to_csv(pred_path, index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_from_config_path(path: str | Path, *, synthetic: bool = False) -> dict[str, Any]:
    cfg = load_config(path)
    return run_experiment(cfg, use_synthetic=synthetic)
