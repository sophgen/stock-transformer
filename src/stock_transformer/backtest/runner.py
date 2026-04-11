"""End-to-end walk-forward experiment: ingest, windows, train, metrics, artifacts."""

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
from sklearn.preprocessing import StandardScaler

from stock_transformer.backtest.metrics import aggregate_fold_metrics, classification_metrics
from stock_transformer.backtest.walkforward import WalkForwardConfig, assert_fold_chronology, generate_folds
from stock_transformer.data.alphavantage import AlphaVantageClient, fetch_candles_for_timeframe
from stock_transformer.data.synthetic import synthetic_random_walk_candles
from stock_transformer.features.sequences import (
    build_direction_labels,
    build_feature_matrix,
    build_windows,
    validate_no_lookahead,
)
from stock_transformer.model.baselines import moving_average_baseline, persistence_probs_on_test
from stock_transformer.model.transformer_classifier import CandleTransformerClassifier, predict_proba


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_windows_from_df(
    df: pd.DataFrame,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray, pd.DataFrame]:
    aligned, X = build_feature_matrix(df)
    y_bar = build_direction_labels(aligned["close"].to_numpy())
    X_win, y_win, end_idx, ts_end = build_windows(
        X, y_bar, aligned["timestamp"], lookback
    )
    validate_no_lookahead(aligned, end_idx, lookback)
    return X_win, y_win, ts_end, end_idx, aligned


def _scale_fold(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    s, L, F = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, F))
    def tr(x: np.ndarray) -> np.ndarray:
        s2, l2, f2 = x.shape
        t = scaler.transform(x.reshape(-1, f2)).reshape(s2, l2, f2)
        return t.astype(np.float32)

    return tr(X_tr), tr(X_va), tr(X_te), scaler


def _train_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: dict[str, Any],
    n_features: int,
    device: torch.device,
) -> CandleTransformerClassifier:
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = CandleTransformerClassifier(
        n_features=n_features,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    bs = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    n = X_tr_t.size(0)
    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            logits = model(X_tr_t[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_va_t)
            vloss = float(loss_fn(logits, y_va_t))
        if vloss < best_val:
            best_val = vloss
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_proba(model: CandleTransformerClassifier, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xt = torch.from_numpy(X[i : i + bs]).to(device)
            logits = model(xt)
            p = predict_proba(logits).cpu().numpy()
            out.append(p)
    return np.concatenate(out, axis=0)


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick threshold maximizing F1 on validation (coarse grid)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        m = classification_metrics(y_true, y_prob, threshold=float(t))
        f1 = m["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def run_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    skip_fetch: bool = False,
) -> dict[str, Any]:
    """
    Run all timeframes in config; write artifacts under ``artifacts_dir``.

    If ``use_synthetic``, ignores Alpha Vantage and uses random-walk candles per timeframe tag.
    """
    cache_dir = Path(config.get("cache_dir", "data"))
    art = Path(config.get("artifacts_dir", "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    run_dir = art / f"run_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(yaml.dump(config, sort_keys=True))

    symbol = str(config["symbol"])
    timeframes = list(config["timeframes"])
    lookback = int(config["lookback"])
    wf = WalkForwardConfig(
        train_bars=int(config["train_bars"]),
        val_bars=int(config["val_bars"]),
        test_bars=int(config["test_bars"]),
        step_bars=int(config["step_bars"]),
    )
    device = torch.device(config.get("device", "cpu"))

    summary: dict[str, Any] = {"timeframes": {}, "run_dir": str(run_dir)}

    client = AlphaVantageClient(cache_dir=cache_dir) if not use_synthetic else None

    for tf in timeframes:
        if use_synthetic or skip_fetch:
            df = synthetic_random_walk_candles(n=1200, symbol=symbol, timeframe=tf, seed=hash(tf) % 2**32)
        else:
            assert client is not None
            df = fetch_candles_for_timeframe(
                client,
                symbol,
                tf,
                use_adjusted_daily=bool(config.get("use_adjusted_daily", True)),
                use_adjusted_monthly=bool(config.get("use_adjusted_monthly", True)),
                intraday_month=config.get("intraday_month"),
                intraday_extended_hours=bool(config.get("intraday_extended_hours", False)),
                intraday_outputsize=str(config.get("intraday_outputsize", "full")),
                daily_outputsize=str(config.get("daily_outputsize", "full")),
            )

        X_win, y_win, ts_end, end_idx, aligned = prepare_windows_from_df(df, lookback)
        closes = aligned["close"].to_numpy(dtype=np.float64)
        n_w = X_win.shape[0]
        folds = generate_folds(n_w, wf)
        if not folds:
            summary["timeframes"][tf] = {"error": "no_folds", "n_windows": n_w}
            continue

        fold_rows: list[dict[str, Any]] = []
        all_preds: list[pd.DataFrame] = []

        for fold in folds:
            assert_fold_chronology(ts_end, fold)

            X_tr, y_tr = X_win[fold.train], y_win[fold.train]
            X_va, y_va = X_win[fold.val], y_win[fold.val]
            X_te, y_te = X_win[fold.test], y_win[fold.test]

            X_tr_s, X_va_s, X_te_s, _ = _scale_fold(X_tr, X_va, X_te)
            F = X_tr_s.shape[2]
            model = _train_model(X_tr_s, y_tr, X_va_s, y_va, config, F, device)

            p_va = _predict_proba(model, X_va_s, device)
            thr = tune_threshold(y_va, p_va)

            p_te = _predict_proba(model, X_te_s, device)
            m_te = classification_metrics(y_te, p_te, threshold=thr)

            pers_probs = persistence_probs_on_test(y_va, y_te)
            m_pers = classification_metrics(y_te, pers_probs, threshold=0.5)

            ma_window = 5
            test_ix = range(fold.test.start, fold.test.stop)
            ma_probs = np.array(
                [
                    moving_average_baseline(closes[: int(end_idx[i]) + 1], window=ma_window)
                    for i in test_ix
                ],
                dtype=np.float64,
            )
            m_ma = classification_metrics(y_te, ma_probs, threshold=0.5)

            row = {"fold_id": fold.fold_id, **{f"test_{k}": v for k, v in m_te.items()}}
            row["threshold"] = thr
            row["test_persistence_accuracy"] = m_pers["accuracy"]
            row["test_ma5_accuracy"] = m_ma["accuracy"]
            fold_rows.append(row)

            pred_df = pd.DataFrame(
                {
                    "timestamp": ts_end.iloc[fold.test].values,
                    "symbol": symbol,
                    "timeframe": tf,
                    "y_true": y_te,
                    "y_prob": p_te,
                    "y_pred": (p_te >= thr).astype(int),
                    "fold_id": fold.fold_id,
                }
            )
            all_preds.append(pred_df)

        agg = aggregate_fold_metrics(fold_rows)
        summary["timeframes"][tf] = {
            "n_windows": n_w,
            "n_folds": len(folds),
            "aggregate": agg,
            "folds": fold_rows,
        }

        pred_path = run_dir / f"predictions__{symbol}__{tf.replace('/', '-')}.csv"
        pd.concat(all_preds, ignore_index=True).to_csv(pred_path, index=False)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_from_config_path(path: str | Path, *, synthetic: bool = False) -> dict[str, Any]:
    cfg = load_config(path)
    return run_experiment(cfg, use_synthetic=synthetic)
