"""Build tensors from aligned OHLCV candles: each (ticker, day) candle becomes a token."""

from __future__ import annotations

import numpy as np
import torch


def build_features(
    aligned: dict[str, "pd.DataFrame"],
    target_symbol: str,
    lookback: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Convert aligned candle DataFrames into model-ready tensors.

    Returns:
        X:       [N, L*S, 5]  input tokens (candle log-returns + log-volume)
        y:       [N, 4]       target (SPY next-day OHLC as log-returns from today's close)
        symbols: ordered list of symbol names (defines the symbol axis)
    """
    import pandas as pd  # noqa: F811

    symbols = list(aligned.keys())
    target_idx = symbols.index(target_symbol.upper())
    S = len(symbols)
    n_days = len(next(iter(aligned.values())))

    ohlcv = np.zeros((n_days, S, 5), dtype=np.float64)
    close_raw = np.zeros((n_days, S), dtype=np.float64)

    for j, sym in enumerate(symbols):
        df = aligned[sym]
        ohlcv[:, j, 0] = df["open"].values
        ohlcv[:, j, 1] = df["high"].values
        ohlcv[:, j, 2] = df["low"].values
        ohlcv[:, j, 3] = df["close"].values
        ohlcv[:, j, 4] = df["volume"].values
        close_raw[:, j] = df["close"].values

    eps = 1e-10
    prev_close = np.roll(close_raw, 1, axis=0)
    prev_close[0, :] = np.nan

    safe_pc = np.where(prev_close > 0, prev_close, eps)
    features = np.zeros((n_days, S, 5), dtype=np.float64)
    features[:, :, 0] = np.log(ohlcv[:, :, 0] / safe_pc + eps)  # open
    features[:, :, 1] = np.log(ohlcv[:, :, 1] / safe_pc + eps)  # high
    features[:, :, 2] = np.log(ohlcv[:, :, 2] / safe_pc + eps)  # low
    features[:, :, 3] = np.log(ohlcv[:, :, 3] / safe_pc + eps)  # close
    features[:, :, 4] = np.log1p(np.clip(ohlcv[:, :, 4], 0, None))  # volume
    features[0, :, :] = 0.0  # day 0 has no prev close

    target_close_today = close_raw[:, target_idx]
    safe_tc = np.where(target_close_today > 0, target_close_today, eps)

    next_ohlcv = np.roll(ohlcv[:, target_idx, :4], -1, axis=0)
    targets = np.log(next_ohlcv / safe_tc[:, None] + eps)
    targets[-1, :] = 0.0  # last day has no next-day target

    # valid range: day 1 (has prev close) through day n-2 (has next-day target)
    valid_start = 1
    valid_end = n_days - 1

    samples_X = []
    samples_y = []

    for t in range(valid_start + lookback - 1, valid_end):
        window = features[t - lookback + 1 : t + 1]  # [L, S, 5]
        flat = window.reshape(lookback * S, 5)  # [L*S, 5]
        samples_X.append(flat)
        samples_y.append(targets[t])

    X = torch.tensor(np.array(samples_X), dtype=torch.float32)
    y = torch.tensor(np.array(samples_y), dtype=torch.float32)

    print(f"  Built {len(X)} samples, each with {lookback}x{S}={lookback*S} tokens")
    print(f"  Target: {target_symbol} next-day OHLC log-returns")

    return X, y, symbols
