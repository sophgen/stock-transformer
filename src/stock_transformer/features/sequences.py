"""Multi-timeframe candle tokenization and autoregressive target construction.

Each candle (minute / hour / day / week / month) is converted to a token with
OHLCV log-return features.  Tokens from every available timeframe are gathered
up to a cutoff timestamp, sorted chronologically, and padded to a fixed
``max_seq_len``.  The target is the **next candle** in the prediction timeframe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical ordering — lower index = coarser grain
TIMEFRAME_IDS: dict[str, int] = {
    "monthly": 0,
    "weekly": 1,
    "daily": 2,
    "60min": 3,
    "30min": 4,
    "15min": 5,
    "5min": 6,
    "1min": 7,
}

N_CANDLE_FEATURES = 5  # open_ret, high_ret, low_ret, close_ret, log_vol


# ── per-candle feature extraction ──────────────────────────────────────────


def candle_log_returns(df: pd.DataFrame) -> np.ndarray:
    """Convert raw OHLCV into scale-free log-return features.

    For row *i* the features are relative to the **previous candle's close**
    (within the same timeframe), making them comparable across timeframes.

    Columns: ``[open_ret, high_ret, low_ret, close_ret, log_vol]``
    """
    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    v = np.clip(df["volume"].values.astype(np.float64), 0, None)

    prev_c = np.empty_like(c)
    prev_c[0] = o[0]
    prev_c[1:] = c[:-1]

    eps = 1e-10
    safe_prev = np.where(prev_c == 0, eps, prev_c)
    features = np.column_stack(
        [
            np.log(o / safe_prev + eps),
            np.log(h / safe_prev + eps),
            np.log(lo / safe_prev + eps),
            np.log(c / safe_prev + eps),
            np.log1p(v),
        ]
    )
    return features.astype(np.float32)


# ── multi-timeframe sample construction ────────────────────────────────────


def build_multitimeframe_samples(
    candles_by_tf: dict[str, pd.DataFrame],
    prediction_tf: str,
    lookbacks: dict[str, int],
    max_seq_len: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Build padded token sequences and targets for autoregressive training.

    For every usable prediction point in ``prediction_tf`` the function:

    1. Collects the most recent *lookback* candles from **each** timeframe
       whose timestamp is ``<=`` the prediction point.
    2. Sorts all collected tokens chronologically.
    3. Pads (or truncates) to ``max_seq_len``.
    4. Computes regression targets (next candle OHLCV returns) and a binary
       direction label.

    Returns
    -------
    X_feat   : ``[N, max_seq_len, 5]``  padded candle-token features
    X_tf_ids : ``[N, max_seq_len]``      timeframe id per token  (long)
    X_mask   : ``[N, max_seq_len]``      padding mask (``True`` = pad)
    y_reg    : ``[N, 5]``                next-candle log-return targets
    y_dir    : ``[N]``                   direction label (1 = up)
    ts_pred  : ``pd.Series``             prediction-point timestamps
    """
    pred_df = candles_by_tf[prediction_tf].sort_values("timestamp").reset_index(drop=True)
    pred_features = candle_log_returns(pred_df)
    pred_ts = pred_df["timestamp"].values
    pred_closes = pred_df["close"].values.astype(np.float64)

    precomputed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for tf, df in candles_by_tf.items():
        df_s = df.sort_values("timestamp").reset_index(drop=True)
        feats = candle_log_returns(df_s)
        ts_arr = df_s["timestamp"].values  # datetime64
        precomputed[tf] = (
            ts_arr,
            feats,
            np.array([TIMEFRAME_IDS.get(tf, len(TIMEFRAME_IDS))] * len(df_s), dtype=np.int64),
        )

    all_X: list[np.ndarray] = []
    all_tf: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    all_yr: list[np.ndarray] = []
    all_yd: list[float] = []
    all_ts: list = []

    for i in range(1, len(pred_df) - 1):
        cutoff = pred_ts[i]

        tokens_feats: list[np.ndarray] = []
        tokens_tf: list[int] = []
        tokens_ts: list = []

        for tf in candles_by_tf:
            ts_arr, feat_arr, tf_id_arr = precomputed[tf]
            lb = lookbacks.get(tf, 32)
            idx_end = int(np.searchsorted(ts_arr, cutoff, side="right")) - 1
            if idx_end < 0:
                continue
            start = max(0, idx_end - lb + 1)
            for j in range(start, idx_end + 1):
                tokens_feats.append(feat_arr[j])
                tokens_tf.append(int(tf_id_arr[j]))
                tokens_ts.append(ts_arr[j])

        if len(tokens_feats) < 2:
            continue

        order = np.argsort(tokens_ts)
        tokens_feats = [tokens_feats[o] for o in order]
        tokens_tf = [tokens_tf[o] for o in order]

        if len(tokens_feats) > max_seq_len:
            tokens_feats = tokens_feats[-max_seq_len:]
            tokens_tf = tokens_tf[-max_seq_len:]

        seq_len = len(tokens_feats)
        feat_pad = np.zeros((max_seq_len, N_CANDLE_FEATURES), dtype=np.float32)
        tf_pad = np.zeros(max_seq_len, dtype=np.int64)
        mask_pad = np.ones(max_seq_len, dtype=bool)

        for k in range(seq_len):
            feat_pad[k] = tokens_feats[k]
            tf_pad[k] = tokens_tf[k]
            mask_pad[k] = False

        target = pred_features[i + 1]
        direction = 1.0 if pred_closes[i + 1] > pred_closes[i] else 0.0

        all_X.append(feat_pad)
        all_tf.append(tf_pad)
        all_mask.append(mask_pad)
        all_yr.append(target)
        all_yd.append(direction)
        all_ts.append(pred_ts[i])

    if not all_X:
        empty_f = np.empty((0, max_seq_len, N_CANDLE_FEATURES), dtype=np.float32)
        empty_tf = np.empty((0, max_seq_len), dtype=np.int64)
        empty_m = np.empty((0, max_seq_len), dtype=bool)
        empty_yr = np.empty((0, N_CANDLE_FEATURES), dtype=np.float32)
        empty_yd = np.empty((0,), dtype=np.float32)
        return empty_f, empty_tf, empty_m, empty_yr, empty_yd, pd.Series(dtype="datetime64[ns]")

    return (
        np.stack(all_X),
        np.stack(all_tf),
        np.stack(all_mask),
        np.stack(all_yr).astype(np.float32),
        np.array(all_yd, dtype=np.float32),
        pd.Series(all_ts),
    )


# ── legacy single-timeframe helpers (kept for backward compat / tests) ─────


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-candle derived features (single timeframe).  Original 4-feature set."""
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    low = df["low"].astype(float)
    v = df["volume"].astype(float).clip(lower=0)

    log_ret = np.log(c / c.shift(1))
    hl_range = (h - low) / c.replace(0, np.nan)
    oc_ratio = (c - o) / c.replace(0, np.nan)

    feat = pd.DataFrame({"log_ret": log_ret, "hl_range": hl_range, "oc_ratio": oc_ratio, "log_vol": np.log1p(v)})
    valid = feat.notna().all(axis=1)
    aligned = df.loc[valid].reset_index(drop=True)
    X = feat.loc[valid].to_numpy(dtype=np.float64)
    return aligned, X


def build_direction_labels(closes: np.ndarray) -> np.ndarray:
    n = len(closes)
    y = np.full(n, np.nan)
    for t in range(n - 1):
        y[t] = 1.0 if closes[t + 1] > closes[t] else 0.0
    return y


def build_windows(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.Series,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    n = X.shape[0]
    xs: list[np.ndarray] = []
    ys: list[float] = []
    ends: list[int] = []
    ts_list: list[pd.Timestamp] = []
    for t in range(lookback - 1, n - 1):
        if np.isnan(y[t]):
            continue
        xs.append(X[t - lookback + 1 : t + 1])
        ys.append(float(y[t]))
        ends.append(t)
        ts_list.append(pd.Timestamp(timestamps.iloc[t]))
    if not xs:
        return (
            np.empty((0, lookback, X.shape[1])),
            np.empty((0,)),
            np.empty((0,), dtype=int),
            pd.Series(dtype="datetime64[ns]"),
        )
    return np.stack(xs), np.asarray(ys, dtype=np.float64), np.asarray(ends, dtype=int), pd.Series(ts_list)


def validate_no_lookahead(df: pd.DataFrame, end_indices: np.ndarray, lookback: int) -> None:
    ts = pd.to_datetime(df["timestamp"])
    for t_end in end_indices:
        start = int(t_end) - lookback + 1
        if start < 0:
            raise AssertionError(f"Invalid window start {start} for end {t_end}")
        w_ts_max = ts.iloc[int(t_end)]
        if ts.iloc[start : int(t_end) + 1].max() != w_ts_max:
            raise AssertionError("Window max timestamp mismatch")
        if int(t_end) >= len(df) - 1:
            raise AssertionError("end index has no future candle")
