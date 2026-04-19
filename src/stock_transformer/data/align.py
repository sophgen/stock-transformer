"""Align per-symbol canonical OHLCV onto a global timestamp index."""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_universe_ohlcv(
    candles_by_symbol: dict[str, pd.DataFrame],
    symbols: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Outer-merge all symbols on ``timestamp``; sort; return wide frame + close matrix.

    Wide columns: ``open__SYM``, ``high__SYM``, ``low__SYM``, ``close__SYM``, ``volume__SYM``.

    Returns
    -------
    panel
 Sorted DataFrame with ``timestamp`` first column.
    close
        ``[n_rows, n_symbols]`` float64, NaN where missing.
    """
    for sym in symbols:
        if sym not in candles_by_symbol:
            raise KeyError(f"Missing candles for symbol {sym}")
    base: pd.DataFrame | None = None
    for sym in symbols:
        df = candles_by_symbol[sym].sort_values("timestamp").copy()
        rename = {
            "open": f"open__{sym}",
            "high": f"high__{sym}",
            "low": f"low__{sym}",
            "close": f"close__{sym}",
            "volume": f"volume__{sym}",
        }
        keep = ["timestamp"] + [c for c in rename if c in df.columns]
        part = df[keep].rename(columns=rename)
        if base is None:
            base = part
        else:
            base = base.merge(part, on="timestamp", how="outer")
    assert base is not None
    base = base.sort_values("timestamp").reset_index(drop=True)
    close_cols = [f"close__{s}" for s in symbols]
    close = base[close_cols].to_numpy(dtype=np.float64)
    return base, close
