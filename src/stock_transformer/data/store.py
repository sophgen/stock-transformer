"""Partitioned canonical candle store (CSV or Parquet)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

Backend = Literal["csv", "parquet"]


class CandleStore:
    """Read/write canonical OHLCV under ``root/canonical/timeframe=<tf>/symbol=<sym>/``."""

    def __init__(self, root: str | Path, *, backend: Backend = "csv") -> None:
        self.root = Path(root)
        self.backend = backend

    def _part_path(self, symbol: str, timeframe: str) -> Path:
        sym = symbol.upper()
        tf = timeframe.lower().replace("/", "-")
        return self.root / "canonical" / f"timeframe={tf}" / f"symbol={sym}" / "part-000"

    def write(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
        path = self._part_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "parquet":
            out = path.with_suffix(".parquet")
            df.to_parquet(out, index=False)
            return out
        out = path.with_suffix(".csv")
        df.to_csv(out, index=False)
        return out

    def read(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self._part_path(symbol, timeframe)
        pq = path.with_suffix(".parquet")
        csv = path.with_suffix(".csv")
        if self.backend == "parquet" and pq.exists():
            return pd.read_parquet(pq)
        if self.backend == "csv" and csv.exists():
            return pd.read_csv(csv, parse_dates=["timestamp"])
        if pq.exists():
            return pd.read_parquet(pq)
        if csv.exists():
            return pd.read_csv(csv, parse_dates=["timestamp"])
        return None

    def exists(self, symbol: str, timeframe: str) -> bool:
        p = self._part_path(symbol, timeframe)
        return p.with_suffix(".parquet").exists() or p.with_suffix(".csv").exists()
