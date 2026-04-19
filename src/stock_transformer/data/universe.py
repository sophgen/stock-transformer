"""Universe configuration: ticker lists, coverage, and label settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class UniverseConfig:
    """Resolved universe experiment settings."""

    symbols: tuple[str, ...]
    target_symbol: str
    timeframe: str
    lookback: int
    min_coverage_symbols: int
    label_mode: str
    raw: dict[str, Any]

    def target_index(self) -> int:
        try:
            return self.symbols.index(self.target_symbol.upper())
        except ValueError as e:
            raise ValueError(f"target_symbol {self.target_symbol!r} not in symbols") from e


def load_universe_config(path: str | Path) -> UniverseConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("universe YAML must be a mapping")
    syms = raw.get("symbols") or []
    if not syms:
        raise ValueError("universe config requires non-empty 'symbols'")
    symbols = tuple(str(s).upper() for s in syms)
    tgt = str(raw.get("target_symbol", symbols[0])).upper()
    if tgt not in symbols:
        raise ValueError(f"target_symbol {tgt!r} must appear in symbols")
    lm = str(raw.get("label_mode", "cross_sectional_return"))
    return UniverseConfig(
        symbols=symbols,
        target_symbol=tgt,
        timeframe=str(raw.get("timeframe", "daily")).lower(),
        lookback=int(raw.get("lookback", 32)),
        min_coverage_symbols=int(raw.get("min_coverage_symbols", max(2, len(symbols) - 1))),
        label_mode=lm,
        raw=raw,
    )


def load_sector_map(path: str | Path) -> tuple[dict[str, str], str]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("sector_map YAML must be a mapping")
    mapping = raw.get("mapping") or {}
    default = str(raw.get("default_sector", "Unknown"))
    out = {str(k).upper(): str(v) for k, v in mapping.items()}
    return out, default


def sectors_for_symbols(
    symbols: tuple[str, ...],
    sector_map: dict[str, str],
    default_sector: str,
) -> np.ndarray:
    d = default_sector
    return np.array([sector_map.get(str(s).upper(), d) for s in symbols], dtype=object)


def membership_table_from_panel(
    timestamps: Any,
    symbols: tuple[str, ...],
    *,
    listings: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] | None = None,
    sector_by_symbol: dict[str, str] | None = None,
    market_cap_bucket_by_symbol: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Point-in-time membership rows (M8 richer schema)."""
    if len(timestamps) == 0:
        return []
    ts_first = pd.Timestamp(timestamps[0])
    ts_last = pd.Timestamp(timestamps[-1])
    sec_map = sector_by_symbol or {}
    cap_map = market_cap_bucket_by_symbol or {}
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        if listings and sym in listings:
            t0, t1 = listings[sym]
            ts_start = pd.Timestamp(t0) if t0 is not None else ts_first
            ts_end = pd.Timestamp(t1) if t1 is not None else ts_last
        else:
            ts_start, ts_end = ts_first, ts_last
        rows.append(
            {
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
                "symbol": sym,
                "active_flag": True,
                "sector": sec_map.get(sym, ""),
                "market_cap_bucket": cap_map.get(sym, ""),
            }
        )
    return rows
