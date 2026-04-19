"""Universe configuration: ticker lists, coverage, and label settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    return UniverseConfig(
        symbols=symbols,
        target_symbol=tgt,
        timeframe=str(raw.get("timeframe", "daily")).lower(),
        lookback=int(raw.get("lookback", 32)),
        min_coverage_symbols=int(raw.get("min_coverage_symbols", max(2, len(symbols) - 1))),
        label_mode=str(raw.get("label_mode", "cross_sectional_return")),
        raw=raw,
    )


def membership_table_from_panel(
    timestamps: Any,
    symbols: tuple[str, ...],
    *,
    sector: str | None = None,
) -> list[dict[str, Any]]:
    """Minimal point-in-time table: one row per symbol spanning full panel range.

    v1 uses a static active window over the aligned history; extend with real
    listings/delistings when metadata is available.
    """
    if len(timestamps) == 0:
        return []
    ts_start = timestamps[0]
    ts_end = timestamps[-1]
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        rows.append(
            {
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
                "symbol": sym,
                "active_flag": True,
                "sector": sector or "",
            }
        )
    return rows
