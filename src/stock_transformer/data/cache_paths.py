"""Deterministic cache paths for raw API responses and canonical candles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _slug(params: dict) -> str:
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def raw_response_path(cache_root: Path, tool_name: str, params: dict) -> Path:
    d = cache_root / "raw" / tool_name.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_slug(params)}.json"


def canonical_candles_path(
    cache_root: Path,
    symbol: str,
    timeframe: str,
    source_tag: str,
) -> Path:
    d = cache_root / "processed" / "candles"
    d.mkdir(parents=True, exist_ok=True)
    safe_tf = timeframe.replace("/", "-")
    return d / f"{symbol.upper()}__{safe_tf}__{source_tag}.csv"
