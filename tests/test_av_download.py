"""Download orchestration (no network; uses temp dirs)."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from stock_transformer.av_download import (
    load_symbol_list,
    normalize_symbol,
    phase4_write_parquet,
    run_download,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def test_config_loader_normalizes_class_shares() -> None:
    s = "brk.b"
    assert normalize_symbol(s) == "BRK-B"
    t = load_symbol_list("BRK.B\n#x\nAAPL  ")
    assert t == ["BRK-B", "AAPL"]


def test_dry_run_planned(tmp_path: Path) -> None:
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(
        dedent(
            f"""
    output_dir: "{tmp_path}/d"
    symbols: ["AAPL"]
    data_types:
      ohlcv: true
      fundamentals: true
      dividends: true
      splits: true
      macro: true
    """
        ),
        encoding="utf-8",
    )
    s = run_download(str(cfgp), dry_run=True)
    assert s.planned_calls > 0
    assert s.calls_made == 0


def test_phase4_idempotent_ohlcv(tmp_path: Path) -> None:
    d = tmp_path / "d"
    raw = d / "raw" / "time_series_daily_adjusted"
    raw.mkdir(parents=True)
    pld = {
        "Meta Data": {"2. Symbol": "A"},
        "Time Series (Daily)": {
            "2020-01-01": {
                "1. open": "1",
                "2. high": "2",
                "3. low": "0.5",
                "4. close": "1.1",
                "5. adjusted close": "1.0",
                "6. volume": "10",
            }
        },
    }
    p = raw / f"AAPL_{'a' * 16}.json"
    p.write_text(json.dumps(pld), encoding="utf-8")
    a = phase4_write_parquet(d)
    p1 = d / "processed" / "ohlcv.parquet"
    b1 = p1.read_bytes() if p1.is_file() else b""
    a2 = phase4_write_parquet(d)
    b2 = p1.read_bytes() if p1.is_file() else b""
    assert a and a2
    assert b1 == b2
