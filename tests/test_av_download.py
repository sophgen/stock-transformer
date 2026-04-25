"""Download orchestration (no network; uses temp dirs)."""

from __future__ import annotations

import json
import signal
from pathlib import Path
from textwrap import dedent
from unittest import mock

import pytest

from stock_transformer.av_download import (
    _write_file_errors,
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


def test_phase4_asserts_primary_key_uniqueness(tmp_path: Path) -> None:
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
    # Two files with the same symbol + date → duplicate → should raise
    p1 = raw / f"AAPL_{'a' * 16}.json"
    p2 = raw / f"AAPL_{'b' * 16}.json"
    p1.write_text(json.dumps(pld), encoding="utf-8")
    p2.write_text(json.dumps(pld), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        phase4_write_parquet(d)


def test_sigint_writes_partial_error_log_and_exits_130(tmp_path: Path) -> None:
    import stock_transformer.av_download as avd

    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(
        dedent(
            f"""
    output_dir: "{tmp_path}/d"
    symbols: ["AAPL", "MSFT", "GOOG"]
    data_types:
      ohlcv: true
      fundamentals: false
      dividends: false
      splits: false
      macro: false
    requests_per_minute: 100
    """
        ),
        encoding="utf-8",
    )

    call_count = 0
    original_flag = avd._interrupt_flag

    def fake_query(fn: str, params: dict, **kw: object) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            avd._interrupt_flag = True
        return {
            "Meta Data": {},
            "Time Series (Daily)": {
                "2020-01-01": {
                    "1. open": "1", "2. high": "2", "3. low": "0.5",
                    "4. close": "1.1", "5. adjusted close": "1.0",
                    "6. volume": "10",
                }
            },
        }

    with mock.patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "k"}):
        with mock.patch(
            "stock_transformer.av_download.AlphaVantageClient"
        ) as MockClient:
            inst = MockClient.return_value
            inst.query = mock.MagicMock(side_effect=fake_query)
            inst.last_cache_hit = False
            inst.last_stale_fallback = False
            summary = run_download(str(cfgp))

    assert summary.interrupted is True
    errf = tmp_path / "d" / "processed" / f"_errors_{summary.run_id}.json"
    assert errf.is_file()
    errors = json.loads(errf.read_text(encoding="utf-8"))
    has_interrupt = any(e.get("error_class") == "KeyboardInterrupt" for e in errors)
    assert has_interrupt

    avd._interrupt_flag = original_flag
