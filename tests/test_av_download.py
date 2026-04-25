"""Download orchestration (no network; uses temp dirs)."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest import mock

import pytest

from stock_transformer.av_download import (
    _default_params_for,
    _macro_stem_from_params,
    _ttl_for,
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


def test_dry_run_makes_no_api_calls(tmp_path: Path) -> None:
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


def test_phase4_idempotent_when_rerun_with_same_cache(tmp_path: Path) -> None:
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


def test_ttl_helpers() -> None:
    ttl = {
        "ohlcv": 1.0, "company_overview": 2.0, "fundamentals": 3.0,
        "etf_profile": 4.0, "corporate_actions": 5.0, "macro": 6.0,
    }
    assert _ttl_for("TIME_SERIES_DAILY_ADJUSTED", ttl) == 1.0
    assert _ttl_for("COMPANY_OVERVIEW", ttl) == 2.0
    assert _ttl_for("INCOME_STATEMENT", ttl) == 3.0
    assert _ttl_for("ETF_PROFILE", ttl) == 4.0
    assert _ttl_for("DIVIDENDS", ttl) == 5.0
    assert _ttl_for("REAL_GDP", ttl) == 6.0
    assert _ttl_for("UNKNOWN_FN", ttl) == 86400.0


def test_default_params_and_macro_stem() -> None:
    assert _default_params_for("TIME_SERIES_DAILY_ADJUSTED", "AAPL") == {
        "symbol": "AAPL", "outputsize": "full", "datatype": "json",
    }
    assert _default_params_for("DIVIDENDS", "AAPL") == {
        "symbol": "AAPL", "datatype": "json",
    }
    assert _default_params_for("REAL_GDP", None) == {"datatype": "json"}
    assert _macro_stem_from_params(
        "TREASURY_YIELD", {"datatype": "json", "interval": "daily", "maturity": "10year"}
    ) == "treasury_yield_10year"
    assert _macro_stem_from_params("CPI", {"interval": "monthly"}) == "cpi"


def test_default_params_macro_required_keys() -> None:
    """Legacy logs that pre-date ``params`` retry with macro-aware fallbacks."""
    ty = _default_params_for("TREASURY_YIELD", None)
    assert ty["interval"] == "daily"
    assert ty["maturity"] == "10year"
    assert _default_params_for("FEDERAL_FUNDS_RATE", None)["interval"] == "daily"
    assert _default_params_for("CPI", None)["interval"] == "monthly"


def test_atomic_write_json_no_partial_on_failure(tmp_path: Path) -> None:
    """A failed atomic write must not leave a partial macro JSON behind."""
    from stock_transformer.av_download import _atomic_write_json

    target = tmp_path / "raw" / "macro" / "real_gdp.json"
    bad_payload = {"x": object()}  # not JSON-serializable -> json.dumps raises
    with pytest.raises(TypeError):
        _atomic_write_json(target, bad_payload)
    assert not target.exists()
    leftovers = list(target.parent.rglob("*.tmp"))
    assert leftovers == []


def test_phase4_macro_pk_uniqueness(tmp_path: Path) -> None:
    """Non-treasury macro parquets must assert (date) uniqueness."""
    d = tmp_path / "d"
    mraw = d / "raw" / "macro"
    mraw.mkdir(parents=True)
    pld = {
        "name": "Real GDP",
        "data": [
            {"date": "2020-01-01", "value": "21000.0"},
            {"date": "2020-01-01", "value": "21000.0"},
        ],
    }
    (mraw / "real_gdp.json").write_text(json.dumps(pld), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        phase4_write_parquet(d)


def test_phase4_macro_non_treasury_writes_parquet(tmp_path: Path) -> None:
    """Single-series macro endpoint produces <stem>.parquet."""
    d = tmp_path / "d"
    mraw = d / "raw" / "macro"
    mraw.mkdir(parents=True)
    pld = {
        "name": "Consumer Price Index",
        "data": [
            {"date": "2020-01-01", "value": "100.0"},
            {"date": "2020-02-01", "value": "100.5"},
        ],
    }
    (mraw / "cpi.json").write_text(json.dumps(pld), encoding="utf-8")
    paths = phase4_write_parquet(d)
    out = d / "processed" / "macro" / "cpi.parquet"
    assert out in paths
    assert out.is_file()


def test_retry_errors_replays_macro_via_stored_params(tmp_path: Path) -> None:
    """A logged macro failure (symbol=None) must be replayable via stored params."""
    cfgp = tmp_path / "c.yaml"
    out = tmp_path / "d"
    cfgp.write_text(
        dedent(
            f"""
    output_dir: "{out}"
    symbols: ["AAPL"]
    data_types:
      ohlcv: false
      fundamentals: false
      dividends: false
      splits: false
      macro: false
    requests_per_minute: 100
    """
        ),
        encoding="utf-8",
    )

    proc = out / "processed"
    proc.mkdir(parents=True)
    err_log = [
        {
            "symbol": None,
            "function": "REAL_GDP",
            "error_class": "RuntimeError",
            "error_message": "rate-limited last time",
            "timestamp": 0,
            "params": {"datatype": "json"},
        }
    ]
    (proc / "_errors_latest.json").write_text(json.dumps(err_log), encoding="utf-8")

    fake_payload = {
        "name": "Real GDP",
        "data": [{"date": "2020-01-01", "value": "21000.0"}],
    }
    seen: list[tuple[str, dict]] = []

    def fake_query(fn: str, params: dict, **kw: object) -> dict:
        seen.append((fn, dict(params)))
        return fake_payload

    with mock.patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "k"}):
        with mock.patch(
            "stock_transformer.av_download.AlphaVantageClient"
        ) as MockClient:
            inst = MockClient.return_value
            inst.query = mock.MagicMock(side_effect=fake_query)
            inst.last_cache_hit = False
            inst.last_stale_fallback = False
            summary = run_download(str(cfgp), retry_errors=True)

    assert summary.errors == 0, summary
    assert len(seen) == 1
    assert seen[0][0] == "REAL_GDP"
    assert seen[0][1]["datatype"] == "json"
    assert (out / "raw" / "macro" / "real_gdp.json").is_file()


def test_retry_errors_skips_run_marker(tmp_path: Path) -> None:
    """The synthetic 'RUN'/KeyboardInterrupt marker should not be retried."""
    cfgp = tmp_path / "c.yaml"
    out = tmp_path / "d"
    cfgp.write_text(
        dedent(
            f"""
    output_dir: "{out}"
    symbols: ["AAPL"]
    data_types:
      ohlcv: false
      fundamentals: false
      dividends: false
      splits: false
      macro: false
    """
        ),
        encoding="utf-8",
    )
    proc = out / "processed"
    proc.mkdir(parents=True)
    err_log = [
        {"symbol": None, "function": "RUN", "error_class": "KeyboardInterrupt",
         "error_message": "sigint", "timestamp": 0},
    ]
    (proc / "_errors_latest.json").write_text(
        json.dumps(err_log), encoding="utf-8"
    )

    with mock.patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "k"}):
        with mock.patch(
            "stock_transformer.av_download.AlphaVantageClient"
        ) as MockClient:
            inst = MockClient.return_value
            inst.query = mock.MagicMock(
                side_effect=AssertionError("must not be called for RUN marker")
            )
            inst.last_cache_hit = False
            inst.last_stale_fallback = False
            summary = run_download(str(cfgp), retry_errors=True)

    assert summary.errors == 0
    inst.query.assert_not_called()


def test_error_log_records_params_for_main_path(tmp_path: Path) -> None:
    """Failures in the main path must record the call params for replay."""
    cfgp = tmp_path / "c.yaml"
    out = tmp_path / "d"
    cfgp.write_text(
        dedent(
            f"""
    output_dir: "{out}"
    symbols: ["AAPL"]
    data_types:
      ohlcv: true
      fundamentals: false
      dividends: false
      splits: false
      macro: false
    """
        ),
        encoding="utf-8",
    )

    def fake_query(fn: str, params: dict, **kw: object) -> dict:
        raise RuntimeError("boom")

    with mock.patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "k"}):
        with mock.patch(
            "stock_transformer.av_download.AlphaVantageClient"
        ) as MockClient:
            inst = MockClient.return_value
            inst.query = mock.MagicMock(side_effect=fake_query)
            inst.last_cache_hit = False
            inst.last_stale_fallback = False
            summary = run_download(str(cfgp))

    assert summary.errors >= 1
    err_path = out / "processed" / "_errors_latest.json"
    rows = json.loads(err_path.read_text(encoding="utf-8"))
    ts = next(r for r in rows if r["function"] == "TIME_SERIES_DAILY_ADJUSTED")
    assert ts["params"]["symbol"] == "AAPL"
    assert ts["params"]["outputsize"] == "full"
    assert ts["params"]["datatype"] == "json"
