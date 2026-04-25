"""Tests for AlphaVantageClient and rate limiter (no network)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import requests

from stock_transformer.data import (
    AlphaVantageClient,
    SlidingWindowRateLimiter,
    _cleanup_tmp_in_raw,
    _is_rate_limit,
    _raw_path,
    _slim_params,
    _slug,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def test_slug_stable_across_apikeys() -> None:
    a = {"function": "X", "apikey": "a", "symbol": "IBM"}
    b = {**a, "apikey": "b"}
    assert _slug(_slim_params(a)) == _slug(_slim_params(b))


def test_rate_limiter_third_needs_sleeper() -> None:
    t = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        return t

    def sleep(dt: float) -> None:
        sleeps.append(dt)
        nonlocal t
        t += dt

    lim = SlidingWindowRateLimiter(2, clock=clock, sleeper=sleep)
    lim.acquire()
    lim.acquire()
    before = t
    lim.acquire()
    assert t > before or sleeps


def test_rate_limit_premium_message() -> None:
    p: dict = {
        "Information": "Our standard API call frequency is 75 requests per minute"
    }
    assert _is_rate_limit(p) is True


def test_startup_cleans_orphan_tmp_files(tmp_path: Path) -> None:
    p = tmp_path / "raw" / "a"
    p.mkdir(parents=True)
    tmp = p / "x.json.tmp"
    tmp.write_text("a", encoding="utf-8")
    _cleanup_tmp_in_raw(tmp_path)
    assert not tmp.exists()


def test_symbol_prefix_in_raw_path() -> None:
    full = {"function": "TIME", "apikey": "k", "symbol": "AAPL"}
    path = _raw_path(Path("data0"), "F", full)
    assert path.name.startswith("AAPL_")
    assert path.suffix == ".json"


@mock.patch.object(
    AlphaVantageClient, "_fetch_from_network", side_effect=OSError("no net")
)
@mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "x"})
def test_stale_cache_on_fetch_failure(
    m_fetch: Any, tmp_path: Path
) -> None:
    full = {"function": "DEMO", "apikey": "x", "symbol": "Z", "a": 1}
    path = _raw_path(tmp_path, "DEMO", full)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"k":1}', encoding="utf-8")
    os.utime(path, (0, 0))
    c = AlphaVantageClient(
        cache_dir=tmp_path,
        requests_per_minute=30,
        wall_time=lambda: 100.0,
    )
    o = c.query("DEMO", {"symbol": "Z", "a": 1}, max_age_sec=1, stale_fallback=True)
    assert o == {"k": 1}
    assert c.last_stale_fallback is True


@mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "k"})
@mock.patch.object(
    AlphaVantageClient, "_atomic_write_cache", side_effect=OSError("boom")
)
def test_atomic_write_error_propagates(
    m_aw: Any, tmp_path: Path
) -> None:
    c = AlphaVantageClient(
        cache_dir=tmp_path, requests_per_minute=100, wall_time=lambda: 0.0
    )
    with (
        mock.patch.object(
            c, "_fetch_from_network", return_value={"ok": True}
        )
    ):
        with pytest.raises(OSError, match="boom"):
            c.query("X", {"symbol": "A"}, max_age_sec=None)
    assert m_aw.called


@mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "k"})
def test_session_reused_across_calls(tmp_path: Path) -> None:
    c = AlphaVantageClient(
        cache_dir=tmp_path, requests_per_minute=100, wall_time=lambda: 0.0
    )
    payload = {"Time Series (Daily)": {"2020-01-01": {"1. open": "1"}}}
    with mock.patch.object(c, "_fetch_from_network", return_value=payload):
        c.query("A", {"symbol": "X1"}, max_age_sec=None)
        c.query("A", {"symbol": "X2"}, max_age_sec=None)
    assert c.session is c._session


def test_rate_limiter_60s_window() -> None:
    t = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        return t

    def sleep(dt: float) -> None:
        sleeps.append(dt)
        nonlocal t
        t += dt

    lim = SlidingWindowRateLimiter(3, clock=clock, sleeper=sleep)
    lim.acquire()  # t=0
    lim.acquire()
    lim.acquire()
    # Window is full (3 in last 60s). Next acquire must sleep.
    sleeps.clear()
    lim.acquire()
    assert sleeps, "4th call should have slept"
    # After sleeping, t should have advanced past the 60s window edge.
    assert t >= 60.0

    # Advance past original window — slots should have expired.
    t = 121.0
    sleeps.clear()
    lim.acquire()
    assert not sleeps, "After window expiry, acquire should not sleep"


@mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "k"})
def test_ttl_boundary_mtime_equals_max_age_is_miss(tmp_path: Path) -> None:
    wall = 200.0
    full = {"function": "DEMO", "apikey": "k", "symbol": "A"}
    path = _raw_path(tmp_path, "DEMO", full)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"cached": true}', encoding="utf-8")
    # mtime = 100, wall = 200, max_age = 100 → age == max_age → miss (not <)
    os.utime(path, (100.0, 100.0))

    c = AlphaVantageClient(
        cache_dir=tmp_path, requests_per_minute=100,
        wall_time=lambda: wall,
    )
    fresh = {"fresh": True}
    with mock.patch.object(c, "_fetch_from_network", return_value=fresh):
        result = c.query("DEMO", {"symbol": "A"}, max_age_sec=100.0)
    assert result == fresh
    assert c.last_cache_hit is False


@mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "k"})
def test_ttl_none_keeps_old_cache_forever(tmp_path: Path) -> None:
    full = {"function": "DEMO", "apikey": "k", "symbol": "B"}
    path = _raw_path(tmp_path, "DEMO", full)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"old": true}', encoding="utf-8")
    os.utime(path, (0, 0))  # very old mtime

    c = AlphaVantageClient(
        cache_dir=tmp_path, requests_per_minute=100,
        wall_time=lambda: 1e9,
    )
    result = c.query("DEMO", {"symbol": "B"}, max_age_sec=None)
    assert result == {"old": True}
    assert c.last_cache_hit is True


@mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "k"})
def test_transient_5xx_retries_and_succeeds(tmp_path: Path) -> None:
    c = AlphaVantageClient(
        cache_dir=tmp_path, requests_per_minute=200,
        wall_time=lambda: 0.0, sleeper=lambda _: None,
    )
    c.retries = 3
    c._transient_backoff = 0.0

    mock_500 = mock.MagicMock()
    mock_500.status_code = 500
    mock_500.raise_for_status.side_effect = requests.HTTPError(
        "500", response=mock_500
    )

    mock_ok = mock.MagicMock()
    mock_ok.status_code = 200
    mock_ok.raise_for_status = mock.MagicMock()
    mock_ok.json.return_value = {"data": "ok"}

    c._session.get = mock.MagicMock(side_effect=[mock_500, mock_ok])
    result = c.query("F", {"symbol": "Z"}, max_age_sec=None)
    assert result == {"data": "ok"}
    assert c._session.get.call_count == 2
