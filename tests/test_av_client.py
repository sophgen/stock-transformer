"""Tests for AlphaVantageClient and rate limiter (no network)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

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
