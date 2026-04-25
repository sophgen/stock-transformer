"""Fetch daily OHLCV candles from Alpha Vantage and align into a multi-ticker panel."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Final

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL: Final = "https://www.alphavantage.co/query"
DEFAULT_QUERY_RETRIES: Final = 5
DEFAULT_REQUESTS_PER_MINUTE: Final = 5
WINDOW_SEC: Final = 60.0

_log = logger


def _slug(params: dict[str, Any]) -> str:
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _slim_params(full: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in full.items() if k != "apikey"}


def _cleanup_tmp_in_raw(cache_root: Path) -> None:
    raw = cache_root / "raw"
    if not raw.is_dir():
        return
    for p in raw.rglob("*.tmp"):
        try:
            p.unlink()
        except OSError:
            pass


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.parent / f"{path.name}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _is_error(payload: dict[str, Any]) -> str | None:
    for key in ("Error Message", "Note", "Information"):
        if key in payload:
            return str(payload[key])
    return None


def _is_rate_limit(payload: dict[str, Any]) -> bool:
    for key in ("Note", "Information"):
        note = payload.get(key)
        if not isinstance(note, str):
            continue
        lower = note.lower()
        if "api call frequency" in lower or "5 calls per minute" in lower:
            return True
        if "requests per minute" in lower or "api rate limit" in lower:
            return True
        if "premium membership" in lower and "call frequency" in lower:
            return True
        if "api rate" in lower or "rate limit" in lower:
            return True
    return False


def _information_suggests_rate_limit(payload: dict[str, Any]) -> bool:
    inf = payload.get("Information")
    if not isinstance(inf, str):
        return False
    l = inf.lower()
    return "min" in l and ("api" in l or "request" in l or "call" in l)


class SlidingWindowRateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be >= 1")
        self._rpm = requests_per_minute
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        now = self._clock()
        # Drop timestamps outside 60s window
        while self._timestamps and now - self._timestamps[0] >= WINDOW_SEC:
            self._timestamps.popleft()
        while len(self._timestamps) >= self._rpm:
            wait = max(0.0, WINDOW_SEC - (now - self._timestamps[0]) + 0.001)
            self._sleeper(wait)
            now = self._clock()
            while self._timestamps and now - self._timestamps[0] >= WINDOW_SEC:
                self._timestamps.popleft()
        self._timestamps.append(now)


def _raw_path(
    cache_root: Path, function: str, full_params: dict[str, Any]
) -> Path:
    d = cache_root / "raw" / function.lower()
    slim = _slim_params(full_params)
    slug = _slug(slim)
    sym = slim.get("symbol")
    if sym is not None and str(sym).strip():
        name = f"{str(sym).upper().replace('.', '-')}_{slug}.json"
    else:
        name = f"{slug}.json"
    return d / name


def _raw_path_csv(
    cache_root: Path, function: str, full_params: dict[str, Any]
) -> Path:
    d = cache_root / "raw" / function.lower()
    slim = _slim_params(full_params)
    slug = _slug(slim)
    return d / f"{slug}.csv"


class AlphaVantageClient:
    """Alpha Vantage REST client with file cache, rate limiting, and optional TTL.

    Caching: params without ``apikey`` are hashed. When ``symbol`` is present, the
    cache filename is ``{SYMBOL}_{hash}.json`` for easier downstream aggregation.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data",
        *,
        requests_per_minute: int | None = None,
        wall_time: Callable[[], float] | None = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self.cache_root = Path(cache_dir)
        self._sleep = sleeper or time.sleep
        self._wall_time: Callable[[], float] = wall_time or time.time
        rpm = requests_per_minute
        if rpm is None:
            env_rpm = os.environ.get("ALPHAVANTAGE_REQUESTS_PER_MINUTE", "")
            rpm = int(env_rpm) if env_rpm else DEFAULT_REQUESTS_PER_MINUTE
        self._limiter = SlidingWindowRateLimiter(
            int(rpm), clock=clock, sleeper=self._sleep
        )
        self.retries = int(
            os.environ.get("ALPHAVANTAGE_QUERY_RETRIES", DEFAULT_QUERY_RETRIES)
        )
        self._session = requests.Session()
        self._last_stale_fallback = False
        self._last_cache_hit = False
        self._transient_backoff = float(
            os.environ.get("ALPHAVANTAGE_TRANSIENT_BASE_SLEEP", "1.0")
        )
        _cleanup_tmp_in_raw(self.cache_root)

    @property
    def last_stale_fallback(self) -> bool:
        return self._last_stale_fallback

    @property
    def last_cache_hit(self) -> bool:
        return self._last_cache_hit

    @property
    def session(self) -> requests.Session:
        return self._session

    def _atomic_write_cache(self, path: Path, text: str) -> None:
        tmp = path.parent / f"{path.name}.tmp"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _cache_fresh(self, path: Path, max_age_sec: float) -> bool:
        """True if the file is newer than ``max_age_sec`` (in seconds)."""
        if not path.is_file():
            return False
        age = self._wall_time() - path.stat().st_mtime
        return age < max_age_sec

    def _read_json_cache(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _fetch_from_network(
        self, function: str, full: dict[str, Any]
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(max(1, self.retries)):
            self._limiter.acquire()
            try:
                resp = self._session.get(BASE_URL, params=full, timeout=60)
                if resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code}", response=resp
                    )
                resp.raise_for_status()
                payload = resp.json()
                err = _is_error(payload)
                if not err:
                    return payload
                if attempt + 1 < self.retries and _is_rate_limit(payload):
                    self._sleep(self._transient_backoff * (2**attempt))
                    continue
                if attempt + 1 < self.retries and _information_suggests_rate_limit(
                    payload
                ):
                    self._sleep(self._transient_backoff * (2**attempt))
                    continue
                raise RuntimeError(f"Alpha Vantage API error: {err}")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_exc = e
                if attempt + 1 < self.retries:
                    self._sleep(self._transient_backoff * (2**attempt))
                    continue
            except requests.HTTPError as e:
                last_exc = e
                r = e.response
                if r is not None and r.status_code >= 500 and attempt + 1 < self.retries:
                    self._sleep(self._transient_backoff * (2**attempt))
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("Exhausted retries")

    def query(
        self,
        function: str,
        params: dict[str, Any],
        *,
        max_age_sec: float | None = None,
        stale_fallback: bool = True,
        no_cache: bool = False,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "Missing ALPHAVANTAGE_API_KEY. "
                "Copy .env.example to .env and set your key."
            )
        self._last_stale_fallback = False
        self._last_cache_hit = False

        full: dict[str, Any] = {"function": function, "apikey": self.api_key, **params}
        path = _raw_path(self.cache_root, function, full)

        if not no_cache and path.is_file():
            if max_age_sec is None:
                out = self._read_json_cache(path)
                if out is not None:
                    self._last_cache_hit = True
                    return out
            elif self._cache_fresh(path, max_age_sec):
                out = self._read_json_cache(path)
                if out is not None:
                    self._last_cache_hit = True
                    return out

        try:
            payload = self._fetch_from_network(function, full)
            err = _is_error(payload)
            if err:
                raise RuntimeError(f"Alpha Vantage API error: {err}")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write_cache(path, json.dumps(payload))
            return payload
        except Exception as e:  # noqa: BLE001 — stale fallback
            if stale_fallback and path.is_file():
                out = self._read_json_cache(path)
                if out is not None:
                    _log.warning(
                        "Stale cache fallback for %s: %s", path.name, e
                    )
                    self._last_stale_fallback = True
                    return out
            raise

    def query_csv(
        self,
        function: str,
        params: dict[str, Any],
        *,
        max_age_sec: float | None = None,
        no_cache: bool = False,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Missing ALPHAVANTAGE_API_KEY. "
                "Copy .env.example to .env and set your key."
            )
        full = {"function": function, "apikey": self.api_key, **params}
        path = _raw_path_csv(self.cache_root, function, full)

        if not no_cache and path.is_file():
            if max_age_sec is None:
                return path.read_text(encoding="utf-8")
            if self._cache_fresh(path, max_age_sec):
                return path.read_text(encoding="utf-8")

        self._limiter.acquire()
        resp = self._session.get(BASE_URL, params=full, timeout=60)
        resp.raise_for_status()
        text = resp.text
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_cache(path, text)
        return text


def _parse_daily(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse AV daily adjusted JSON into a DataFrame."""
    series_key = None
    for k in payload:
        if isinstance(k, str) and k.startswith("Time Series"):
            series_key = k
            break
    if not series_key:
        raise ValueError(f"No time series in response: {list(payload.keys())}")

    rows: list[dict[str, Any]] = []
    for date_str, fields in payload[series_key].items():
        rows.append({
            "timestamp": pd.Timestamp(date_str),
            "open": float(fields["1. open"]),
            "high": float(fields["2. high"]),
            "low": float(fields["3. low"]),
            "close": float(fields["5. adjusted close"]),
            "volume": float(
                fields.get("6. volume", fields.get("5. volume", 0)) or 0
            ),
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def fetch_universe(
    symbols: list[str],
    cache_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    """Fetch daily adjusted candles for each symbol, return {symbol: DataFrame}."""
    client = AlphaVantageClient(cache_dir=cache_dir)
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym = sym.upper()
        print(f"  Fetching {sym}...")
        params = {
            "symbol": sym,
            "outputsize": "full",
            "datatype": "json",
        }
        payload = client.query("TIME_SERIES_DAILY_ADJUSTED", params, max_age_sec=None)
        result[sym] = _parse_daily(payload)
    return result


def align_universe(candles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Inner-join all symbols on date, return aligned DataFrames sharing the same index."""
    date_sets = [set(df["timestamp"]) for df in candles.values()]
    common = sorted(set.intersection(*date_sets))
    if not common:
        raise ValueError("No overlapping dates across symbols")

    common_set = set(common)
    aligned: dict[str, pd.DataFrame] = {}
    for sym, df in candles.items():
        mask = df["timestamp"].isin(common_set)
        aligned[sym] = df[mask].sort_values("timestamp").reset_index(drop=True)

    print(f"  Aligned {len(candles)} symbols on {len(common)} common trading days")
    print(f"  Date range: {common[0].date()} to {common[-1].date()}")
    return aligned
