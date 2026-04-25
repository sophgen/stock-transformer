"""Orchestrate bulk AlphaVantage downloads and Parquet exports."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import yaml

from stock_transformer.av_parsers import (
    assert_unique,
    detect_asset_type,
    ohlcv_long_from_time_series,
    parse_company_overview,
    parse_dividends,
    parse_earnings,
    parse_etf_profile,
    parse_financial_statement,
    parse_macro,
    parse_splits,
)
from stock_transformer.data import AlphaVantageClient


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` atomically via tmp + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp"
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

OnError = Literal["skip", "abort"]

_log = logging.getLogger("av_download")
_interrupt_flag = False
_handler_installed = False

_FUNDAMENTAL_FNS: frozenset[str] = frozenset(
    {"INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS"}
)
_NEEDS_ENTITLEMENT: frozenset[str] = frozenset(
    {"TIME_SERIES_DAILY_ADJUSTED", "COMPANY_OVERVIEW"}
)
_MACRO_FNS: frozenset[str] = frozenset(
    {
        "REAL_GDP", "REAL_GDP_PER_CAPITA", "INFLATION", "RETAIL_SALES",
        "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL", "CPI",
        "FEDERAL_FUNDS_RATE", "TREASURY_YIELD",
    }
)


def _ttl_for(fn: str, ttl: dict[str, float]) -> float:
    if fn == "TIME_SERIES_DAILY_ADJUSTED":
        return ttl["ohlcv"]
    if fn == "COMPANY_OVERVIEW":
        return ttl["company_overview"]
    if fn in _FUNDAMENTAL_FNS:
        return ttl["fundamentals"]
    if fn == "ETF_PROFILE":
        return ttl["etf_profile"]
    if fn in {"DIVIDENDS", "SPLITS"}:
        return ttl["corporate_actions"]
    if fn in _MACRO_FNS:
        return ttl["macro"]
    return 86400.0


def _default_params_for(fn: str, sym: str | None) -> dict[str, Any]:
    if fn == "TIME_SERIES_DAILY_ADJUSTED":
        return {"symbol": sym, "outputsize": "full", "datatype": "json"}
    if fn in _FUNDAMENTAL_FNS | {"COMPANY_OVERVIEW", "ETF_PROFILE",
                                  "DIVIDENDS", "SPLITS"}:
        return {"symbol": sym, "datatype": "json"}
    # Macro endpoints with required params: spec defaults from
    # configs/download.yaml::macro. Used as a fallback when retrying
    # an old error log entry that pre-dates the ``params`` field.
    if fn == "TREASURY_YIELD":
        return {"datatype": "json", "interval": "daily", "maturity": "10year"}
    if fn == "FEDERAL_FUNDS_RATE":
        return {"datatype": "json", "interval": "daily"}
    if fn == "CPI":
        return {"datatype": "json", "interval": "monthly"}
    return {"datatype": "json"}


def _macro_stem_from_params(fn: str, params: dict[str, Any]) -> str:
    if fn == "TREASURY_YIELD":
        mat = str(params.get("maturity", "")).strip() or "unknown"
        return f"treasury_yield_{mat}"
    return fn.lower()


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sigint_handler(_signo: int, _frame: Any) -> None:
    global _interrupt_flag
    _interrupt_flag = True
    for h in logging.root.handlers:
        h.flush()
    _log.warning("Interrupt requested; will stop after the current work unit.")


@dataclass(frozen=True)
class DownloadSummary:
    run_id: str
    elapsed_sec: float
    calls_made: int
    cache_hits: int
    stale_fallbacks: int
    errors: int
    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    planned_calls: int = 0
    interrupted: bool = False
    on_error: str = "skip"


@dataclass
class _RunState:
    run_id: str
    on_error: OnError = "skip"
    no_cache: bool = False
    stale_fallback: bool = True
    calls_made: int = 0
    cache_hits: int = 0
    stale_fallbacks: int = 0
    err_rows: list[dict[str, Any]] = field(default_factory=list)


def normalize_symbol(s: str) -> str:
    t = s.strip()
    if not t or t.startswith("#"):
        return ""
    return t.upper().replace(".", "-")


def load_symbol_list(
    text: str, *, inline: list[str] | None = None
) -> list[str]:
    if inline is not None:
        return [normalize_symbol(x) for x in inline if normalize_symbol(x)]
    out: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        n = normalize_symbol(line)
        if n:
            out.append(n)
    return out


def _macro_jobs_from_config(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    m = (cfg or {}).get("macro") or {}
    tmat = m.get("treasury_yield_maturities") or [
        "3month", "2year", "5year", "7year", "10year", "30year",
    ]
    ty_int = m.get("treasury_yield_interval", "daily")
    ffr = m.get("fed_funds_interval", "daily")
    cpi = m.get("cpi_interval", "monthly")
    bj: dict[str, str] = {"datatype": "json"}
    jobs: list[dict[str, Any]] = []
    for fn in (
        "REAL_GDP",
        "REAL_GDP_PER_CAPITA",
        "INFLATION",
        "RETAIL_SALES",
        "DURABLES",
        "UNEMPLOYMENT",
        "NONFARM_PAYROLL",
    ):
        jobs.append({"function": fn, "params": {**bj}, "out_stem": fn.lower()})
    jobs.append(
        {"function": "CPI", "params": {**bj, "interval": cpi}, "out_stem": "cpi"}
    )
    jobs.append(
        {
            "function": "FEDERAL_FUNDS_RATE",
            "params": {**bj, "interval": ffr},
            "out_stem": "federal_funds_rate",
        }
    )
    for mat in tmat:
        jobs.append(
            {
                "function": "TREASURY_YIELD",
                "params": {**bj, "interval": ty_int, "maturity": mat},
                "out_stem": f"treasury_yield_{mat}",
            }
        )
    return jobs


def _ttl_map(cfg: dict[str, Any]) -> dict[str, float]:
    c = (cfg or {}).get("cache_ttl") or {}
    return {
        "ohlcv": float(c.get("ohlcv", 86400)),
        "company_overview": float(c.get("company_overview", 604800)),
        "fundamentals": float(c.get("fundamentals", 2592000)),
        "etf_profile": float(c.get("etf_profile", 2592000)),
        "corporate_actions": float(c.get("corporate_actions", 604800)),
        "macro": float(c.get("macro", 86400)),
    }


def _ent(cfg: dict[str, Any]) -> dict[str, str]:
    e = (cfg or {}).get("entitlement", "delayed")
    return {"entitlement": str(e)} if e else {}


def _record(
    st: _RunState, c: AlphaVantageClient, *, is_hit: bool | None = None
) -> None:
    if is_hit is not None:
        if is_hit:
            st.cache_hits += 1
        else:
            st.calls_made += 1
        return
    if c.last_stale_fallback:
        st.stale_fallbacks += 1
        return
    if c.last_cache_hit:
        st.cache_hits += 1
    else:
        st.calls_made += 1


def _symbol_from_stem(stem: str) -> str:
    if "_" not in stem:
        return stem
    a, b = stem.rsplit("_", 1)
    if len(b) == 16 and all(
        c in "0123456789abcdef" for c in b.lower()
    ):
        return a
    return stem


def phase4_write_parquet(root: Path) -> list[Path]:
    raw, proc = root / "raw", root / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    tdirf = proc / "fundamentals"
    tdirf.mkdir(parents=True, exist_ok=True)
    tdir0 = proc / "corporate_actions"
    tdir0.mkdir(parents=True, exist_ok=True)
    tmac = proc / "macro"
    tmac.mkdir(parents=True, exist_ok=True)

    odir = raw / "time_series_daily_adjusted"
    parts: list[pd.DataFrame] = []
    if odir.is_dir():
        for jf in sorted(odir.glob("*.json")):
            try:
                pld = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not pld or pld.get("Error Message") or pld.get("Note"):
                continue
            sm = _symbol_from_stem(jf.stem)
            df = ohlcv_long_from_time_series(pld, sm)
            if not df.empty:
                parts.append(df)
    if parts:
        o = pd.concat(parts, ignore_index=True)
        p = proc / "ohlcv.parquet"
        o.to_parquet(p, index=False)
        assert_unique(o, ["symbol", "date"], "ohlcv")
        written.append(p)

    cov = raw / "company_overview"
    cacc: list[pd.DataFrame] = []
    if cov.is_dir():
        for jf in sorted(cov.glob("*.json")):
            try:
                pld = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not pld or pld.get("Error Message"):
                continue
            dfc = parse_company_overview(pld, _symbol_from_stem(jf.stem))
            if not dfc.empty:
                cacc.append(dfc)
    if cacc:
        cdf = pd.concat(cacc, ignore_index=True)
        po = tdirf / "company_overview.parquet"
        cdf.to_parquet(po, index=False)
        assert_unique(cdf, ["symbol"], "company_overview")
        written.append(po)

    def _fin(
        dname: str, kind: str, oname: str, keys: list[str]
    ) -> None:
        sub = raw / dname
        if not sub.is_dir():
            return
        acc: list[pd.DataFrame] = []
        for jf in sorted(sub.glob("*.json")):
            try:
                pld = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not pld or pld.get("Error Message"):
                continue
            if not pld.get("annualReports") and not pld.get("quarterlyReports"):
                continue
            dfp = parse_financial_statement(
                pld, _symbol_from_stem(jf.stem), cast(Any, kind)
            )
            if not dfp.empty:
                acc.append(dfp)
        if not acc:
            return
        df0 = pd.concat(acc, ignore_index=True)
        out = tdirf / f"{oname}.parquet"
        df0.to_parquet(out, index=False)
        assert_unique(df0, keys, oname)
        written.append(out)

    _fin(
        "income_statement", "income", "income_statement",
        ["symbol", "fiscalDateEnding", "frequency"],
    )
    _fin(
        "balance_sheet", "balance", "balance_sheet",
        ["symbol", "fiscalDateEnding", "frequency"],
    )
    _fin(
        "cash_flow", "cashflow", "cash_flow",
        ["symbol", "fiscalDateEnding", "frequency"],
    )

    ed = raw / "earnings"
    if ed.is_dir():
        ea: list[pd.DataFrame] = []
        for jf in sorted(ed.glob("*.json")):
            try:
                pld = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not pld or pld.get("Error Message"):
                continue
            ea.append(parse_earnings(pld, _symbol_from_stem(jf.stem)))
        if ea:
            de = pd.concat(ea, ignore_index=True)
            if not de.empty:
                out = tdirf / "earnings.parquet"
                de.to_parquet(out, index=False)
                assert_unique(
                    de, ["symbol", "fiscalDateEnding", "frequency"], "earnings"
                )
                written.append(out)

    etf = raw / "etf_profile"
    if etf.is_dir():
        p_over, p_h = [], []
        for jf in sorted(etf.glob("*.json")):
            try:
                pld = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not pld or pld.get("Error Message"):
                continue
            o, h = parse_etf_profile(pld, _symbol_from_stem(jf.stem))
            if not o.empty:
                p_over.append(o)
            if not h.empty:
                p_h.append(h)
        if p_over:
            pth = tdirf / "etf_profile.parquet"
            pd.concat(p_over, ignore_index=True).to_parquet(pth, index=False)
            written.append(pth)
        if p_h:
            pth2 = tdirf / "etf_holdings.parquet"
            pd.concat(p_h, ignore_index=True).to_parquet(pth2, index=False)
            written.append(pth2)

    for dname, pnm, pfun, ukeys in (
        ("dividends", "dividends", parse_dividends, None),
        ("splits", "splits", parse_splits, None),
    ):
        sub0 = raw / dname
        if not sub0.is_dir():
            continue
        a2: list[pd.DataFrame] = []
        for jf in sorted(sub0.glob("*.json")):
            try:
                pld = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not pld or pld.get("Error Message"):
                continue
            a2.append(pfun(pld, _symbol_from_stem(jf.stem)))
        if not a2:
            continue
        dfa = pd.concat(a2, ignore_index=True)
        if dfa.empty:
            continue
        out = tdir0 / f"{pnm}.parquet"
        dfa.to_parquet(out, index=False)
        if pnm == "dividends" and "ex_dividend_date" in dfa.columns:
            assert_unique(dfa, ["symbol", "ex_dividend_date"], pnm)
        if pnm == "splits" and "effective_date" in dfa.columns:
            assert_unique(dfa, ["symbol", "effective_date"], pnm)
        written.append(out)

    mdir = raw / "macro"
    t_all: list[pd.DataFrame] = []
    if mdir.is_dir():
        for jf in sorted(mdir.glob("*.json")):
            try:
                pld = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not pld or pld.get("Error Message") or pld.get("Note"):
                continue
            stem = jf.stem
            if len(stem) == 16 and all(
                c in "0123456789abcdef" for c in stem.lower()
            ):
                continue
            name0 = str(pld.get("name", stem))[:200]
            mdf0 = parse_macro(
                pld, name0, stem_hint=stem
            )
            if mdf0.empty:
                continue
            is_tr = (
                "treasury" in stem.lower()
                or "treasury" in name0.lower()
                or (
                    "maturity" in mdf0.columns
                    and bool(mdf0["maturity"].notna().any())
                )
            )
            if is_tr and "maturity" in mdf0.columns and "date" in mdf0.columns:
                t_all.append(mdf0)
            else:
                safe_stem = re.sub(r"[^a-zA-Z0-9_]+", "_", stem)[:100]
                if "date" in mdf0.columns:
                    assert_unique(mdf0, ["date"], f"macro:{safe_stem}")
                pth0 = tmac / f"{safe_stem}.parquet"
                mdf0.to_parquet(pth0, index=False)
                written.append(pth0)
    if t_all:
        tyd = pd.concat(t_all, ignore_index=True)
        if "maturity" in tyd.columns and "date" in tyd.columns:
            assert_unique(tyd, ["maturity", "date"], "treasury_yield")
        pty = tmac / "treasury_yield.parquet"
        tyd.to_parquet(pty, index=False)
        written.append(pty)

    return written


def _load_retry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _q(
    c: AlphaVantageClient, st: _RunState, fn: str, p: dict[str, Any], max_age: float | None
) -> dict[str, Any]:
    return c.query(
        fn, p, max_age_sec=max_age, stale_fallback=st.stale_fallback, no_cache=st.no_cache
    )


def _q_err(
    st: _RunState,
    sym: str | None,
    fn: str,
    e: Exception,
    params: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {
        "symbol": sym,
        "function": fn,
        "error_class": type(e).__name__,
        "error_message": str(e),
        "timestamp": time.time(),
    }
    if params is not None:
        row["params"] = params
    st.err_rows.append(row)


def run_download(
    config_path: str,
    *,
    dry_run: bool = False,
    retry_errors: bool = False,
    no_cache: bool = False,
    symbols_override: list[str] | None = None,
) -> DownloadSummary:
    global _interrupt_flag, _handler_installed
    _interrupt_flag = False
    with open(config_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    out_root = Path(cfg.get("output_dir", "data"))
    proc = out_root / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id()
    t0 = time.time()
    types = (cfg or {}).get("data_types") or {}
    on_err = cast(
        OnError, str((cfg or {}).get("on_error", "skip") or "skip")
    )
    st_f = bool((cfg or {}).get("stale_fallback", True))
    st = _RunState(
        run_id=run_id, on_error=on_err, no_cache=no_cache, stale_fallback=st_f
    )
    rpm = int((cfg or {}).get("requests_per_minute", 65))
    ttl = _ttl_map(cfg)
    ent = _ent(cfg)
    if symbols_override is not None:
        syms = load_symbol_list("", inline=symbols_override)
    else:
        inline = (cfg or {}).get("symbols")
        if inline:
            syms = load_symbol_list("", inline=cast(list, inline))
        else:
            sf = Path((cfg or {}).get("symbols_file", "configs/sp500.txt"))
            if not sf.is_file():
                raise FileNotFoundError(f"symbols file not found: {sf}")
            syms = load_symbol_list(sf.read_text(encoding="utf-8"))
    if not syms:
        return DownloadSummary(
            run_id=run_id, elapsed_sec=0.0, calls_made=0, cache_hits=0,
            stale_fallbacks=0, errors=0, output_paths=(), on_error=on_err,
        )

    logf = proc / f"_run_{run_id}.log"
    fh = logging.FileHandler(str(logf), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    _log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(sh)
    _log.setLevel(logging.DEBUG)

    if not _handler_installed:
        try:
            signal.signal(signal.SIGINT, _sigint_handler)
            _handler_installed = True
        except (ValueError, OSError, AttributeError):
            pass

    f_type = bool(types.get("fundamentals", True))
    n = len(syms)
    planned = 0
    if types.get("ohlcv", True):
        planned += n
    if types.get("dividends", True):
        planned += n
    if types.get("splits", True):
        planned += n
    if f_type:
        planned += n * 5
    if types.get("macro", True):
        planned += len(_macro_jobs_from_config(cfg))

    if dry_run:
        eta_min = planned / max(rpm, 1)
        _log.info(
            "dry_run planned_calls~%s (approx), estimated_time~%.0fm at %d req/min",
            planned, eta_min, rpm,
        )
        _log.removeHandler(fh)
        _log.removeHandler(sh)
        fh.close()
        return DownloadSummary(
            run_id=run_id, elapsed_sec=0.0, calls_made=0, cache_hits=0,
            stale_fallbacks=0, errors=0, output_paths=(), planned_calls=planned,
            on_error=on_err,
        )

    client = AlphaVantageClient(
        out_root, requests_per_minute=rpm
    )

    try:
        return _run_download_body(
            client, st, cfg, syms, proc, out_root, logf,
            ttl, ent, types, on_err, retry_errors, run_id, rpm, t0, planned,
        )
    finally:
        _log.removeHandler(fh)
        _log.removeHandler(sh)
        fh.close()


def _run_download_body(
    client: AlphaVantageClient,
    st: _RunState,
    cfg: dict[str, Any],
    syms: list[str],
    proc: Path,
    out_root: Path,
    logf: Path,
    ttl: dict[str, float],
    ent: dict[str, str],
    types: dict[str, Any],
    on_err: OnError,
    retry_errors: bool,
    run_id: str,
    rpm: int,
    t0: float,
    planned: int,
) -> DownloadSummary:
    global _interrupt_flag

    f_type = bool(types.get("fundamentals", True))
    split: dict[str, str] = {}

    retry_list = _load_retry(proc / "_errors_latest.json")
    if retry_errors and not retry_list:
        _log.warning("retry_errors but _errors_latest.json empty or missing")

    if retry_errors and retry_list:
        n_retry = len(retry_list)
        for i, r in enumerate(retry_list, start=1):
            if _interrupt_flag:
                break
            fn = str(r.get("function", ""))
            sym = r.get("symbol")
            if not fn or fn == "RUN":
                continue
            params_log = r.get("params")
            if isinstance(params_log, dict) and params_log:
                p = dict(params_log)
            elif sym:
                p = _default_params_for(fn, str(sym))
            elif fn in _MACRO_FNS:
                p = _default_params_for(fn, None)
            else:
                continue
            params = {**p, **ent} if fn in _NEEDS_ENTITLEMENT else p
            max_age = _ttl_for(fn, ttl)
            try:
                pld = _q(client, st, fn, params, max_age)
                _record(st, client)
                status = "hit" if client.last_cache_hit else (
                    "stale" if client.last_stale_fallback else "fetch"
                )
                _log.info(
                    "[retry %d/%d] %s %s %s",
                    i, n_retry, sym or "-", fn, status,
                )
                if fn in _MACRO_FNS:
                    mdir2 = out_root / "raw" / "macro"
                    stem0 = _macro_stem_from_params(fn, params)
                    _atomic_write_json(mdir2 / f"{stem0}.json", pld)
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "[retry %d/%d] %s %s error: %s",
                    i, n_retry, sym or "-", fn, e,
                )
                _q_err(
                    st, str(sym) if sym else None, fn, e, params=params,
                )
        if not _interrupt_flag:
            paths = phase4_write_parquet(out_root)
        else:
            paths = []
        elapsed = time.time() - t0
        _write_file_errors(st, proc)
        return DownloadSummary(
            run_id=run_id, elapsed_sec=elapsed, calls_made=st.calls_made,
            cache_hits=st.cache_hits, stale_fallbacks=st.stale_fallbacks,
            errors=len(st.err_rows), output_paths=tuple(paths),
            interrupted=bool(_interrupt_flag),
            on_error=on_err,
        )

    def _w(
        fn: str,
        sym0: str | None,
        e: Exception,
        params: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "symbol": sym0,
            "function": fn,
            "error_class": type(e).__name__,
            "error_message": str(e),
            "timestamp": time.time(),
        }
        if params is not None:
            row["params"] = params
        st.err_rows.append(row)
        if on_err == "skip":
            _log.warning("skip %s %s: %s", fn, sym0, e)
        else:
            _log.error("abort on %s %s: %s", fn, sym0, e)
            raise e

    _call_i = 0

    def _progress(sym: str | None, endpoint: str, status: str) -> None:
        nonlocal _call_i
        _call_i += 1
        elapsed_so_far = time.time() - t0
        if _call_i > 1 and elapsed_so_far > 0:
            eta_sec = elapsed_so_far / (_call_i - 1) * max(planned - _call_i, 0)
            eta_m = eta_sec / 60
            eta_str = f"eta {eta_m:.0f}m"
        else:
            eta_str = "eta --"
        _log.info("[%d/%d] %s %s %s %s", _call_i, planned, sym or "-", endpoint, status, eta_str)

    if f_type and not _interrupt_flag:
        for s in syms:
            if _interrupt_flag:
                break
            params_co = {"symbol": s, "datatype": "json", **ent}
            try:
                pld = _q(
                    client, st, "COMPANY_OVERVIEW",
                    params_co,
                    ttl["company_overview"],
                )
                _record(st, client)
                _progress(s, "COMPANY_OVERVIEW", "hit" if client.last_cache_hit else "fetch")
                split[s] = str(detect_asset_type(pld))
            except Exception as e:  # noqa: BLE001
                _progress(s, "COMPANY_OVERVIEW", "error")
                _w("COMPANY_OVERVIEW", s, e, params=params_co)
        sp_path = proc / "_universe_split.json"
        out_split = {
            "Common Stock": [x for x, v in split.items() if v == "Common Stock"],
            "ETF": [x for x, v in split.items() if v == "ETF"],
            "Unknown": [x for x, v in split.items() if v not in (
                "Common Stock", "ETF",
            )],
        }
        sp_path.write_text(json.dumps(out_split, indent=2), encoding="utf-8")
    else:
        for s in syms:
            split[s] = "Unknown"

    for s in syms:
        if _interrupt_flag:
            break
        if not (types or {}).get("ohlcv", True):
            continue
        params_ts = {
            "symbol": s, "outputsize": "full", "datatype": "json", **ent
        }
        try:
            _q(
                client, st, "TIME_SERIES_DAILY_ADJUSTED",
                params_ts,
                ttl["ohlcv"],
            )
            _record(st, client)
            _progress(s, "OHLCV", "hit" if client.last_cache_hit else "fetch")
        except Exception as e:  # noqa: BLE001
            _progress(s, "OHLCV", "error")
            _w("TIME_SERIES_DAILY_ADJUSTED", s, e, params=params_ts)

    for s in syms:
        if _interrupt_flag:
            break
        if f_type and split.get(s) == "Common Stock":
            for fna in (
                "INCOME_STATEMENT",
                "BALANCE_SHEET",
                "CASH_FLOW",
                "EARNINGS",
            ):
                params_fn = {"symbol": s, "datatype": "json"}
                try:
                    _q(
                        client, st, fna, params_fn,
                        ttl["fundamentals"],
                    )
                    _record(st, client)
                    _progress(s, fna, "hit" if client.last_cache_hit else "fetch")
                except Exception as e:  # noqa: BLE001
                    _progress(s, fna, "error")
                    _w(fna, s, e, params=params_fn)
        elif f_type and split.get(s) == "ETF":
            params_etf = {"symbol": s, "datatype": "json"}
            try:
                _q(
                    client, st, "ETF_PROFILE",
                    params_etf,
                    ttl["etf_profile"],
                )
                _record(st, client)
                _progress(s, "ETF_PROFILE", "hit" if client.last_cache_hit else "fetch")
            except Exception as e:  # noqa: BLE001
                _progress(s, "ETF_PROFILE", "error")
                _w("ETF_PROFILE", s, e, params=params_etf)

    for s in syms:
        if _interrupt_flag:
            break
        if types.get("dividends", True):
            params_div = {"symbol": s, "datatype": "json"}
            try:
                _q(
                    client, st, "DIVIDENDS",
                    params_div,
                    ttl["corporate_actions"],
                )
                _record(st, client)
                _progress(s, "DIVIDENDS", "hit" if client.last_cache_hit else "fetch")
            except Exception as e:  # noqa: BLE001
                _progress(s, "DIVIDENDS", "error")
                _w("DIVIDENDS", s, e, params=params_div)
        if types.get("splits", True):
            params_sp = {"symbol": s, "datatype": "json"}
            try:
                _q(
                    client, st, "SPLITS",
                    params_sp,
                    ttl["corporate_actions"],
                )
                _record(st, client)
                _progress(s, "SPLITS", "hit" if client.last_cache_hit else "fetch")
            except Exception as e:  # noqa: BLE001
                _progress(s, "SPLITS", "error")
                _w("SPLITS", s, e, params=params_sp)

    if types.get("macro", True) and not _interrupt_flag:
        mdir2 = out_root / "raw" / "macro"
        for job in _macro_jobs_from_config(cfg):
            if _interrupt_flag:
                break
            fn0 = str(job["function"])
            p0: dict[str, str] = dict(job["params"])
            try:
                pld = _q(
                    client, st, fn0, p0, ttl["macro"],
                )
                _record(st, client)
                _progress(None, fn0, "hit" if client.last_cache_hit else "fetch")
                stem0 = str(job.get("out_stem", fn0))
                _atomic_write_json(mdir2 / f"{stem0}.json", pld)
            except Exception as e:  # noqa: BLE001
                _progress(None, fn0, "error")
                _w(fn0, None, e, params=p0)

    if not _interrupt_flag:
        paths = phase4_write_parquet(out_root)
    else:
        paths = []
    if _interrupt_flag:
        st.err_rows.append(
            {
                "symbol": None, "function": "RUN", "error_class": "KeyboardInterrupt",
                "error_message": "sigint", "timestamp": time.time()
            }
        )
    _write_file_errors(st, proc)
    _log.info("DownloadSummary: calls=%s hits=%s stale=%s err=%s",
              st.calls_made, st.cache_hits, st.stale_fallbacks, len(st.err_rows))
    elapsed = time.time() - t0
    with open(logf, "a", encoding="utf-8") as lfa:
        lfa.write(
            f"\nSummary: {st.calls_made=}, {st.cache_hits=}, "
            f"{st.stale_fallbacks=}, errors={len(st.err_rows)} paths={paths}\n"
        )
    return DownloadSummary(
        run_id=run_id,
        elapsed_sec=elapsed,
        calls_made=st.calls_made,
        cache_hits=st.cache_hits,
        stale_fallbacks=st.stale_fallbacks,
        errors=len(st.err_rows),
        output_paths=tuple(paths) if paths else tuple(),
        planned_calls=planned,
        interrupted=bool(_interrupt_flag),
        on_error=on_err,
    )


def _write_file_errors(st: _RunState, proc: Path) -> None:
    content = json.dumps(st.err_rows, indent=2, default=str)
    (proc / f"_errors_{st.run_id}.json").write_text(content, encoding="utf-8")
    (proc / "_errors_latest.json").write_text(content, encoding="utf-8")
