"""Parse AlphaVantage JSON payloads into analysis-ready :class:`pandas.DataFrame` objects."""

from __future__ import annotations

from typing import Any, Final, Literal, cast

import pandas as pd

_ASSET_STOCK: Final = "Common Stock"
_ASSET_ETF: Final = "ETF"
_ASSET_UNKNOWN: Final = "Unknown"

AssetType = Literal["Common Stock", "ETF", "Unknown"]


def _coerce_numeric_dataframe(
    df: pd.DataFrame, skip_cols: set[str] | None = None
) -> pd.DataFrame:
    if skip_cols is None:
        skip_cols = set()
    out = df.copy()
    for c in out.columns:
        if c in skip_cols:
            continue
        s = out[c].map(
            lambda v: None
            if v in (None, "None", "none", "")
            or (isinstance(v, str) and v.strip() in ("", "None", "none"))
            else v
        )
        out[c] = pd.to_numeric(s, errors="coerce")
    return out


def parse_company_overview(
    payload: dict[str, Any], symbol: str
) -> pd.DataFrame:
    if not payload or (
        "Symbol" not in payload and "Name" not in payload
    ):
        return pd.DataFrame()
    row: dict[str, Any] = {"symbol": symbol}
    for k, v in payload.items():
        if not isinstance(k, str) or k == "_original_error":
            continue
        row[k] = v
    df = pd.DataFrame([row])
    str_keep: set[str] = {
        "symbol", "Name", "Description", "CIK", "AssetType", "Address",
        "Exchange", "Currency", "Country", "Sector", "Industry", "FiscalYearEnd",
        "LatestQuarter", "DividendDate", "ExDividendDate", "DividendYear",
        "DividendMonth", "52WeekHighDate", "52WeekLowDate", "LastSplitFactor",
        "LastSplitDate", "Name",
    }
    for c in list(df.columns):
        if c in str_keep or c.endswith("Date"):
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(
                df[c].replace("None", None), errors="coerce"
            )
    return df


def detect_asset_type(payload: dict[str, Any]) -> AssetType:
    at = payload.get("AssetType")
    if not isinstance(at, str):
        return cast(AssetType, _ASSET_UNKNOWN)
    a = at.strip().lower()
    if "etf" in a or a == "etf":
        return cast(AssetType, _ASSET_ETF)
    if "common stock" in a or a == "stock":
        return cast(AssetType, _ASSET_STOCK)
    return cast(AssetType, _ASSET_UNKNOWN)


def parse_financial_statement(
    payload: dict[str, Any],
    symbol: str,
    _kind: Literal["income", "balance", "cashflow"],
) -> pd.DataFrame:
    ar = payload.get("annualReports")
    qr = payload.get("quarterlyReports")
    rows: list[dict[str, Any]] = []
    for lab, rlist in (("annual", ar), ("quarterly", qr)):
        if not rlist or not isinstance(rlist, list):
            continue
        for d in rlist:
            if not isinstance(d, dict):
                continue
            item = dict(d)
            item["symbol"] = symbol
            item["frequency"] = "annual" if lab == "annual" else "quarterly"
            if "fiscalDateEnding" in item:
                item["fiscalDateEnding"] = pd.Timestamp(item["fiscalDateEnding"])
            rows.append(item)
    if not rows:
        return pd.DataFrame()
    return _coerce_numeric_dataframe(
        pd.DataFrame(rows),
        skip_cols={"symbol", "fiscalDateEnding", "reportedCurrency", "frequency"},
    )


def parse_earnings(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
    annual = payload.get("annualEarnings", []) or []
    qtr = payload.get("quarterlyEarnings", []) or []
    rows: list[dict[str, Any]] = []
    for d in annual:
        if isinstance(d, dict):
            r = {**d, "symbol": symbol, "frequency": "annual"}
            if "fiscalDateEnding" in r:
                r["fiscalDateEnding"] = pd.Timestamp(r["fiscalDateEnding"])
            rows.append(r)
    for d in qtr:
        if isinstance(d, dict):
            r = {**d, "symbol": symbol, "frequency": "quarterly"}
            if "fiscalDateEnding" in r:
                r["fiscalDateEnding"] = pd.Timestamp(r["fiscalDateEnding"])
            if "date" in r and "fiscalDateEnding" not in r:
                r["fiscalDateEnding"] = pd.Timestamp(r["date"])
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    return _coerce_numeric_dataframe(
        pd.DataFrame(rows),
        skip_cols={"symbol", "fiscalDateEnding", "frequency"},
    )


def parse_etf_profile(
    payload: dict[str, Any], symbol: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = payload.get("holdings")
    if not h or not isinstance(h, list):
        over = {k: v for k, v in payload.items() if k != "holdings"}
        over["symbol"] = symbol
        df1 = pd.DataFrame([over])
        for c in list(df1.columns):
            if c in {"symbol", "name", "Name", "Description", "ETF", "netAssets"}:
                continue
            if df1[c].dtype == object:
                df1[c] = pd.to_numeric(
                    df1[c].replace("None", None), errors="coerce"
                )
        return df1, pd.DataFrame()
    hrows: list[dict[str, Any]] = []
    for d in h:
        if not isinstance(d, dict):
            continue
        hrows.append({**d, "etf_symbol": symbol})
    odf = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "name": str(
                    payload.get("name")
                    or payload.get("Name")
                    or payload.get("netAssets", "")
                ),
            }
        ]
    )
    hdf = pd.DataFrame(hrows)
    for c in hdf.columns:
        if c in {"symbol", "etf_symbol", "name", "name:", "ticker", "cik", "ticker1"}:
            continue
        if hdf[c].dtype == object:
            hdf[c] = pd.to_numeric(hdf[c].replace("None", None), errors="coerce")
    return odf, hdf


def parse_dividends(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
    data = payload.get("data", [])
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        r = {**d, "symbol": symbol}
        if "ex_dividend_date" in r:
            r["ex_dividend_date"] = pd.Timestamp(r["ex_dividend_date"])
        elif "date" in r:
            r["ex_dividend_date"] = pd.Timestamp(r["date"])
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return _coerce_numeric_dataframe(
        pd.DataFrame(rows), skip_cols={"symbol", "ex_dividend_date", "payment_date"}
    )


def parse_splits(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
    data = payload.get("data", [])
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        r = {**d, "symbol": symbol}
        if "effective_date" not in r and "date" in r:
            r["effective_date"] = pd.Timestamp(r["date"])
        if "effective_date" in r and isinstance(r["effective_date"], str):
            r["effective_date"] = pd.Timestamp(r["effective_date"])
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return _coerce_numeric_dataframe(
        pd.DataFrame(rows), skip_cols={"symbol", "effective_date", "label"}
    )


def parse_macro(
    payload: dict[str, Any],
    series_name: str,
    *,
    maturity: str | None = None,
    stem_hint: str | None = None,
) -> pd.DataFrame:
    if not payload or "data" not in payload:
        return pd.DataFrame()
    data = payload.get("data", [])
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    mat = maturity
    if mat is None and payload.get("maturity"):
        mat = str(payload["maturity"])
    if mat is None and stem_hint and "treasury_yield_" in stem_hint.lower():
        mat = stem_hint.lower().split("treasury_yield_", 1)[-1]
    rows: list[dict[str, Any]] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        r = {**d, "series": series_name}
        if "date" in d:
            r["date"] = pd.Timestamp(d["date"])
        if mat is not None:
            r["maturity"] = mat
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return _coerce_numeric_dataframe(
        pd.DataFrame(rows), skip_cols={"date", "series", "maturity", "name"}
    )


def ohlcv_long_from_time_series(
    payload: dict[str, Any], symbol: str
) -> pd.DataFrame:
    series_key: str | None = None
    for k in payload:
        if isinstance(k, str) and k.startswith("Time Series"):
            series_key = k
            break
    if not series_key:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for date_str, fields in (payload[series_key] or {}).items():
        if not isinstance(fields, dict):
            continue
        d = {
            "symbol": symbol,
            "date": pd.Timestamp(date_str),
            "open": float(fields.get("1. open", 0) or 0),
            "high": float(fields.get("2. high", 0) or 0),
            "low": float(fields.get("3. low", 0) or 0),
            "close": float(fields.get("4. close", 0) or 0),
            "adjusted_close": float(
                fields.get("5. adjusted close", fields.get("4. close", 0)) or 0
            ),
            "volume": float(
                fields.get("6. volume", fields.get("5. volume", 0)) or 0
            ),
            "dividend_amount": float(fields.get("7. dividend amount", 0) or 0),
            "split_coefficient": float(
                str(fields.get("8. split coefficient", "0") or 0)
            ),
        }
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def assert_unique(
    df: pd.DataFrame, key_cols: list[str], name: str
) -> None:
    if df.empty:
        return
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing key columns: {missing}")
    dupes = df.duplicated(subset=key_cols, keep=False)
    if dupes.any():
        n = int(dupes.sum())
        raise ValueError(
            f"{name}: {n} duplicate rows on {key_cols}, sample: "
            f"{df.loc[dupes, key_cols].head(10).to_dict()!r}"
        )
