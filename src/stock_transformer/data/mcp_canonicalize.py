"""Normalize MCP-wrapped Alpha Vantage payloads before canonicalization."""

from __future__ import annotations

from typing import Any


def _has_time_series(d: dict[str, Any]) -> bool:
    return any(isinstance(k, str) and k.startswith("Time Series") for k in d)


def unwrap_mcp_alphavantage_payload(wrapped: dict[str, Any]) -> dict[str, Any]:
    """Strip MCP/tool nesting so :func:`canonicalize_series` / intraday see AV-shaped JSON."""
    if _has_time_series(wrapped):
        return wrapped
    if "tool_result" in wrapped and isinstance(wrapped["tool_result"], dict):
        return unwrap_mcp_alphavantage_payload(wrapped["tool_result"])
    if "payload" in wrapped and isinstance(wrapped["payload"], dict):
        return unwrap_mcp_alphavantage_payload(wrapped["payload"])
    if "result" in wrapped and isinstance(wrapped["result"], dict):
        return unwrap_mcp_alphavantage_payload(wrapped["result"])
    for v in wrapped.values():
        if isinstance(v, dict) and _has_time_series(v):
            return v
        if isinstance(v, dict):
            inner = unwrap_mcp_alphavantage_payload(v)
            if inner is not v or _has_time_series(inner):
                return inner
    return wrapped
