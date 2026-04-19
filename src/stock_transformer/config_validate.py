"""Validate experiment config dicts before running (fail fast with clear errors)."""

from __future__ import annotations

from typing import Any

_ALLOWED_LABEL_MODES = frozenset(
    {
        "cross_sectional_return",
        "raw_return",
        "equal_weighted_return",
        "sector_neutral_return",
    }
)
_ALLOWED_LOSS = frozenset({"mse", "listnet", "approx_ndcg"})
_ALLOWED_STORE = frozenset({"csv", "parquet"})
_ALLOWED_DATA_SOURCE = frozenset({"rest", "mcp"})


def validate_experiment_config(cfg: dict[str, Any] | None) -> None:
    """Raise ``ValueError`` if the config is unusable for the declared ``experiment_mode``."""
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError("Config must be a non-empty YAML mapping")
    mode = str(cfg.get("experiment_mode", "single_symbol")).lower()
    if mode == "universe":
        validate_universe_config(cfg)
    else:
        validate_single_symbol_config(cfg)


def validate_universe_config(cfg: dict[str, Any] | None) -> None:
    """Validate a universe experiment dict (``experiment_mode: universe``)."""
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError("Config must be a non-empty YAML mapping")
    _validate_universe(cfg)


def validate_single_symbol_config(cfg: dict[str, Any] | None) -> None:
    """Validate a single-symbol experiment dict."""
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError("Config must be a non-empty YAML mapping")
    _validate_single_symbol(cfg)


def _validate_universe(cfg: dict[str, Any]) -> None:
    syms = cfg.get("symbols") or []
    if not syms:
        raise ValueError("universe config requires non-empty 'symbols'")
    n_sym = len(syms)
    for key in ("train_bars", "val_bars", "test_bars", "step_bars"):
        if key not in cfg:
            raise ValueError(f"universe config missing required key '{key}'")
        if int(cfg[key]) < 1:
            raise ValueError(f"'{key}' must be >= 1")
    if int(cfg.get("lookback", 0)) < 2:
        raise ValueError("'lookback' must be >= 2")
    mcs = int(
        cfg.get("min_coverage_symbols", max(2, n_sym - 1) if n_sym else 2),
    )
    if mcs < 1:
        raise ValueError("'min_coverage_symbols' must be >= 1")
    if mcs > n_sym:
        raise ValueError("'min_coverage_symbols' cannot exceed len(symbols)")
    lm = str(cfg.get("label_mode", "cross_sectional_return")).lower()
    if lm not in _ALLOWED_LABEL_MODES:
        raise ValueError(
            f"Unknown label_mode {lm!r}; expected one of {sorted(_ALLOWED_LABEL_MODES)}",
        )
    loss = str(cfg.get("loss", "mse")).lower()
    if loss not in _ALLOWED_LOSS:
        raise ValueError(f"Unknown loss {loss!r}; expected one of {sorted(_ALLOWED_LOSS)}")
    st = cfg.get("store")
    if st is not None and str(st).lower() not in _ALLOWED_STORE:
        raise ValueError(f"Unknown store {st!r}; expected 'csv' or 'parquet'")
    ds = str(cfg.get("data_source", "rest")).lower()
    if ds not in _ALLOWED_DATA_SOURCE:
        raise ValueError(f"Unknown data_source {ds!r}; expected 'rest' or 'mcp'")


def _validate_single_symbol(cfg: dict[str, Any]) -> None:
    if not str(cfg.get("symbol", "")).strip():
        raise ValueError("single-symbol config requires non-empty 'symbol'")
    tfs = cfg.get("timeframes")
    if not tfs or not isinstance(tfs, (list, tuple)):
        raise ValueError("single-symbol config requires non-empty 'timeframes' list")
    for key in ("train_bars", "val_bars", "test_bars", "step_bars"):
        if key not in cfg:
            raise ValueError(f"config missing required key '{key}'")
        if int(cfg[key]) < 1:
            raise ValueError(f"'{key}' must be >= 1")
