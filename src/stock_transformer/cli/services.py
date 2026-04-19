"""Orchestration callable from Click without embedding Click in runners.

Imports the ``stock_transformer.cli`` package at runtime inside functions so tests can
``patch("stock_transformer.cli.run_experiment", ...)`` and exercise the same code path as users.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from stock_transformer.backtest.progress import ProgressCallback
from stock_transformer.config_validate import format_validation_error


def run_backtest(
    config_path: Path,
    *,
    synthetic: bool,
    device: str | None,
    seed: int | None,
    dry_run: bool,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    import stock_transformer.cli as stx_cli

    cfg = stx_cli.prepare_backtest_config(config_path, device=device, seed=seed)
    mode = str(cfg.get("experiment_mode") or "single_symbol").lower()
    if mode == "universe":
        return stx_cli.run_universe_experiment(cfg, use_synthetic=synthetic, dry_run=dry_run, progress=progress)
    return stx_cli.run_experiment(cfg, use_synthetic=synthetic, dry_run=dry_run, progress=progress)


def run_fetch(cache_dir: str, symbols: list[str], *, refresh: bool) -> None:
    from stock_transformer.data.fetch_cmd import fetch_universe_sample_data

    fetch_universe_sample_data(cache_dir, symbols, refresh=refresh)


def run_sweep(config_path: Path, *, use_synthetic: bool) -> dict[str, Any]:
    import stock_transformer.cli as stx_cli

    with open(config_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    return stx_cli.run_loss_sweep(base, config_path=config_path, use_synthetic=use_synthetic)


def validate_config_file(config_path: Path) -> None:
    from stock_transformer.backtest.env_config import apply_stx_env_overrides
    from stock_transformer.backtest.runner import load_config
    from stock_transformer.config_models import coerce_experiment_config

    raw = load_config(config_path)
    if raw is None or not isinstance(raw, dict):
        raise ValueError("Config must be a non-empty YAML mapping")
    apply_stx_env_overrides(raw)
    coerce_experiment_config(raw)


def validation_error_message(exc: ValidationError, *, config_path: Path) -> str:
    return format_validation_error(exc, path_hint=str(config_path))
