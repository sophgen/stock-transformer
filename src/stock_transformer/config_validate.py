"""Validate experiment config dicts before running (fail fast with clear errors)."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from stock_transformer.config_models import (
    SingleSymbolExperimentConfig,
    UniverseExperimentConfig,
    coerce_experiment_config,
)


def _validation_to_value_error(exc: ValidationError) -> ValueError:
    parts: list[str] = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e["loc"])
        parts.append(f"{loc}: {e['msg']}")
    return ValueError("; ".join(parts))


def format_validation_error(exc: ValidationError, *, path_hint: str = "config") -> str:
    """Human-readable multi-line message for CLI users (bullet list)."""
    lines = [f"Error: invalid configuration ({path_hint})"]
    for e in exc.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "(root)"
        lines.append(f"  • {loc}: {e['msg']}")
    return "\n".join(lines)


def validate_experiment_config(cfg: dict[str, Any] | None) -> None:
    """Raise ``ValueError`` if the config is unusable for the declared ``experiment_mode``."""
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError("Config must be a non-empty YAML mapping")
    try:
        coerce_experiment_config(cfg)
    except ValidationError as e:
        raise _validation_to_value_error(e) from e


def validate_universe_config(cfg: dict[str, Any] | None) -> None:
    """Validate a universe experiment dict (``experiment_mode: universe``)."""
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError("Config must be a non-empty YAML mapping")
    try:
        UniverseExperimentConfig.model_validate(cfg)
    except ValidationError as e:
        raise _validation_to_value_error(e) from e


def validate_single_symbol_config(cfg: dict[str, Any] | None) -> None:
    """Validate a single-symbol experiment dict."""
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError("Config must be a non-empty YAML mapping")
    try:
        SingleSymbolExperimentConfig.model_validate(cfg)
    except ValidationError as e:
        raise _validation_to_value_error(e) from e
