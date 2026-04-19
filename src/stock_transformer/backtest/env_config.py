"""Merge ``STX_*`` environment variables into raw YAML dicts before Pydantic coercion.

Precedence is enforced in :func:`stock_transformer.backtest.runner.run_from_config_path`:
CLI flags override these values, which override keys from the file.
"""

from __future__ import annotations

import os
from typing import Any

# env var name -> top-level YAML key
_ENV_OVERRIDES: dict[str, str] = {
    "STX_DEVICE": "device",
    "STX_CACHE_DIR": "cache_dir",
    "STX_ARTIFACTS_DIR": "artifacts_dir",
    "STX_EPOCHS": "epochs",
}


def apply_stx_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply non-empty ``STX_*`` variables to ``cfg`` in place."""
    for env_key, yaml_key in _ENV_OVERRIDES.items():
        v = os.environ.get(env_key, "").strip()
        if not v:
            continue
        if yaml_key == "epochs":
            cfg[yaml_key] = int(v)
        else:
            cfg[yaml_key] = v
    return cfg
