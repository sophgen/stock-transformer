"""Artifact writers shared by single-symbol and universe runners.

Centralizing JSON/CSV snapshots keeps I/O consistent and makes the runners easier
to test by mocking these helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def save_config_snapshot(run_dir: Path, config: dict[str, Any]) -> None:
    """Write ``config_snapshot.yaml`` for reproducibility."""
    (run_dir / "config_snapshot.yaml").write_text(yaml.dump(config, sort_keys=True))


def save_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    """Write ``summary.json`` (same structure the runners have always produced)."""
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def save_fold_payload(run_dir: Path, folds_payload: dict[str, Any]) -> None:
    """Universe runs: persist fold index ranges and timestamps."""
    (run_dir / "folds.json").write_text(json.dumps(folds_payload, indent=2, default=str))


def save_predictions_csv(
    path: Path,
    df: pd.DataFrame | None,
    *,
    columns: list[str],
) -> None:
    """Write a predictions CSV, emitting an empty file with headers if there are no rows."""
    if df is not None and len(df):
        df.to_csv(path, index=False)
    else:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
