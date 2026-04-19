"""Shared helpers for walk-forward runs (paths, git metadata, fold error capture)."""

from __future__ import annotations

import json
import subprocess
import traceback
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


def git_head_short(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()[:40]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def allocate_run_dir(artifacts: Path, prefix: str) -> Path:
    """Create ``artifacts / {prefix}_{utc}_{uuid8}`` and return it."""
    artifacts.mkdir(parents=True, exist_ok=True)
    ts_part = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts / f"{prefix}_{ts_part}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def fold_error_record(fold_id: int, exc: BaseException) -> dict[str, Any]:
    """Structured fold failure for ``summary.json`` and logs."""
    return {
        "fold_id": fold_id,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def append_fold_error_log(run_dir: Path, record: dict[str, Any]) -> None:
    with (run_dir / "fold_errors.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
