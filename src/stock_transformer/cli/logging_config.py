"""Root logging setup for the ``stx`` process.

Library code under ``stock_transformer`` should use ``logging.getLogger(__name__)``; this module
configures the root handler so global ``-v`` / ``-q`` and optional ``STX_LOG_LEVEL`` behave
consistently for the CLI without importing Click in training code. ISO-like timestamps make
log lines sortable and grep-friendly in long backtests and CI artifacts.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_logging(
    verbose: int,
    *,
    quiet: bool,
    log_file: Path | None = None,
) -> None:
    """Apply verbosity, ``STX_LOG_LEVEL`` (when verbosity is neutral), and optional file logging."""
    if quiet:
        level = logging.WARNING
    else:
        level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    env_ll = os.environ.get("STX_LOG_LEVEL", "").strip().upper()
    if env_ll and not quiet and verbose == 0:
        level = getattr(logging, env_ll, level)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
        handlers=handlers,
    )
