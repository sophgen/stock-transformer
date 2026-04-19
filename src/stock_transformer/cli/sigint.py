"""Process-level signal handling for long-running CLI commands.

We translate SIGINT to :exc:`KeyboardInterrupt` so library code can use normal ``try``/``except``
paths and the CLI can exit with **130** without printing a traceback. SIGTERM (common from
process managers and containers) exits with **143** (128 + 15), which matches common POSIX
expectations for graceful termination.
"""

from __future__ import annotations

import signal
from typing import Any


def install_signal_handlers() -> None:
    """Install handlers once per ``stx`` invocation (after root logging is configured)."""

    def _int_handler(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    def _term_handler(_signum: int, _frame: Any) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGINT, _int_handler)
    signal.signal(signal.SIGTERM, _term_handler)


def install_sigint_handler() -> None:
    """Backward-compatible alias for :func:`install_signal_handlers`."""
    install_signal_handlers()
