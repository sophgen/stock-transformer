"""Process-level SIGINT handling for long-running CLI commands.

We convert SIGINT to ``KeyboardInterrupt`` so library code can use normal ``try``/``except``
paths and the CLI can exit with 130 without printing a traceback to users.
"""

from __future__ import annotations

import signal
from typing import Any


def install_sigint_handler() -> None:
    def _handler(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)
