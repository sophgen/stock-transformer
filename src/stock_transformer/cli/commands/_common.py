"""Small helpers shared by Click command modules.

Keeping ``cli_exit`` here avoids duplicating ``SystemExit`` handling across files and
keeps each command file focused on its user-facing strings and option wiring.
"""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


def cli_exit(code: int, message: str | None = None) -> None:
    """Print ``message`` to stdout or stderr, then raise ``SystemExit(code)``.

    stderr is used for non-success codes so stdout stays reserved for JSON / tables
    in composable shell pipelines.
    """
    if message:
        click.echo(message, err=(code != 0))
    raise SystemExit(code)
