"""Click callbacks for user-supplied values (clear errors before heavy work runs).

Failing fast here keeps stack traces out of user-facing output and matches the
``flag > env > file > defaults`` story: we validate CLI tokens before merging YAML.
"""

from __future__ import annotations

from typing import Any

import click


def device_option(ctx: click.Context, param: click.Parameter, value: Any) -> str | None:
    """Reject whitespace-only ``--device`` so we never write a useless value into the merged config.

    PyTorch still validates the device name when the session starts; this step only
    catches the common ``--device ""`` mistake with an immediate, local error.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        raise click.BadParameter(
            "device must not be empty (omit the flag or pass e.g. cpu, cuda:0).", ctx=ctx, param=param
        )
    return s


def normalize_fetch_symbols(ctx: click.Context, param: click.Parameter, value: Any) -> tuple[str, ...]:
    """Normalize ``--symbols`` values after Click collects them into one tuple.

    For ``multiple=True`` options, Click invokes the callback once with the full
    tuple (or ``()`` when the flag is omitted). We strip, uppercase, and reject
    blanks so the data layer always receives clean tickers consistent with YAML.
    """
    if not value:
        return ()
    out: list[str] = []
    for raw in value:
        s = str(raw).strip().upper()
        if not s:
            raise click.BadParameter(
                "each --symbols value must be non-empty (example: --symbols MSTR).", ctx=ctx, param=param
            )
        out.append(s)
    return tuple(out)
