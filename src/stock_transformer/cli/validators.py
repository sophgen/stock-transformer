"""Click callbacks for user-supplied values (clear errors before heavy work runs)."""

from __future__ import annotations

from typing import Any

import click


def device_option(ctx: click.Context, param: click.Parameter, value: Any) -> str | None:
    """Reject empty ``--device`` strings; PyTorch validates the name later."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        raise click.BadParameter(
            "device must not be empty (omit the flag or pass e.g. cpu, cuda:0).", ctx=ctx, param=param
        )
    return s
