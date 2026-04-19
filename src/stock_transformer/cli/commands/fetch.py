"""``stx fetch`` — Alpha Vantage download + canonical cache writes.

Network and filesystem work lives in :mod:`stock_transformer.data.fetch_cmd`; this
module only wires flags and maps failures to exit codes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from stock_transformer.cli.commands._common import cli_exit
from stock_transformer.cli.output import style_text
from stock_transformer.cli.services import run_fetch
from stock_transformer.cli.validators import cache_dir_option, normalize_fetch_symbols
from stock_transformer.data.fetch_cmd import DEFAULT_UNIVERSE

logger = logging.getLogger(__name__)


def register_fetch(cli: click.Group) -> None:
    """Attach ``fetch`` to the root group."""

    @cli.command("fetch")
    @click.option(
        "--cache-dir",
        type=click.Path(path_type=Path, file_okay=False),
        default=Path("data"),
        callback=cache_dir_option,
        show_default=True,
        help="Root for raw/ and canonical/.",
    )
    @click.option(
        "--symbols",
        multiple=True,
        callback=normalize_fetch_symbols,
        help="Ticker symbols (repeatable). Default: pilot universe.",
    )
    @click.option(
        "--refresh",
        is_flag=True,
        help="Re-download and overwrite canonical CSV.",
    )
    @click.pass_context
    def fetch_cmd(ctx: click.Context, cache_dir: Path, symbols: tuple[str, ...], refresh: bool) -> None:
        """Fetch daily-adjusted OHLCV for symbols into the local cache."""
        syms = list(symbols) if symbols else list(DEFAULT_UNIVERSE)
        try:
            run_fetch(str(cache_dir), syms, refresh=refresh)
        except KeyboardInterrupt:
            logger.warning("Fetch interrupted")
            cli_exit(130)
        except Exception as e:
            click.echo(style_text(str(e), fg="red", ctx=ctx), err=True)
            cli_exit(1)
