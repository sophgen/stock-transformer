"""Click subcommands registered on the root ``stx`` group.

Splitting commands by domain keeps :mod:`stock_transformer.cli.app` short and makes it
obvious where to add a new subcommand without editing an oversized module. Registration
order is: backtest, fetch (and nested ``data`` group), sweep, config, validate, version,
completion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stock_transformer.cli.commands.backtest import register_backtest
from stock_transformer.cli.commands.completion import register_completion
from stock_transformer.cli.commands.config_group import build_config_group
from stock_transformer.cli.commands.fetch import register_fetch
from stock_transformer.cli.commands.sweep import register_sweep
from stock_transformer.cli.commands.tools import register_validate, register_version
from stock_transformer.cli.output import default_config_path

if TYPE_CHECKING:
    import click


def register_all_commands(cli: click.Group) -> None:
    """Attach every subcommand and nested group to ``cli``."""
    dc = default_config_path
    register_backtest(cli, default_config=dc)
    register_fetch(cli)
    register_sweep(cli)
    cli.add_command(build_config_group(default_config=dc))
    register_validate(cli)
    register_version(cli)
    register_completion(cli)
