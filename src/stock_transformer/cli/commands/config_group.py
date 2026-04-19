"""``stx config`` — inspect effective merged configuration.

These commands help operators diff file + env + defaults without running training;
they call :func:`stock_transformer.backtest.runner.prepare_backtest_config` for the
same merge path as ``stx backtest``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import click
import yaml
from pydantic import ValidationError

from stock_transformer.cli.commands._common import cli_exit
from stock_transformer.cli.output import config_diff_from_merged, style_text
from stock_transformer.cli.services import validation_error_message
from stock_transformer.config_models import SingleSymbolExperimentConfig, UniverseExperimentConfig


def build_config_group(*, default_config: Callable[[], Path]) -> click.Group:
    """Return the nested ``config`` group with ``show`` and ``diff`` subcommands."""

    @click.group(
        "config",
        invoke_without_command=True,
        context_settings={"help_option_names": ["-h", "--help"]},
        short_help="Inspect merged YAML after env + validation (effective config).",
    )
    @click.pass_context
    def config_group(ctx: click.Context) -> None:
        """Inspect merged effective configuration (flag > env > file > defaults)."""
        if ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())
            ctx.exit(0)

    @config_group.command("show")
    @click.option(
        "-c",
        "--config",
        "config_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=default_config,
        show_default="configs/default.yaml or $STX_CONFIG",
        help="Experiment YAML.",
    )
    @click.pass_context
    def config_show_cmd(ctx: click.Context, config_path: Path) -> None:
        """Print merged, validated config as YAML."""
        import stock_transformer.cli as stx_cli

        try:
            cfg = stx_cli.prepare_backtest_config(config_path)
            click.echo(yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False))
        except ValidationError as e:
            click.echo(style_text(validation_error_message(e, config_path=config_path), fg="red", ctx=ctx), err=True)
            cli_exit(1)

    @config_group.command("diff")
    @click.option(
        "-c",
        "--config",
        "config_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=default_config,
        show_default="configs/default.yaml or $STX_CONFIG",
        help="Experiment YAML.",
    )
    @click.pass_context
    def config_diff_cmd(ctx: click.Context, config_path: Path) -> None:
        """Print keys that differ from Pydantic defaults for this experiment mode."""
        import stock_transformer.cli as stx_cli

        try:
            cfg = stx_cli.prepare_backtest_config(config_path)
            diff = config_diff_from_merged(
                cfg, single_cls=SingleSymbolExperimentConfig, universe_cls=UniverseExperimentConfig
            )
            if not diff:
                click.echo("(no differences from model defaults)")
                return
            click.echo(yaml.safe_dump(diff, sort_keys=True, default_flow_style=False))
        except ValidationError as e:
            click.echo(style_text(validation_error_message(e, config_path=config_path), fg="red", ctx=ctx), err=True)
            cli_exit(1)

    return config_group
