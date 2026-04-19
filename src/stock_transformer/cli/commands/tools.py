"""Meta commands: ``validate`` (CI-friendly) and ``version`` (explicit duplicate of ``--version``).

``validate`` exists so pipelines can fail fast on bad YAML without allocating GPUs;
``version`` mirrors ``stx --version`` for users who prefer a subcommand.
"""

from __future__ import annotations

from pathlib import Path

import click
from pydantic import ValidationError

from stock_transformer.cli.commands._common import cli_exit
from stock_transformer.cli.output import style_text, version_string
from stock_transformer.cli.services import validate_config_file, validation_error_message


def register_validate(cli: click.Group) -> None:
    """Attach ``validate`` to the root group."""

    @cli.command("validate")
    @click.option(
        "-c",
        "--config",
        "config_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        required=True,
        help="Experiment YAML to validate only (no training).",
    )
    @click.pass_context
    def validate_cmd(ctx: click.Context, config_path: Path) -> None:
        """Load and validate a config (Pydantic); exit 0 or 1."""
        try:
            validate_config_file(config_path)
        except ValidationError as e:
            click.echo(style_text(validation_error_message(e, config_path=config_path), fg="red", ctx=ctx), err=True)
            cli_exit(1)
        except ValueError as e:
            click.echo(style_text(str(e), fg="red", ctx=ctx), err=True)
            cli_exit(1)
        click.echo(f"OK: {config_path}")


def register_version(cli: click.Group) -> None:
    """Attach ``version`` to the root group."""

    @cli.command("version")
    def version_cmd() -> None:
        """Print version and runtime device info (same as ``stx --version``)."""
        click.echo(version_string())
