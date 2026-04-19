"""``stx sweep`` — ranking-loss comparison on a universe YAML.

Delegates to :func:`stock_transformer.backtest.loss_sweep.run_loss_sweep` via
:mod:`stock_transformer.cli.services` so sweep logic stays importable without Click.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from pydantic import ValidationError

from stock_transformer.cli.commands._common import cli_exit
from stock_transformer.cli.output import format_sweep_table, style_text
from stock_transformer.cli.services import run_sweep, validation_error_message


def register_sweep(cli: click.Group) -> None:
    """Attach ``sweep`` to the root group."""

    @cli.command("sweep")
    @click.option(
        "-c",
        "--config",
        "config_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("configs/universe.yaml"),
        show_default=True,
        help="Universe experiment YAML.",
    )
    @click.option("--synthetic", is_flag=True, help="Use synthetic universe data.")
    @click.option(
        "-o",
        "--output-format",
        type=click.Choice(["text", "json"], case_sensitive=False),
        default="text",
        show_default=True,
        help="text: comparison table; json: merged sweep summary.",
    )
    @click.pass_context
    def sweep_cmd(ctx: click.Context, config_path: Path, synthetic: bool, output_format: str) -> None:
        """Run loss sweep (mse, listnet, approx_ndcg) and merge summaries."""
        try:
            merged = run_sweep(config_path, use_synthetic=synthetic)
            fmt = output_format.lower()
            if fmt == "json":
                click.echo(json.dumps(merged, indent=2, default=str))
            else:
                click.echo(format_sweep_table(merged))
        except ValidationError as e:
            click.echo(style_text(validation_error_message(e, config_path=config_path), fg="red", ctx=ctx), err=True)
            cli_exit(1)
        except KeyboardInterrupt:
            cli_exit(130)
