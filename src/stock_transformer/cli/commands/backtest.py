"""``stx backtest`` — walk-forward experiment dispatch (single-symbol or universe).

The handler stays thin: merge/validate config via :mod:`stock_transformer.cli.services`,
then format results with :mod:`stock_transformer.cli.output` so runners stay testable
without Click.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import click
from pydantic import ValidationError

from stock_transformer.cli.commands._common import cli_exit
from stock_transformer.cli.output import emit_backtest_result, style_text
from stock_transformer.cli.progress_display import backtest_progress_callback
from stock_transformer.cli.services import run_backtest, validation_error_message
from stock_transformer.cli.validators import device_option

logger = logging.getLogger(__name__)


def register_backtest(cli: click.Group, *, default_config: Callable[[], Path]) -> None:
    """Attach ``backtest`` to ``cli`` (keeps the root :mod:`stock_transformer.cli.app` minimal)."""

    @cli.command("backtest")
    @click.option(
        "-c",
        "--config",
        "config_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=default_config,
        show_default="configs/default.yaml or $STX_CONFIG",
        help="Experiment YAML.",
    )
    @click.option("--synthetic", is_flag=True, help="Use synthetic candles (no API).")
    @click.option(
        "--device",
        default=None,
        metavar="NAME",
        callback=device_option,
        help="PyTorch device (overrides YAML and STX_DEVICE).",
    )
    @click.option(
        "--seed",
        type=int,
        default=None,
        help="Override YAML seed (after STX_SEED / file).",
    )
    @click.option(
        "-o",
        "--output-format",
        type=click.Choice(["text", "json"], case_sensitive=False),
        default="text",
        show_default=True,
        help="text: one-line summary; json: full summary dict on stdout.",
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Resolve data and print fold plan only (no training).",
    )
    @click.pass_context
    def backtest_cmd(
        ctx: click.Context,
        config_path: Path,
        synthetic: bool,
        device: str | None,
        seed: int | None,
        output_format: str,
        dry_run: bool,
    ) -> None:
        """Run a walk-forward experiment from config (single-symbol or universe)."""
        try:
            progress = backtest_progress_callback(ctx, dry_run=dry_run)
            summary = run_backtest(
                config_path,
                synthetic=synthetic,
                device=device,
                seed=seed,
                dry_run=dry_run,
                progress=progress,
            )
        except ValidationError as e:
            click.echo(style_text(validation_error_message(e, config_path=config_path), fg="red", ctx=ctx), err=True)
            cli_exit(1)
        except KeyboardInterrupt:
            logger.warning("Interrupted")
            cli_exit(130)
        except ValueError as e:
            click.echo(style_text(str(e), fg="red", ctx=ctx), err=True)
            cli_exit(1)

        emit_backtest_result(summary, output_format=output_format.lower(), ctx=ctx)
