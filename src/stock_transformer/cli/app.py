"""Click application: command tree, options, and thin handlers that delegate to ``services``."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from stock_transformer.cli.logging_config import setup_logging
from stock_transformer.cli.output import (
    config_diff_from_merged,
    default_config_path,
    emit_backtest_result,
    format_sweep_table,
    style_text,
    version_string,
)
from stock_transformer.cli.progress_display import backtest_progress_callback
from stock_transformer.cli.services import (
    run_backtest,
    run_fetch,
    run_sweep,
    validate_config_file,
    validation_error_message,
)
from stock_transformer.cli.sigint import install_sigint_handler
from stock_transformer.cli.validators import device_option
from stock_transformer.config_models import SingleSymbolExperimentConfig, UniverseExperimentConfig
from stock_transformer.data.fetch_cmd import DEFAULT_UNIVERSE

logger = logging.getLogger(__name__)


def _exit(code: int, message: str | None = None) -> None:
    if message:
        click.echo(message, err=(code != 0))
    raise SystemExit(code)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option("-v", "--verbose", count=True, help="More log detail (-v INFO, -vv DEBUG).")
@click.option("-q", "--quiet", is_flag=True, help="Only warnings and errors.")
@click.option(
    "--log-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Also append logs to this file.",
)
@click.option("--no-color", is_flag=True, help="Disable styled output (or set NO_COLOR).")
@click.option(
    "--rich",
    "use_rich",
    is_flag=True,
    help="Use Rich for fold/epoch lines when the 'rich' package is installed.",
)
@click.version_option(version=version_string(), prog_name="stx")
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: int,
    quiet: bool,
    log_file: Path | None,
    no_color: bool,
    use_rich: bool,
) -> None:
    """Leakage-safe walk-forward experiments (single-symbol or universe) and data helpers."""
    load_dotenv()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)
    ctx.ensure_object(dict)
    ctx.obj["no_color"] = no_color
    ctx.obj["quiet"] = quiet
    ctx.obj["use_rich"] = use_rich
    setup_logging(verbose, quiet=quiet, log_file=log_file)
    install_sigint_handler()


@cli.command("backtest")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=default_config_path,
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
        _exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        _exit(130)
    except ValueError as e:
        click.echo(style_text(str(e), fg="red", ctx=ctx), err=True)
        _exit(1)

    emit_backtest_result(summary, output_format=output_format.lower(), ctx=ctx)


@cli.command("fetch")
@click.option(
    "--cache-dir",
    default="data",
    show_default=True,
    help="Root for raw/ and canonical/.",
)
@click.option(
    "--symbols",
    multiple=True,
    help="Ticker symbols (repeatable). Default: pilot universe.",
)
@click.option(
    "--refresh",
    is_flag=True,
    help="Re-download and overwrite canonical CSV.",
)
@click.pass_context
def fetch_cmd(ctx: click.Context, cache_dir: str, symbols: tuple[str, ...], refresh: bool) -> None:
    """Fetch daily-adjusted OHLCV for symbols into the local cache."""
    syms = list(symbols) if symbols else list(DEFAULT_UNIVERSE)
    try:
        run_fetch(cache_dir, syms, refresh=refresh)
    except KeyboardInterrupt:
        logger.warning("Fetch interrupted")
        _exit(130)
    except Exception as e:
        click.echo(style_text(str(e), fg="red", ctx=ctx), err=True)
        _exit(1)


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
        _exit(1)
    except KeyboardInterrupt:
        _exit(130)


@click.group("config", invoke_without_command=True)
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
    default=default_config_path,
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
        _exit(1)


@config_group.command("diff")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=default_config_path,
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
        _exit(1)


cli.add_command(config_group)


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
        _exit(1)
    except ValueError as e:
        click.echo(style_text(str(e), fg="red", ctx=ctx), err=True)
        _exit(1)
    click.echo(f"OK: {config_path}")


@cli.command("version")
def version_cmd() -> None:
    """Print version and runtime device info (same as ``stx --version``)."""
    click.echo(version_string())


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``stx``; returns a process exit code."""
    try:
        cli.main(args=argv, prog_name="stx", standalone_mode=False)
        return 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1


def main_backtest_compat(argv: list[str] | None = None) -> int:
    """Legacy ``stx-backtest`` entry: equivalent to ``stx backtest …``."""
    args = list(sys.argv[1:] if argv is None else argv)
    return main(["backtest", *args])
