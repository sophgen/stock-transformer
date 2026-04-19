"""Click CLI: ``stx`` with subcommands for backtest, fetch, sweep, validate, and version."""

from __future__ import annotations

import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from stock_transformer import __version__
from stock_transformer.backtest.env_config import apply_stx_env_overrides
from stock_transformer.backtest.loss_sweep import run_loss_sweep
from stock_transformer.backtest.runner import load_config, run_from_config_path
from stock_transformer.config_models import coerce_experiment_config
from stock_transformer.config_validate import format_validation_error
from stock_transformer.data.fetch_cmd import DEFAULT_UNIVERSE, fetch_universe_sample_data

logger = logging.getLogger(__name__)


def setup_logging(verbose: int, *, quiet: bool) -> None:
    """Configure root logging from global ``-v`` / ``-q`` flags (CLI overrides env noise)."""
    if quiet:
        level = logging.WARNING
    else:
        level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _version_string() -> str:
    """Runtime version string for ``--version`` (torch build and resolved auto device)."""
    import torch

    from stock_transformer.device import resolve_device

    dev = resolve_device("auto")
    return f"stock-transformer {__version__} (torch {torch.__version__}, auto→{dev})"


def _install_sigint() -> None:
    """Avoid dumping a traceback on Ctrl+C during long training runs."""

    def _handler(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


def _summary_exit_code(summary: dict[str, Any]) -> int:
    if summary.get("fold_errors"):
        return 2
    err = summary.get("error")
    if err in ("partial_failure", "no_folds"):
        return 2
    return 0


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option("-v", "--verbose", count=True, help="More log detail (-v INFO, -vv DEBUG).")
@click.option("-q", "--quiet", is_flag=True, help="Only warnings and errors.")
@click.option("--no-color", is_flag=True, help="Reserved for future styled output.", hidden=True)
@click.version_option(version=_version_string(), prog_name="stx")
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: bool, no_color: bool) -> None:
    """Leakage-safe walk-forward experiments (single-symbol or universe) and data helpers."""
    load_dotenv()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)
    ctx.ensure_object(dict)
    ctx.obj["no_color"] = no_color
    setup_logging(verbose, quiet=quiet)
    _install_sigint()


@cli.command("backtest")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/default.yaml"),
    show_default=True,
    help="Experiment YAML.",
)
@click.option("--synthetic", is_flag=True, help="Use synthetic candles (no API).")
@click.option(
    "--device",
    default=None,
    metavar="NAME",
    help="PyTorch device (overrides YAML and STX_DEVICE).",
)
def backtest_cmd(config_path: Path, synthetic: bool, device: str | None) -> None:
    """Run a walk-forward experiment from config (single-symbol or universe)."""
    try:
        summary = run_from_config_path(config_path, synthetic=synthetic, device=device)
    except ValidationError as e:
        click.echo(format_validation_error(e, path_hint=str(config_path)), err=True)
        raise SystemExit(1) from e
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        raise SystemExit(130) from None
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from e

    click.echo(f"Run complete. Artifacts: {summary.get('run_dir')}")
    raise SystemExit(_summary_exit_code(summary))


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
def fetch_cmd(cache_dir: str, symbols: tuple[str, ...], refresh: bool) -> None:
    """Fetch daily-adjusted OHLCV for symbols into the local cache."""
    syms = list(symbols) if symbols else list(DEFAULT_UNIVERSE)
    try:
        fetch_universe_sample_data(cache_dir, syms, refresh=refresh)
    except KeyboardInterrupt:
        logger.warning("Fetch interrupted")
        raise SystemExit(130) from None
    except Exception as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from e


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
def sweep_cmd(config_path: Path, synthetic: bool) -> None:
    """Run loss sweep (mse, listnet, approx_ndcg) and merge summaries."""
    try:
        with open(config_path, encoding="utf-8") as f:
            base = yaml.safe_load(f)
        merged = run_loss_sweep(base, config_path=config_path, use_synthetic=synthetic)
        click.echo(json.dumps(merged, indent=2, default=str))
    except ValidationError as e:
        click.echo(format_validation_error(e, path_hint=str(config_path)), err=True)
        raise SystemExit(1) from e
    except KeyboardInterrupt:
        raise SystemExit(130) from None


@cli.command("validate")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Experiment YAML to validate only (no training).",
)
def validate_cmd(config_path: Path) -> None:
    """Load and validate a config (Pydantic); exit 0 or 1."""
    try:
        raw = load_config(config_path)
        if raw is None or not isinstance(raw, dict):
            raise ValueError("Config must be a non-empty YAML mapping")
        apply_stx_env_overrides(raw)
        coerce_experiment_config(raw)
    except ValidationError as e:
        click.echo(format_validation_error(e, path_hint=str(config_path)), err=True)
        raise SystemExit(1) from e
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from e
    click.echo(f"OK: {config_path}")


@cli.command("version")
def version_cmd() -> None:
    """Print version and runtime device info (same as ``stx --version``)."""
    click.echo(_version_string())


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


if __name__ == "__main__":
    raise SystemExit(main())
