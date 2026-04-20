"""Click application: root group, global options, and command registration.

Handlers live under :mod:`stock_transformer.cli.commands`; this file defines process-wide
CLI policy (logging, signals, dotenv) once per invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from stock_transformer.cli.commands import register_all_commands
from stock_transformer.cli.logging_config import setup_logging
from stock_transformer.cli.output import version_string
from stock_transformer.cli.sigint import install_signal_handlers


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    short_help="Walk-forward experiments, config tools, and data helpers.",
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
    """Leakage-safe walk-forward experiments (single-symbol or universe) and data helpers.

    Root options run before subcommands so library loggers inherit a consistent level and
    long jobs install signal handling once per process.
    """
    load_dotenv()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)
    ctx.ensure_object(dict)
    ctx.obj["no_color"] = no_color
    ctx.obj["quiet"] = quiet
    ctx.obj["use_rich"] = use_rich
    setup_logging(verbose, quiet=quiet, log_file=log_file)
    install_signal_handlers()


register_all_commands(cli)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``stx``; returns a process exit code (for scripts and ``python -m``).

    Using ``standalone_mode=False`` keeps Click from calling ``sys.exit`` inside the
    library so embedders can inspect the integer code. :exc:`KeyboardInterrupt` maps to
    **130** when it reaches this boundary (subcommands normally catch it after signal
    handlers translate SIGINT).
    """
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
    except KeyboardInterrupt:
        return 130


def main_backtest_compat(argv: list[str] | None = None) -> int:
    """Legacy ``stx-backtest`` entry: equivalent to ``stx backtest …``.

    Preserved so packaging, docs, and user muscle memory keep working while the
    preferred invocation is the ``backtest`` subcommand on ``stx``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    return main(["backtest", *args])
