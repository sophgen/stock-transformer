"""Click CLI: ``stx`` with subcommands for backtest, fetch, sweep, validate, and version."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError
from pydantic_core import PydanticUndefined

from stock_transformer import __version__
from stock_transformer.backtest.env_config import apply_stx_env_overrides
from stock_transformer.backtest.loss_sweep import run_loss_sweep
from stock_transformer.backtest.progress import ProgressCallback
from stock_transformer.backtest.runner import load_config, prepare_backtest_config, run_experiment
from stock_transformer.backtest.universe_runner import run_universe_experiment
from stock_transformer.config_models import (
    SingleSymbolExperimentConfig,
    UniverseExperimentConfig,
    coerce_experiment_config,
)
from stock_transformer.config_validate import format_validation_error
from stock_transformer.data.fetch_cmd import DEFAULT_UNIVERSE, fetch_universe_sample_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StxResult:
    """Structured exit payload (pair with :func:`_exit` for tooling)."""

    code: int
    message: str | None = None


def _exit(code: int, message: str | None = None) -> None:
    if message:
        click.echo(message, err=(code != 0))
    raise SystemExit(code)


def _default_config_path() -> Path:
    return Path(os.environ.get("STX_CONFIG", "configs/default.yaml"))


def setup_logging(
    verbose: int,
    *,
    quiet: bool,
    log_file: Path | None = None,
) -> None:
    """Configure root logging from global ``-v`` / ``-q``, optional ``STX_LOG_LEVEL``, and ``--log-file``."""
    if quiet:
        level = logging.WARNING
    else:
        level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    env_ll = os.environ.get("STX_LOG_LEVEL", "").strip().upper()
    if env_ll and not quiet and verbose == 0:
        level = getattr(logging, env_ll, level)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=handlers,
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


def _use_color(ctx: click.Context | None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if ctx and ctx.obj and ctx.obj.get("no_color"):
        return False
    return True


def _style_text(text: str, *, fg: str, ctx: click.Context | None) -> str:
    if not _use_color(ctx):
        return text
    return click.style(text, fg=fg)


def _emit_backtest_result(
    summary: dict[str, Any],
    *,
    output_format: str,
    ctx: click.Context | None,
) -> None:
    code = _summary_exit_code(summary)
    if output_format == "json":
        click.echo(json.dumps(summary, indent=2, default=str))
    else:
        if summary.get("dry_run") and summary.get("fold_plan"):
            n_s = summary.get("n_samples")
            n_f = summary.get("n_folds")
            if n_s is not None and n_f is not None:
                click.echo(f"Dry run: {n_s} samples in index, {n_f} fold(s). Fold boundaries:")
            else:
                click.echo("Dry run: fold boundaries:")
            click.echo(
                yaml.safe_dump(
                    summary["fold_plan"],
                    sort_keys=True,
                    default_flow_style=False,
                    allow_unicode=True,
                ).rstrip()
            )
        if code == 0:
            msg = f"Run complete. Artifacts: {summary.get('run_dir')}"
            click.echo(_style_text(msg, fg="green", ctx=ctx))
        else:
            err = summary.get("error", "run_failed")
            msg = f"Run finished with issues ({err}). Artifacts: {summary.get('run_dir')}"
            click.echo(_style_text(msg, fg="yellow", ctx=ctx))
    _exit(code)


def _fmt_metric_cell(x: Any) -> str:
    return "—" if x is None else f"{float(x):.3f}"


def _sweep_metrics_row(agg: dict[str, Any] | None) -> tuple[str, str, str]:
    if not agg:
        return ("—", "—", "—")
    sp = agg.get("spearman_mean_mean")
    nd = agg.get("ndcg3_mean_mean")
    hit = agg.get("top2_hit_mean")
    return (_fmt_metric_cell(sp), _fmt_metric_cell(nd), _fmt_metric_cell(hit))


def _format_sweep_table(merged: dict[str, Any]) -> str:
    by_loss = merged.get("by_loss") or {}
    lines = [
        f"{'Loss':<12} | {'Spearman':>10} | {'NDCG@3':>8} | {'Hit@2':>8}",
        f"{'-' * 12}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}",
    ]
    for loss in ("mse", "listnet", "approx_ndcg"):
        block = by_loss.get(loss)
        agg = block.get("aggregate") if isinstance(block, dict) else None
        sp, nd, hit = _sweep_metrics_row(agg if isinstance(agg, dict) else None)
        lines.append(f"{loss:<12} | {sp:>10} | {nd:>8} | {hit:>8}")
    return "\n".join(lines)


def _pydantic_default_for_field(field_info: Any) -> Any:
    if field_info.default_factory is not None:
        return field_info.default_factory()
    if field_info.default is not PydanticUndefined:
        return field_info.default
    return PydanticUndefined


def _config_diff_from_merged(cfg: dict[str, Any]) -> dict[str, Any]:
    mode = str(cfg.get("experiment_mode") or "single_symbol").lower()
    cls = UniverseExperimentConfig if mode == "universe" else SingleSymbolExperimentConfig
    diff: dict[str, Any] = {}
    for name, finfo in cls.model_fields.items():
        dv = _pydantic_default_for_field(finfo)
        if dv is PydanticUndefined:
            continue
        if name not in cfg:
            continue
        if cfg[name] != dv:
            diff[name] = {"value": cfg[name], "default": dv}
    return diff


class StxCliProgress:
    """Per-fold / per-epoch lines on stderr; optional Rich styling when installed (``--rich``)."""

    def __init__(self, *, use_rich: bool) -> None:
        self._use_rich = use_rich
        self._console: Any = None
        if use_rich:
            try:
                from rich.console import Console

                self._console = Console(stderr=True, highlight=False)
            except ImportError:
                self._use_rich = False

    def on_fold_start(self, fold_id: int, total_folds: int) -> None:
        msg = f"[stx] fold {fold_id + 1}/{total_folds}"
        if self._console is not None:
            self._console.print(f"[bold cyan]{msg}[/]")
        else:
            click.echo(msg, err=True)

    def on_epoch_end(
        self,
        fold_id: int,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        tl = float(metrics.get("train_loss", float("nan")))
        vl = float(metrics.get("val_loss", float("nan")))
        line = f"[stx] fold {fold_id} epoch {epoch + 1}/{total_epochs} train_loss={tl:.6f} val_loss={vl:.6f}"
        if self._console is not None:
            self._console.print(line)
        else:
            click.echo(line, err=True)

    def on_fold_end(self, fold_id: int, summary: dict[str, Any]) -> None:
        line = f"[stx] fold {fold_id} training complete"
        if self._console is not None:
            self._console.print(f"[green]{line}[/]")
        else:
            click.echo(line, err=True)


def _backtest_progress(ctx: click.Context, *, dry_run: bool) -> ProgressCallback | None:
    """``None`` when quiet, dry-run, or progress disabled."""
    root = ctx.find_root()
    root.ensure_object(dict)
    if dry_run or root.obj.get("quiet"):
        return None
    return StxCliProgress(use_rich=bool(root.obj.get("use_rich")))


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option("-v", "--verbose", count=True, help="More log detail (-v INFO, -vv DEBUG).")
@click.option("-q", "--quiet", is_flag=True, help="Only warnings and errors.")
@click.option(
    "--log-file", type=click.Path(path_type=Path, dir_okay=False), default=None, help="Also append logs to this file."
)
@click.option("--no-color", is_flag=True, help="Disable styled output (or set NO_COLOR).")
@click.option(
    "--rich",
    "use_rich",
    is_flag=True,
    help="Use Rich for fold/epoch lines when the 'rich' package is installed.",
)
@click.version_option(version=_version_string(), prog_name="stx")
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
    _install_sigint()


@cli.command("backtest")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=_default_config_path,
    show_default="configs/default.yaml or $STX_CONFIG",
    help="Experiment YAML.",
)
@click.option("--synthetic", is_flag=True, help="Use synthetic candles (no API).")
@click.option(
    "--device",
    default=None,
    metavar="NAME",
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
        cfg = prepare_backtest_config(config_path, device=device, seed=seed)
        mode = str(cfg.get("experiment_mode") or "single_symbol").lower()
        progress = _backtest_progress(ctx, dry_run=dry_run)
        if mode == "universe":
            summary = run_universe_experiment(cfg, use_synthetic=synthetic, dry_run=dry_run, progress=progress)
        else:
            summary = run_experiment(cfg, use_synthetic=synthetic, dry_run=dry_run, progress=progress)
    except ValidationError as e:
        click.echo(
            _style_text(format_validation_error(e, path_hint=str(config_path)), fg="red", ctx=ctx),
            err=True,
        )
        _exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        _exit(130)
    except ValueError as e:
        click.echo(_style_text(str(e), fg="red", ctx=ctx), err=True)
        _exit(1)

    _emit_backtest_result(summary, output_format=output_format.lower(), ctx=ctx)


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
        fetch_universe_sample_data(cache_dir, syms, refresh=refresh)
    except KeyboardInterrupt:
        logger.warning("Fetch interrupted")
        _exit(130)
    except Exception as e:
        click.echo(_style_text(str(e), fg="red", ctx=ctx), err=True)
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
        with open(config_path, encoding="utf-8") as f:
            base = yaml.safe_load(f)
        merged = run_loss_sweep(base, config_path=config_path, use_synthetic=synthetic)
        fmt = output_format.lower()
        if fmt == "json":
            click.echo(json.dumps(merged, indent=2, default=str))
        else:
            click.echo(_format_sweep_table(merged))
    except ValidationError as e:
        click.echo(
            _style_text(format_validation_error(e, path_hint=str(config_path)), fg="red", ctx=ctx),
            err=True,
        )
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
    default=_default_config_path,
    show_default="configs/default.yaml or $STX_CONFIG",
    help="Experiment YAML.",
)
@click.pass_context
def config_show_cmd(ctx: click.Context, config_path: Path) -> None:
    """Print merged, validated config as YAML."""
    try:
        cfg = prepare_backtest_config(config_path)
        click.echo(yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False))
    except ValidationError as e:
        click.echo(
            _style_text(format_validation_error(e, path_hint=str(config_path)), fg="red", ctx=ctx),
            err=True,
        )
        _exit(1)


@config_group.command("diff")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=_default_config_path,
    show_default="configs/default.yaml or $STX_CONFIG",
    help="Experiment YAML.",
)
@click.pass_context
def config_diff_cmd(ctx: click.Context, config_path: Path) -> None:
    """Print keys that differ from Pydantic defaults for this experiment mode."""
    try:
        cfg = prepare_backtest_config(config_path)
        diff = _config_diff_from_merged(cfg)
        if not diff:
            click.echo("(no differences from model defaults)")
            return
        click.echo(yaml.safe_dump(diff, sort_keys=True, default_flow_style=False))
    except ValidationError as e:
        click.echo(
            _style_text(format_validation_error(e, path_hint=str(config_path)), fg="red", ctx=ctx),
            err=True,
        )
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
        raw = load_config(config_path)
        if raw is None or not isinstance(raw, dict):
            raise ValueError("Config must be a non-empty YAML mapping")
        apply_stx_env_overrides(raw)
        coerce_experiment_config(raw)
    except ValidationError as e:
        click.echo(
            _style_text(format_validation_error(e, path_hint=str(config_path)), fg="red", ctx=ctx),
            err=True,
        )
        _exit(1)
    except ValueError as e:
        click.echo(_style_text(str(e), fg="red", ctx=ctx), err=True)
        _exit(1)
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
