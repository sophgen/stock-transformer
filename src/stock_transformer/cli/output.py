"""Stdout/stderr presentation for CLI commands (no training or I/O to disk artifacts).

Keeping formatting here lets tests assert on pure functions and keeps Click handlers thin.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from stock_transformer import __version__


def version_string() -> str:
    """Human-readable build string for ``--version`` (torch and resolved auto device)."""
    import torch

    from stock_transformer.device import resolve_device

    dev = resolve_device("auto")
    return f"stock-transformer {__version__} (torch {torch.__version__}, auto→{dev})"


def use_color(ctx: click.Context | None) -> bool:
    """Respect NO_COLOR and CLI ``--no-color`` so logs stay machine-friendly in CI."""
    if os.environ.get("NO_COLOR"):
        return False
    if ctx and ctx.obj and ctx.obj.get("no_color"):
        return False
    return True


def style_text(text: str, *, fg: str, ctx: click.Context | None) -> str:
    """Apply ANSI color only when the user did not disable styling (TTY-friendly defaults)."""
    if not use_color(ctx):
        return text
    return click.style(text, fg=fg)


def summary_exit_code(summary: dict[str, Any]) -> int:
    """Map runner summary dicts to process exit codes (see README exit-code table)."""
    if summary.get("fold_errors"):
        return 2
    err = summary.get("error")
    if err in ("partial_failure", "no_folds"):
        return 2
    return 0


def emit_backtest_result(
    summary: dict[str, Any],
    *,
    output_format: str,
    ctx: click.Context | None,
) -> None:
    """Print JSON or text summary and terminate with the code implied by ``summary``."""
    code = summary_exit_code(summary)
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
            click.echo(style_text(msg, fg="green", ctx=ctx))
        else:
            err = summary.get("error", "run_failed")
            msg = f"Run finished with issues ({err}). Artifacts: {summary.get('run_dir')}"
            click.echo(style_text(msg, fg="yellow", ctx=ctx))
    raise SystemExit(code)


def fmt_metric_cell(x: Any) -> str:
    """Render a numeric metric or an em dash placeholder for sweep tables."""
    return "—" if x is None else f"{float(x):.3f}"


def _sweep_metrics_row(agg: dict[str, Any] | None) -> tuple[str, str, str]:
    """Format one sweep table row (internal helper for :func:`format_sweep_table`)."""
    if not agg:
        return ("—", "—", "—")
    sp = agg.get("spearman_mean_mean")
    nd = agg.get("ndcg3_mean_mean")
    hit = agg.get("top2_hit_mean")
    return (fmt_metric_cell(sp), fmt_metric_cell(nd), fmt_metric_cell(hit))


def format_sweep_table(merged: dict[str, Any]) -> str:
    """Fixed-column text table for loss sweep (readable in terminals without Rich)."""
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


def pydantic_default_for_field(field_info: Any) -> Any:
    """Resolve a field's default for ``config diff`` (factory vs static default)."""
    if field_info.default_factory is not None:
        return field_info.default_factory()
    if field_info.default is not PydanticUndefined:
        return field_info.default
    return PydanticUndefined


def config_diff_from_merged(
    cfg: dict[str, Any], *, single_cls: type[BaseModel], universe_cls: type[BaseModel]
) -> dict[str, Any]:
    """Highlight non-default keys for the active experiment mode (``config diff`` UX)."""
    mode = str(cfg.get("experiment_mode") or "single_symbol").lower()
    cls = universe_cls if mode == "universe" else single_cls
    diff: dict[str, Any] = {}
    for name, finfo in cls.model_fields.items():
        dv = pydantic_default_for_field(finfo)
        if dv is PydanticUndefined:
            continue
        if name not in cfg:
            continue
        if cfg[name] != dv:
            diff[name] = {"value": cfg[name], "default": dv}
    return diff


def default_config_path() -> Path:
    """Default ``-c`` when omitted: repo convention or ``STX_CONFIG`` override."""
    return Path(os.environ.get("STX_CONFIG", "configs/default.yaml"))
