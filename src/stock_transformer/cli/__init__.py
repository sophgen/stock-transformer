"""``stx`` command-line interface (Click).

The package splits parsing (:mod:`stock_transformer.cli.app` and
:mod:`stock_transformer.cli.commands`), presentation (:mod:`stock_transformer.cli.output`),
and runner orchestration (:mod:`stock_transformer.cli.services`) so training code stays
free of CLI imports. Subcommands include ``backtest``, ``config``, ``fetch`` (also under
``data fetch``), ``sweep``, ``validate``, ``version``, and ``completion``.

Symbols re-exported from :mod:`stock_transformer.backtest.runner` (and related modules) exist so
tests and scripts can ``patch("stock_transformer.cli.run_experiment", ...)`` without reaching into
implementation modules.
"""

from __future__ import annotations

from stock_transformer.backtest.env_config import apply_stx_env_overrides
from stock_transformer.backtest.loss_sweep import run_loss_sweep
from stock_transformer.backtest.runner import load_config, prepare_backtest_config, run_experiment
from stock_transformer.backtest.universe_runner import run_universe_experiment
from stock_transformer.cli.app import cli, main, main_backtest_compat
from stock_transformer.cli.logging_config import setup_logging
from stock_transformer.cli.progress_display import StxCliProgress
from stock_transformer.cli.types import StxResult
from stock_transformer.config_models import coerce_experiment_config

__all__ = [
    "StxCliProgress",
    "StxResult",
    "apply_stx_env_overrides",
    "cli",
    "coerce_experiment_config",
    "load_config",
    "main",
    "main_backtest_compat",
    "prepare_backtest_config",
    "run_experiment",
    "run_loss_sweep",
    "run_universe_experiment",
    "setup_logging",
]
