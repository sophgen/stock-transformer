# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CLI: `stock_transformer.cli.commands` package — one module per subcommand (`backtest`, `fetch`, `sweep`, `config`, `validate`, `version`) with shared `cli_exit` helper; root `app.py` only defines the group, globals, and `register_all_commands`.
- CLI: SIGTERM handler exits with **143** (containers / process managers); ISO-like timestamps in default stderr log format.
- CLI: `-o` as a short form of `--output-format` on `stx backtest` and `stx sweep`; `short_help` on root and `stx config` groups; `--cache-dir` validated as a non-empty path before fetch runs.
- PyPI-oriented `keywords` and `classifiers` in `pyproject.toml`.
- Docstrings clarifying config merge order in `prepare_backtest_config`, `run_from_config_path`, and `env_config`.
- `stx fetch --symbols`: validate non-empty tokens and normalize to uppercase (Click passes the full tuple to `multiple=True` callbacks); `stx config -h` matches root `-h`/`--help`.
- Dev install (`pip install -e ".[dev]"` / `uv sync --extra dev`) now includes **Rich** and **click-man** in the same extra (aligned with CI `make man-check`).
- Expanded “why” docstrings on CLI modules, `prepare_backtest_config`, and README per-command examples.
- CLI package layout: `src/stock_transformer/cli/` (`app`, `services`, `output`, `logging_config`, `progress_display`, `validators`, `sigint`); `python -m stock_transformer.cli` entry via `cli/__main__.py`; `--device` empty-string validation.
- Documentation: README table of contents, environment-variable table, CLI package layout; CONTRIBUTING layering and testing notes.
- CLI: `stx config show` / `stx config diff`; `backtest --output-format`, `--dry-run`, `--seed`; global `--log-file` / `--no-color` / `--rich`; expanded `STX_*` env vars (`STX_SEED`, `STX_BATCH_SIZE`, `STX_LOG_LEVEL`, `STX_CONFIG`).
- `ProgressCallback` + per-fold/per-epoch stderr lines; `StxResult` helper; `prepare_backtest_config` and explicit universe vs single-symbol dispatch; `run_universe_from_config_path` / `run_single_symbol_from_config_path`; dry-run fold plans for both modes.
- Sweep text table output; subprocess helpers in tests for exit codes 2 and 130; `CliRunner(catch_exceptions=False)` fixture; fast CliRunner universe test.
- `man/stx.1` via `click-man`; `Makefile` targets `man` / `man-check`; `Dockerfile`; Dependabot; macOS CI smoke; **mypy** `disallow_untyped_defs`.
- Click-based CLI `stx` with subcommands: `backtest`, `fetch`, `sweep`, `validate`, `version`.
- Global flags `-v` / `-vv`, `-q`, and `--version`; SIGINT handling without noisy tracebacks.
- `stock_transformer.device.resolve_device` module; `batch_predict` on classifier and ranker models.
- Shared `RunContext`, artifact helpers, `STX_*` env overrides, and config typo warnings for unknown YAML keys.
- `inference_batch_size` in validated configs; `py.typed` marker for type consumers.
- `CONTRIBUTING.md`, shell completion notes under `completions/`, and optional GitHub Actions release workflow.

### Changed

- Monolithic `cli.py` replaced by the `stock_transformer.cli` package (same entry points: `stx`, `stx-backtest`).
- Entry point `stx` (legacy `stx-backtest` forwards to `stx backtest`).
- Training loops share `_run_supervised_epochs`; structured logging in runners.

### Deprecated

- `CandleTransformerClassifier` emits `DeprecationWarning` (use `CandleTransformer`).

## [0.1.0] - 2026-04-19

### Added

- Single-symbol multi-timeframe candle transformer with walk-forward evaluation.
- Universe / cross-sectional ranker mode with Spearman, Kendall, NDCG, baselines, and optional portfolio simulation.
- Alpha Vantage ingestion with caching; synthetic data for tests and offline runs.
- Pydantic-validated YAML configs; artifact layout (`summary.json`, per-fold logs, predictions CSV).
