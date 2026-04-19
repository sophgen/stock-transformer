# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Click-based CLI `stx` with subcommands: `backtest`, `fetch`, `sweep`, `validate`, `version`.
- Global flags `-v` / `-vv`, `-q`, and `--version`; SIGINT handling without noisy tracebacks.
- `stock_transformer.device.resolve_device` module; `batch_predict` on classifier and ranker models.
- Shared `RunContext`, artifact helpers, `STX_*` env overrides, and config typo warnings for unknown YAML keys.
- `inference_batch_size` in validated configs; `py.typed` marker for type consumers.
- `CONTRIBUTING.md`, shell completion notes under `completions/`, and optional GitHub Actions release workflow.

### Changed

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
