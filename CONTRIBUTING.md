# Contributing

## Development setup

The repo uses **uv** (same as CI). From the project root:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src/stock_transformer
```

Alternatively, create a venv and `pip install -e ".[dev]"`.

**Locked dependencies:** `uv.lock` pins transitive versions for reproducible installs; bump it with `uv lock` when you change `pyproject.toml`.

## Configuration precedence

For keys touched by both files and the environment, the merge order is documented in
`prepare_backtest_config` in `src/stock_transformer/backtest/runner.py`: **YAML file →
non-empty `STX_*` env vars → explicit `device` / `seed` from library or CLI**. In the
`stx backtest` command, pass `--device` / `--seed` to override env and file without
editing YAML.

## Exit codes (CLI)

| Code | When |
|------|------|
| 0 | Success |
| 1 | Bad or missing config, validation error, or other CLI/runtime error |
| 2 | Partial experiment failure (e.g. fold errors) or `no_folds` — inspect `summary.json` |
| 130 | SIGINT / Ctrl+C after best-effort cleanup |

## Architecture (high level)

```text
stx (cli package)  →  services (orchestration)  →  prepare_backtest_config → run_experiment | run_universe_experiment
        │                      │
        │                      └── fetch / sweep / validate
        └── output, logging, progress (no training imports)
                      ↓
         single-symbol runner          universe_runner
                      ↓                       ↓
            data + features + model + walkforward + training
                      ↓
              artifacts (summary.json, CSV, logs)
```

### Layering rules

- **`stock_transformer/cli/`** parses arguments, configures logging, formats stdout/stderr, and translates validation errors into user-facing text. It must not implement tensor math or walk-forward logic.
- **`stock_transformer/backtest/`** orchestrates data, folds, training, and metrics; optional `ProgressCallback` (`backtest/progress.py`) lets the CLI print fold/epoch lines without importing Click inside the training loop.
- **`stock_transformer/model/`** holds `nn.Module` implementations; device resolution lives in `device.py`.

Library callers can use `prepare_backtest_config` plus `run_experiment` / `run_universe_experiment`, or the wrappers `run_from_config_path`, `run_single_symbol_from_config_path`, and `run_universe_from_config_path` exported from `stock_transformer.backtest`.

**Why `cli/services.py` imports `stock_transformer.cli` at runtime:** integration tests patch `stock_transformer.cli.run_experiment` (and similar names). Resolving runners through the public CLI package keeps those patches aligned with user-facing behavior.

## Coding conventions

- **Ruff** for lint + format; **mypy** with `disallow_untyped_defs` on `src/stock_transformer`.
- Prefer **`logging`** in library modules; the CLI prints short summaries and tables.
- **Click:** short and long flags (`-c`/`--config`), sensible defaults, `BadParameter` for fixable input mistakes before heavy work.
- **Signals:** long commands install a SIGINT handler that raises `KeyboardInterrupt` so we exit with **130** without dumping a traceback.
- Avoid bare `except` without a comment; if you must catch broadly, state why (e.g. fold isolation).

## Adding a model or loss

1. Implement the module under `src/stock_transformer/model/`.
2. Wire training in `backtest/training.py` (reuse `_run_supervised_epochs` when the loop matches).
3. Extend Pydantic config in `config_models.py` if new hyperparameters are required.
4. Add synthetic or small-config tests under `tests/` so CI stays fast and offline.

## Testing policy

- Prefer **synthetic** data in tests (no API keys, deterministic).
- Golden CSV / JSON fixtures live under `tests/golden/`; update them deliberately when outputs change.
- **Unit tests** target pure helpers (`cli/output.py`, config validation) and runners with mocks.
- **CLI tests** use `click.testing.CliRunner` and **subprocess** invocations (`python -m stock_transformer.cli …`) so exit codes and patching behavior match production.

## Pull requests

- Describe behavior change and risk (especially walk-forward or data semantics).
- Note if configs or artifacts change shape.
- Ensure CI passes (Python 3.11 and 3.12 in GitHub Actions).
