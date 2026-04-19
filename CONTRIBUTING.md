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

## Architecture (high level)

```text
stx (cli.py)  →  run_from_config_path / validate / fetch / sweep
                      ↓
              run_experiment_dispatch
                      ↓
         single-symbol runner          universe_runner
                      ↓                       ↓
            data + features + model + walkforward + training
                      ↓
              artifacts (summary.json, CSV, logs)
```

- **CLI** (`src/stock_transformer/cli.py`) parses arguments, configures logging, and catches validation errors with readable messages. It should not embed training logic.
- **Runners** (`backtest/runner.py`, `backtest/universe_runner.py`) orchestrate data, folds, training, and metrics; they call shared helpers (`backtest/artifacts.py`, `backtest/context.py`, `device.py`).
- **Models** (`model/`) implement `nn.Module` code only; device resolution lives in `device.py`, batched inference in `batch_predict` helpers.

## Adding a model or loss

1. Implement the module under `src/stock_transformer/model/`.
2. Wire training in `backtest/training.py` (reuse `_run_supervised_epochs` when the loop matches).
3. Extend Pydantic config in `config_models.py` if new hyperparameters are required.
4. Add synthetic or small-config tests under `tests/` so CI stays fast and offline.

## Testing policy

- Prefer **synthetic** data in tests (no API keys, deterministic).
- Golden CSV / JSON fixtures live under `tests/golden/`; update them deliberately when outputs change.
- CLI behavior is covered with subprocess and `click.testing.CliRunner` where appropriate.

## Style

- **Ruff** for lint + format; **mypy** on `src/stock_transformer`.
- Prefer `logging` in library code under `src/`; the CLI prints user-facing summaries.
- Avoid broad `except` without comment; fold-level isolation uses `BLE001` with an explicit rationale.

## Pull requests

- Describe behavior change and risk (especially walk-forward or data semantics).
- Note if configs or artifacts change shape.
- Ensure CI passes (Python 3.11 and 3.12 in GitHub Actions).
