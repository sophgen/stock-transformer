# stock-transformer

Multi-timeframe autoregressive candle transformer **and** a **multi-ticker universe** ranker with walk-forward backtesting. The primary interface is the **`stx`** CLI (Click).

Requires **Python 3.11+**.

## What it does

- **Single-symbol mode (default):** Each OHLCV candle from multiple timeframes is a token; a causal Transformer predicts the next candle (regression + direction).
- **Universe mode (`experiment_mode: universe`):** A ranker scores every symbol on a shared calendar with strict point-in-time features and cross-sectional targets.

See **`plan.md`** for milestones and **`CHANGELOG.md`** for release notes.

## Installation

**Recommended** (matches CI and `uv.lock`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
```

The `stx` command is installed into the project environment. Use `uv run stx …` or activate the venv created by `uv sync`.

**pip / editable install:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies include **PyTorch**, **pandas**, **PyArrow**, **Pydantic**, and **Click**.

## Quickstart (synthetic, no API key)

```bash
uv run stx backtest --synthetic
uv run stx backtest --synthetic -c configs/universe.yaml
```

## Live data (Alpha Vantage)

Put `ALPHAVANTAGE_API_KEY` in a **`.env`** file at the repo root (see `.env.example`) or export it. The CLI loads it via `python-dotenv`.

```bash
uv run stx fetch
uv run stx backtest -c configs/sample.yaml
```

## CLI reference (`stx`)

Global options (before the subcommand):

| Option | Meaning |
|--------|---------|
| `-h`, `--help` | Help for `stx` or the subcommand. |
| `--version` | Version, PyTorch build, and resolved `auto` device. |
| `-v` | INFO logging. |
| `-vv` | DEBUG logging. |
| `-q`, `--quiet` | Warnings and errors only. |

Subcommands:

| Command | Purpose |
|---------|---------|
| `stx backtest` | Run walk-forward experiment from YAML (single-symbol or universe). |
| `stx fetch` | Download daily-adjusted OHLCV for the default pilot universe into `cache_dir`. |
| `stx sweep` | Run universe experiment for each ranking loss and merge `by_loss` (see `backtest/loss_sweep.py`). |
| `stx validate` | Load and validate YAML only (no training). Useful in CI. |
| `stx version` | Same information as `stx --version`. |

### `stx backtest`

| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--config` | `configs/default.yaml` | Experiment YAML path. |
| `--synthetic` | off | Use built-in synthetic data (no API). |
| `--device` | (from env / YAML) | PyTorch device override: `auto`, `cpu`, `mps`, `cuda`, `cuda:N`. |

**Legacy:** `stx-backtest` is an alias for `stx backtest` (same flags).

### `stx fetch`

| Option | Default | Description |
|--------|---------|-------------|
| `--cache-dir` | `data` | Root for `raw/` and `canonical/`. |
| `--symbols` | MSTR IBIT COIN QQQ | Repeatable symbol list. |
| `--refresh` | off | Force re-download and overwrite canonical CSV. |

### `stx sweep`

| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--config` | `configs/universe.yaml` | Universe YAML. |
| `--synthetic` | off | Synthetic universe data. |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success. |
| 1 | Invalid or missing config, validation error, or runtime error. |
| 2 | Partial failure (e.g. fold errors) or no folds — inspect `summary.json`. |
| 130 | Interrupted (Ctrl+C) after best-effort artifact write. |

## Configuration precedence

For keys supported by both file and environment:

1. **CLI flags** (e.g. `stx backtest --device cpu`)
2. **Environment variables** `STX_DEVICE`, `STX_CACHE_DIR`, `STX_ARTIFACTS_DIR`, `STX_EPOCHS`
3. **YAML** values
4. **Pydantic defaults** in `config_models.py`

Unknown YAML keys log a **warning** (possible typo) but are ignored after validation of known fields.

## Example commands

```bash
# Validate config before a long run
uv run stx validate -c configs/universe.yaml

# Verbose single-symbol run
uv run stx -v backtest --synthetic -c configs/default.yaml

# Fetch only pilot symbols, custom cache
uv run stx fetch --cache-dir data --symbols MSTR --symbols QQQ

# Loss sweep (universe config)
uv run stx sweep --synthetic -c configs/universe.yaml
```

## Shell completion (bash)

Generate completion script (requires Click):

```bash
_STX_COMPLETE=bash_source stx > ~/.stx-complete.bash
echo 'source ~/.stx-complete.bash' >> ~/.bashrc
```

A short note lives in `completions/stx.bash`.

## Troubleshooting

- **Missing API key:** `stx fetch` / live `backtest` need `ALPHAVANTAGE_API_KEY` in the environment or `.env`.
- **MPS not available:** If YAML sets `device: mps` on non-Apple hardware, use `stx backtest --device cpu` or set `STX_DEVICE=cpu`.
- **CUDA / MPS errors:** Use `--device cpu` for reproducible CPU-only runs.
- **Config validation:** Run `stx validate -c your.yaml` for fast feedback; Pydantic errors are printed as bullet lists.

## Configuration files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Single-symbol multi-timeframe |
| `configs/universe.yaml` | Universe ranker |
| `configs/sample.yaml` | Smaller universe smoke run |

Universe highlights: `symbols`, `label_mode`, `store`, `data_source`, `loss`, optional `portfolio_sim`, `inference_batch_size` for batched inference.

## Artifacts

Runs write under `artifacts_dir` (default `artifacts/`): `config_snapshot.yaml`, `summary.json`, `training_log_fold_*.csv`, prediction CSVs, and universe extras (`feature_schema.json`, `folds.json`, `universe_membership.json`). On fold failures, see `fold_errors.log`.

## Tests and quality

```bash
uv run pytest -q
uv run pytest -q --cov=stock_transformer --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src/stock_transformer
```

CI runs Ruff, mypy, and pytest on Python **3.11** and **3.12**. A release workflow (`.github/workflows/release.yml`) builds and can publish on `v*` tags when `PYPI_API_TOKEN` is configured.

## Project structure

```text
src/stock_transformer/
├── cli.py                 # stx entrypoint (Click)
├── device.py              # resolve_device (no torch.nn imports)
├── config_models.py       # Pydantic
├── config_validate.py
├── data/                  # Alpha Vantage, alignment, cache, fetch helpers
├── features/              # Sequences, universe tensor
├── model/                 # Transformers, baselines, losses
└── backtest/              # Walk-forward, metrics, training, runners, artifacts
```

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup, architecture notes, and conventions.
