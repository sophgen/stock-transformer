# stock-transformer

Multi-timeframe autoregressive candle transformer **and** a **multi-ticker universe** ranker with walk-forward backtesting. The primary interface is the **`stx`** CLI (Click).

Requires **Python 3.11+**.

## Contents

- [What it does](#what-it-does)
- [Installation](#installation)
- [Quickstart](#quickstart-synthetic-no-api-key)
- [Live data](#live-data-alpha-vantage)
- [CLI reference](#cli-reference-stx)
- [Configuration precedence](#configuration-precedence)
- [Environment variables](#environment-variables)
- [Examples](#example-commands)
- [Shell completion](#shell-completion-bash-zsh-fish)
- [Man pages](#man-pages)
- [Docker](#docker)
- [Troubleshooting](#troubleshooting)
- [Configuration files](#configuration-files)
- [Artifacts](#artifacts)
- [Tests and quality](#tests-and-quality)
- [CLI package layout](#cli-package-layout)
- [Project structure](#project-structure)
- [Contributing](#contributing)

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
| `--log-file` | Append logs to a file (in addition to stderr). |
| `--no-color` | Disable styled output (or set `NO_COLOR`). |
| `--rich` | Use Rich for fold/epoch progress lines on stderr when the `rich` package is installed (dev env includes it). |

During training, **`stx backtest`** prints per-fold and per-epoch lines on stderr (suppressed by `-q`). Hooks live in `backtest/progress.py` and are wired from the CLI.

Subcommands (by theme: **experiments** — `backtest`, `sweep`; **configuration** — `config`; **data** — `fetch` or `data fetch`; **meta** — `validate`, `version`, `completion`):

| Command | Purpose |
|---------|---------|
| `stx backtest` | Run walk-forward experiment from YAML (single-symbol or universe). |
| `stx config show` | Print merged, validated config as YAML (`-c` path). |
| `stx config diff` | Print keys that differ from Pydantic defaults for that mode. |
| `stx fetch` | Download daily-adjusted OHLCV for the default pilot universe into `cache_dir`. |
| `stx data fetch` | Same as `stx fetch` (grouped under `data` for discoverability). |
| `stx sweep` | Run universe experiment for each ranking loss and merge `by_loss` (see `backtest/loss_sweep.py`). |
| `stx validate` | Load and validate YAML only (no training). Useful in CI. |
| `stx version` | Same information as `stx --version`. |
| `stx completion` | Print a shell tab-completion script (`bash`, `zsh`, or `fish`). |

### `stx backtest`

| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--config` | `configs/default.yaml` or `$STX_CONFIG` | Experiment YAML path. |
| `--synthetic` | off | Use built-in synthetic data (no API). |
| `--device` | (from env / YAML) | PyTorch device override: `auto`, `cpu`, `mps`, `cuda`, `cuda:N`. |
| `--seed` | (from env / YAML) | Override `seed` for a quick reproducibility check. |
| `-o`, `--output-format` | `text` | `text` — one-line summary (and in `--dry-run`, sample/fold counts plus YAML fold boundaries on stdout); `json` — full summary dict on stdout. |
| `--dry-run` | off | Resolve data, write `folds.json` / `summary.json`, print fold plan to stdout in text mode, exit without training. |

**Legacy:** `stx-backtest` is an alias for `stx backtest` (same flags).

### `stx config`

| Subcommand | Purpose |
|------------|---------|
| `stx config show` | Merged effective config (after env + validation) as YAML. |
| `stx config diff` | Keys whose values differ from Pydantic defaults for that mode. |

Both take `-c/--config` (same default as `stx backtest`).

### `stx fetch` / `stx data fetch`

The same command is available as **`stx fetch`** or **`stx data fetch`** (identical flags).

| Option | Default | Description |
|--------|---------|-------------|
| `--cache-dir` | `data` | Root for `raw/` and `canonical/`. |
| `--symbols` | MSTR IBIT COIN QQQ | Repeatable symbol list (strip + uppercased). Empty tokens are rejected. |
| `--refresh` | off | Force re-download and overwrite canonical CSV. |

### `stx completion`

| Argument | Description |
|----------|-------------|
| `SHELL` | One of `bash`, `zsh`, `fish`. Prints a script to stdout; source it from your shell profile (or save to e.g. `/etc/bash_completion.d/stx`). |

### `stx sweep`

| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--config` | `configs/universe.yaml` | Universe YAML. |
| `--synthetic` | off | Synthetic universe data. |
| `-o`, `--output-format` | `text` | `text` — comparison table; `json` — merged sweep object. |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success. |
| 1 | Invalid or missing config, validation error, or runtime error. |
| 2 | Partial failure (e.g. fold errors) or no folds — inspect `summary.json`. |
| 130 | Interrupted (Ctrl+C) after best-effort artifact write. |
| 143 | SIGTERM (containers / process managers) — graceful exit without a traceback. |

## Configuration precedence

For keys supported by both file and environment:

1. **CLI flags** (e.g. `stx backtest --device cpu --seed 99`)
2. **Environment variables** — `STX_DEVICE`, `STX_CACHE_DIR`, `STX_ARTIFACTS_DIR`, `STX_EPOCHS`, `STX_SEED`, `STX_BATCH_SIZE`, `STX_LOG_LEVEL` (used when `-v`/`-q` are not set), and `STX_CONFIG` (default path for `-c` when you omit it)
3. **YAML** values
4. **Pydantic defaults** in `config_models.py`

Unknown YAML keys log a **warning** (possible typo) but are ignored after validation of known fields.

## Environment variables

| Variable | Role |
|----------|------|
| `STX_CONFIG` | Default path for `-c/--config` when omitted (otherwise `configs/default.yaml`). |
| `STX_DEVICE` | Device string merged into YAML before validation (`cpu`, `mps`, `cuda`, …). Overridden by `stx backtest --device`. |
| `STX_CACHE_DIR` | Cache root for data (merged into config). |
| `STX_ARTIFACTS_DIR` | Run output directory (merged into config). |
| `STX_EPOCHS` | Integer epochs override. |
| `STX_SEED` | Integer seed override (overridden by `--seed`). |
| `STX_BATCH_SIZE` | Integer batch size override. |
| `STX_LOG_LEVEL` | When you neither pass `-v`/`-vv` nor `-q`, sets root log level (`DEBUG`, `INFO`, …). |
| `NO_COLOR` | Disable ANSI colors (same idea as `--no-color`). |
| `ALPHAVANTAGE_API_KEY` | Required for live `fetch` / non-synthetic runs that hit the API. |

## Example commands

One minimal example per command (combine flags as needed):

```bash
# Root help (same as: stx -h)
uv run stx --help

# Walk-forward backtest (synthetic; no API key)
uv run stx backtest --synthetic
uv run stx backtest --synthetic -c configs/universe.yaml -o json

# Effective config after merge + validation (stdout is YAML)
uv run stx config show -c configs/default.yaml

# Keys that differ from Pydantic defaults for this experiment mode
uv run stx config diff -c configs/universe.yaml

# Download daily OHLCV into ./data (requires ALPHAVANTAGE_API_KEY unless you only inspect help)
uv run stx fetch --cache-dir data --symbols MSTR --symbols QQQ
uv run stx data fetch --cache-dir data --symbols MSTR

# Bash completion script (install to ~/.bashrc or similar)
uv run stx completion bash | head -3

# Compare ranking losses on a universe config
uv run stx sweep --synthetic -c configs/universe.yaml

# Fast CI check: validate YAML only
uv run stx validate -c configs/universe.yaml

# Version string (also: stx --version)
uv run stx version

# Verbose single-symbol run (INFO logs on stderr)
uv run stx -v backtest --synthetic -c configs/default.yaml
```

## Shell completion (bash, zsh, fish)

**Preferred:** use the built-in generator (prints the same Click completion script as the manual env-var flow):

```bash
stx completion bash > ~/.stx-complete.bash
echo 'source ~/.stx-complete.bash' >> ~/.bashrc
```

Alternatively, set the completion env var and run `stx` (requires `stx` on your `PATH` or `uv run stx`):

```bash
_STX_COMPLETE=bash_source stx > ~/.stx-complete.bash
echo 'source ~/.stx-complete.bash' >> ~/.bashrc
```

Committed stubs for reference: `completions/stx.bash`, `completions/stx.zsh`, `completions/stx.fish` (regenerate with `stx completion <shell>` or the `_STX_COMPLETE=…` pattern).

## Man pages

`make man` generates `man/*.1` (root `stx.1` plus per-command pages such as `stx-backtest.1`, `stx-fetch.1`, `stx-data.1`) via **click-man** (dev dependency). CI runs **`make man-check`** so the committed `man/` tree matches a fresh click-man run.

```bash
make man
make man-check
```

## Docker

Build a self-contained image (PyTorch and dependencies are installed by `pip`; image is large):

```bash
docker build -t stock-transformer .
docker run --rm stock-transformer --help
```

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

CI runs Ruff, **mypy** (with `disallow_untyped_defs`), pytest with **≥80%** coverage, **`make man-check`**, on Python **3.11** and **3.12** (Ubuntu), plus a **macOS** smoke job for the CLI. **Dependabot** is configured for GitHub Actions and pip (`.github/dependabot.yml`). A release workflow (`.github/workflows/release.yml`) builds and can publish on `v*` tags when `PYPI_API_TOKEN` is configured.

## CLI package layout

The `stx` command is implemented under `src/stock_transformer/cli/` so parsing, formatting, and orchestration stay separate from training code:

| Module | Responsibility |
|--------|----------------|
| `cli/app.py` | Root Click group, global options, `register_all_commands`, `main` / `main_backtest_compat`. |
| `cli/commands/` | One module per subcommand (`backtest`, `fetch` + `data` group, `sweep`, `config`, `validate`, `version`, `completion`) — options and thin handlers only. |
| `cli/services.py` | Calls runners via the `stock_transformer.cli` package so tests can patch `run_experiment` and friends. |
| `cli/output.py` | Text/JSON summaries, sweep tables, version string. |
| `cli/logging_config.py` | Root logging for `-v` / `-q` / `--log-file`. |
| `cli/progress_display.py` | Optional Rich stderr lines (`ProgressCallback`). |
| `cli/validators.py` | Click callbacks (e.g. non-empty `--device`). |
| `cli/sigint.py` | SIGINT / SIGTERM handlers (exit 130 / 143 without tracebacks). |

Core walk-forward logic remains in `backtest/` and must not import the CLI.

## Project structure

```text
src/stock_transformer/
├── cli/                   # stx: app.py, services, output, logging, progress
├── device.py              # resolve_device (no torch.nn imports)
├── config_models.py       # Pydantic
├── config_validate.py
├── data/                  # Alpha Vantage, alignment, cache, fetch helpers
├── features/              # Sequences, universe tensor
├── model/                 # Transformers, baselines, losses
└── backtest/              # Walk-forward, metrics, training, ProgressCallback, runners, artifacts
```

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup, architecture notes, and conventions.
