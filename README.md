# stock-transformer

Multi-timeframe autoregressive candle transformer **and** a **multi-ticker universe** ranker with walk-forward backtesting.

Requires **Python 3.11+**.

## Concept

### Single-symbol mode (default)

Each OHLCV candle — from any timeframe (minute, hour, day, week, month) — is
treated as a **token**. All past candles up to a prediction point are fed into
a causal Transformer encoder. The model predicts the **next candle** — both its
OHLCV log-returns (regression) and its direction (classification: up / down).

### Universe mode (`experiment_mode: universe`)

Several tickers share a **global timestamp grid** per timeframe. At each time
`t`, the model sees a lookback tensor `[lookback, num_symbols, features]` with
**masks** for missing rows, and is trained to score or rank symbols using
**cross-sectional targets** (e.g. forward return minus the peer median at `t`).
This matches the design in `plan.md` (pilot: e.g. `MSTR` with predictors such as
`IBIT`, `COIN`, `QQQ`).

The backtest enforces **strict point-in-time discipline**: features use only data
≤ `t`; labels use returns from `t` to `t+1`. Walk-forward folds use the same
global time cuts for every symbol.

## Setup

**Recommended** (matches CI and `uv.lock`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or install uv your way
uv sync --extra dev
```

Run the CLI with `uv run stx-backtest …` or activate the project venv that `uv sync` creates.

**Alternative** (pip):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies include **PyTorch**, **pandas**, **PyArrow**, and **Pydantic** (configs are validated and defaulted before each run).

On **Apple Silicon** the default config uses `device: "auto"` which picks MPS
when available. Override to `"cpu"` or `"cuda"` if needed.

## Quick start — synthetic data (no API key required)

Single-symbol pipeline:

```bash
uv run stx-backtest --synthetic
# or: stx-backtest --synthetic
```

Universe / cross-sectional pipeline:

```bash
uv run stx-backtest --synthetic -c configs/universe.yaml
```

## Live data via Alpha Vantage

```bash
export ALPHAVANTAGE_API_KEY=your_key_here
uv run stx-backtest -c configs/default.yaml
uv run stx-backtest -c configs/universe.yaml
```

The client fetches candles per timeframe, respects throttling, caches raw JSON,
and writes canonical data under `cache_dir` (CSV and optional **partitioned**
`csv` / `parquet` store when `store` is set in universe config). Universe mode
fetches every symbol in the YAML list.

## CLI

- `-c` / `--config` — experiment YAML (default: `configs/default.yaml`).
- `--synthetic` — skip the API and use built-in random-walk data.

Exit codes: **0** success, **1** missing/unreadable config, **2** partial failure
(e.g. a walk-forward fold raised) or no folds — see `summary.json` and optional
`fold_errors.log` under the run directory.

## Configuration

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Single-symbol multi-timeframe next-candle prediction |
| `configs/universe.yaml` | Multi-ticker universe, cross-sectional labels, ranker |

YAML is validated with **Pydantic**; omitted keys get defaults aligned with these
files (so small config snippets in tests still work).

**Universe highlights:**

- `symbols`, `target_symbol` (reporting key), `timeframe`, `lookback`, `min_coverage_symbols`
- `label_mode`: `cross_sectional_return`, `raw_return`, `equal_weighted_return`, `sector_neutral_return` (sector map required for the last)
- `store`: `csv` or `parquet`; `data_source`: `rest` or `mcp`
- `loss`: `mse`, `listnet`, `approx_ndcg`
- Optional **`portfolio_sim`** block (top‑k book on test folds; see `configs/universe.yaml`)
- Optional training controls (defaults preserve prior behavior): `early_stopping_patience` (0 = off), `lr_reduce_on_plateau`, `lr_scheduler_patience`, `lr_scheduler_factor`, `lr_scheduler_min_lr`

## Architecture

**Single-symbol:** multi-timeframe token sequence → causal Transformer →
regression + direction heads.

**Universe:** per-symbol **temporal** Transformer over lookback → **cross-sectional**
Transformer over symbols → one **score per ticker**. Training uses the chosen
ranking loss on masked finite targets; evaluation includes Spearman, Kendall,
NDCG@k, top‑k hit rate, and baselines (momentum, mean reversion, equal scores,
optional linear / GBT tabular rankers).

## Artifacts

Each run writes under `artifacts_dir` (default `artifacts/`): `config_snapshot.yaml`,
`summary.json` (includes `git_sha` when available), per-fold predictions CSV, and
universe-only files such as `feature_schema.json`, `folds.json`,
`universe_membership.json`. Training emits **`training_log_fold_<id>.csv`**
(epoch / train & val loss / learning rate). On fold failures, **`fold_errors.log`**
and `summary.json` entries include a **traceback**.

## Tests and development

```bash
uv run pytest -q
uv run pytest -q --cov=stock_transformer --cov-report=term-missing   # as in CI
uv run ruff check .
uv run ruff format --check .
uv run mypy src/stock_transformer
```

CI (GitHub Actions) runs **Ruff** (lint + format check), **mypy**, and **pytest**
with coverage on Python **3.11** and **3.12**.

## Project structure

```
src/stock_transformer/
├── cli.py              # stx-backtest entrypoint
├── config_models.py    # Pydantic experiment schemas + coercion
├── config_validate.py  # Public validate_* helpers (used in tests)
├── data/               # Alpha Vantage client, alignment, universe helpers, synthetic
├── features/           # Multi-timeframe tokens; universe tensor assembly
├── labels/             # Cross-sectional return and bucket helpers
├── model/              # CandleTransformer, TransformerRanker, baselines, losses
└── backtest/           # Walk-forward splits, metrics, training loops, runners
```

See **`plan.md`** for the milestone tracker, glossary, and roadmap (e.g. further
store/label/invariant work).
