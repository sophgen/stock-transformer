# `stx` CLI Guide for Opportunity Exploration

A task-oriented reference for driving the `stx` CLI from an automated agent
(e.g. Claude Code) to explore equity/ETF opportunities. This document focuses
on *what to run, what it produces, and how to read the output* rather than on
internals.

For a high-level overview see [`README.md`](../README.md). For release history
see [`CHANGELOG.md`](../CHANGELOG.md).

---

## Contents

1. [Mental model](#1-mental-model)
2. [Setup in one minute](#2-setup-in-one-minute)
3. [The two experiment modes](#3-the-two-experiment-modes)
4. [Command reference](#4-command-reference)
5. [Config files (YAML schema)](#5-config-files-yaml-schema)
6. [Environment variables & precedence](#6-environment-variables--precedence)
7. [Artifact layout & how to parse results](#7-artifact-layout--how-to-parse-results)
8. [Exit codes & JSON output contract](#8-exit-codes--json-output-contract)
9. [Exploration recipes](#9-exploration-recipes)
10. [Automation tips for Claude Code](#10-automation-tips-for-claude-code)
11. [Troubleshooting & known gotchas](#11-troubleshooting--known-gotchas)
12. [Glossary](#12-glossary)

---

## 1. Mental model

`stx` is **one binary with two experiment modes**:

| Mode | Question it answers | Primary metric family |
|------|--------------------|-----------------------|
| `single_symbol` | "Will the next candle of **this** ticker go up/down, and by how much?" | Classification (accuracy, F1, ROC-AUC, Brier) + regression (MAE, RMSE, directional accuracy) |
| `universe` | "Which of **these N** tickers will rank highest in next-period return?" | Cross-sectional ranking (Spearman, Kendall, NDCG@k, top-k hit), optional portfolio P&L |

Both modes are trained with **walk-forward** cross-validation — the time
series is cut into contiguous `train / val / test` folds that roll forward, so
every reported metric is strictly out-of-sample. Leakage is prevented at the
sample-index level (features at time `t`, label for return `t → t+1`).

All experiments are driven by a **YAML config** (validated by Pydantic). Every
run writes a timestamped directory under `artifacts/` that contains the config
snapshot, per-fold metrics, predictions CSV, and fold-boundary JSON.

Flow for opportunity exploration:

```
pick/edit YAML  →  stx validate   (fast check)
               →  stx config show (see effective merged config)
               →  stx backtest --dry-run  (confirm fold plan)
               →  stx backtest -o json    (actually run, parse summary)
               →  stx sweep -o json       (compare ranking losses)
               →  read artifacts/<run>/…  (inspect predictions / folds)
```

---

## 2. Setup in one minute

```bash
# Install uv once, then sync the repo's pinned environment.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev

# Optional: live data requires an Alpha Vantage key.
cp .env.example .env
echo 'ALPHAVANTAGE_API_KEY=your_key_here' >> .env

# Sanity check — no API key needed:
uv run stx --version
uv run stx backtest --synthetic --dry-run
```

All commands below assume `uv run stx …`. You can also `source .venv/bin/activate`
and drop the `uv run` prefix.

> **Python ≥ 3.11 is required.** Runtime stack: PyTorch, pandas, PyArrow,
> Pydantic, Click.

---

## 3. The two experiment modes

### 3.1 Single-symbol mode

- `experiment_mode: single_symbol` (or omit — it is the default).
- Example config: [`configs/default.yaml`](../configs/default.yaml).
- Model: causal Transformer over **multi-timeframe candle tokens**
  (`monthly`, `weekly`, `daily`, `60min`, …).
- Prediction: next candle of `prediction_timeframe`
  (default `daily`) — joint **regression** (close return) + **direction**.
- Primary loss: `alpha * MSE + (1 - alpha) * BCE` (`loss_alpha` in YAML).

Use when you have a hypothesis about one ticker and want a calibrated answer
to *"does the model beat chance on this series?"* before going cross-sectional.

### 3.2 Universe / ranker mode

- `experiment_mode: universe` is **required**.
- Example configs: [`configs/universe.yaml`](../configs/universe.yaml),
  [`configs/sample.yaml`](../configs/sample.yaml).
- Model: temporal + cross-sectional Transformer ranker that scores every
  symbol at every aligned timestamp and ranks them.
- Label options (`label_mode`):
  - `cross_sectional_return` — return demeaned across the universe at each `t` (default, recommended).
  - `raw_return` — plain next-period return.
  - `equal_weighted_return` — return minus equal-weighted basket return.
  - `sector_neutral_return` — return minus same-sector mean (requires `sector_map_path`).
- Loss options (`loss`): `mse`, `listnet`, `approx_ndcg`.
- Optional `portfolio_sim` block converts test-fold rankings into a simulated
  long-only or long-short book with transaction costs.

Use for opportunity exploration across a basket (sector, theme, correlated ETFs).
`stx sweep` automates comparing the three ranking losses side-by-side.

---

## 4. Command reference

### Global (before the subcommand)

```
stx [GLOBAL_OPTIONS] COMMAND [OPTIONS]
```

| Option | Effect |
|--------|--------|
| `-h`, `--help` | Help for `stx` or the current subcommand. |
| `--version` | Print `stock-transformer X.Y.Z (torch A.B.C, auto→<device>)`. |
| `-v` / `-vv` | `INFO` / `DEBUG` logs on stderr. |
| `-q`, `--quiet` | Suppress info logs (warnings + errors only). |
| `--log-file PATH` | Also append logs to this file. |
| `--no-color` | Disable ANSI styling (also via `NO_COLOR` env var). |
| `--rich` | Use the `rich` package for per-fold/epoch progress lines when installed. |

### `stx backtest`

Walk-forward experiment (auto-dispatches single-symbol vs universe based on
`experiment_mode` in the YAML).

```
stx backtest [-c CONFIG] [--synthetic] [--device NAME] [--seed N]
             [-o text|json] [--dry-run]
```

| Option | Default | Purpose |
|--------|---------|---------|
| `-c`, `--config` | `$STX_CONFIG` or `configs/default.yaml` | Experiment YAML. |
| `--synthetic` | off | Use built-in synthetic candles (no API). Great for smoke tests. |
| `--device` | YAML/env | `auto`, `cpu`, `mps`, `cuda`, `cuda:N`. |
| `--seed` | YAML/env | Reproducibility override; wins over `STX_SEED`. |
| `-o`, `--output-format` | `text` | `text` one-liner; `json` full summary dict on stdout. |
| `--dry-run` | off | Resolve data, write `folds.json` / partial `summary.json`, print fold plan, exit without training. |

**Legacy alias:** `stx-backtest …` forwards to `stx backtest …`.

### `stx sweep`

Run the ranking-loss comparison for a universe config (`mse`, `listnet`,
`approx_ndcg`). Each loss becomes a sub-run; a merged summary is printed.

```
stx sweep [-c CONFIG] [--synthetic] [-o text|json]
```

| Option | Default | Purpose |
|--------|---------|---------|
| `-c`, `--config` | `configs/universe.yaml` | Universe YAML. |
| `--synthetic` | off | Synthetic universe data. |
| `-o` | `text` | `text` = fixed-column table; `json` = merged dict with a `by_loss` key. |

### `stx config show` / `stx config diff`

Inspect the merged effective configuration *without* training.

```
stx config show -c CONFIG   # full merged YAML (flag > env > file > defaults)
stx config diff -c CONFIG   # only keys that differ from Pydantic defaults
```

Great for confirming what env/flag overrides actually did to a file.

### `stx fetch` / `stx data fetch`

Download daily-adjusted OHLCV via Alpha Vantage; writes raw JSON to
`<cache_dir>/raw/` and canonical CSV (partitioned by timeframe + symbol) to
`<cache_dir>/canonical/`. Requires `ALPHAVANTAGE_API_KEY`.

```
stx fetch [--cache-dir DIR] [--symbols SYM]... [--refresh]
stx data fetch [--cache-dir DIR] [--symbols SYM]... [--refresh]   # identical
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--cache-dir` | `data` | Root for `raw/` and `canonical/`. |
| `--symbols` | `MSTR IBIT COIN QQQ` (pilot universe) | Repeatable. Strips + uppercases. Blank tokens are rejected. |
| `--refresh` | off | Re-download and overwrite canonical CSV. |

### `stx validate`

Loads the YAML, applies `STX_*` env overrides, runs full Pydantic validation.
**No training, no GPU, no network.** Ideal for CI gates.

```
stx validate -c CONFIG
```

Exit `0` on success, `1` on validation error.

### `stx version`

Same output as `stx --version`. Useful for scripts that want a subcommand.

### `stx completion <bash|zsh|fish>`

Prints a shell completion script. Install via `eval "$(stx completion zsh)"`
or redirect to a file and source it from your rc file.

---

## 5. Config files (YAML schema)

Pydantic enforces the schema (unknown keys produce a warning). The two
top-level schemas are `SingleSymbolExperimentConfig` and
`UniverseExperimentConfig` (see
[`src/stock_transformer/config_models.py`](../src/stock_transformer/config_models.py)).

### 5.1 Shared keys (both modes)

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `experiment_mode` | `"single_symbol" \| "universe"` | `single_symbol` | Switches Pydantic schema and runner. |
| `train_bars` | int ≥ 1 | required | Training window per fold (in aligned samples). |
| `val_bars` | int ≥ 1 | required | Validation window per fold. |
| `test_bars` | int ≥ 1 | required | Test window per fold. |
| `step_bars` | int ≥ 1 | required | Fold stride (how far each new fold rolls). |
| `d_model` / `nhead` / `dim_feedforward` / `dropout` | | 64 / 4 / 128 / 0.1 | Transformer width. `nhead` must divide `d_model`. |
| `epochs` / `batch_size` / `learning_rate` | | 10 / 32 / 5e-4 | Training loop. |
| `early_stopping_patience` | int ≥ 0 | `0` (off) | Stop if val metric does not improve. |
| `lr_reduce_on_plateau` | bool | `false` | Optional LR scheduler. |
| `device` | str | `auto` | `auto` / `cpu` / `mps` / `cuda[:N]`. |
| `seed` | int | `42` | |
| `artifacts_dir` | str | `artifacts` | Run outputs root. |
| `cache_dir` | str | `data` | Data cache root. |
| `use_adjusted_daily/weekly/monthly` | bool | `true` | Use Alpha Vantage adjusted series. |
| `intraday_month` / `intraday_extended_hours` / `intraday_outputsize` / `daily_outputsize` | | `null` / `false` / `full` / `full` | Passed to Alpha Vantage. |
| `inference_batch_size` | int ≥ 1 | 256 / 128 | Batched inference. |

### 5.2 Single-symbol extras

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `symbol` | str (non-empty) | required | Ticker. |
| `timeframes` | list[str] | required | Input token timeframes (e.g. `[monthly, weekly, daily, 60min]`). |
| `prediction_timeframe` | str | `daily` | Which timeframe's next candle is predicted. |
| `lookbacks` | `dict[str, int]` | `{}` | Per-timeframe history length. |
| `lookback` | int ≥ 1 | `32` | Fallback if `lookbacks[tf]` missing. |
| `max_seq_len` | int ≥ 1 | `256` | Token sequence pad/truncate. |
| `num_layers` | int ≥ 1 | `2` | Transformer depth. |
| `loss_alpha` | float in `[0,1]` | `0.5` | Weight for regression vs direction loss. |
| `default_threshold` | float | `0.5` | Classifier threshold (can be tuned per-fold from val). |
| `synthetic_n_daily` | int | `1200` | Synthetic series length when `--synthetic`. |

### 5.3 Universe extras

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `symbols` | list[str] | required | Universe members (upper-cased). |
| `target_symbol` | str | first symbol | Primary label row / highlighted in metrics. |
| `timeframe` | str | `daily` | Single prediction timeframe. |
| `lookback` | int ≥ 2 | required | History window. |
| `min_coverage_symbols` | int \| null | `max(2, N-1)` | Minimum symbols with valid OHLCV at `t` and `t+1`. |
| `label_mode` | see §3.2 | `cross_sectional_return` | |
| `sector_map_path` | str \| null | `null` | Required if `label_mode = sector_neutral_return`. |
| `store` | `"csv" \| "parquet" \| null` | `null` | Canonical store backend. |
| `loss` | `mse \| listnet \| approx_ndcg` | `mse` | Ranking loss. |
| `data_source` | `rest \| mcp` | `rest` | How to load Alpha Vantage payloads. |
| `features` | list[str] \| null | 5 default OHLCV-derived features | Custom feature columns. |
| `portfolio_sim` | dict \| null | `null` | `{enabled, book, top_k, transaction_cost_one_way_bps}`. |
| `num_temporal_layers` / `num_cross_layers` | int ≥ 1 | 2 / 1 | Ranker depth. |
| `save_models` | bool | `false` | Persist PyTorch weights per fold. |
| `synthetic_n_bars` | int | `600` | Synthetic calendar length. |

---

## 6. Environment variables & precedence

**Precedence (highest → lowest):** CLI flag → `STX_*` env var → YAML → Pydantic default.

| Variable | Purpose |
|----------|---------|
| `STX_CONFIG` | Default `-c` path when omitted. |
| `STX_DEVICE` | Device override for `backtest` / `sweep`. |
| `STX_CACHE_DIR` | Data cache root. |
| `STX_ARTIFACTS_DIR` | Run output root. |
| `STX_EPOCHS` | Integer epochs override. |
| `STX_SEED` | Integer seed override (CLI `--seed` wins). |
| `STX_BATCH_SIZE` | Integer batch-size override. |
| `STX_LOG_LEVEL` | Root log level when `-v/-vv/-q` not passed (`DEBUG`, `INFO`, …). |
| `NO_COLOR` | Disable ANSI styling. |
| `ALPHAVANTAGE_API_KEY` | Required for live `fetch` / non-synthetic runs. |

Loaded from a `.env` file at the repo root via `python-dotenv`.

---

## 7. Artifact layout & how to parse results

Every run writes `<artifacts_dir>/<prefix>_YYYYMMDD_HHMMSS[_shortsha]/`.
`prefix` is `run` for single-symbol and `universe_run` for universe mode.

### 7.1 Always produced

| File | Content |
|------|---------|
| `config_snapshot.yaml` | Exact merged config used for the run. |
| `summary.json` | Per-fold metrics + `aggregate` (mean/std) + run metadata. Primary machine-readable output. |
| `training_log_fold_<k>.csv` | Per-epoch train/val loss for fold `k`. |
| `folds.json` | Fold boundaries (`i_start`, `i_end`, `timestamp_start`, `timestamp_end` for each of `train/val/test`). |

### 7.2 Single-symbol extras

| File | Content |
|------|---------|
| `predictions__<SYMBOL>.csv` | `timestamp, y_true_return, y_pred_return, y_pred_prob, y_pred_label, fold_id`. |

### 7.3 Universe extras

| File | Content |
|------|---------|
| `predictions_universe.csv` | `timestamp, symbol, timeframe, y_true_raw_return, y_true_relative_return, y_score, y_rank_pred, y_rank_true, fold_id`. |
| `feature_schema.json` | `{features: [...], n, hash, git_sha}`. Stable hash of the feature set for reproducibility. |
| `universe_membership.json` | `[{timestamp_start, timestamp_end, symbol, active_flag, sector, market_cap_bucket}]`. |
| `fold_errors.log` | Present only when at least one fold failed. |

### 7.4 Summary schema (single-symbol)

```jsonc
{
  "symbol": "IBM",
  "prediction_timeframe": "daily",
  "timeframes": ["monthly", "weekly", "daily"],
  "n_samples": 798,
  "n_folds": 16,
  "device": "cpu",
  "run_dir": "artifacts/run_...",
  "folds": [
    {
      "fold_id": 0,
      "threshold": 0.1,
      "test_cls_accuracy": 0.5,
      "test_cls_precision": 0.5,
      "test_cls_recall": 1.0,
      "test_cls_f1": 0.667,
      "test_cls_brier": 0.265,
      "test_cls_roc_auc": 0.569,
      "test_reg_mae": 0.426,
      "test_reg_rmse": 0.814,
      "test_reg_directional_accuracy": 0.6
    }
    // … one object per fold
  ],
  "aggregate": { "test_cls_accuracy_mean": 0.5, "test_cls_accuracy_std": 0.08, "…": "…" }
}
```

### 7.5 Summary schema (universe)

```jsonc
{
  "experiment": "universe",
  "symbols": ["MSTR", "IBIT", "COIN", "QQQ"],
  "target_symbol": "MSTR",
  "timeframe": "daily",
  "n_samples": 567,
  "n_folds": 12,
  "device": "mps",
  "run_dir": "artifacts/universe_run_...",
  "folds": [
    {
      "fold_id": 0,
      "spearman_mean": 0.26,
      "kendall_mean": 0.26,
      "ndcg3_mean": 0.91,
      "top2_hit": 1.0,
      "target_mae": 0.06,
      "target_rmse": 0.07,
      "baseline_momentum_spearman_mean": 0.10,
      "baseline_mean_reversion_spearman_mean": -0.05,
      "baseline_equal_spearman_mean": 0.0,
      "baseline_linear_spearman_mean": 0.08,
      "baseline_gbt_spearman_mean": 0.12
    }
    // … one object per fold
  ],
  "aggregate": { "spearman_mean_mean": 0.08, "…": "…" }
}
```

### 7.6 Sweep schema (universe)

```jsonc
{
  "by_loss": {
    "mse":        { "aggregate": { "spearman_mean_mean": 0.04, "ndcg3_mean_mean": 0.88, "top2_hit_mean": 0.73 }, "run_dir": "..." },
    "listnet":    { "aggregate": { "...": "..." }, "run_dir": "..." },
    "approx_ndcg":{ "aggregate": { "...": "..." }, "run_dir": "..." }
  },
  "config_path": "configs/universe.yaml"
}
```

### 7.7 Canonical data cache

After `stx fetch`:

```
data/
├── raw/<endpoint>/<sha>.json            # cached Alpha Vantage payloads
└── canonical/timeframe=<tf>/symbol=<SYM>/part-000.csv
    # columns: timestamp,symbol,timeframe,open,high,low,close,volume
```

Partitioned layout is Hive-style and readable with pandas / pyarrow /
DuckDB (`SELECT … FROM read_csv_auto('data/canonical/...')`).

---

## 8. Exit codes & JSON output contract

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Invalid/missing config, Pydantic validation error, or unrecoverable runtime error. |
| `2` | Partial failure (e.g. one or more folds errored, or `error in {partial_failure, no_folds}`). **`summary.json` is still written.** |
| `130` | SIGINT / Ctrl+C (best-effort artifact flush, no traceback). |
| `143` | SIGTERM (container friendly). |

When `-o json` is used, **stdout is pure JSON** (the summary dict), which
makes it easy to pipe into `jq` or consume programmatically:

```bash
uv run stx backtest --synthetic -o json | jq '.aggregate.test_cls_accuracy_mean'
uv run stx sweep    --synthetic -o json | jq '.by_loss | to_entries | map({loss:.key, spearman:.value.aggregate.spearman_mean_mean})'
```

Logs go to **stderr** and never pollute the JSON payload. In `text` mode the
final line is a one-liner such as `Run complete. Artifacts: artifacts/run_...`.

---

## 9. Exploration recipes

Copy-paste-runnable workflows for an automated agent.

### R1. Smoke test (no API key, no GPU, ~seconds)

```bash
uv run stx validate   -c configs/universe.yaml
uv run stx backtest   --synthetic -c configs/sample.yaml -o json > /tmp/smoke.json
jq '.aggregate' /tmp/smoke.json
```

### R2. Fetch a pilot universe from Alpha Vantage

```bash
echo 'ALPHAVANTAGE_API_KEY=...' >> .env
uv run stx fetch --cache-dir data --symbols MSTR --symbols IBIT --symbols COIN --symbols QQQ
ls data/canonical/timeframe=daily/
```

### R3. Single-symbol sanity check for a new ticker

```bash
cp configs/default.yaml /tmp/exp_NVDA.yaml
sed -i '' 's/symbol: "IBM"/symbol: "NVDA"/' /tmp/exp_NVDA.yaml
uv run stx fetch --symbols NVDA
uv run stx -v backtest -c /tmp/exp_NVDA.yaml -o json > /tmp/exp_NVDA.json
jq '{n_folds, mean_acc: .aggregate.test_cls_accuracy_mean, dir_acc: .aggregate.test_reg_directional_accuracy_mean}' /tmp/exp_NVDA.json
```

### R4. Compare ranking losses on a custom basket

1. Copy `configs/universe.yaml` → `configs/basket_ai.yaml`, edit `symbols` and `target_symbol`.
2. Fetch data: `uv run stx fetch --symbols NVDA --symbols AMD --symbols TSM --symbols AVGO`.
3. Run: `uv run stx sweep -c configs/basket_ai.yaml -o json > /tmp/sweep.json`.
4. Pick the winning loss by `spearman_mean_mean`, then run `stx backtest`
   with that `loss` for full artifacts.

```bash
jq -r '.by_loss
       | to_entries
       | sort_by(-.value.aggregate.spearman_mean_mean)[0]
       | "\(.key)\t\(.value.aggregate.spearman_mean_mean)"' /tmp/sweep.json
```

### R5. Dry-run to plan folds before a long training job

```bash
uv run stx backtest -c configs/universe.yaml --dry-run -o text
# or machine-readable:
uv run stx backtest -c configs/universe.yaml --dry-run -o json \
  | jq '{n_samples, n_folds, first: .fold_plan["0"], last: .fold_plan | to_entries | last.value}'
```

This writes `folds.json` + partial `summary.json` without ever calling the
training loop — perfect for checking window sizes against history length.

### R6. Re-rank an already-finished run

Every run's `predictions_universe.csv` has `y_score` and `y_rank_true`; you
can apply any post-hoc selection strategy offline:

```python
import pandas as pd
df = pd.read_csv("artifacts/universe_run_20260419_205207_75dd4283/predictions_universe.csv")
top = (
    df.dropna(subset=["y_score"])
      .assign(rank=lambda d: d.groupby("timestamp")["y_score"].rank(method="first", ascending=False))
)
pnl = top.query("rank <= 2").groupby("timestamp")["y_true_raw_return"].mean()
print(pnl.describe(), pnl.cumsum().iloc[-1])
```

### R7. Enable an in-sample portfolio simulation

Uncomment / add to a universe YAML:

```yaml
portfolio_sim:
  enabled: true
  book: "long_only"    # or "long_short"
  top_k: 2
  transaction_cost_one_way_bps: 1.0
```

`summary.json` then includes a `portfolio_sim` block alongside the ranking metrics.

### R8. Reproducibility / ablation

```bash
for s in 0 1 2 3 4; do
  uv run stx backtest --synthetic -c configs/universe.yaml --seed "$s" -o json \
    > "/tmp/seed_${s}.json"
done
jq -s 'map({seed: .seed, sp: .aggregate.spearman_mean_mean})' /tmp/seed_*.json
```

### R9. CI-style validation only (fast, no GPU)

```bash
uv run stx validate -c configs/universe.yaml
uv run stx validate -c configs/default.yaml
uv run stx config diff -c configs/universe.yaml   # shows what differs from model defaults
```

### R10. Inspect the effective config after env overrides

```bash
STX_SEED=7 STX_EPOCHS=2 uv run stx config show -c configs/universe.yaml \
  | grep -E '^(seed|epochs):'
```

---

## 10. Automation tips for Claude Code

- **Prefer `-o json`** for `backtest` and `sweep` — it is stable, well-typed,
  and safe to `jq` against. Logs go to stderr only.
- **Use `--synthetic` first.** It needs no API key and runs in seconds — a
  perfect smoke test before burning quota on a real universe.
- **Always validate before training.** `stx validate -c cfg.yaml` returns
  in milliseconds and gives bullet-list Pydantic errors. CI-friendly.
- **Dry-run to sanity-check windows.** `stx backtest --dry-run` writes
  `folds.json` + partial `summary.json` and prints the full fold plan as
  YAML to stdout. Use it to verify that `train_bars + val_bars + test_bars`
  fits the available history before training kicks off.
- **Avoid rerunning identical configs.** Each run is written to a new
  timestamped directory; stale runs are never overwritten.
- **Always set `--device cpu` in sandboxed environments** (no GPU) — or set
  `STX_DEVICE=cpu`. `auto` tries MPS on Apple Silicon and CUDA on NVIDIA,
  which can fail in containers.
- **Parse `run_dir` from JSON output** to locate artifacts without shelling:
  `jq -r .run_dir /tmp/result.json`.
- **Exit codes matter.** `2` means "finished with fold errors — inspect
  `summary.json.fold_errors`", not "crashed".
- **Environment isolation.** If you are iterating on many configs,
  `STX_ARTIFACTS_DIR=/tmp/stx_scratch` keeps the main `artifacts/` clean.
- **Capture stdout separately** when running `backtest` with `-o json` so
  logs don't corrupt your JSON:
  `uv run stx -q backtest --synthetic -o json > out.json 2> err.log`.
- **Do not import the CLI from training code.** The `cli/` package depends
  on `backtest/`, not the other way around; for programmatic use call
  `stock_transformer.backtest.run_from_config_path(...)` directly.

### Suggested agent loop (pseudo)

```
1. edit YAML under configs/
2. stx validate -c <yaml>                  # fail-fast
3. stx config diff -c <yaml>               # confirm what's non-default
4. stx backtest -c <yaml> --dry-run -o json   # confirm fold plan
5. stx backtest -c <yaml> -o json > out.json
   → on exit 0: parse aggregate metrics → decide next change
   → on exit 2: read artifacts/<run>/fold_errors.log, narrow universe / widen windows
   → on exit 1: stderr has the Pydantic bullet list → fix YAML, go to 2
6. (universe only) stx sweep -c <yaml> -o json to compare losses
7. drill into artifacts/<run_dir>/predictions_universe.csv for case studies
```

---

## 11. Troubleshooting & known gotchas

| Symptom | Likely cause / fix |
|---------|-------------------|
| `ALPHAVANTAGE_API_KEY not set` | Populate `.env` or export the variable. |
| `device` errors on Linux / containers | Set `STX_DEVICE=cpu` or pass `--device cpu`. |
| `Unknown config key 'foo' (typo?)` warning | Extra YAML key was ignored. Check spelling — unknown keys are silently dropped after the warning. |
| Exit `2` and empty `folds.json` | History too short for `train_bars + val_bars + test_bars`. Lower the window sizes or fetch more data. |
| `n_folds` very small | `step_bars` too large relative to usable samples; lower it. |
| NaN metrics in early folds | Universe fold has too few symbols with valid coverage; raise `min_coverage_symbols` or widen history. |
| CI complains about `make man-check` | Run `make man` and commit; `click-man` regenerates per-command `man/*.1`. |
| Weird ANSI in log files | Pass `--no-color` or set `NO_COLOR=1`. |

---

## 12. Glossary

- **Walk-forward**: contiguous, time-ordered `train / val / test` splits that
  roll forward by `step_bars`. No future leakage.
- **Aligned sample index**: one row per prediction timestamp where all
  required symbols/timeframes have valid OHLCV (universe mode enforces a
  `min_coverage_symbols` threshold).
- **Cross-sectional return**: a symbol's return minus the universe mean
  return at the same timestamp (demeaned / relative return).
- **ListNet / ApproxNDCG**: pairwise/listwise ranking losses; often improve
  top-k hit rate vs plain MSE on returns.
- **Brier score**: mean squared error of probability predictions
  (lower = better, `0.25` is random for a balanced binary target).
- **Spearman / Kendall**: rank-correlation metrics between predicted and
  true per-timestamp orderings.
- **NDCG@k**: rank quality weighted toward the top of the list — cares most
  about getting the top few right.
- **Top-k hit**: fraction of timestamps where the true best symbol appears
  in the predicted top-`k`.
- **Baseline metrics in `summary.json`**: `momentum`, `mean_reversion`,
  `equal`, `linear`, `gbt` — non-deep baselines that the Transformer ranker
  must beat to be worth using.
