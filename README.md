# stock-transformer

Multi-timeframe autoregressive candle transformer **and** a **multi-ticker universe** ranker with walk-forward backtesting.

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

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

On **Apple Silicon** the default config uses `device: "auto"` which picks MPS
when available. Override to `"cpu"` or `"cuda"` if needed.

## Quick start — synthetic data (no API key required)

Single-symbol pipeline:

```bash
stx-backtest --synthetic
```

Universe / cross-sectional pipeline:

```bash
stx-backtest --synthetic -c configs/universe.yaml
```

## Live data via Alpha Vantage

```bash
export ALPHAVANTAGE_API_KEY=your_key_here
stx-backtest -c configs/default.yaml
stx-backtest -c configs/universe.yaml
```

The client fetches candles per timeframe, respects throttling, caches raw JSON,
and writes canonical CSVs. Universe mode loops all symbols in the YAML list.

## Configuration

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Single-symbol multi-timeframe next-candle prediction |
| `configs/universe.yaml` | Multi-ticker universe, cross-sectional labels, ranker |

Key universe settings: `symbols`, `target_symbol`, `timeframe`, `lookback`,
`min_coverage_symbols`, `label_mode` (`cross_sectional_return` or `raw_return`).

## Architecture

**Single-symbol:** multi-timeframe token sequence → causal Transformer →
regression + direction heads.

**Universe:** per-symbol **temporal** Transformer over lookback → **cross-sectional**
Transformer over symbols → one **score per ticker**. Training loss is masked MSE
on finite cross-sectional targets; evaluation includes Spearman, NDCG@k, and
top-k hit rate, plus momentum / equal-score baselines.

## Tests

```bash
pytest -v
```

## Project structure

```
src/stock_transformer/
├── data/           # Alpha Vantage client, alignment, universe helpers, synthetic
├── features/       # Multi-timeframe tokens; universe tensor assembly
├── labels/         # Cross-sectional return and bucket helpers
├── model/          # CandleTransformer, TransformerRanker, baselines
└── backtest/       # Walk-forward splits, metrics, single + universe runners
```

See `plan.md` for the full target architecture (parquet store, sector-neutral
labels, richer leakage tests, and portfolio simulation in later phases).
