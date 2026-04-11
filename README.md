# stock-transformer

Multi-timeframe autoregressive candle transformer with walk-forward backtesting.

## Concept

Each OHLCV candle — from any timeframe (minute, hour, day, week, month) — is
treated as a **token**.  All past candles up to a prediction point are fed into
a causal Transformer encoder (like autoregressive language modeling, but for
price candles).  The model predicts the **next candle** — both its OHLCV
log-returns (regression) and its direction (classification: up / down).

The backtest enforces **strict point-in-time discipline**: at every prediction
step the model only sees candles whose timestamps are ≤ the cutoff.  Walk-forward
folds ensure train / validation / test splits never overlap in time.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

On **Apple Silicon** the default config uses `device: "auto"` which picks MPS
when available.  Override to `"cpu"` or `"cuda"` if needed.

## Quick start — synthetic data (no API key required)

```bash
stx-backtest --synthetic
```

## Live data via Alpha Vantage

```bash
export ALPHAVANTAGE_API_KEY=your_key_here
stx-backtest -c configs/default.yaml
```

The pipeline fetches candles for every timeframe listed in the config, builds
multi-timeframe token sequences, and runs a walk-forward backtest.

## Configuration

See `configs/default.yaml`.  Key settings:

| Setting | Description |
|---|---|
| `timeframes` | List of timeframes to ingest as tokens |
| `prediction_timeframe` | Timeframe of the candle being predicted |
| `lookbacks` | Per-timeframe number of past candles to include |
| `max_seq_len` | Maximum token sequence length (pad / truncate) |
| `loss_alpha` | Weighting: `α·MSE + (1−α)·BCE` |
| `device` | `"auto"` / `"mps"` / `"cuda"` / `"cpu"` |

## Architecture

```
Multi-timeframe candle sequence (sorted by timestamp):
  [month_t-3, ..., week_t-2, ..., day_t-10, ..., hour_t-5, ...]
  Each token = [open_ret, high_ret, low_ret, close_ret, log_vol] + timeframe embedding

       ↓  causal Transformer encoder (only attends to past)

  Last token representation
       ↓
  ┌─────────────┐    ┌─────────────┐
  │ Regression   │    │ Direction   │
  │ head (OHLCV) │    │ head (↑/↓)  │
  └─────────────┘    └─────────────┘
```

## Tests

```bash
pytest -v
```

## Project structure

```
src/stock_transformer/
├── data/           # Alpha Vantage client, canonicalization, synthetic generator
├── features/       # Multi-timeframe tokenization, log-return features
├── model/          # CandleTransformer, baselines, device resolver
└── backtest/       # Walk-forward splits, metrics, experiment runner
```
