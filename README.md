# Stock Transformer

Predict SPY's next-day candle (open, high, low, close) using a transformer that treats each (ticker, day) candle as a token.

## How it works

Daily OHLCV candles from multiple tickers are each treated as a token -- similar to words in a sentence. A transformer encoder processes a window of these candle tokens across all tickers, then predicts SPY's next-day open/high/low/close from the final-timestep SPY token.

## Setup

1. Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

2. Create your `.env` file:
   ```bash
   cp .env.example .env
   ```
   Then add your key to `.env`:
   ```
   ALPHAVANTAGE_API_KEY=your_key_here
   ```

3. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

## Run

```bash
python run.py
```

Or with a custom config:

```bash
python run.py -c path/to/config.yaml
```

The first run fetches data from Alpha Vantage (5 API calls, one per symbol). All responses are cached in `data/` so subsequent runs are instant.

## Configuration

All settings live in `configs/default.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `symbols` | SPY, QQQ, AAPL, MSFT, GOOGL | Tickers to use as input tokens |
| `target_symbol` | SPY | Symbol to predict |
| `lookback` | 30 | Days of history per sample |
| `train_pct` | 0.7 | Training split |
| `val_pct` | 0.15 | Validation split (rest is test) |
| `d_model` | 64 | Transformer embedding dimension |
| `nhead` | 4 | Attention heads |
| `num_layers` | 3 | Transformer encoder layers |
| `dropout` | 0.1 | Dropout rate |
| `epochs` | 50 | Max training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 0.001 | AdamW learning rate |
| `seed` | 42 | Random seed |

## Project structure

```
run.py                          Entry point
configs/default.yaml            Configuration
src/stock_transformer/
    data.py                     Alpha Vantage fetch + date alignment
    features.py                 Log-return features + tensor builder
    model.py                    CandleTransformer (encoder + prediction head)
    train.py                    Training loop, evaluation, device/seed helpers
```
