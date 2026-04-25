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

The first run fetches data from Alpha Vantage (5 API calls, one per symbol). All responses are cached under `data/raw/` (see `ALPHAVANTAGE_REQUESTS_PER_MINUTE` in your environment; default 5 for free tier) so subsequent runs are instant.

## Bulk download (data analysis)

The bulk-download pipeline pulls OHLCV, fundamentals, corporate actions, and macro time series for a larger symbol universe (default ~500 S&P 500 tickers) in an API-rate-aware way, then aggregates everything into Parquet for downstream analysis. Full design lives in [`docs/data_download.md`](docs/data_download.md).

### Prerequisites

1. An AlphaVantage **Premium 75** API key in `.env` (the bulk path is sized for the Premium 75 plan; a free-tier key will be throttled by AlphaVantage long before the limiter kicks in).
2. Dependencies installed (`pip install -e .` — covers `pyarrow` for Parquet writes).

### Configure what to download

The download manifest is [`configs/download.yaml`](configs/download.yaml). The defaults are sensible; the most common knobs:

| Key | Default | What it controls |
|---|---|---|
| `symbols_file` | `configs/sp500.txt` | One ticker per line. Class-share punctuation is normalized (`BRK.B` ↔ `BRK-B`). |
| `symbols` | _(unset)_ | Optional inline list; takes precedence over `symbols_file`. |
| `data_types.{ohlcv,fundamentals,dividends,splits,macro}` | all `true` | Toggle endpoint families. `fundamentals: true` gates the COMPANY_OVERVIEW classifier (equity vs. ETF routing). |
| `requests_per_minute` | `65` | Sliding-window cap (15 % safety margin under the 75-req/min Premium tier). |
| `entitlement` | `delayed` | Premium 75 plan tier; OHLCV daily bars are unaffected. |
| `on_error` | `skip` | `skip` (per-symbol error isolation) or `abort`. |
| `stale_fallback` | `true` | Serve stale cache when a TTL refetch fails (logs a warning, increments the `stale_fallbacks` counter). |
| `cache_ttl.*` | see file | Per-endpoint freshness windows. |

The default symbol universe lives in [`configs/sp500.txt`](configs/sp500.txt) (snapshot of the index — refresh as needed; it's rebalanced quarterly).

### Estimate the cost first

Always run a dry run first to see how many calls and how much wall-time the run will take:

```bash
python scripts/download_data.py --dry-run
```

Sample output for the full kitchen sink against the bundled sp500 list:

```
dry_run planned_calls~4031 (approx), estimated_time~62m at 65 req/min
```

No network calls are made during a dry run.

### Run the download

```bash
python scripts/download_data.py
# or with an explicit manifest:
python scripts/download_data.py -c configs/download.yaml
```

What happens:

1. **Phase 0 — load & normalize.** Reads the manifest and the symbol list.
2. **Phase 1 — classifier** (skipped if `fundamentals: false`). Fetches `COMPANY_OVERVIEW` for each symbol, buckets by `AssetType`, persists the routing to `data/processed/_universe_split.json`.
3. **Phase 2 — per-symbol data.** OHLCV + dividends + splits for everyone; income/balance/cashflow/earnings for equities; `ETF_PROFILE` for ETFs.
4. **Phase 3 — macro.** Iterates the macro endpoints (GDP, CPI, fed funds, treasury yields ×6 maturities, etc.).
5. **Phase 4 — Parquet aggregation.** Always rebuilds from the current state of `data/raw/`, asserts primary-key uniqueness, writes `data/processed/{ohlcv,fundamentals/*,corporate_actions/*,macro/*}.parquet`.

Progress prints one line per call (`[i/N] symbol endpoint cache_status eta`), and a per-run log file `data/processed/_run_{run_id}.log` captures everything at DEBUG level.

### Common workflows

```bash
# Quick smoke test against a tiny universe (no need to edit the manifest)
python scripts/download_data.py --symbols AAPL,MSFT,GOOGL

# Re-run only the calls that failed last time
# (replays from data/processed/_errors_latest.json with stored params)
python scripts/download_data.py --retry-errors

# Force a full refresh, ignoring TTL on every endpoint
python scripts/download_data.py --no-cache

# Use a different manifest
python scripts/download_data.py -c configs/my_universe.yaml
```

### Resume, interrupt, and re-run safety

- **Cache.** Every API response is written atomically to `data/raw/<endpoint>/<symbol>_<hash>.json` (apikey is excluded from the hash). Re-running picks up where the cache left off; cache hits are free.
- **TTL.** Each endpoint family has its own freshness window (OHLCV 24 h, COMPANY_OVERVIEW 7 d, fundamentals 30 d, etc.). Within the window, calls are cache-only.
- **SIGINT (Ctrl-C).** The handler stops after the in-flight call, flushes logs, writes the partial error log to `data/processed/_errors_{run_id}.json`, prints a summary, and the CLI exits with status `130`. Phase 4 is skipped on interrupt — re-running picks up from cache.
- **Stale-cache fallback.** If a TTL-expired refetch fails (network blip, transient AV error), the cached file is returned and a warning is logged. Watch the `stale_fallbacks` counter in the summary before treating data as fresh.

### Output layout

```
data/
  raw/                                      one JSON per (endpoint, symbol-or-job)
    time_series_daily_adjusted/
    company_overview/
    income_statement/  balance_sheet/  cash_flow/  earnings/
    etf_profile/
    dividends/  splits/
    macro/
  processed/
    ohlcv.parquet                           (symbol, date)
    fundamentals/
      company_overview.parquet
      income_statement.parquet              (symbol, fiscalDateEnding, frequency)
      balance_sheet.parquet
      cash_flow.parquet
      earnings.parquet
      etf_profile.parquet  etf_holdings.parquet
    corporate_actions/
      dividends.parquet                     (symbol, ex_dividend_date)
      splits.parquet                        (symbol, effective_date)
    macro/
      real_gdp.parquet  cpi.parquet  federal_funds_rate.parquet  ...
      treasury_yield.parquet                (maturity, date)
    _universe_split.json                    phase-1 equity/ETF/unknown buckets
    _errors_{run_id}.json                   per-run error log (function/symbol/params/error)
    _errors_latest.json                     symlink-equivalent for --retry-errors
    _run_{run_id}.log                       full INFO/DEBUG log
```

### Troubleshooting

- `Missing ALPHAVANTAGE_API_KEY` — set it in `.env` or your shell.
- "Stale cache fallback for ..." warnings — AlphaVantage refused or timed out; the cached copy is being served. Investigate if `stale_fallbacks > 0` in the summary.
- Phase 4 raises `ValueError: ... duplicate ...` — a primary-key violation in the aggregated data; usually means two cache files map to the same `(symbol, date)` etc. Check `data/raw/<endpoint>/` for duplicates.
- Hitting the 75-req/min cap anyway — lower `requests_per_minute` in the manifest, or check that no other process is consuming the same key.

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
