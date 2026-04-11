# stock-transformer

Leakage-safe autoregressive **next-candle direction** forecasting with a causal Transformer and **walk-forward (rolling-origin) evaluation**. Uses Alpha Vantage OHLCV series (same endpoints as the Cursor **Alpha Vantage MCP** tools: `TIME_SERIES_INTRADAY`, `TIME_SERIES_DAILY` / `TIME_SERIES_DAILY_ADJUSTED`, `TIME_SERIES_MONTHLY` / `TIME_SERIES_MONTHLY_ADJUSTED`).

## Layout

- `src/stock_transformer/data/` — REST ingestion (mirrors MCP), canonical candles, synthetic data for tests
- `src/stock_transformer/features/` — feature matrix, direction labels, sliding windows, lookahead checks
- `src/stock_transformer/model/` — causal `CandleTransformerClassifier`, naive baselines
- `src/stock_transformer/backtest/` — walk-forward splits, metrics, `runner` orchestration
- `configs/default.yaml` — experiment template
- `tests/` — temporal integrity and smoke tests

## Setup

```bash
cd /path/to/stock-transformer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Set `ALPHAVANTAGE_API_KEY` for live data (see [Alpha Vantage](https://www.alphavantage.co/support/#api-key)).

## Run

**Synthetic smoke (no API key):**

```bash
stx-backtest -c configs/default.yaml --synthetic
```

**Live data:**

```bash
export ALPHAVANTAGE_API_KEY=...
stx-backtest -c configs/default.yaml
```

Artifacts: `artifacts/run_<timestamp>/` — `config_snapshot.yaml`, `summary.json`, per-timeframe `predictions__*.csv`.

## MCP vs Python

In Cursor, discover tools with MCP `TOOL_LIST` → `TOOL_GET` → `TOOL_CALL`. This package calls the same Alpha Vantage **HTTP API** so you can run backtests from the CLI/CI without the MCP runtime.

## Tests

```bash
pytest
```

## Assumptions

- **Hour candles** use intraday `interval=60min` (Alpha Vantage intraday intervals: 1, 5, 15, 30, 60 minutes).
- **Forecast-only** phase: classification metrics on held-out test slices; no trading simulation.
- Intraday history is often fetched **per month** via `month=YYYY-MM` when you need long intraday backtests (extend config / loop in a future iteration).
