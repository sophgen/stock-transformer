# Stock Transformer

Predict SPY's next-day candle (open, high, low, close) using a transformer that treats each (ticker, day) candle as a token.

## Quick start

```bash
cp .env.example .env              # add your ALPHAVANTAGE_API_KEY
uv pip install -e .               # install deps into venv
python run.py                     # uses configs/default.yaml
python run.py -c my_config.yaml   # custom config
```

Data is cached locally after first fetch (5 API calls for 5 symbols).
