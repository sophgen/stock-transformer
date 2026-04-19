# Stock Transformer Cross-Sectional Backtest Plan

> **Authoritative status lives in [§Milestone tracker](#milestone-tracker).**
> Everything else in this document is the *target design*. When code and plan
> disagree on a completed milestone, code wins and the plan is updated to match.

## Scope and fixed decisions

- **Prediction target:** next-period **cross-sectional ticker performance** over
  a universe. The single-symbol autoregressive head is kept as a reference
  ablation but is no longer the primary path.
- **Primary label (v1):** next-period **relative return** per ticker, demeaned
  by the **nanmedian** of the live cross-section at the same timestamp.
- **Initial prediction task:** score all tickers at time `t` by expected return
  from `t` to `t+1`; report ranking quality.
- **Backtest mode:** forecast evaluation only for v1 — no position sizing or
  PnL simulation.
- **Supported timeframes:** `1min, 5min, 15min, 30min, 60min, daily, weekly,
  monthly`. The `timeframe` field in YAML accepts any of these; it maps to
  Alpha Vantage endpoints via the table in [§Alpha Vantage data plan](#alpha-vantage-data-plan).
- **Data source:** Alpha Vantage REST + on-disk cache. MCP discovery is **out
  of scope until M12** (see [§Milestone tracker](#milestone-tracker)).
- **System design constraint:** the full modeling pipeline is **multi-ticker
  end-to-end**. Inputs, labels, splits, normalization, and metrics are defined
  over the whole universe at each timestamp.
- **End-state:** train on multiple tickers (shared history, aligned timestamps,
  proper masking) and use the model to score or rank the whole cross-section.
  The pilot basket (`MSTR` + predictors) is a small-universe **drill-down** of
  the same ranker — not a separate direction classifier
  (see [§Pilot](#pilot-small-universe-drill-down-for-mstr)).

## Glossary

Shared vocabulary. Any new module should use these terms exactly.

| Term | Meaning |
|---|---|
| **Panel** | Outer-joined OHLCV dataframe across the universe on a single global timestamp index per timeframe. Columns are `timestamp` + `{field}__{symbol}`. |
| **Row / bar** | One aligned timestamp on the panel. |
| **Sample** | One training example anchored at prediction timestamp `t`: a lookback window `[t-L+1 … t]` across the full universe. |
| **Fold** | A contiguous `(train, val, test)` slice of the *sample index* produced by `WalkForwardConfig`. |
| **Live** (at `t`, for `s`) | `isfinite(raw_return[t, s])` — i.e. both `close[t, s]` and `close[t+1, s]` are finite. |
| **Coverage** (at `t`) | `sum_s live(t, s)`. A row is kept iff coverage ≥ `min_coverage_symbols`. |
| **Canonical candle** | One OHLCV row in the uniform schema `timestamp, symbol, timeframe, open, high, low, close, volume`, regardless of whether it came from REST or MCP. |
| **Feature schema hash** | `sha256(json.dumps(feature_names, sort_keys=True))[:16]`; changes invalidate cached tensors and is stamped in `feature_schema.json`. |
| **Target symbol** | YAML `target_symbol`: a *reporting key* for per-symbol drill-downs; does **not** gate training or coverage. |

## Conventions

Pin these once; every module honours them.

- **Timestamps:** UTC-naive `pd.Timestamp`. Intraday bars are stamped at bar
  close. All panels share a single global index per timeframe.
- **Symbol order:** driven by the YAML `symbols:` list; uppercased at load;
  immutable within a run; snapshot to `universe_membership.json`.
- **Label indexing:** at row `t`, `y[t, s] = close[t+1, s] / close[t, s] - 1`.
  The last row is always NaN. Matches `labels/cross_sectional.py::raw_returns_forward`.
- **Cross-sectional target (`label_mode`):**
  - `raw_return` — no demeaning.
  - `cross_sectional_return` — subtract `nanmedian` across live peers at `t` (v1 default).
  - `equal_weighted_return` *(later)* — subtract nanmean across live peers.
  - `sector_neutral_return` *(later)* — subtract nanmedian within the ticker's sector.
- **Mask polarity:** in feature tensors, `mask[t, s] == True` means
  **padding / invalid** (matches `~row_valid` in `features/universe_tensor.py`).
  Label validity is expressed separately as `torch.isfinite(y)`; loss and
  metrics must combine the two.
- **Coverage rule:** a prediction timestamp `t` is kept iff
  `sum(isfinite(raw_return[t])) >= min_coverage_symbols`. This is evaluated on
  forward return (i.e. both `t` and `t+1` must be finite for that symbol).
  `target_symbol` is **not** required to be live — it is a reporting key only;
  its metrics will be NaN on rows where MSTR is masked.
- **Tensor shapes and dtypes:**
  - Feature block per sample: `X[n] ∈ [L, S, F]` (L = lookback, S = #symbols).
  - Batched: `X ∈ [N, L, S, F]`, `mask ∈ [N, L, S]`, `y ∈ [N, S]`.
  - `X`: `float32`. `mask`: `bool` (True = padded/invalid). `y`: `float32`,
    NaN where label undefined. `raw_ret`: `float32`. `end_row`: `int64`.
  - F is tracked by `N_UNIVERSE_FEATURES` in `features/universe_tensor.py`;
    will grow when cross-sectional features land (M9a) — record
    `feature_schema_hash` in `feature_schema.json` (see M7b).
- **Deterministic ordering & seeding:** every training run calls
  `torch.manual_seed(cfg.seed)` and `np.random.seed(cfg.seed)`; DataLoaders use
  `torch.Generator` seeded from the same value.
- **Device:** `device: "auto"` prefers MPS on Apple Silicon, else CUDA, else
  CPU. Ops unsupported on MPS fall back to CPU with a single-line warning.

## Config reference

Authoritative list of YAML keys consumed by `configs/universe.yaml` (primary)
and `configs/default.yaml` (single-symbol reference). New keys introduced by a
future milestone are tagged with the milestone id.

### Universe config (`configs/universe.yaml`)

| Key | Type | Default | Intro | Description |
|---|---|---|---|---|
| `experiment_mode` | `"universe" \| "single_symbol"` | `"single_symbol"` | M5 | Dispatch in CLI. Missing → single-symbol. |
| `symbols` | `list[str]` | — | M1 | Ordered, uppercased at load; fixes symbol axis for the run. |
| `target_symbol` | `str` | first of `symbols` | M1 | Reporting-only drill-down key; does **not** drive training. |
| `timeframe` | `"1min"\|"5min"\|"15min"\|"30min"\|"60min"\|"daily"\|"weekly"\|"monthly"` | `"daily"` | M2 | Maps to AV endpoint per [§Alpha Vantage data plan](#alpha-vantage-data-plan). |
| `lookback` | `int >= 2` | `32` | M3 | L in `[L, S, F]`. |
| `min_coverage_symbols` | `int` | `max(2, len(symbols)-1)` | M3 | Drop `t` where `sum_s live(t, s) < this`. |
| `label_mode` | `"cross_sectional_return"\|"raw_return"\|"equal_weighted_return"\|"sector_neutral_return"` | `"cross_sectional_return"` | M4 / M9b | Last two land in M9b. |
| `use_adjusted_daily` / `use_adjusted_weekly` / `use_adjusted_monthly` | `bool` | `true` | M2 | Select AV adjusted endpoint. |
| `intraday_month` | `str \| null` | `null` | M2 | AV `month=YYYY-MM` slice for intraday. |
| `intraday_extended_hours` | `bool` | `false` | M2 | Include pre/post-market bars. |
| `intraday_outputsize` | `"compact"\|"full"` | `"full"` | M2 | AV outputsize. |
| `daily_outputsize` | `"compact"\|"full"` | `"full"` | M2 | AV outputsize. |
| `cache_dir` | `str` | `"data"` | M2 | Root for raw + canonical caches. |
| `store` | `"csv"\|"parquet"` | `"csv"` | M8 | Canonical storage backend. |
| `data_source` | `"rest"\|"mcp"` | `"rest"` | M12 | Gate MCP path without breaking REST. |
| `train_bars` / `val_bars` / `test_bars` / `step_bars` | `int` | — | M5 | `WalkForwardConfig` over the sample index. |
| `d_model` / `nhead` / `num_temporal_layers` / `num_cross_layers` / `dim_feedforward` / `dropout` | numeric | — | M5 | `TransformerRanker` hyperparameters. |
| `epochs` / `batch_size` / `learning_rate` | numeric | — | M5 | Optimiser. |
| `loss` | `"mse"\|"listnet"\|"approx_ndcg"` | `"mse"` | M10 | Training loss ablation. |
| `features` | `list[str]` | implicit v1 list | M9a | Explicit feature enumeration; bumps `feature_schema_hash`. |
| `sector_map_path` | `str` | `"configs/sector_map.yaml"` | M9b | Required when `label_mode == "sector_neutral_return"`. |
| `device` | `"auto"\|"cpu"\|"cuda"\|"mps"` | `"auto"` | M5 | Resolved by `resolve_device`. |
| `seed` | `int` | `42` | M5 | Seeds torch, numpy, DataLoaders. |
| `artifacts_dir` | `str` | `"artifacts"` | M5 | Root for per-run artifact directory. |
| `save_models` | `bool` | `false` | M5 | When true, write `model_state_fold_<id>.pt`. |
| `synthetic_n_bars` | `int` | `600` | M5 | Length of the synthetic panel under `--synthetic`. |

Unknown keys are tolerated but logged at load time (`configs/validate.py` is
future work; out of scope for M7).

## Objective definition

- At each timestamp `t`, the model observes a lookback window over the entire
  ticker universe and predicts which tickers will outperform / underperform
  over the next horizon.
- Baseline target for ticker `i` at timestamp `t`:
  - `raw_return(i, t) = close(i, t+1) / close(i, t) - 1`
  - `cross_sectional_return(i, t) = raw_return(i, t) - median_j(raw_return(j, t))`
- Prefer a **continuous score target** for ranking, with optional
  bucketization for classification:
  - Regression target: future cross-sectional return (v1).
  - Bucket target: top `q%`, middle bucket, bottom `q%` within the universe at
    timestamp `t` (`labels/cross_sectional.py::bucket_labels_by_quantile`).
- `label_mode` in YAML selects which target variant is used for loss and for
  the primary ranking metric.

## Ticker universe

- Universe defined in `configs/universe.yaml` (v1: fixed watchlist).
- Future sources (M8+): sector basket; point-in-time index membership.
- Filtering criteria (documented here, enforced progressively):
  - Minimum history length.
  - Minimum average volume.
  - Price floor.
  - Listing status at the relevant historical timestamp.
- Survivorship bias guardrails:
  - Do not train only on today's survivors when representing a historical
    benchmark — use the point-in-time membership table from M8.
  - Record the effective universe membership for every fold and timestamp range
    in `universe_membership.json` per run.
- Per-ticker gaps (holidays, halts, late listings, delistings) are handled via
  the mask — never forward-filled.
- `min_coverage_symbols` drops timestamps with insufficient live peers
  (see [§Conventions](#conventions)).

## Project scaffold

```
src/stock_transformer/
├── data/
│   ├── alphavantage.py        # REST client + per-symbol fetch + universe batch
│   ├── canonicalize.py        # AV payload → canonical OHLCV schema
│   ├── cache_paths.py         # raw + canonical cache paths
│   ├── align.py               # outer-join global timestamp alignment
│   ├── universe.py            # UniverseConfig, membership table
│   └── synthetic.py           # seedable fake candles for tests and CI
├── features/
│   ├── sequences.py           # single-symbol multi-timeframe token builder
│   └── universe_tensor.py     # [N, L, S, F] samples + masks
├── labels/
│   └── cross_sectional.py     # raw/relative forward returns, bucket labels
├── model/
│   ├── transformer_classifier.py  # single-symbol CandleTransformer (reference)
│   ├── transformer_ranker.py      # temporal + cross-sectional attention
│   └── baselines.py               # equal score, momentum rank
└── backtest/
    ├── walkforward.py         # fold generation + chronology checks
    ├── metrics.py             # regression + ranking metrics, aggregation
    ├── runner.py              # single-symbol experiment
    └── universe_runner.py     # universe experiment (primary)
configs/
├── default.yaml               # single-symbol reference
└── universe.yaml              # multi-ticker universe (primary)
tests/
├── test_sequences.py
├── test_multitimeframe.py
├── test_walkforward.py
├── test_data_integrity.py
├── test_runner_synthetic.py
├── test_cross_sectional_labels.py
├── test_universe_tensor.py
└── test_universe_runner_synthetic.py
```

## Alpha Vantage data plan

- **REST client (current):** `AlphaVantageClient.query` hits the public REST
  API, caches raw JSON, and emits canonical CSV via `canonicalize_*` helpers.
  Batching over a universe goes through `fetch_candles_for_universe`, which
  respects `min_interval_sec` throttling with retries and exponential backoff.
- **Storage path today:** canonical CSV under `data/canonical/<symbol>/<timeframe>.csv`.
  Good enough through M7. M8 introduces partitioned parquet behind a
  `store: csv|parquet` config flag with read-through compatibility:

  ```
  data/canonical/
    timeframe=daily/symbol=MSTR/part-000.parquet
    timeframe=daily/symbol=IBIT/part-000.parquet
  ```

- **Timeframe → endpoint mapping:**

  | `timeframe` value | AV function | Notes |
  |---|---|---|
  | `1min`, `5min`, `15min`, `30min`, `60min` | `TIME_SERIES_INTRADAY` | `interval` = value verbatim |
  | `daily` | `TIME_SERIES_DAILY` or `_ADJUSTED` (config-driven) | Adjusted = default for equities |
  | `weekly` | `TIME_SERIES_WEEKLY` or `_ADJUSTED` | |
  | `monthly` | `TIME_SERIES_MONTHLY` or `_ADJUSTED` | |

- **Canonical candle schema** (both REST and any future MCP path):
  `timestamp, symbol, timeframe, open, high, low, close, volume`.
- **Raw vs canonical:** raw payloads cached as received for replay; canonical
  outputs written as typed CSV (parquet in M8).
- **Rate limits:** honour `min_interval_sec`; a run fetching 4 symbols × 1
  timeframe should stay well under 5 calls/minute. Tests never hit the network
  (see [§Runtime baseline](#runtime-baseline)).
- **Universe-membership table (M8):** `timestamp_start, timestamp_end, symbol,
  active_flag, sector, market_cap_bucket` — supports point-in-time filtering
  and sector-neutral evaluation. v1 stub in `data/universe.py::membership_table_from_panel`.

## Leakage-safe dataset construction

- Build data on a **global timestamp index** per timeframe, not as isolated
  per-symbol samples.
- For each timestamp `t`:
  - Gather the full universe cross-section of tickers eligible at `t`.
  - Build a lookback tensor covering `[t-L+1 … t]` for all eligible symbols.
  - Predict each symbol's outcome from `t` to `t+1`.
- Core training object per sample:
  - `X_t ∈ [L, S, F]`
  - `mask_t ∈ [L, S]` bool — `True` means padded / invalid.
  - `y_t ∈ [S]` next-period cross-sectional return (or raw, per `label_mode`).
- Never forward-fill future information.
- Mask handles missing candles — only using information available up to `t`.
- Enforce `min_coverage_symbols` at each `t` (see [§Conventions](#conventions)).
- Build labels *after* the timestamp universe at `t` is fixed, so the ranking
  target is computed against the correct contemporaneous peer set.
- Symbol order deterministic per [§Conventions](#conventions).

## Feature construction

- **Per-ticker temporal features (implemented, v1):** OHLC log-returns vs
  previous close, `log1p(volume)`. `N_UNIVERSE_FEATURES = 5`.
- **Planned additional temporal features (M9):**
  - Realized log-return families at multiple horizons.
  - Rolling volatility.
  - Intraperiod range.
  - Volume change.
- **Cross-sectional features (M9):** at each timestamp, per symbol —
  percentile rank of return / volume / volatility within the live universe;
  z-score vs the cross-section; relative strength vs equal-weighted universe;
  relative volume vs median.
- **Static / slow metadata features (M9, when sector source lands):** sector,
  industry, market-cap bucket.
- **Ticker embedding:** included as one feature family; must not be the only
  per-ticker identity signal so the model cannot reduce to
  "single-ticker OHLCV + ticker ID".
- Every feature schema change bumps `feature_schema_hash` in
  `feature_schema.json` (see [§Per-run artifacts](#per-run-artifacts)).

## Walk-forward backtest protocol

- Rolling-origin evaluation: `train → val → test`, advance by a fixed step,
  repeat. Driven by `WalkForwardConfig(train_bars, val_bars, test_bars, step_bars)`.
- Splits are **global in time**: same calendar cutoffs for every ticker.
- Universe membership inside each fold is determined using information
  available at that fold's timestamps only.
- For each fold:
  - Build train/val/test tensors from all eligible timestamps and symbols.
  - Fit scaling / normalization on the **training cross-section only** (M9).
  - Train the model to score the full universe at each timestamp.
  - Tune thresholds / hyperparameters on validation only.
  - Freeze parameters and report test metrics.
- Aggregate metrics across folds with mean / std and per-fold breakdowns.
- Also report:
  - Per-ticker breakdowns.
  - Per-sector breakdowns (M9).
  - Metrics by universe size and coverage level.

```mermaid
flowchart LR
  universe[PointInTimeUniverse] --> rawData[AlphaVantageData]
  rawData --> canonicalCandles[CanonicalCandles_AllTickers]
  universe --> membership[UniverseMembershipTable]
  canonicalCandles --> align[GlobalTimestampAlignment]
  membership --> align
  align --> tensors[UniverseTensors_N_L_S_F]
  tensors --> labels[CrossSectionalLabels]
  labels --> folds[RollingOriginFolds]
  folds --> model[TransformerRanker]
  model --> eval[RankAndBucketEvaluation]
  eval --> metrics[AggregateMetrics]
```

## Model and training plan

- v1 architecture (`model/transformer_ranker.py`):
  - Per-symbol **temporal encoder** (causal Transformer) over the lookback window.
  - **Cross-sectional attention** block over symbol representations at `t`.
  - Prediction head: one score per ticker.
- Causal masking over time ensures information from `t+1` never leaks into the
  representation at `t`.
- Symbol masks ensure absent tickers do not contaminate attention or loss.
- **v1 loss:** masked MSE on the cross-sectional target (`label_mode` selects
  the target variant). Ranking losses are an explicit ablation in M10 under
  `loss: mse|listnet|approx_ndcg`.
- Ablation ladder (kept for comparability):
  - Temporal-only per symbol (single-asset head).
  - Temporal + ticker embedding.
  - Temporal + full cross-sectional attention (v1 primary).

## Baselines

- **Equal-score** — no relative edge (implemented: `model/baselines.py::equal_score_baseline`).
- **Momentum rank** — rank by recent trailing return (implemented: `momentum_rank_scores`).
- **Mean-reversion rank** — rank by negative trailing return
  (M6b, `model/baselines.py::mean_reversion_rank_scores`).
- **Linear cross-sectional** — ridge on flattened tabular features
  (M6b, `model/baselines_tabular.py::fit_linear_cs_ranker`).
- **Gradient-boosted tree ranker** — LightGBM on flattened tabular features
  (M6b, `model/baselines_tabular.py::fit_gbt_ranker`).

## Evaluation outputs (forecasting only)

- **Primary ranking metrics** (per timestamp, averaged across folds):
  - Spearman rank correlation of predicted scores vs realized next-period returns.
  - Kendall rank correlation.
  - Top-k hit rate.
  - Precision@k / Recall@k for top-bucket prediction.
  - NDCG@k.
- **Secondary metrics:**
  - Bucket classification accuracy (when bucket labels are used).
  - Regression MAE / RMSE for relative return.
  - Calibration diagnostics if scores are converted to probabilities.
- **Diagnostic breakdowns:**
  - Per-ticker.
  - Per-sector (M9).
  - Per-timeframe.
  - Per-fold.
  - By market regime (future).
- **Predictions table (`predictions_universe.csv`), long format** (M7b):
  columns `timestamp, symbol, timeframe, y_true_raw_return,
  y_true_relative_return, y_score, y_rank_pred, y_rank_true, fold_id`.
  Ranks are computed per `(fold_id, timestamp)` over the finite subset of
  `y_score` (resp. `y_true_*`) using `scipy.stats.rankdata(method="average")`,
  **descending** (higher value → rank `1`). NaN score/truth → NaN rank.

  Example (3 symbols, 1 timestamp, fold 0):

  ```csv
  timestamp,symbol,timeframe,y_true_raw_return,y_true_relative_return,y_score,y_rank_pred,y_rank_true,fold_id
  2024-01-02,MSTR,daily,0.021,0.008,0.014,1,1,0
  2024-01-02,IBIT,daily,0.013,0.000,0.009,2,2,0
  2024-01-02,COIN,daily,-0.011,-0.024,-0.006,3,3,0
  ```

  *(The current runner writes a wide variant; M7b migrates to this long schema
  and keeps a golden-file test comparing against a frozen fixture.)*

## Guardrails and tests

- **Reproducibility checks:**
  - Fixed random seeds.
  - Config snapshot saved with each run.
  - Universe snapshot saved for each fold.
  - Feature schema hash saved with each run.
- **Data integrity checks:**
  - Monotonic timestamps per symbol and on the aligned panel.
  - No duplicated `(symbol, timeframe, timestamp)` keys.
  - Cross-sectional label distributions are sensible (finite mean, bounded variance).
  - Universe coverage and live-symbol count reported for every fold.

### Leakage test matrix

All live in `tests/test_leakage_universe.py` (M7a). Each test uses synthetic
data from `data.synthetic.synthetic_universe_candles` so no network calls are
needed.

**Shared fixture** (pytest fixture `universe_panel`):

```python
symbols = ("MSTR", "IBIT", "COIN", "QQQ")
candles = synthetic_universe_candles(
    n_bars=400, symbols=symbols, timeframe="daily", seed=7,
)
panel, close = align_universe_ohlcv(candles, symbols)
```

For `test_pit_universe_membership`, `IBIT` is additionally masked to be
absent for rows `[0, 100)` by setting its OHLCV to NaN on those rows before
alignment.

| Test | Assertion |
|---|---|
| `test_features_do_not_reference_future` | Scramble panel rows `> t`; rebuild samples; `X` is bitwise identical. |
| `test_label_uses_only_t_to_tplus1` | Perturb `close[t+2]`; `y[t]` unchanged. |
| `test_fold_boundaries_monotonic` | For every fold: `max(ts[train]) < min(ts[val]) < min(ts[test])`. |
| `test_pit_universe_membership` | Symbol listed only from `t=100`; all samples with `t<100` have `mask[:, :, sym] == True`. |
| `test_coverage_drop` | Force `live.sum() < min_coverage_symbols` for a row; that row is absent from samples. |
| `test_deterministic_symbol_order` | Shuffle YAML `symbols`; after re-sorting outputs, results identical. |
| `test_train_scaling_fit_on_train_only` | (M9a) Scaler statistics depend only on the train slice. Skipped in M7a, unskipped when M9a lands. |
| `test_target_symbol_not_required_live` | Row where only MSTR is masked survives; MSTR metrics are NaN, peers are finite. |

## CLI contract

- Entrypoint: `stx-backtest [-c CONFIG] [--synthetic]`.
- Dispatch by `experiment_mode` in YAML (landing in M7b — today's CLI always
  calls the single-symbol runner):
  - `"single_symbol"` (or absent) → `backtest/runner.py::run_from_config_path`.
  - `"universe"` → `backtest/universe_runner.py::run_universe_from_config_path`.
- Exit codes (final target, M7b):
  - `0` — success.
  - `1` — missing / unreadable config.
  - `2` — at least one fold raised; `summary.json["error"]` is populated and
    partial results for completed folds are still written.
- `summary.json` is always written, even on partial failure.
- `--synthetic` forces the synthetic data path for both modes (CI / smoke test).
- On success, the CLI prints exactly one line to stdout:
  `Run complete. Artifacts: <run_dir>` (machine-parseable by downstream jobs).

## Runtime baseline

- Python ≥ 3.11 *(target; `pyproject.toml` currently pins `>=3.10`. M6b bumps it.)*
- PyTorch ≥ 2.2, numpy ≥ 1.26, pandas ≥ 2.2, pyyaml ≥ 6 *(target; M6b bumps pins
  in `pyproject.toml` from the current 2.0 / 1.24 / 2.0 / 6.0).*
- M6b adds scikit-learn ≥ 1.4 and lightgbm ≥ 4.3 (pinned in `pyproject.toml`).
- M8 adds pyarrow ≥ 15.
- Apple Silicon: `device: "auto"` prefers MPS; ops unsupported on MPS fall
  back to CPU with a single-line warning at model build time.
- **No network calls in tests.** Live Alpha Vantage requests are gated behind
  `ALPHAVANTAGE_API_KEY`; the client raises at construction time if the env
  var is missing and the config is not `--synthetic`.

## Per-run artifacts

Every run writes to `artifacts/universe_run_<UTC-timestamp>/`:

| File | Contents |
|---|---|
| `config_snapshot.yaml` | Fully merged effective config (defaults + overrides). |
| `universe_membership.json` | Per-symbol first/last valid row on the aligned panel. |
| `feature_schema.json` | Feature names, `N_UNIVERSE_FEATURES`, git SHA, `feature_schema_hash`. |
| `folds.json` | `fold_id → {train, val, test}` timestamp ranges and sample indices. |
| `summary.json` | Aggregate + per-fold metrics, aggregate baselines, run metadata. |
| `predictions_universe.csv` | Long-format predictions (see [§Evaluation outputs](#evaluation-outputs-forecasting-only)). |
| `model_state_fold_<id>.pt` | Optional; off by default, enable via `save_models: true`. |

## Milestone tracker

> This is the authoritative status. Tick boxes as milestones land.
> Each open milestone has explicit **Definition of Done (DoD)**: artifacts to
> ship, commands that must succeed, and tests that must pass.

- [x] **M1 — Universe config + point-in-time stub.** `configs/universe.yaml`,
      `data/universe.py::UniverseConfig`, `membership_table_from_panel`.
- [x] **M2 — Batch ingest + global alignment.** `fetch_candles_for_universe`,
      `data/align.py::align_universe_ohlcv`.
- [x] **M3 — Universe tensor assembly.** `features/universe_tensor.py`
      (`[N, L, S, F]` + mask).
- [x] **M4 — Cross-sectional targets.** `labels/cross_sectional.py`
      (median demean + bucket labels).
- [x] **M5 — Walk-forward universe runner.** `backtest/universe_runner.py`,
      extended `backtest/metrics.py` with Spearman / NDCG@k / top-k hit rate.
- [x] **M6a — Simple baselines.** Equal-score + momentum rank in
      `model/baselines.py`, wired into the runner.

---

> **Template for each open milestone.** Every open milestone below uses this
> shape so an agent can execute it without guessing:
>
> - **Files** — new / edited paths and module-level signatures.
> - **Config keys** — YAML additions (see [§Config reference](#config-reference)).
> - **Tests** — files + assertion summary.
> - **Commands** — must succeed on a clean checkout.
> - **Artifacts** — what changes under `artifacts/universe_run_*`.
> - **Out of scope** — explicit non-goals.

---

- [ ] **M6b — Tabular baselines.**
  - **Files:**
    - `features/tabular.py` (new):
      ```python
      def flatten_universe_sample(
          X: np.ndarray,          # [N, L, S, F]  float32
          mask: np.ndarray,        # [N, L, S]    bool, True = padded
          y: np.ndarray,           # [N, S]        float32, NaN = no label
      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
          """Return (X_flat, y_flat, group_ids, sym_ids).

          One row per (n, s) where mask[n, -1, s] is False AND isfinite(y[n, s]).
          X_flat.shape == (M, L*F); group_ids[m] == n; sym_ids[m] == s.
          """
      ```
    - `model/baselines_tabular.py` (new):
      ```python
      def fit_linear_cs_ranker(
          X_tr, y_tr, groups_tr, *, alpha: float = 1.0
      ) -> "LinearCSRanker": ...

      def fit_gbt_ranker(
          X_tr, y_tr, groups_tr, *, params: dict | None = None
      ) -> "GBTRanker": ...
      # Both expose .predict(X) -> np.ndarray[M] used to rebuild [N, S] scores.
      ```
    - `model/baselines.py` (edit): add `mean_reversion_rank_scores(close, end_rows, lookback)`.
    - `backtest/universe_runner.py` (edit): call new baselines per fold.
    - `pyproject.toml` (edit): bump `scikit-learn>=1.4`, add `lightgbm>=4.3`,
      bump `numpy>=1.26`, `pandas>=2.2`, `torch>=2.2`, `requires-python=">=3.11"`.
  - **Config keys:** none new (baselines always run).
  - **Tests:** `tests/test_baselines_tabular.py`
    - `test_flatten_round_trip_shape`: `X_flat.shape[0] == (~mask[:,-1,:] & isfinite(y)).sum()`.
    - `test_linear_ranker_beats_equal_on_synthetic`: Spearman > 0 on seeded toy data.
    - `test_gbt_ranker_runs_on_synthetic`: no NaN in predictions; finite Spearman.
    - `test_mean_reversion_baseline_is_negative_of_momentum` on a pure-momentum synthetic panel.
  - **Commands:**
    - `pytest tests/test_baselines_tabular.py -v`
    - `stx-backtest --synthetic -c configs/universe.yaml`
  - **Artifacts:** `summary.json` per-fold row gains
    `baseline_linear_spearman_mean`, `baseline_gbt_spearman_mean`,
    `baseline_mean_reversion_spearman_mean`; aggregates mirror these.
  - **Out of scope:** ranking-specific losses (→ M10), cross-sectional features (→ M9a).

---

- [ ] **M7a — Leakage test matrix.**
  - **Files:** `tests/test_leakage_universe.py` — the full matrix from
    [§Leakage test matrix](#leakage-test-matrix), using the shared fixture.
  - **Config keys:** none.
  - **Tests:** all rows of the matrix are present, parametrised where natural
    (one test function per row). Tests that depend on M9a (`test_train_scaling_fit_on_train_only`)
    are marked `pytest.mark.skip(reason="awaits M9a")` with a TODO pointing at M9a.
  - **Commands:** `pytest tests/test_leakage_universe.py -v`.
  - **Artifacts:** none (test-only).
  - **Out of scope:** schema migration (→ M7b).

---

- [ ] **M7b — Predictions long schema + run artifacts.**
  - **Files:**
    - `backtest/universe_runner.py` (edit): emit long-format
      `predictions_universe.csv` as specified in
      [§Evaluation outputs](#evaluation-outputs-forecasting-only); drop the
      wide format; also emit `feature_schema.json` and `folds.json`.
    - `features/universe_tensor.py` (edit): add
      `feature_schema() -> dict` returning `{"features": [...], "n": N_UNIVERSE_FEATURES, "hash": <sha256>}`.
    - `cli.py` (edit): dispatch on `experiment_mode`; exit `2` on fold exception
      with partial `summary.json` still written; keep the one-line success print.
  - **Config keys:** `save_models` (bool, default `false`).
  - **Tests:**
    - `tests/test_predictions_schema.py`: golden-file comparison against
      `tests/golden/predictions_universe.csv`; round-trip ranks are inverse of
      descending `rankdata(method="average")`.
    - `tests/test_cli_dispatch.py`: `experiment_mode: "universe"` routes to
      `universe_runner`; missing config → exit `1`; forced fold failure → exit `2`
      with `summary.json` present.
    - `tests/test_run_artifacts.py`: every run writes
      `{config_snapshot.yaml, universe_membership.json, feature_schema.json, folds.json, summary.json, predictions_universe.csv}`.
  - **Commands:**
    - `pytest -v`
    - `stx-backtest --synthetic -c configs/universe.yaml` (exit 0)
  - **Artifacts:** `feature_schema.json` and `folds.json` added; predictions file is long-format.
  - **Out of scope:** parquet store (→ M8).

---

- [ ] **M8 — Partitioned parquet store + richer membership.**
  - **Files:**
    - `data/store.py` (new): `CandleStore(backend: Literal["csv","parquet"])`
      with `.write(symbol, timeframe, df)` and `.read(symbol, timeframe) -> df`.
      Parquet layout: `data/canonical/timeframe=<tf>/symbol=<sym>/part-000.parquet`.
    - `data/universe.py` (edit): extend `membership_table_from_panel` to accept
      an optional `listings: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]`
      and emit rows `timestamp_start, timestamp_end, symbol, active_flag, sector, market_cap_bucket`.
    - `pyproject.toml`: add `pyarrow>=15`.
  - **Config keys:** `store` (default `"csv"`).
  - **Tests:** `tests/test_candle_store.py` — write→read round-trip bit-equal on
    a synthetic candle df for both backends; `tests/test_membership_richer.py`.
  - **Commands:** `pytest tests/test_candle_store.py tests/test_membership_richer.py -v`;
    `stx-backtest -c configs/universe.yaml` with `store: parquet`.
  - **Artifacts:** no change to artifact layout.
  - **Out of scope:** sector features (→ M9).

---

- [ ] **M9a — Cross-sectional features + train-only scaler.**
  - **Files:**
    - `features/cross_sectional.py` (new): `percentile_rank`, `zscore`,
      `relative_strength`, `relative_volume` — each `(panel, symbols) -> np.ndarray[n_rows, n_symbols]`.
    - `features/universe_tensor.py` (edit): `build_universe_samples` accepts
      `features: list[str]` and composes the feature stack accordingly;
      `N_UNIVERSE_FEATURES` becomes a derived constant `len(features)`.
    - `features/scaling.py` (new):
      ```python
      class TrainOnlyScaler:
          def fit(self, X: np.ndarray, mask: np.ndarray) -> "TrainOnlyScaler": ...
          def transform(self, X: np.ndarray) -> np.ndarray: ...
      # Statistics computed over (n, l, s) where ~mask; per-feature mean/std.
      ```
      Wired into `universe_runner.py` per fold — fit on train only, apply to val/test.
  - **Config keys:** `features` (list of feature names; default is today's 5-feature list).
  - **Tests:**
    - Unit tests for each feature function (shape, NaN propagation).
    - `tests/test_leakage_universe.py::test_train_scaling_fit_on_train_only` — unskip from M7a.
    - `tests/test_feature_schema_hash.py`: swapping `features` order or content changes the hash.
  - **Commands:** `pytest -v`; `stx-backtest --synthetic -c configs/universe.yaml`.
  - **Artifacts:** `feature_schema.json` reflects new list + hash; `summary.json` unchanged.
  - **Out of scope:** sector-neutral labels (→ M9b).

---

- [ ] **M9b — Sector-neutral labels + sector map.**
  - **Files:**
    - `configs/sector_map.yaml` (new):
      ```yaml
      version: 1
      default_sector: "Unknown"
      mapping:
        MSTR: "Information Technology"
        IBIT: "Crypto"
        COIN: "Financials"
        QQQ:  "Index"
      ```
    - `labels/cross_sectional.py` (edit): extend `cross_sectional_targets` with
      `"equal_weighted_return"` (nanmean) and `"sector_neutral_return"`
      (nanmedian within-sector, requires `sectors: np.ndarray[S]`).
    - `data/universe.py` (edit): `load_sector_map(path) -> dict[str, str]`,
      `sectors_for_symbols(symbols, sector_map) -> np.ndarray[S]`.
    - `backtest/universe_runner.py` (edit): build per-sector breakdowns in summary.
  - **Config keys:** `label_mode` admits two more values; `sector_map_path`.
  - **Tests:** `tests/test_sector_neutral_labels.py` — synthetic 2-sector panel
    where within-sector median demeaning zeros out the mean per sector-row.
    `tests/test_summary_per_sector.py` — `summary.json["aggregate"]["per_sector"]` exists.
  - **Commands:** `pytest -v`; `stx-backtest --synthetic -c configs/universe.yaml`
    with `label_mode: sector_neutral_return`.
  - **Artifacts:** `summary.json` gains `aggregate.per_sector` block.
  - **Out of scope:** dynamic/time-varying sector membership (future).

---

- [ ] **M10 — Ranking loss ablation.**
  - **Files:**
    - `model/losses.py` (new):
      ```python
      def masked_mse(pred: Tensor, target: Tensor) -> Tensor: ...
      def listnet_loss(pred: Tensor, target: Tensor, *, mask: Tensor) -> Tensor: ...
      def approx_ndcg_loss(pred: Tensor, target: Tensor, *, mask: Tensor, alpha: float = 10.0) -> Tensor: ...
      ```
      All operate on `pred/target: [N, S]` and a label mask
      (`isfinite(target) & ~padded`). `listnet_loss` uses softmax over valid entries.
    - `backtest/universe_runner.py` (edit): dispatch via `loss` config; move the
      inline `_masked_mse` into `model/losses.py`.
  - **Config keys:** `loss: mse|listnet|approx_ndcg` (default `"mse"`).
  - **Tests:** `tests/test_ranking_losses.py` — gradient sanity; a perfect
    ranker drives each loss to a known minimum on a small fixture.
  - **Commands:** `pytest tests/test_ranking_losses.py -v`; three synthetic
    runs, one per `loss`, and `summary.json["by_loss"]` aggregates both.
  - **Artifacts:** `summary.json` gains top-level `by_loss` when multiple losses
    are run via a dedicated sweep script (`scripts/sweep_loss.py`, also new).
  - **Out of scope:** portfolio simulation (→ M11).

---

- [ ] **M11 — (Future) Portfolio construction + trading simulation.**
  - Gated on M9a/M9b+M10 producing ranking quality demonstrably above baselines
    on ≥ 2 walk-forward horizons.
  - Scope: top-k long-only and long/short books; transaction-cost stub;
    turnover reporting. No code to be written before gate is met.

---

- [ ] **M12 — (Optional) MCP AlphaVantage path.**
  - Parallel branch for schema discovery via `TOOL_LIST` / `TOOL_GET` / `TOOL_CALL`.
  - Canonical candle schema must be byte-identical to the REST path
    (`tests/test_mcp_rest_parity.py` — same seed, same symbols, same timeframe → equal panels).
  - Gated behind `data_source: rest|mcp` in YAML; default remains `"rest"`.

## Pilot: small-universe drill-down for `MSTR`

The pilot is a **small-universe special case of the v1 ranker**, not a separate
direction classifier. A 3–4 symbol basket (target + predictors) is used to
smoke-test the full pipeline — tensor layout, masking, alignment, leakage
rules, metrics — before scaling the watchlist. `target_symbol: "MSTR"` is a
**reporting / drill-down key only**: the model trains on the full
cross-section and loss ignores it unless `loss` is switched by a future
ablation.

A dedicated MSTR-directional head is explicitly **not** part of v1; it is
listed as an optional ablation under M10 and only begins if the cross-sectional
ranker fails to outperform baselines.

**Suggested predictor tickers (verify availability on your Alpha Vantage plan):**

| Symbol | Role |
|--------|------|
| **IBIT** | Spot Bitcoin ETF — tight link to the primary driver of MSTR's asset value narrative. |
| **COIN** | Crypto exchange equity — high beta to crypto cycles and liquidity. |
| **QQQ**  | Nasdaq-100 proxy — tech / risk-on regime that often co-moves with speculative growth names. |

Reasonable alternatives if a symbol is missing or illiquid on your feed:
**GBTC** (Bitcoin trust), **MARA** or **RIOT** (miners, higher idiosyncratic noise).
For v1, prefer **IBIT + COIN** (2 tickers) and add **QQQ** as a third once
alignment and masking are stable.

**Modeling note:** tensor layout, masking, loss, metrics, and leakage rules
are identical to the full-universe configuration; only `len(symbols)` changes.
Scaling from the pilot basket to a full watchlist must not require any code
changes outside `configs/universe.yaml`.
