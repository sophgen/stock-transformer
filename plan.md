# Stock Transformer Cross-Sectional Backtest Plan

> **Authoritative status lives in [§Milestone tracker](#milestone-tracker).**
> Everything else in this document is the *target design*. When code and plan
> disagree on a completed milestone, code wins and the plan is updated to match.

## Architecture at a glance

The project treats candles as tokens for a Transformer, but lifts the idea from
a single ticker to a **panel of tickers** so the model has a contemporaneous
peer set at every timestamp.

- A **token** is one OHLCV bar for one `(symbol, timeframe)` at one timestamp,
  encoded as log-returns relative to the prior bar close plus `log1p(volume)`.
- A **sample** at prediction timestamp `t` is the full `[L, S, F]` block of
  tokens: `L` timestamps of lookback × `S` symbols × `F` features, with a
  boolean mask marking padding / invalid positions.
- The model is a **temporal Transformer per symbol** (causal over `L`) followed
  by a **cross-sectional Transformer over symbols** at the final step, emitting
  one score per ticker per timestamp.
- Training supervises those scores against **cross-sectional forward returns**
  (`y[t, s] = close[t+1, s] / close[t, s] - 1`, optionally demeaned by the
  live-peer nanmedian at `t`).
- Evaluation is **ranking-first** (Spearman, NDCG@k, top-k hit rate) and
  regression-secondary (MAE / RMSE). There is **no PnL simulation in v1**.

Two prediction heads exist in the repo:

| Head | Status | Purpose |
|---|---|---|
| `TransformerRanker` (universe, multi-ticker) | **v1 primary** | Predicts cross-sectional scores over `S` symbols at each `t`. |
| `CandleTransformer` (single-symbol autoregressive, multi-timeframe tokens) | Reference / ablation | Predicts next-candle OHLCV log-returns + direction for one ticker. Kept for comparability; not on the critical path. |

If a future ablation needs a per-ticker "up/down + magnitude" directional
head, it plugs in alongside `CandleTransformer` without changing the universe
pipeline (see [§Pilot](#pilot-small-universe-drill-down-for-mstr)).

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
| **Sample index** | The ordered sequence of prediction timestamps kept by `build_universe_samples` after `min_coverage_symbols` filtering. `N = len(sample_index)`. All `*_bars` / `step_bars` YAML keys count **positions in this index**, not raw panel rows. |
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

## Invariants (cross-module contracts)

Short checklist any new module must preserve. Each invariant has a test that
covers it (test names in parentheses).

1. **No future leakage in features.** Features for sample at `t` depend only
   on panel rows `≤ t`. (`test_features_do_not_reference_future`, M7a)
2. **Label indexing is `t → t+1`.** `y[t, s]` never reads `close[t+k, s]` for
   `k ≥ 2`. (`test_label_uses_only_t_to_tplus1`, M7a)
3. **Fold chronology.** For every fold `f`:
   `max(ts[train_f]) < min(ts[val_f]) < min(ts[test_f])`.
   (`test_fold_boundaries_monotonic`, M7a; already enforced at runtime by
   `backtest/walkforward.py::assert_fold_chronology`)
4. **Mask polarity is uniform.** Every tensor named `mask` or `padding_mask`
   uses `True = padding / invalid`. Label validity is tracked separately as
   `isfinite(y)`.
5. **Symbol axis is immutable and deterministic.** Order comes from YAML,
   uppercased at load, snapshotted to `universe_membership.json`; shuffling
   YAML order and re-sorting outputs yields identical results.
   (`test_deterministic_symbol_order`, M7a)
6. **Coverage rule is a hard filter, not a soft weight.** A sample at `t` is
   produced iff `sum_s isfinite(raw_return[t, s]) ≥ min_coverage_symbols`.
   (`test_coverage_drop`, M7a)
7. **`target_symbol` is reporting-only.** Drops / metrics for the target key
   are emitted but never affect training-set construction.
   (`test_target_symbol_not_required_live`, M7a)
8. **Train-only normalisation.** Any scaler fits on train-fold statistics only
   and applies frozen to val / test. (`test_train_scaling_fit_on_train_only`,
   skipped until M9a)
9. **Reproducibility.** Given a config + seed, `summary.json["aggregate"]` is
   bit-stable on the same hardware device.
10. **`summary.json` is always written**, even on partial failure (exit code
    `2`).

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
| `bucket_q` | `float ∈ (0, 0.5)` | `0.33` | M4 | Top/bottom quantile fraction for `bucket_labels_by_quantile`; diagnostic-only in v1. |
| `use_adjusted_daily` / `use_adjusted_weekly` / `use_adjusted_monthly` | `bool` | `true` | M2 | Select AV adjusted endpoint. |
| `intraday_month` | `str \| null` | `null` | M2 | AV `month=YYYY-MM` slice for intraday. |
| `intraday_extended_hours` | `bool` | `false` | M2 | Include pre/post-market bars. |
| `intraday_outputsize` | `"compact"\|"full"` | `"full"` | M2 | AV outputsize. |
| `daily_outputsize` | `"compact"\|"full"` | `"full"` | M2 | AV outputsize. |
| `cache_dir` | `str` | `"data"` | M2 | Root for raw + canonical caches. |
| `store` | `"csv"\|"parquet"` | `"csv"` | M8 | Canonical storage backend. |
| `data_source` | `"rest"\|"mcp"` | `"rest"` | M12 | Gate MCP path without breaking REST. |
| `train_bars` / `val_bars` / `test_bars` / `step_bars` | `int` | — | M5 | `WalkForwardConfig` lengths; **units are sample-index positions, not raw bars** (see [§Glossary](#glossary)). Name kept for backward compatibility with existing configs / tests. |
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
  - Bucket target: three buckets within the universe at timestamp `t` via
    `labels/cross_sectional.py::bucket_labels_by_quantile(values, q=bucket_q)`.
    Encoding (per the implementation):
    - `2` — top `bucket_q` fraction (best realized return).
    - `1` — middle bucket.
    - `0` — bottom `bucket_q` fraction.
    - `NaN` — fewer than 3 finite peers at that row.

    `bucket_q ∈ (0, 0.5)` (default `0.33`). Buckets are diagnostics only in
    v1 — training loss operates on the continuous target.
- `label_mode` in YAML selects which continuous target variant is used for
  loss and for the primary ranking metric.

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

**Current layout** (what exists on disk today, through M6a). Files added by
later milestones are listed under "Planned additions" and in each milestone's
**Files** section.

```
src/stock_transformer/
├── cli.py                     # stx-backtest (exit 2 on partial fold failure)
├── data/
│   ├── alphavantage.py        # REST + optional MCP unwrap + partitioned store
│   ├── canonicalize.py        # AV payload → canonical OHLCV schema
│   ├── mcp_canonicalize.py    # unwrap MCP-wrapped AV JSON
│   ├── store.py               # CandleStore csv|parquet under canonical/
│   ├── cache_paths.py         # raw + legacy canonical cache paths
│   ├── align.py               # outer-join global timestamp alignment
│   ├── universe.py          # UniverseConfig, sector map, membership table
│   └── synthetic.py           # seedable fake candles for tests and CI
├── features/
│   ├── sequences.py           # single-symbol multi-timeframe token builder
│   ├── universe_tensor.py     # [N, L, S, F] samples + masks + feature_schema
│   ├── cross_sectional.py     # optional CS numeric planes for the tensor
│   ├── scaling.py             # TrainOnlyScaler (train-fold fit)
│   └── tabular.py             # flatten_universe_sample for baselines
├── labels/
│   └── cross_sectional.py     # raw / CS / EW / sector-neutral targets
├── model/
│   ├── transformer_classifier.py  # single-symbol CandleTransformer (reference)
│   ├── transformer_ranker.py      # temporal + cross-sectional attention
│   ├── baselines.py               # equal, momentum, mean-reversion ranks
│   ├── baselines_tabular.py       # ridge + hist GBDT on flat windows
│   └── losses.py                  # mse, listnet, approx_ndcg
└── backtest/
    ├── walkforward.py         # fold generation + chronology checks
    ├── metrics.py             # regression + ranking metrics, per-sector
    ├── runner.py              # single-symbol experiment
    └── universe_runner.py     # universe experiment (primary)
configs/
├── default.yaml               # single-symbol reference
├── universe.yaml              # multi-ticker universe (primary)
└── sector_map.yaml            # static sector tags (M9b)
scripts/
└── sweep_loss.py              # optional multi-loss summary (M10)
tests/
├── golden/predictions_universe.csv
├── fixtures/av_raw/           # MCP parity JSON
├── test_sequences.py
├── test_multitimeframe.py
├── test_walkforward.py
├── test_data_integrity.py
├── test_runner_synthetic.py
├── test_cross_sectional_labels.py
├── test_universe_tensor.py
├── test_universe_runner_synthetic.py
├── test_baselines_tabular.py
├── test_leakage_universe.py
├── test_predictions_schema.py
├── test_cli_dispatch.py
├── test_run_artifacts.py
├── test_candle_store.py
├── test_membership_richer.py
├── test_feature_schema_hash.py
├── test_sector_neutral_labels.py
├── test_summary_per_sector.py
├── test_ranking_losses.py
└── test_mcp_rest_parity.py
```

**Planned additions by milestone** (each milestone below is authoritative):

| Milestone | Paths |
|---|---|
| M6b | `src/stock_transformer/features/tabular.py`, `src/stock_transformer/model/baselines_tabular.py`, `tests/test_baselines_tabular.py` |
| M7a | `tests/test_leakage_universe.py` |
| M7b | `tests/test_predictions_schema.py`, `tests/test_cli_dispatch.py`, `tests/test_run_artifacts.py`, `tests/golden/predictions_universe.csv` |
| M8  | `src/stock_transformer/data/store.py`, `tests/test_candle_store.py`, `tests/test_membership_richer.py` |
| M9a | `src/stock_transformer/features/cross_sectional.py`, `src/stock_transformer/features/scaling.py`, `tests/test_feature_schema_hash.py` |
| M9b | `configs/sector_map.yaml`, `tests/test_sector_neutral_labels.py`, `tests/test_summary_per_sector.py` |
| M10 | `src/stock_transformer/model/losses.py`, `scripts/sweep_loss.py`, `tests/test_ranking_losses.py` |
| M12 | `src/stock_transformer/data/mcp_canonicalize.py` + `tests/test_mcp_rest_parity.py`, `tests/fixtures/av_raw/*.json` |

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

Two layers: **numeric features** (per `(t, s)` floats that enter the `[L, S, F]`
tensor and are listed in the YAML `features` key from M9a onward) and **model
components** (always-on identity signals inside the Transformer that are
independent of the feature list).

### Numeric features (tensor columns, counted by `F`)

- **Per-ticker temporal features (implemented, v1):** OHLC log-returns vs
  previous close, `log1p(volume)`. `N_UNIVERSE_FEATURES = 5`.
- **Planned additional temporal features (M9a):**
  - Realized log-return families at multiple horizons.
  - Rolling volatility.
  - Intraperiod range.
  - Volume change.
- **Cross-sectional features (M9a):** at each timestamp, per symbol —
  percentile rank of return / volume / volatility within the live universe;
  z-score vs the cross-section; relative strength vs equal-weighted universe;
  relative volume vs median.
- **Static / slow metadata features (M9b, when sector source lands):** sector
  one-hot, market-cap bucket one-hot. Industry is optional.

### Model components (always on, not counted by `F`)

- **Ticker embedding** (`nn.Embedding(n_symbols, d_model)` inside
  `TransformerRanker.sym_embed`): injects per-symbol identity into the temporal
  encoder. It is a **model component**, not a YAML-listed feature; it does not
  contribute to `F` or to `feature_schema_hash`.
- **Temporal positional embedding** (`nn.Embedding(max_seq_len, d_model)`):
  same treatment — model-internal, not feature-listed.

> **Guardrail:** numeric features must carry cross-sectional signal on their
> own. The ticker embedding alone should not let the model reduce to
> "single-ticker OHLCV + ticker ID"; the M9a cross-sectional features
> (percentile rank, z-score, relative strength) are the explicit fix.

### Schema hashing

- `feature_schema_hash = sha256(json.dumps(feature_names, sort_keys=True))[:16]`
  is computed over the **numeric feature list only** (not model components).
- Landing in M7b: `features/universe_tensor.py::feature_schema() -> dict`
  returns `{"features": [...], "n": F, "hash": "<16 hex>"}`; the runner stamps
  this to `feature_schema.json` on every run.
- Any change to the numeric feature list bumps the hash and invalidates cached
  tensors.

## Walk-forward backtest protocol

- Rolling-origin evaluation: `train → val → test`, advance by a fixed step,
  repeat. Driven by `WalkForwardConfig(train_bars, val_bars, test_bars, step_bars)`.
- Splits are **global in time**: same calendar cutoffs for every ticker.
- Universe membership inside each fold is determined using information
  available at that fold's timestamps only.
- For each fold:
  - Build train/val/test tensors from all eligible timestamps and symbols.
  - Fit scaling / normalization on the **training cross-section only** (M9a).
  - Train the model to score the full universe at each timestamp.
  - Tune thresholds / hyperparameters on validation only (v1: early-stopping
    by best val MSE; richer HP search is out of scope).
  - Freeze parameters and report test metrics.
- Aggregate metrics across folds with mean / std and per-fold breakdowns.
- Also report:
  - Per-ticker breakdowns.
  - Per-sector breakdowns (M9b).
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
  - **Spearman** rank correlation of predicted scores vs realized next-period
    returns. Skipped for rows where `isfinite(score) & isfinite(y)` has fewer
    than `min_coverage_symbols` entries or where either side has zero variance.
  - **Kendall** rank correlation (M7b; same filter as Spearman).
  - **Top-k hit rate** — `|top_k(score) ∩ top_k(y_true_raw_return)| > 0` counted
    per row over the finite subset; reported as a fraction of valid rows.
    Ties broken by original index order (argsort stable).
  - **Precision@k / Recall@k** for top-bucket prediction (where bucket label
    is computed via `bucket_labels_by_quantile`).
  - **NDCG@k** — relevance = `max(0, y_true_raw_return − min_live(y_true_raw_return))`
    shifted to non-negative per row, DCG `= Σ rel_i / log2(i + 2)`, normalised
    by the ideal DCG on the same row; returns `NaN` when ideal DCG is `0`.
- **Secondary metrics:**
  - Bucket classification accuracy — computed iff `bucket_q` is set and scores
    are bucketized via the same quantile rule; off by default in v1.
  - Regression MAE / RMSE for relative return (via `masked_regression_metrics`).
  - Calibration diagnostics if scores are converted to probabilities (future).
- **Diagnostic breakdowns:**
  - Per-ticker.
  - Per-sector (M9b).
  - Per-timeframe.
  - Per-fold.
  - By market regime (future).
- **Predictions table (`predictions_universe.csv`), long format** (M7b):
  - Columns and dtypes:
    - `timestamp` — ISO 8601, UTC-naive, same format as the panel index
      (`2024-01-02` for daily; `2024-01-02T13:30:00` for intraday).
    - `symbol` — uppercase string.
    - `timeframe` — lowercase string matching the YAML `timeframe`.
    - `y_true_raw_return` — float, `raw_return(i, t)`; `NaN` when the forward
      return is undefined (missing `close[t, s]` or `close[t+1, s]`).
    - `y_true_relative_return` — float, `raw_return` minus the live-peer
      nanmedian at `t`; `NaN` iff `y_true_raw_return` is `NaN`.
    - `y_score` — float, model score for `(t, s)`; `NaN` where the padding
      mask is set at the last step (`mask[:, -1, :] == True`).
    - `y_rank_pred` — integer-valued float; rank of `y_score` within
      `(fold_id, timestamp)` over the **finite subset** of `y_score`.
    - `y_rank_true` — integer-valued float; rank of
      **`y_true_raw_return`** within `(fold_id, timestamp)` over the finite
      subset. (Because per-row median demeaning preserves ranks, ranking by
      `y_true_relative_return` is identical; we pick raw for provenance.)
    - `fold_id` — integer from `WalkForwardConfig`.
  - Rank rule: `scipy.stats.rankdata(method="average")`, **descending** (higher
    value → rank `1`). NaN in the source value → NaN in the rank.
  - Row ordering: `(fold_id ASC, timestamp ASC, symbol ASC)`; the symbol order
    within a timestamp is the YAML order.

  Example (3 symbols, 1 timestamp, fold 0):

  ```csv
  timestamp,symbol,timeframe,y_true_raw_return,y_true_relative_return,y_score,y_rank_pred,y_rank_true,fold_id
  2024-01-02,MSTR,daily,0.021,0.008,0.014,1,1,0
  2024-01-02,IBIT,daily,0.013,0.000,0.009,2,2,0
  2024-01-02,COIN,daily,-0.011,-0.024,-0.006,3,3,0
  ```

  *(The current runner writes a wide variant `y_true_relative_<SYM>` /
  `y_true_raw_<SYM>` / `y_score_<SYM>` with one row per `(fold_id, timestamp)`;
  M7b migrates to this long schema and keeps a golden-file test comparing
  against a frozen fixture at `tests/golden/predictions_universe.csv`.)*

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
- scipy ≥ 1.11 (required for `scipy.stats.rankdata` used in the predictions
  long schema; add alongside the M7b migration if not already pulled in
  transitively by scikit-learn).
- M6b adds scikit-learn ≥ 1.4 and lightgbm ≥ 4.3 (pinned in `pyproject.toml`).
- M8 adds pyarrow ≥ 15.
- Apple Silicon: `device: "auto"` prefers MPS; ops unsupported on MPS fall
  back to CPU with a single-line warning at model build time.
- **No network calls in tests.** Live Alpha Vantage requests are gated behind
  `ALPHAVANTAGE_API_KEY`; the client raises at construction time if the env
  var is missing and the config is not `--synthetic`.
- **Known limitation (synthetic):** `data.synthetic.synthetic_universe_candles`
  always emits a business-day index regardless of the `timeframe` argument.
  Tests that exercise intraday timeframes must either use `daily` for the
  synthetic fixture or extend the helper to honour `timeframe`. The `timeframe`
  column on the returned frame is set correctly and downstream code does not
  parse the index cadence, so this does not affect correctness for daily-only
  flows. Tracked for a follow-up alongside M9a feature work.

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

- [x] **M6b — Tabular baselines.**
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
    - `pyproject.toml` (edit): bump `scikit-learn>=1.4`,
      bump `numpy>=1.26`, `pandas>=2.2`, `torch>=2.2`, `requires-python=">=3.11"`.
      *(GBT baseline uses `sklearn.ensemble.HistGradientBoostingRegressor` for portability;
      native LightGBM was dropped after segfaults on Python 3.13 / macOS.)*
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

- [x] **M7a — Leakage test matrix.**
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

- [x] **M7b — Predictions long schema + run artifacts.**
  - **Files:**
    - `backtest/universe_runner.py` (edit): emit long-format
      `predictions_universe.csv` as specified in
      [§Evaluation outputs](#evaluation-outputs-forecasting-only); drop the
      wide format; also emit `feature_schema.json` and `folds.json`.
    - `features/universe_tensor.py` (edit): add
      `feature_schema() -> dict` returning `{"features": [...], "n": N_UNIVERSE_FEATURES, "hash": <sha256>}`.
    - `cli.py` (edit): dispatch on `experiment_mode`; exit `2` on fold exception
      with partial `summary.json` still written; keep the one-line success print.
    - `pyproject.toml` (edit): add `scipy>=1.11` to `dependencies` (used for
      `scipy.stats.rankdata` in the long schema).
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

- [x] **M8 — Partitioned parquet store + richer membership.**
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

- [x] **M9a — Cross-sectional features + train-only scaler.**
  - **Files:**
    - `features/cross_sectional.py` (new): `percentile_rank`, `zscore_cross_section`,
      `relative_strength_vs_ew`, `relative_volume_vs_median`, plus trailing-return / vol helpers.
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

- [x] **M9b — Sector-neutral labels + sector map.**
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
    - `data/universe.py` (edit): `load_sector_map(path) -> tuple[dict[str, str], str]`,
      `sectors_for_symbols(symbols, sector_map, default_sector) -> np.ndarray[S]`.
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

- [x] **M10 — Ranking loss ablation.**
  - **Files:**
    - `model/losses.py` (new):
      ```python
      def masked_mse(pred: Tensor, target: Tensor) -> Tensor: ...
      def listnet_loss(pred: Tensor, target: Tensor, *, mask: Tensor) -> Tensor: ...
      def approx_ndcg_loss(pred: Tensor, target: Tensor, *, mask: Tensor, alpha: float = 10.0) -> Tensor: ...
      ```
      All operate on `pred/target: [N, S]` and a label mask (**`True = valid`**:
      `isfinite(target) & ~padded_last_step`). `listnet_loss` applies softmax
      over the valid subset per row (masked positions filled with `-inf` before
      softmax). `approx_ndcg_loss` uses the smoothed rank surrogate with
      temperature `alpha`. **Edge cases:**
      - Row with fewer than 2 valid entries → contributes `0` to the loss and
        is excluded from the denominator (so gradients from that row are zero).
      - If **all** rows in the batch are degenerate, each loss returns
        `torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)`.
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

- [x] **M12 — (Optional) MCP AlphaVantage path.**
  - Parallel branch for schema discovery via `TOOL_LIST` / `TOOL_GET` / `TOOL_CALL`.
  - **Parity test scope (`tests/test_mcp_rest_parity.py`):** the parity target
    is the **canonicalizer**, not the upstream data. The test feeds a single
    frozen raw payload (checked in under `tests/fixtures/av_raw/`) through
    both the REST canonicalizer (`canonicalize_series` / `canonicalize_intraday`)
    and the MCP canonicalizer, and asserts the resulting canonical DataFrames
    are byte-identical (same dtypes, same row order, same `timestamp` values).
    It does **not** make live calls to either path.
  - Gated behind `data_source: rest|mcp` in YAML; default remains `"rest"`.
  - **Shipped:** `data/mcp_canonicalize.py::unwrap_mcp_alphavantage_payload`, `data_source: mcp`
    unwrap in `fetch_candles_for_timeframe`, and `tests/fixtures/av_raw/` + `tests/test_mcp_rest_parity.py`.

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
