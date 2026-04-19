# Refactor Plan: `stock-transformer` → Production-Ready CLI Tool

> Audit date: 2026-04-19
> Scope: CLI redesign, code quality, docs, packaging, CI/CD

---

## Executive Summary

`stock-transformer` is a well-structured ML experiment harness with solid Pydantic
validation, walk-forward evaluation discipline, and reasonable test coverage. The
main gaps are: (1) a flat CLI with no subcommands, (2) monolithic runner functions
with duplicated patterns, (3) no structured logging, (4) standalone scripts that
should be first-class CLI commands, and (5) missing contributor/release docs.

This plan is ordered by **impact** — each section is an independent work unit.

---

## 1. CLI Structure & Interface

### Current State

- Single entrypoint `stx-backtest` via raw `argparse` with three flags (`-c`, `--synthetic`, `--device`).
- `scripts/fetch_sample_data.py` and `scripts/sweep_loss.py` are separate `argparse` scripts with their own `main()` — not discoverable as part of the tool.
- No verbosity control, no `--version`, no shell completions.
- Exit codes (0/1/2) are well-defined but the error messages are bare `print()` to stderr.

### Proposed Changes

**1.1 — Migrate to `click` (or `typer`) with subcommands**

Replace the flat `argparse` parser with a `click.Group` command tree:

```
stx backtest   -c CONFIG [--synthetic] [--device NAME]   # current stx-backtest
stx fetch      [--cache-dir DIR] [--symbols SYM…] [--refresh]
stx sweep      -c CONFIG [--synthetic]
stx validate   -c CONFIG                                  # dry-run: load + validate only
stx version                                               # print version + torch/device info
```

- Absorb `scripts/fetch_sample_data.py` → `stx fetch`.
- Absorb `scripts/sweep_loss.py` → `stx sweep`.
- Add `stx validate` for config-only checks (useful in CI or before long runs).
- Keep `scripts/` as thin wrappers that call the click commands (backward compat), or remove them.

**1.2 — Add global flags at the group level**

```
stx [--verbose / -v / -vv] [--quiet / -q] [--no-color] COMMAND …
```

`-v` sets `logging.INFO`, `-vv` sets `logging.DEBUG`, `--quiet` suppresses
everything below `WARNING`. `--no-color` disables Rich/colorama output if added.

**1.3 — Add `--version` flag**

```
stx --version   →   stock-transformer 0.1.0 (torch 2.x, device auto→mps)
```

Read from `stock_transformer.__version__` and enrich with runtime info.

**1.4 — Shell completion generation**

Click provides `_STX_COMPLETE=bash_source stx` etc. Document it in the README.
Alternatively ship static completion scripts under `completions/`.

**1.5 — Input validation with actionable errors**

Currently Pydantic `ValidationError` propagates as an unformatted traceback. Catch
it in the CLI layer and print a human-readable bullet list:

```
Error: invalid config configs/bad.yaml
  • d_model (64) must be divisible by nhead (5)
  • train_bars: value is not a valid integer
```

---

## 2. Code Quality — Separation of Concerns

### Current State

- `runner.py::run_experiment` (200 lines) and `universe_runner.py::run_universe_experiment` (350+ lines) each do **all** of: config coercion, data loading, tokenization, fold generation, model construction, training, inference, metric computation, prediction assembly, artifact I/O.
- Both runners duplicate: run-dir allocation, config snapshot, fold error handling, summary JSON assembly, prediction CSV writing.
- `resolve_device()` lives in `model/transformer_classifier.py` but is imported by both runners and tests — it belongs in a shared utility.
- Hardcoded batch sizes (256, 128) in `_predict` / `_predict_ranker`.

### Proposed Changes

**2.1 — Extract a shared `RunContext` dataclass**

```python
@dataclasses.dataclass
class RunContext:
    run_dir: Path
    device: torch.device
    config: dict[str, Any]
    git_sha: str
    summary: dict[str, Any]   # mutable, accumulates results
```

Both `run_experiment` and `run_universe_experiment` create a `RunContext` at the
top, pass it through, and dump `summary.json` at the bottom via a single
`RunContext.finalize()` method. This eliminates ~40 lines of duplicated boilerplate.

**2.2 — Move `resolve_device` to a top-level `device.py` module**

`src/stock_transformer/device.py` — one function, no model imports. Update all
import sites. Keeps `model/transformer_classifier.py` focused on the model.

**2.3 — Extract prediction batch-inference into model modules**

Move `_predict` → `model/transformer_classifier.py::batch_predict` and
`_predict_ranker` → `model/transformer_ranker.py::batch_predict`. Accept an
optional `batch_size` parameter (default 256 / 128). This makes inference
importable without importing the runner.

**2.4 — Extract artifact I/O helpers**

Create `backtest/artifacts.py`:

```python
def save_config_snapshot(run_dir, config): ...
def save_predictions_csv(run_dir, records, columns, filename): ...
def save_summary(run_dir, summary): ...
def save_fold_payload(run_dir, folds, ts_pred): ...
```

Both runners call these instead of inlining `json.dumps` / `pd.to_csv` patterns.

**2.5 — Break `run_universe_experiment` into phases**

The 350-line function should become:

```python
def run_universe_experiment(config, *, use_synthetic=False):
    ctx = _setup_universe_run(config)
    panel, close, X, mask, y, ... = _load_and_build_features(ctx, use_synthetic)
    fold_results = _walk_forward_loop(ctx, X, mask, y, close, ...)
    return _assemble_summary(ctx, fold_results)
```

Each phase is independently testable.

---

## 3. Structured Logging

### Current State

- Progress and errors are communicated via `print()`, JSON artifact files, and CSV training logs.
- No `logging` module usage anywhere in `src/`.
- Silent runs make debugging slow — you have to read `summary.json` or `fold_errors.log` after the fact.

### Proposed Changes

**3.1 — Add `logging` throughout**

Replace all `print()` in `src/` with `logger.info()` / `logger.warning()` /
`logger.error()`. Key log points:

- Config loaded and coerced (INFO)
- Data fetched / synthetic generated (INFO with row counts)
- Fold N started / completed / failed (INFO / ERROR)
- Training epoch progress (DEBUG — only visible with `-vv`)
- Final summary metrics (INFO)

**3.2 — Configure in the CLI layer**

```python
def _setup_logging(verbosity: int, quiet: bool) -> None:
    level = logging.WARNING if quiet else [logging.WARNING, logging.INFO, logging.DEBUG][min(verbosity, 2)]
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)s — %(message)s", level=level)
```

Called once in the click group callback.

**3.3 — Keep artifact files as-is**

`summary.json`, `fold_errors.log`, `training_log_fold_*.csv` stay — they are the
structured record. Logging complements them with real-time progress.

---

## 4. Signal Handling & Graceful Shutdown

### Current State

- `Ctrl+C` during a long training run produces a full Python traceback.
- No cleanup of partial artifacts.

### Proposed Changes

**4.1 — Catch `KeyboardInterrupt` in CLI**

```python
try:
    summary = run_from_config_path(...)
except KeyboardInterrupt:
    logger.warning("Interrupted — saving partial results")
    # write whatever folds completed so far
    sys.exit(130)
```

**4.2 — Save partial summary on interrupt**

In the walk-forward loop, check for a `threading.Event` or simply wrap the loop
body with a try/except for `KeyboardInterrupt`, write `summary["error"] = "interrupted"`,
save `summary.json`, and exit cleanly.

---

## 5. Config Hierarchy Hardening

### Current State

- Good: CLI `--device` → `STX_DEVICE` env → YAML `device` → `"auto"` default.
- Weak: only `device` has this full cascade. Other knobs (e.g. `cache_dir`,
  `artifacts_dir`, `epochs`) cannot be overridden from the environment.
- `extra="ignore"` in Pydantic silently drops unknown YAML keys — a typo in a
  config field (e.g. `epcohs: 20`) is silently lost, which is a footgun.

### Proposed Changes

**5.1 — Warn on unknown keys (don't silently ignore)**

Change `extra="ignore"` → `extra="forbid"` in the Pydantic models, **or** keep
`"ignore"` but log a warning when extra keys are present:

```python
@model_validator(mode="before")
@classmethod
def warn_unknown(cls, values):
    known = set(cls.model_fields)
    for k in set(values) - known:
        logger.warning("Unknown config key %r (typo?)", k)
    return values
```

**5.2 — Support `STX_*` env overrides for common knobs**

Define a convention: `STX_DEVICE`, `STX_CACHE_DIR`, `STX_ARTIFACTS_DIR`,
`STX_EPOCHS`. Apply in `run_from_config_path` before coercion:

```python
_ENV_OVERRIDES = {"STX_DEVICE": "device", "STX_CACHE_DIR": "cache_dir", ...}
for env_key, cfg_key in _ENV_OVERRIDES.items():
    v = os.environ.get(env_key, "").strip()
    if v:
        cfg[cfg_key] = v
```

---

## 6. Training Loop De-duplication

### Current State

`train_candle_transformer` and `train_transformer_ranker` in `training.py` share
~80% of their structure: optimizer setup, scheduler creation, epoch loop, early
stopping, best-state tracking, CSV logging. The differences are: loss computation
and model forward signature.

### Proposed Changes

**6.1 — Extract a generic training harness**

```python
def _train_loop(
    model: nn.Module,
    train_step: Callable[[nn.Module, Tensor], Tensor],  # returns loss
    val_step: Callable[[nn.Module], float],
    cfg: dict,
    device: torch.device,
    log_path: Path | None = None,
) -> nn.Module:
    ...  # optimizer, scheduler, epoch loop, early stopping, CSV logging
```

`train_candle_transformer` and `train_transformer_ranker` become thin wrappers
that define `train_step` / `val_step` closures and call `_train_loop`.

This cuts ~100 lines of duplication and makes it trivial to add new model types.

---

## 7. Documentation

### Current State

- `README.md`: solid (setup, modes, CLI, config, architecture, artifacts, tests).
- No `CONTRIBUTING.md`, no `CHANGELOG.md`.
- Module/function docstrings exist on most public APIs; a few are missing or terse.
- No man pages, no shell completion docs.

### Proposed Changes

**7.1 — `CONTRIBUTING.md`**

Contents:

- Dev setup (`uv sync --extra dev`)
- Architecture overview (diagram: CLI → runner → {data, features, model} → artifacts)
- How to add a new model, loss, or feature
- Testing policy (synthetic-only in CI, golden file updates)
- Coding conventions (ruff, mypy, no `print()` in `src/`)
- PR checklist

**7.2 — `CHANGELOG.md`**

Start with `## [Unreleased]` and backfill a `## [0.1.0]` entry covering the
current feature set. Follow [Keep a Changelog](https://keepachangelog.com/).

**7.3 — Update `README.md`**

- Add subcommand reference table (after CLI migration).
- Add `stx --version` example.
- Add "Configuration Precedence" section (CLI > env > YAML > defaults).
- Add shell completion install instructions.
- Add a "Troubleshooting" section (common errors: missing API key, MPS not available, etc.).

**7.4 — Docstring audit**

Add/improve docstrings on:

- `backtest/run_helpers.py` functions (brief but good — just add return types).
- `data/align.py::align_universe_ohlcv` — document the returned `panel` schema.
- `features/universe_tensor.py::build_universe_samples` — document shape/semantics of every return value.
- `backtest/portfolio_sim.py::simulate_topk_portfolio` — document the keys in the returned dict.

---

## 8. Testing Gaps

### Current State

- Good synthetic E2E coverage for both pipelines.
- CLI tested via subprocess (exit codes, device override).
- Config validation well-tested.
- Missing: integration test for `stx fetch` / `stx sweep` (currently untested scripts), negative config tests (unknown keys, type mismatches beyond what exists), `KeyboardInterrupt` behavior.

### Proposed Changes

**8.1 — CLI integration tests for new subcommands**

After migrating to click, add tests that invoke each subcommand:

```python
def test_stx_fetch_help():
    result = runner.invoke(cli, ["fetch", "--help"])
    assert result.exit_code == 0

def test_stx_validate_good_config(tmp_path):
    ...

def test_stx_validate_bad_config_exit_1(tmp_path):
    ...
```

**8.2 — Unit tests for extracted helpers**

Once `RunContext`, `artifacts.py`, and `_train_loop` are extracted, write focused
unit tests for each (no model training needed — mock the model).

**8.3 — Add a test for unknown config key warning**

```python
def test_unknown_key_warns(caplog):
    raw = {**valid_config, "epcohs": 20}
    coerce_experiment_config(raw)
    assert "Unknown config key" in caplog.text
```

---

## 9. Distribution & Packaging

### Current State

- `pyproject.toml` is clean: setuptools build, `src`-layout, `[project.scripts]` entrypoint.
- CI runs lint + type check + tests on Python 3.11/3.12.
- No release workflow, no version bumping, no PyPI publish step.

### Proposed Changes

**9.1 — Update `[project.scripts]` for the new CLI**

```toml
[project.scripts]
stx = "stock_transformer.cli:main"
```

(Drop `stx-backtest` or keep as alias for backward compat.)

**9.2 — Add CI release workflow**

```yaml
# .github/workflows/release.yml
on:
  push:
    tags: ["v*"]
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

**9.3 — Add `py.typed` marker**

Create `src/stock_transformer/py.typed` (empty file) so downstream consumers get
type information from mypy.

**9.4 — Pin minimum dependency versions more tightly**

Current: `torch>=2.2`. Consider upper bounds or tested-range comments in
`pyproject.toml` to avoid silent breakage on major bumps.

---

## 10. Minor Quality-of-Life Improvements

| Item | Location | Change |
|------|----------|--------|
| Hardcoded batch sizes | `runner.py:66`, `universe_runner.py:95` | Make configurable via `inference_batch_size` config key |
| `load_config` duplicated | `runner.py:47`, `universe_runner.py:479` | Use single `load_config` from runner (universe already imports runner) |
| `BLE001` broad except | `runner.py:266`, `universe_runner.py:399` | Already intentional (fold isolation), but add a brief comment explaining the contract |
| `_safe_nanmean` | `universe_runner.py:68` | Move to `metrics.py` where it logically belongs |
| Legacy `CandleTransformerClassifier` | `transformer_classifier.py:107` | Deprecation warning + removal timeline (it's dead code if no external consumers) |
| `run_universe_from_config_path` | `universe_runner.py:478` | Unused in the codebase (dead code); remove or wire into CLI |
| Redundant `load_universe_config_from_dict` | `universe_runner.py:461` | Overlaps with `coerce_universe_config`; consolidate |

---

## Suggested Priority Order

| Phase | Sections | Estimated Effort |
|-------|----------|------------------|
| **Phase 1** | §1 (CLI) + §7.1–7.2 (CONTRIBUTING, CHANGELOG) | 1–2 days |
| **Phase 2** | §2 (Separation of concerns) + §6 (Training dedup) | 1–2 days |
| **Phase 3** | §3 (Logging) + §4 (Signal handling) + §5 (Config) | 0.5–1 day |
| **Phase 4** | §8 (Tests) + §9 (Packaging/CI) + §10 (QoL) | 0.5–1 day |

Each phase is independently mergeable. Phase 1 has the highest user-facing impact.
Phase 2 has the highest maintainability impact.

---

## Files Touched (Expected)

```
NEW    src/stock_transformer/device.py
NEW    src/stock_transformer/backtest/artifacts.py
NEW    src/stock_transformer/backtest/context.py
NEW    CONTRIBUTING.md
NEW    CHANGELOG.md
NEW    src/stock_transformer/py.typed
NEW    .github/workflows/release.yml

EDIT   src/stock_transformer/cli.py              — click migration, logging setup
EDIT   src/stock_transformer/backtest/runner.py   — extract RunContext, artifacts, phases
EDIT   src/stock_transformer/backtest/universe_runner.py — same
EDIT   src/stock_transformer/backtest/training.py — generic train loop
EDIT   src/stock_transformer/backtest/metrics.py  — absorb _safe_nanmean
EDIT   src/stock_transformer/model/transformer_classifier.py — batch_predict, remove resolve_device
EDIT   src/stock_transformer/model/transformer_ranker.py     — batch_predict
EDIT   src/stock_transformer/config_models.py     — unknown-key warning
EDIT   pyproject.toml                             — click dep, entrypoint rename, py.typed
EDIT   README.md                                  — updated CLI docs
EDIT   .github/workflows/ci.yml                   — (minor: add release job reference)
EDIT   tests/test_cli_dispatch.py                 — click test runner
EDIT   scripts/fetch_sample_data.py               — thin wrapper or remove
EDIT   scripts/sweep_loss.py                      — thin wrapper or remove

DELETE (candidates, optional)
       src/stock_transformer/model/transformer_classifier.py::CandleTransformerClassifier  (dead code)
       src/stock_transformer/backtest/universe_runner.py::run_universe_from_config_path     (unused)
```
