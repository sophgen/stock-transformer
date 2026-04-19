# CLI Refactor Plan — `stx`

Status: **Draft**
Date: 2026-04-19

---

## 1. Current-state audit

The project already has significant CLI infrastructure in place. This section maps
what exists to each guideline area so the refactor focuses only on real gaps.

### What already works well

| Area | Status | Evidence |
|------|--------|----------|
| Click-based CLI with subcommands | Done | `cli.py` — `backtest`, `fetch`, `sweep`, `validate`, `version` |
| Short/long flag variants | Done | `-c/--config`, `-v/--verbose`, `-q/--quiet`, `-h/--help` |
| Sensible defaults | Done | `configs/default.yaml` as default config; Pydantic fills the rest |
| `--help` at every level | Done | Click `context_settings={"help_option_names": ["-h", "--help"]}` |
| Exit codes | Done | 0/1/2/130 — tested in `test_cli_dispatch.py` |
| Config hierarchy (flag > env > YAML > default) | Done | `env_config.py` + CLI flag overrides + Pydantic defaults |
| Structured logging with verbosity | Done | `-v` INFO, `-vv` DEBUG, `-q` WARNING |
| SIGINT handling | Done | `_install_sigint` + exit 130 |
| Pydantic validation with clear errors | Done | `format_validation_error` renders bullet list |
| pyproject.toml with entry points | Done | `stx` and legacy `stx-backtest` |
| CI (GitHub Actions) | Done | lint + type-check + test on 3.11/3.12 |
| Release workflow | Done | `v*` tag → build → PyPI |
| README with quickstart + reference | Done | Covers install, commands, config, troubleshooting |
| CONTRIBUTING.md | Done | Setup, architecture, testing, style |
| CHANGELOG.md (Keep a Changelog) | Done | Two entries; Unreleased + 0.1.0 |
| Shell completion | Done | `completions/stx.bash`, instructions in README |
| Separation of CLI / logic / I/O | Mostly done | CLI parses, runners orchestrate, models compute |
| Unit + integration tests for CLI | Partial | 6 tests in `test_cli_dispatch.py`; mix of subprocess + CliRunner |
| Unknown-key typo warnings | Done | `warn_unknown_keys_single` / `warn_unknown_keys_universe` model validators |

### What's missing or incomplete

| Gap | Severity | Notes |
|-----|----------|-------|
| **No `--output-format` flag** | Medium | Machine consumers want JSON; humans want tables. Currently `sweep` dumps JSON, `backtest` prints one line. |
| **No `stx config show` / `stx config diff` subcommand** | Low | Useful for debugging merged config (flag > env > YAML > default). |
| **No progress indicators** | Medium | Multi-fold training is silent for minutes; a progress bar or periodic status line helps. |
| **Incomplete env-var coverage** | Low | Only 4 `STX_*` vars wired; `STX_SEED`, `STX_LOG_LEVEL`, `STX_CONFIG` missing. |
| **`backtest` doesn't dispatch on `experiment_mode`** | High | Rule 30 says this is a target for M7b; `run_from_config_path` currently handles it internally but CLI doesn't expose this as two explicit paths. |
| **Subprocess tests are slow** | Low | `test_cli_universe_synthetic_exit_0` spawns a full training run. A thin `CliRunner` mock path would be faster for CI. |
| **No man page generation** | Low | Click can produce man pages via `click-man`; not wired. |
| **No `--dry-run` mode** | Medium | Validate config + resolve data + print fold plan without training. Goes beyond `validate`. |
| **Mixed `raise SystemExit` and `ctx.exit`** | Low | `backtest_cmd` raises `SystemExit` directly; other commands let Click handle return. Inconsistent pattern. |
| **No `--log-file` option** | Low | All output goes to stderr via `logging.basicConfig`; no file sink option. |
| **`py.typed` marker exists but mypy strict is off** | Low | `ignore_missing_imports = true`; public API types are loose. |
| **Version is duplicated** | Low | `__init__.py` and `pyproject.toml` both hard-code `"0.1.0"`. |

---

## 2. Refactor plan

### Phase 1 — CLI dispatch and consistency (high priority)

**Goal:** Every subcommand behaves uniformly; experiment mode dispatch lives in the CLI layer.

#### 1a. Unify experiment-mode dispatch in `backtest_cmd`

`backtest_cmd` should read YAML, detect `experiment_mode`, and call the
right runner (`runner.run_from_config_path` vs
`universe_runner.run_universe_from_config_path`). The runner modules should
not re-detect the mode.

```
File: src/stock_transformer/cli.py (backtest_cmd)

- Load raw YAML → apply_stx_env_overrides → coerce_experiment_config
- If experiment_mode == "universe": call universe runner
- Else: call single-symbol runner
- Both runners return a summary dict with the same shape
```

*Tests:*
- `test_cli_dispatch.py::test_backtest_dispatches_universe` — universe config → universe runner called.
- `test_cli_dispatch.py::test_backtest_dispatches_single` — default config → single-symbol runner called.

#### 1b. Normalize exit-code handling

Replace bare `raise SystemExit(N)` in command functions with a shared helper
that Click's result callback can consume. Pattern:

```python
class StxResult:
    code: int
    message: str | None

def _exit(code: int, message: str | None = None) -> None:
    if message:
        click.echo(message, err=(code != 0))
    raise SystemExit(code)
```

#### 1c. Add `--output-format` to `backtest` and `sweep`

```
--output-format text|json   (default: text)
```

- `text`: current human-friendly line (`Run complete. Artifacts: …`).
- `json`: print full `summary.json` to stdout (pipe-friendly).

*Tests:* assert stdout is valid JSON when `--output-format json`.

---

### Phase 2 — New subcommands and flags (medium priority)

#### 2a. `stx config` subgroup

| Subcommand | Purpose |
|------------|---------|
| `stx config show -c PATH` | Print fully merged config (flag > env > file > default) as YAML. |
| `stx config diff -c PATH` | Show only keys that differ from Pydantic defaults. |

These are read-only, no training. Helpful for debugging "what will actually run."

#### 2b. `stx backtest --dry-run`

Resolve data (or synthetic), build fold plan, print fold boundaries and
sample counts, then exit 0 without training. Uses the same code path as a
real run up to the training loop.

#### 2c. `--log-file PATH`

Add a global option that attaches a `FileHandler` to the root logger in
`setup_logging`. Useful for headless / CI runs where stderr is noisy.

#### 2d. `--seed` flag on `backtest`

Override YAML `seed` from the CLI for quick reproducibility checks.

#### 2e. Expand `STX_*` environment variables

| Env var | YAML key | Type |
|---------|----------|------|
| `STX_SEED` | `seed` | int |
| `STX_LOG_LEVEL` | (logging level) | str |
| `STX_CONFIG` | (default config path) | str |
| `STX_BATCH_SIZE` | `batch_size` | int |

---

### Phase 3 — UX polish (medium priority)

#### 3a. Progress reporting

Add a callback protocol that runners can call per-fold and per-epoch:

```python
class ProgressCallback(Protocol):
    def on_fold_start(self, fold_id: int, total_folds: int) -> None: ...
    def on_epoch_end(self, fold_id: int, epoch: int, total_epochs: int, metrics: dict) -> None: ...
    def on_fold_end(self, fold_id: int, summary: dict) -> None: ...
```

CLI wires this to a simple stderr progress line (or `rich.progress` bar
behind `--rich` if the dependency is present).

Quiet mode (`-q`) suppresses progress entirely.

#### 3b. Colored and styled output

Conditionally use `click.style` for:
- Error messages (red)
- Warnings (yellow)
- Success summary (green)

Gate on `--no-color` flag (already reserved, hidden) and `NO_COLOR` env var.

#### 3c. Table output for `sweep`

Instead of dumping raw JSON, format a comparison table when `--output-format text`:

```
Loss      | Spearman | NDCG@2 | Hit@2
mse       |   0.412  |  0.73  |  0.65
listnet   |   0.438  |  0.76  |  0.68
approx    |   0.401  |  0.71  |  0.63
```

---

### Phase 4 — Code quality and testing (medium priority)

#### 4a. CLI integration test harness

Create `tests/conftest.py` fixtures:

```python
@pytest.fixture
def stx_runner():
    """CliRunner pre-configured with catch_exceptions=False."""
    return CliRunner(mix_stderr=False)

@pytest.fixture
def quick_universe_yaml(tmp_path):
    """Write a minimal universe config for fast CLI tests."""
    ...
```

Migrate subprocess-heavy tests to `CliRunner` where possible. Keep one
subprocess test per subcommand as a true integration test.

#### 4b. Expand CLI test coverage

Target tests (in `test_cli_dispatch.py` or split into `test_cli_*.py`):

| Test | Assertion |
|------|-----------|
| `test_backtest_json_output` | `--output-format json` → valid JSON on stdout |
| `test_backtest_dry_run` | `--dry-run` → fold plan printed, no model files |
| `test_config_show` | Merged config printed as valid YAML |
| `test_config_diff` | Only non-default keys shown |
| `test_log_file_created` | `--log-file /tmp/x.log` → file exists after run |
| `test_seed_flag_overrides_yaml` | `--seed 99` → summary shows seed=99 |
| `test_unknown_subcommand_exit_2` | `stx bogus` → exit 2, helpful message |
| `test_version_output` | `stx --version` → contains `stock-transformer` |
| `test_quiet_suppresses_info` | `-q` → no INFO lines in stderr |

#### 4c. Tighten mypy

- Set `strict = true` in `[tool.mypy]` or at least enable `disallow_untyped_defs`.
- Add return-type annotations to all public functions in `cli.py`, runners, and config modules.
- Fix the resulting errors incrementally (one module per PR).

#### 4d. Single-source version

Use `importlib.metadata` in `__init__.py`:

```python
from importlib.metadata import version
__version__ = version("stock-transformer")
```

Remove the hard-coded `"0.1.0"` string. `pyproject.toml` remains the sole source.

---

### Phase 5 — Distribution and packaging (lower priority)

#### 5a. Man page generation

Add `click-man` to dev dependencies and a Makefile / script target:

```bash
uv run click-man stx --target man/
```

Ship generated man pages under `man/` in the repo. Add a CI step that
regenerates and checks for drift.

#### 5b. Shell completion for zsh and fish

Extend `completions/` with generated scripts:

```bash
_STX_COMPLETE=zsh_source stx > completions/stx.zsh
_STX_COMPLETE=fish_source stx > completions/stx.fish
```

Document in README under a "Shell completion" section (currently bash-only).

#### 5c. Docker image

Add a minimal `Dockerfile`:

```dockerfile
FROM python:3.11-slim
COPY . /app
RUN pip install /app
ENTRYPOINT ["stx"]
```

Document in README for users who don't want to install Python/PyTorch locally.

#### 5d. CI enhancements

| Addition | File | Purpose |
|----------|------|---------|
| Smoke test on macOS | `ci.yml` | Catch MPS-related regressions |
| `--cov-fail-under=80` | `ci.yml` | Coverage gate |
| Dependabot config | `.github/dependabot.yml` | Automated dep updates |
| Release notes generation | `release.yml` | Auto-generate from CHANGELOG on tag |

---

## 3. Architectural principles for the refactor

### Layer separation (already mostly in place — formalize it)

```
┌──────────────────────────────────────────────┐
│  CLI layer  (cli.py)                         │
│  - Parse args, setup logging, catch errors   │
│  - No training logic, no tensor ops          │
├──────────────────────────────────────────────┤
│  Orchestration layer  (backtest/*.py)        │
│  - Runners, walk-forward, fold management    │
│  - Calls model/feature/label modules         │
│  - Returns summary dicts (pure data)         │
├──────────────────────────────────────────────┤
│  Core layer  (model/, features/, labels/)    │
│  - Pure computation, no I/O, no CLI deps     │
│  - Testable with synthetic tensors only      │
├──────────────────────────────────────────────┤
│  I/O layer  (data/, backtest/artifacts.py)   │
│  - File reads/writes, API calls, caching     │
│  - Isolated behind interfaces                │
└──────────────────────────────────────────────┘
```

**Rule:** imports flow downward only. `cli.py` may import from `backtest/`
and `config_models.py`. Core and I/O layers must not import from `cli.py`.

### Config resolution order (already implemented — keep it)

```
CLI flags  →  STX_* env vars  →  YAML file  →  Pydantic defaults
         (highest priority)                    (lowest priority)
```

---

## 4. File change map

Estimated files touched per phase, to help scope PRs.

| Phase | Files modified | Files added |
|-------|---------------|-------------|
| 1a | `cli.py`, `runner.py` | — |
| 1b | `cli.py` | — |
| 1c | `cli.py` | — |
| 2a | `cli.py` | — |
| 2b | `cli.py`, `universe_runner.py`, `runner.py` | — |
| 2c | `cli.py` | — |
| 2d | `cli.py` | — |
| 2e | `env_config.py` | — |
| 3a | `cli.py`, `universe_runner.py`, `runner.py`, `training.py` | — |
| 3b | `cli.py` | — |
| 3c | `cli.py` | — |
| 4a | `test_cli_dispatch.py` | `tests/conftest.py` (if not present) |
| 4b | `test_cli_dispatch.py` | — |
| 4c | `pyproject.toml`, multiple `*.py` | — |
| 4d | `__init__.py` | — |
| 5a | `pyproject.toml` | `man/stx.1`, script |
| 5b | — | `completions/stx.zsh`, `completions/stx.fish` |
| 5c | — | `Dockerfile` |
| 5d | `ci.yml`, `release.yml` | `.github/dependabot.yml` |

---

## 5. Suggested PR sequence

Each PR should be independently mergeable and leave the tool fully functional.

| PR | Phase | Scope | Risk |
|----|-------|-------|------|
| **PR-1** | 1a + 1b | Dispatch + exit-code cleanup | Medium — touches the hot path |
| **PR-2** | 1c | `--output-format` flag | Low |
| **PR-3** | 2a | `stx config show/diff` | Low — new commands, no existing behavior changes |
| **PR-4** | 2b + 2d + 2e | `--dry-run`, `--seed`, env vars | Low |
| **PR-5** | 2c + 3b | `--log-file`, colored output | Low |
| **PR-6** | 3a | Progress callbacks | Medium — threads through runners |
| **PR-7** | 3c | Sweep table output | Low |
| **PR-8** | 4a + 4b | Test harness + expanded coverage | Low |
| **PR-9** | 4c | mypy strict | Low — type-only changes |
| **PR-10** | 4d | Single-source version | Low |
| **PR-11** | 5a + 5b | Man pages + completions | Low |
| **PR-12** | 5c + 5d | Docker + CI hardening | Low |

---

## 6. Out of scope for this refactor

These items are explicitly **not** part of the CLI refactor to avoid scope creep:

- New model architectures or training features (stay on milestone track in `plan.md`).
- MCP data-source wiring (M12).
- Portfolio simulation changes (M11).
- New ranking losses (M10).
- Deleting the single-symbol reference path.
- Rewriting runners or model code — only interface changes where needed for CLI.
- GUI or web dashboard.

---

## 7. Acceptance criteria

The refactor is complete when:

1. `stx --help`, `stx <cmd> --help` cover all commands and flags.
2. `stx backtest` dispatches correctly for both experiment modes.
3. `--output-format json` produces valid, parseable JSON for `backtest` and `sweep`.
4. `stx config show` and `stx config diff` work.
5. `--dry-run` prints fold plan without training.
6. All exit codes (0/1/2/130) are tested with both `CliRunner` and subprocess.
7. `stx --version` reports a single-sourced version.
8. CI passes with `--cov-fail-under=80`.
9. README documents every subcommand, flag, env var, and exit code.
10. CONTRIBUTING.md reflects the updated architecture.
11. Shell completions exist for bash, zsh, and fish.
