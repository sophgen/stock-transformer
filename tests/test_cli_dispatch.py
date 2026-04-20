"""CLI exit codes, universe dispatch, and Click subcommands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from stock_transformer.cli import cli

REPO = Path(__file__).resolve().parents[1]


def test_cli_missing_config_exit_1():
    r = subprocess.run(
        [sys.executable, "-m", "stock_transformer.cli", "backtest", "-c", str(REPO / "nope.yaml")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1


def test_cli_universe_synthetic_exit_0(tmp_path):
    cfg = yaml.safe_load((REPO / "configs" / "universe.yaml").read_text())
    cfg["artifacts_dir"] = str(tmp_path / "a")
    cfg["device"] = "cpu"
    cfg["train_bars"] = 50
    cfg["val_bars"] = 15
    cfg["test_bars"] = 15
    cfg["step_bars"] = 15
    cfg["epochs"] = 1
    cfg["synthetic_n_bars"] = 200
    p = tmp_path / "u.yaml"
    p.write_text(yaml.dump(cfg))
    r = subprocess.run(
        [sys.executable, "-m", "stock_transformer.cli", "backtest", "--synthetic", "-c", str(p)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_cli_device_flag_overrides_yaml(tmp_path):
    """``--device`` must win over YAML (e.g. file says mps, CI runs on cpu)."""
    cfg = yaml.safe_load((REPO / "configs" / "universe.yaml").read_text())
    cfg["artifacts_dir"] = str(tmp_path / "a")
    cfg["device"] = "mps"
    cfg["train_bars"] = 50
    cfg["val_bars"] = 15
    cfg["test_bars"] = 15
    cfg["step_bars"] = 15
    cfg["epochs"] = 1
    cfg["synthetic_n_bars"] = 200
    p = tmp_path / "u.yaml"
    p.write_text(yaml.dump(cfg))
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "stock_transformer.cli",
            "backtest",
            "--synthetic",
            "-c",
            str(p),
            "--device",
            "cpu",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_cli_partial_failure_exit_2_subprocess():
    """Exit 2 via real subprocess (acceptance: subprocess + CliRunner coverage)."""
    script = REPO / "tests" / "cli_subprocess_helpers" / "backtest_exit_2.py"
    r = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2, r.stderr + r.stdout


def test_cli_keyboard_interrupt_exit_130_subprocess():
    """Exit 130 via real subprocess (matches Ctrl+C handling path)."""
    script = REPO / "tests" / "cli_subprocess_helpers" / "backtest_exit_130.py"
    r = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 130, r.stderr + r.stdout


def test_cli_keyboard_interrupt_exit_130_cli_runner(monkeypatch):
    import stock_transformer.cli as cli_mod

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, "run_experiment", interrupt)
    r = CliRunner().invoke(cli_mod.cli, ["backtest", "-c", str(REPO / "configs" / "default.yaml")])
    assert r.exit_code == 130


def test_cli_partial_failure_exit_2(tmp_path, monkeypatch):
    import stock_transformer.cli as cli_mod

    def boom(*_a, **_k):
        return {"run_dir": str(tmp_path), "fold_errors": [{"fold_id": 0, "error": "x"}]}

    monkeypatch.setattr(cli_mod, "run_experiment", boom)
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["backtest", "-c", str(REPO / "configs" / "default.yaml")])
    assert result.exit_code == 2


def test_stx_cli_progress_runs_without_error() -> None:
    """Smoke-test stderr progress helper (Rich optional)."""
    from stock_transformer.cli import StxCliProgress

    p = StxCliProgress(use_rich=False)
    p.on_fold_start(0, 3)
    p.on_epoch_end(0, 0, 5, {"train_loss": 1.0, "val_loss": 0.5, "lr": 1e-3})
    p.on_fold_end(0, {"spearman_mean": 0.1})


def test_backtest_universe_fast_cli_runner(quick_universe_yaml, tmp_path, monkeypatch):
    """Fast path: CliRunner + mock (see subprocess tests for full training)."""
    import stock_transformer.cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "run_universe_experiment",
        lambda *a, **k: {"run_dir": str(tmp_path / "r"), "experiment": "universe"},
    )
    r = CliRunner().invoke(cli_mod.cli, ["backtest", "--synthetic", "-c", str(quick_universe_yaml)])
    assert r.exit_code == 0


def test_backtest_dispatches_universe(monkeypatch, tmp_path):
    import stock_transformer.cli as cli_mod

    called: dict[str, bool] = {}

    def uni(*_a, **_k):
        called["universe"] = True
        return {"run_dir": str(tmp_path), "experiment": "universe"}

    def single(*_a, **_k):
        called["single"] = True
        return {"run_dir": str(tmp_path)}

    monkeypatch.setattr(cli_mod, "run_universe_experiment", uni)
    monkeypatch.setattr(cli_mod, "run_experiment", single)
    cfg = yaml.safe_load((REPO / "configs" / "universe.yaml").read_text())
    cfg["artifacts_dir"] = str(tmp_path / "a")
    p = tmp_path / "x.yaml"
    p.write_text(yaml.dump(cfg))
    r = CliRunner().invoke(cli_mod.cli, ["backtest", "-c", str(p), "--synthetic"])
    assert r.exit_code == 0
    assert called.get("universe") and not called.get("single")


def test_backtest_dispatches_single(monkeypatch, tmp_path):
    import stock_transformer.cli as cli_mod

    called: dict[str, bool] = {}

    def uni(*_a, **_k):
        called["universe"] = True
        return {"run_dir": str(tmp_path)}

    def single(*_a, **_k):
        called["single"] = True
        return {"run_dir": str(tmp_path)}

    monkeypatch.setattr(cli_mod, "run_universe_experiment", uni)
    monkeypatch.setattr(cli_mod, "run_experiment", single)
    r = CliRunner().invoke(cli_mod.cli, ["backtest", "-c", str(REPO / "configs" / "default.yaml"), "--synthetic"])
    assert r.exit_code == 0
    assert called.get("single") and not called.get("universe")


def test_sweep_json_output(monkeypatch, tmp_path):
    import stock_transformer.cli as cli_mod

    fake = {
        "experiment": "sweep_loss",
        "by_loss": {},
        "config": None,
        "sweep_dir": str(tmp_path),
        "summary_path": str(tmp_path / "summary.json"),
    }
    monkeypatch.setattr(cli_mod, "run_loss_sweep", lambda *a, **k: fake)
    r = CliRunner().invoke(
        cli_mod.cli,
        ["sweep", "--synthetic", "-c", str(REPO / "configs" / "universe.yaml"), "--output-format", "json"],
    )
    assert r.exit_code == 0
    json.loads(r.stdout)


def test_backtest_json_output(monkeypatch, tmp_path):
    import stock_transformer.cli as cli_mod

    summary = {"run_dir": str(tmp_path / "r"), "n_folds": 0, "error": "no_folds"}

    monkeypatch.setattr(cli_mod, "run_experiment", lambda *_a, **_k: summary)
    r = CliRunner().invoke(
        cli_mod.cli,
        ["backtest", "-c", str(REPO / "configs" / "default.yaml"), "--synthetic", "--output-format", "json"],
    )
    assert r.exit_code == 2
    json.loads(r.stdout)


def test_backtest_json_output_short_o_flag(monkeypatch, tmp_path):
    """``-o`` is an alias for ``--output-format`` (POSIX-style short long pair)."""
    import stock_transformer.cli as cli_mod

    summary = {"run_dir": str(tmp_path / "r")}

    monkeypatch.setattr(cli_mod, "run_experiment", lambda *_a, **_k: summary)
    r = CliRunner().invoke(
        cli_mod.cli,
        ["backtest", "-c", str(REPO / "configs" / "default.yaml"), "--synthetic", "-o", "json"],
    )
    assert r.exit_code == 0
    json.loads(r.stdout)


def test_backtest_dry_run(monkeypatch, tmp_path):
    import stock_transformer.cli as cli_mod

    def dry(*_a, **_k):
        assert _k.get("dry_run") is True
        return {
            "run_dir": str(tmp_path),
            "dry_run": True,
            "n_samples": 120,
            "n_folds": 1,
            "fold_plan": {"0": {"train": {"i_start": 0, "i_end": 49}}},
        }

    monkeypatch.setattr(cli_mod, "run_experiment", dry)
    r = CliRunner().invoke(cli_mod.cli, ["backtest", "-c", str(REPO / "configs" / "default.yaml"), "--dry-run"])
    assert r.exit_code == 0
    assert "Dry run:" in r.stdout
    assert "120" in r.stdout
    assert "fold_plan" not in r.stdout  # YAML keys are fold ids, not the string fold_plan


def test_config_show(stx_runner, tmp_path):
    import stock_transformer.cli as cli_mod

    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    cfg["artifacts_dir"] = str(tmp_path / "a")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(cfg))
    r = stx_runner.invoke(cli_mod.cli, ["config", "show", "-c", str(p)])
    assert r.exit_code == 0
    yaml.safe_load(r.stdout)


def test_config_diff(stx_runner, tmp_path):
    import stock_transformer.cli as cli_mod

    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    cfg["artifacts_dir"] = str(tmp_path / "a")
    cfg["seed"] = 12345
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(cfg))
    r = stx_runner.invoke(cli_mod.cli, ["config", "diff", "-c", str(p)])
    assert r.exit_code == 0
    assert "seed" in r.stdout


def test_log_file_created(tmp_path, monkeypatch):
    import stock_transformer.cli as cli_mod

    log_path = tmp_path / "run.log"

    def fake(*_a, **_k):
        return {"run_dir": str(tmp_path / "r")}

    monkeypatch.setattr(cli_mod, "run_experiment", fake)
    r = CliRunner().invoke(
        cli_mod.cli,
        ["--log-file", str(log_path), "backtest", "-c", str(REPO / "configs" / "default.yaml"), "--synthetic"],
    )
    assert r.exit_code == 0
    assert log_path.is_file()


def test_seed_flag_overrides_yaml(monkeypatch, tmp_path):
    import stock_transformer.cli as cli_mod

    seen: dict[str, int] = {}

    def prep(path, *, device=None, seed=None):
        from stock_transformer.backtest.runner import prepare_backtest_config as real

        out = real(path, device=device, seed=seed)
        seen["seed"] = int(out["seed"])
        return out

    def single(cfg, **_k):
        return {"run_dir": str(tmp_path), "seed": cfg.get("seed")}

    monkeypatch.setattr(cli_mod, "prepare_backtest_config", prep)
    monkeypatch.setattr(cli_mod, "run_experiment", single)
    r = CliRunner().invoke(
        cli_mod.cli,
        ["backtest", "-c", str(REPO / "configs" / "default.yaml"), "--synthetic", "--seed", "99"],
    )
    assert r.exit_code == 0
    assert seen.get("seed") == 99


def test_unknown_subcommand_exit_2():
    r = CliRunner().invoke(cli, ["not-a-command"], catch_exceptions=False)
    assert r.exit_code == 2


def test_version_output():
    r = CliRunner().invoke(cli, ["--version"])
    assert r.exit_code == 0
    assert "stock-transformer" in r.output


def test_quiet_suppresses_info(monkeypatch, tmp_path):
    import stock_transformer.cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "run_experiment",
        lambda *_a, **_k: {"run_dir": str(tmp_path)},
    )
    r = CliRunner().invoke(
        cli_mod.cli,
        ["-q", "backtest", "-c", str(REPO / "configs" / "default.yaml"), "--synthetic"],
    )
    assert r.exit_code == 0
    assert "INFO" not in (r.stderr or "")


def test_stx_fetch_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["fetch", "--help"])
    assert result.exit_code == 0
    assert "cache-dir" in result.output.lower() or "--cache-dir" in result.output


def test_stx_data_fetch_matches_fetch(monkeypatch):
    """``stx data fetch`` uses the same command object as ``stx fetch`` (shared flags)."""
    import stock_transformer.cli.commands.fetch as fetch_mod

    calls: list[str] = []

    def capture(cache_dir, symbols, *, refresh):
        calls.append("run")

    monkeypatch.setattr(fetch_mod, "run_fetch", capture)
    runner = CliRunner()
    r1 = runner.invoke(cli, ["fetch", "--symbols", "MSTR"], catch_exceptions=False)
    r2 = runner.invoke(cli, ["data", "fetch", "--symbols", "MSTR"], catch_exceptions=False)
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert len(calls) == 2


def test_stx_completion_bash_prints_script():
    runner = CliRunner()
    r = runner.invoke(cli, ["completion", "bash"], catch_exceptions=False)
    assert r.exit_code == 0
    out = r.stdout or ""
    assert "_STX_COMPLETE" in out and "stx" in out.lower()


def test_main_keyboard_interrupt_returns_130(monkeypatch):
    from stock_transformer.cli import app as app_mod

    def raise_ki(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(app_mod.cli, "main", raise_ki)
    assert app_mod.main([]) == 130


def test_fetch_rejects_blank_symbol_after_strip():
    runner = CliRunner()
    for args in (["fetch", "--symbols", ""], ["fetch", "--symbols", "   "]):
        r = runner.invoke(cli, args, catch_exceptions=False)
        assert r.exit_code != 0
        out = (r.output + (r.stderr or "")).lower()
        assert "non-empty" in out


def test_cache_dir_callback_rejects_whitespace_only_path():
    """``cache_dir_option`` runs after Click's Path conversion; reject useless whitespace paths."""
    import click as click_mod

    from stock_transformer.cli.validators import cache_dir_option

    ctx = click_mod.Context(click_mod.Command("fetch"))
    param = click_mod.Option(["--cache-dir"])
    with pytest.raises(click_mod.BadParameter):
        cache_dir_option(ctx, param, Path("   "))


def test_fetch_normalizes_symbols_to_uppercase(monkeypatch):
    import stock_transformer.cli.commands.fetch as fetch_cmd_mod

    seen: list[list[str]] = []

    def capture(cache_dir, symbols, *, refresh):
        seen.append(list(symbols))

    monkeypatch.setattr(fetch_cmd_mod, "run_fetch", capture)
    runner = CliRunner()
    r = runner.invoke(cli, ["fetch", "--symbols", "mstr"], catch_exceptions=False)
    assert r.exit_code == 0
    assert seen == [["MSTR"]]


def test_config_group_accepts_dash_h():
    runner = CliRunner()
    r = runner.invoke(cli, ["config", "-h"], catch_exceptions=False)
    assert r.exit_code == 0
    assert "show" in r.output and "diff" in r.output


def test_stx_validate_good_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "-c", str(REPO / "configs" / "default.yaml")])
    assert result.exit_code == 0
