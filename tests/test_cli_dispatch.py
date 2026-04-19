"""M7b CLI exit codes and universe dispatch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_cli_missing_config_exit_1():
    r = subprocess.run(
        [sys.executable, "-m", "stock_transformer.cli", "-c", str(REPO / "nope.yaml")],
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
        [sys.executable, "-m", "stock_transformer.cli", "--synthetic", "-c", str(p)],
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


def test_cli_partial_failure_exit_2(tmp_path, monkeypatch):
    import stock_transformer.cli as cli_mod

    def boom(*_a, **_k):
        return {"run_dir": str(tmp_path), "fold_errors": [{"fold_id": 0, "error": "x"}]}

    monkeypatch.setattr(cli_mod, "run_from_config_path", boom)
    code = cli_mod.main(["-c", str(REPO / "configs" / "default.yaml")])
    assert code == 2
