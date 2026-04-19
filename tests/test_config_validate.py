"""Config validation (fail-fast unknown keys / bad shapes)."""

from __future__ import annotations

import pytest

from stock_transformer.config_validate import (
    validate_single_symbol_config,
    validate_universe_config,
)


def test_universe_rejects_empty_symbols():
    with pytest.raises(ValueError, match="symbols"):
        validate_universe_config({"experiment_mode": "universe", "symbols": []})


def test_universe_requires_walkforward_keys():
    with pytest.raises(ValueError, match="train_bars"):
        validate_universe_config(
            {
                "experiment_mode": "universe",
                "symbols": ["A", "B"],
                "lookback": 4,
            },
        )


def test_universe_rejects_bad_min_coverage():
    with pytest.raises(ValueError, match="min_coverage_symbols"):
        validate_universe_config(
            {
                "experiment_mode": "universe",
                "symbols": ["A", "B", "C"],
                "train_bars": 10,
                "val_bars": 5,
                "test_bars": 5,
                "step_bars": 5,
                "lookback": 4,
                "min_coverage_symbols": 10,
            },
        )


def test_single_symbol_requires_symbol_and_timeframes():
    with pytest.raises(ValueError, match="symbol"):
        validate_single_symbol_config(
            {"timeframes": ["daily"], "train_bars": 1, "val_bars": 1, "test_bars": 1, "step_bars": 1}
        )
    with pytest.raises(ValueError, match="timeframes"):
        validate_single_symbol_config({"symbol": "X", "train_bars": 1, "val_bars": 1, "test_bars": 1, "step_bars": 1})
