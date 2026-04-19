"""M7b long predictions schema and golden self-consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def test_golden_predictions_self_consistent():
    path = Path(__file__).resolve().parent / "golden" / "predictions_universe.csv"
    df = pd.read_csv(path)
    expected_cols = [
        "timestamp",
        "symbol",
        "timeframe",
        "y_true_raw_return",
        "y_true_relative_return",
        "y_score",
        "y_rank_pred",
        "y_rank_true",
        "fold_id",
    ]
    assert list(df.columns) == expected_cols
    for (_, _), g in df.groupby(["fold_id", "timestamp"], sort=False):
        r_raw = g["y_true_raw_return"].to_numpy(dtype=np.float64)
        m = np.isfinite(r_raw)
        if m.sum() < 1:
            continue
        exp = rankdata(-r_raw[m], method="average")
        got = g["y_rank_true"].to_numpy(dtype=np.float64)[m]
        np.testing.assert_allclose(exp, got, rtol=1e-6, atol=1e-6)
