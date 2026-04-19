"""PyTorch device resolution from config/CLI strings."""

from __future__ import annotations

import pytest
import torch

from stock_transformer.model.transformer_classifier import resolve_device


def test_resolve_cpu_explicit():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_auto_returns_valid_device():
    d = resolve_device("auto")
    assert d.type in ("mps", "cuda", "cpu")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_resolve_mps_explicit():
    assert resolve_device("mps").type == "mps"


def test_resolve_mps_raises_when_unavailable():
    if torch.backends.mps.is_available():
        pytest.skip("MPS is available")
    with pytest.raises(ValueError, match="mps"):
        resolve_device("mps")
