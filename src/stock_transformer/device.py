"""Resolve PyTorch device strings from config or CLI (no model imports).

Keeping this module free of ``nn.Module`` dependencies avoids import cycles and
lets tests and runners choose a device without pulling in the transformer stack.
"""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Map config/CLI device string to ``torch.device``.

    ``auto`` prefers MPS (Apple Silicon) when built and available, then CUDA, else CPU.
    Accepts ``cpu``, ``mps``, ``cuda``, ``cuda:N``, or any string ``torch.device`` understands.
    """
    raw = str(name).strip()
    key = raw.lower()
    if key == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if key == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "device is 'mps' but MPS is not available. "
                "Use device: auto or cpu, or a PyTorch build with MPS support."
            )
        return torch.device("mps")
    if key == "cuda" or key.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("device requests CUDA but CUDA is not available. Use device: cpu or mps (Apple Silicon).")
        return torch.device(raw)
    if key == "cpu":
        return torch.device("cpu")
    return torch.device(raw)
