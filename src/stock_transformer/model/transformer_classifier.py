"""Causal Transformer encoder for binary next-candle direction."""

from __future__ import annotations

import torch
import torch.nn as nn


class CandleTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, n_features]
        returns logits [batch]
        """
        h = self.input_proj(x)
        seq_len = h.size(1)
        # Explicit causal mask (required when using causal attention in recent PyTorch)
        causal = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=h.device, dtype=h.dtype
        )
        h = self.encoder(h, mask=causal, is_causal=False)
        h = self.norm(h[:, -1, :])
        return self.head(h).squeeze(-1)


def predict_proba(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)
