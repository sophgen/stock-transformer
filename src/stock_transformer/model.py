"""CandleTransformer: treats each (ticker, day) candle as a token to predict SPY's next candle."""

from __future__ import annotations

import torch
import torch.nn as nn


class CandleTransformer(nn.Module):
    def __init__(
        self,
        n_symbols: int,
        lookback: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        target_symbol_idx: int = 0,
    ) -> None:
        super().__init__()
        self.n_symbols = n_symbols
        self.lookback = lookback
        self.target_symbol_idx = target_symbol_idx
        self.seq_len = lookback * n_symbols

        self.feat_proj = nn.Linear(5, d_model)
        self.ticker_embed = nn.Embedding(n_symbols, d_model)
        self.pos_embed = nn.Embedding(lookback, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L*S, 5] candle features

        Returns:
            [B, 4] predicted OHLC log-returns for target symbol
        """
        B = x.shape[0]
        S = self.n_symbols
        L = self.lookback

        projected = self.feat_proj(x)  # [B, L*S, d_model]

        ticker_ids = torch.arange(S, device=x.device).repeat(L)  # [L*S]
        pos_ids = torch.arange(L, device=x.device).repeat_interleave(S)  # [L*S]

        projected = projected + self.ticker_embed(ticker_ids) + self.pos_embed(pos_ids)

        encoded = self.encoder(projected)  # [B, L*S, d_model]

        target_token_idx = (L - 1) * S + self.target_symbol_idx
        target_repr = encoded[:, target_token_idx, :]  # [B, d_model]

        return self.head(target_repr)  # [B, 4]
