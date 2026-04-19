"""Transformer ranker: per-symbol temporal encoding + cross-sectional mixing → scores."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerRanker(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_symbols: int,
        d_model: int = 64,
        nhead: int = 4,
        num_temporal_layers: int = 2,
        num_cross_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.n_symbols = n_symbols
        self.d_model = d_model

        self.feat_proj = nn.Linear(n_features, d_model)
        self.sym_embed = nn.Embedding(n_symbols, d_model)
        self.temporal_pos = nn.Embedding(max_seq_len, d_model)

        t_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            self.temporal_enc = nn.TransformerEncoder(
                t_layer, num_layers=num_temporal_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.temporal_enc = nn.TransformerEncoder(t_layer, num_layers=num_temporal_layers)

        c_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            self.cross_enc = nn.TransformerEncoder(
                c_layer, num_layers=num_cross_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.cross_enc = nn.TransformerEncoder(c_layer, num_layers=num_cross_layers)

        self.norm = nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            ``[B, L, S, F]`` features.
        padding_mask
            ``[B, L, S]`` bool — True for missing / pad positions.
        """
        B, L, S, _ = x.shape
        if S != self.n_symbols:
            raise ValueError(f"Expected S={self.n_symbols}, got {S}")

        sym_ids = torch.arange(S, device=x.device, dtype=torch.long).view(1, 1, S).expand(B, L, S)

        h = self.feat_proj(x)
        h = h + self.sym_embed(sym_ids)
        pos_ids = torch.arange(L, device=x.device, dtype=torch.long)
        pos_e = self.temporal_pos(pos_ids).view(1, L, 1, self.d_model)
        h = h + pos_e

        h = h.permute(0, 2, 1, 3).reshape(B * S, L, self.d_model)
        pm = padding_mask.permute(0, 2, 1).reshape(B * S, L)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=x.device, dtype=h.dtype)
        pad_attn = pm.to(dtype=h.dtype) * torch.finfo(h.dtype).min
        h = self.temporal_enc(h, mask=causal, src_key_padding_mask=pad_attn)
        h = h.view(B, S, L, self.d_model)

        lengths = (~padding_mask).sum(dim=1)
        last_ix = (lengths - 1).clamp(min=0)
        dead = lengths <= 0
        b_idx = torch.arange(B, device=x.device)[:, None].expand(B, S)
        s_idx = torch.arange(S, device=x.device)[None, :].expand(B, S)
        z = h[b_idx, s_idx, last_ix[b_idx, s_idx], :]
        z = torch.where(dead[..., None], torch.zeros_like(z), z)

        cross_pad = padding_mask[:, -1, :] | dead
        z = self.cross_enc(z, src_key_padding_mask=cross_pad)
        z = self.norm(z)
        scores = self.score_head(z).squeeze(-1)
        return scores
