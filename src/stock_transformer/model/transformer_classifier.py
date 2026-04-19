"""Autoregressive candle transformer: multi-timeframe tokens → next candle prediction.

Each candle (from any timeframe) is a token carrying OHLCV log-return features
plus a learned timeframe embedding.  A causal Transformer encoder attends only to
past tokens, and the last-token representation feeds:

* a **regression head** — predicts the next candle's OHLCV log-returns, and
* a **direction head** — binary logit for next close up / down.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CandleTransformer(nn.Module):
    def __init__(
        self,
        n_candle_features: int = 5,
        n_timeframes: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.candle_proj = nn.Linear(n_candle_features, d_model)
        self.timeframe_embed = nn.Embedding(n_timeframes, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.input_drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_candle_features),
        )
        self.direction_head = nn.Linear(d_model, 1)

    def forward(
        self,
        candle_features: torch.Tensor,
        timeframe_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        candle_features : ``[B, S, n_candle_features]``
        timeframe_ids   : ``[B, S]``  long
        padding_mask    : ``[B, S]``  bool — ``True`` for padded positions

        Returns
        -------
        candle_pred     : ``[B, n_candle_features]``  next-candle return prediction
        direction_logit : ``[B]``                     binary logit (up / down)
        """
        B, S, _ = candle_features.shape

        h = self.candle_proj(candle_features)
        h = h + self.timeframe_embed(timeframe_ids)
        positions = torch.arange(S, device=h.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embed(positions)
        h = self.input_drop(h)

        causal = nn.Transformer.generate_square_subsequent_mask(S, device=h.device, dtype=h.dtype)
        pad = padding_mask.to(dtype=h.dtype) * torch.finfo(h.dtype).min if padding_mask is not None else None
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad)

        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1
            last = h[torch.arange(B, device=h.device), lengths]
        else:
            last = h[:, -1, :]

        last = self.norm(last)
        candle_pred = self.regression_head(last)
        direction_logit = self.direction_head(last).squeeze(-1)
        return candle_pred, direction_logit


# ---------------------------------------------------------------------------
# Backward-compat aliases so older imports keep working
# ---------------------------------------------------------------------------


class CandleTransformerClassifier(CandleTransformer):
    """Legacy wrapper — delegates to :class:`CandleTransformer`."""

    def __init__(self, n_features: int, **kwargs) -> None:
        kwargs.setdefault("n_candle_features", n_features)
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        B, S, _ = x.shape
        tf_ids = torch.zeros(B, S, dtype=torch.long, device=x.device)
        _, direction_logit = super().forward(x, tf_ids)
        return direction_logit


def predict_proba(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


predict_direction_proba = predict_proba


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


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
            raise ValueError(
                "device requests CUDA but CUDA is not available. Use device: cpu or mps (Apple Silicon)."
            )
        return torch.device(raw)
    if key == "cpu":
        return torch.device("cpu")
    return torch.device(raw)
