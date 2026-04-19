"""Training losses for cross-sectional rankers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
    """MSE over positions where ``mask`` is True (valid)."""
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
    err = (pred - torch.nan_to_num(target, nan=0.0)) ** 2
    return err[mask].mean()


def listnet_loss(pred: torch.Tensor, target: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
    """ListNet-style cross-entropy between target softmax (from gains) and pred softmax."""
    B, _S = pred.shape
    parts: list[torch.Tensor] = []
    for i in range(B):
        m = mask[i]
        if int(m.sum().item()) < 2:
            continue
        logits = pred[i].masked_fill(~m, float("-inf"))
        log_p = F.log_softmax(logits[m], dim=0)
        y = torch.nan_to_num(target[i, m], nan=0.0).float()
        tgt = F.softmax(y * 10.0, dim=0)
        parts.append(-(tgt * log_p).sum())
    if not parts:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
    return torch.stack(parts).mean()


def approx_ndcg_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor,
    alpha: float = 10.0,
) -> torch.Tensor:
    """Pairwise smooth loss weighted by gain differences (Chapelle-style surrogate)."""
    B, _S = pred.shape
    parts: list[torch.Tensor] = []
    a = float(alpha)
    for i in range(B):
        m = mask[i]
        nv = int(m.sum().item())
        if nv < 2:
            continue
        s = pred[i, m]
        y = torch.nan_to_num(target[i, m], nan=0.0).float()
        g = torch.relu(y - y.min())
        if float(g.max().item()) < 1e-12:
            continue
        order = torch.argsort(g, descending=True)
        discounts = torch.log2(torch.arange(2, nv + 2, device=pred.device, dtype=s.dtype))
        idcg = (g[order] / discounts).sum().clamp(min=1e-12)
        g_row = g.unsqueeze(1)
        g_col = g.unsqueeze(0)
        gain_diff = g_row - g_col
        pair_mask = gain_diff > 0
        s_row = s.unsqueeze(1)
        s_col = s.unsqueeze(0)
        sdiff = s_row - s_col
        contrib = F.softplus(-a * sdiff) * gain_diff * pair_mask.to(dtype=s.dtype)
        n_pairs = pair_mask.sum()
        if n_pairs.item() == 0:
            continue
        parts.append(contrib.sum() / (idcg * n_pairs.to(dtype=s.dtype)))
    if not parts:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
    return torch.stack(parts).mean()


def training_loss(
    name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask_valid: torch.Tensor,
    ndcg_alpha: float = 10.0,
) -> torch.Tensor:
    """``mask_valid``: True = finite target and not padded at last step."""
    n = name.lower().strip()
    if n == "mse":
        return masked_mse(pred, target, mask=mask_valid)
    if n == "listnet":
        return listnet_loss(pred, target, mask=mask_valid)
    if n in ("approx_ndcg", "approx-ndcg"):
        return approx_ndcg_loss(pred, target, mask=mask_valid, alpha=ndcg_alpha)
    raise ValueError(f"Unknown loss: {name}")
