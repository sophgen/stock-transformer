"""M10 ranking / listwise losses."""

from __future__ import annotations

import torch

from stock_transformer.model.losses import approx_ndcg_loss, listnet_loss, masked_mse


def _fixture_batch():
    pred = torch.tensor([[1.0, 2.0, 0.5, float("nan")]], requires_grad=True)
    tgt = torch.tensor([[0.1, 0.3, 0.2, float("nan")]])
    m = torch.tensor([[True, True, True, False]])
    return pred, tgt, m


def test_masked_mse_grad():
    pred, tgt, m = _fixture_batch()
    loss = masked_mse(pred, tgt, mask=m)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad[:, :3]).all()


def test_listnet_grad():
    pred, tgt, m = _fixture_batch()
    loss = listnet_loss(pred, tgt, mask=m)
    loss.backward()
    assert pred.grad is not None


def test_approx_ndcg_grad():
    pred, tgt, m = _fixture_batch()
    loss = approx_ndcg_loss(pred, tgt, mask=m, alpha=5.0)
    loss.backward()
    assert pred.grad is not None


def test_degenerate_row_zero_loss():
    pred = torch.tensor([[1.0, float("nan")]], requires_grad=True)
    tgt = torch.tensor([[0.1, float("nan")]])
    m = torch.tensor([[True, False]])
    loss = listnet_loss(pred, tgt, mask=m)
    assert loss.item() == 0.0
    assert loss.requires_grad


def test_masked_mse_zero_when_pred_equals_target():
    pred = torch.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
    tgt = pred.detach().clone()
    m = torch.tensor([[True, True, True]])
    loss = masked_mse(pred, tgt, mask=m)
    assert loss.item() == 0.0


def test_listnet_lower_when_pred_aligns_with_targets():
    tgt = torch.tensor([[0.1, 0.5, 0.2], [0.0, 1.0, 2.0]])
    m = torch.tensor([[True, True, True], [True, True, True]])
    pred_good = tgt * 10.0
    pred_bad = torch.zeros_like(tgt)
    assert listnet_loss(pred_good, tgt, mask=m) < listnet_loss(pred_bad, tgt, mask=m)


def test_approx_ndcg_lower_when_pred_tracks_gains():
    tgt = torch.tensor([[3.0, 1.0, 2.0]])
    m = torch.tensor([[True, True, True]])
    pred_good = tgt.clone()
    pred_bad = -tgt
    assert approx_ndcg_loss(pred_good, tgt, mask=m, alpha=5.0) < approx_ndcg_loss(pred_bad, tgt, mask=m, alpha=5.0)
