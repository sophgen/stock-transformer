"""Training loop, evaluation, and device/seed helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_data(
    X: torch.Tensor,
    y: torch.Tensor,
    train_pct: float,
    val_pct: float,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Chronological train/val/test split."""
    n = len(X)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = (X[:train_end], y[:train_end])
    val = (X[train_end:val_end], y[train_end:val_end])
    test = (X[val_end:], y[val_end:])

    print(f"  Split: train={train_end}, val={val_end - train_end}, test={n - val_end}")
    return train, val, test


def train_model(
    model: nn.Module,
    train_data: tuple[torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    patience: int = 5,
) -> nn.Module:
    """Train with early stopping on validation loss."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(train_data[0].to(device), train_data[1].to(device))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_X, val_y = val_data[0].to(device), val_data[1].to(device)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_y).item()

        avg_train = train_loss / max(n_batches, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        if epoch <= 5 or epoch % 5 == 0 or no_improve == 0:
            print(f"  Epoch {epoch:3d}  train={avg_train:.6f}  val={val_loss:.6f}{marker}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(
    model: nn.Module,
    test_data: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> None:
    """Evaluate on test set and print metrics."""
    model.eval()
    test_X, test_y = test_data[0].to(device), test_data[1].to(device)

    with torch.no_grad():
        pred = model(test_X)

    mse_per = ((pred - test_y) ** 2).mean(dim=0).cpu().numpy()
    mse_total = ((pred - test_y) ** 2).mean().item()

    pred_close_sign = (pred[:, 3] > 0).cpu().numpy()
    true_close_sign = (test_y[:, 3] > 0).cpu().numpy()
    direction_acc = (pred_close_sign == true_close_sign).mean()

    labels = ["open", "high", "low", "close"]
    print("\n  === Test Results ===")
    for i, name in enumerate(labels):
        print(f"  MSE {name:>5s}: {mse_per[i]:.6f}")
    print(f"  MSE total: {mse_total:.6f}")
    print(f"  Direction accuracy (close): {direction_acc:.1%}")
    print(f"  Test samples: {len(test_y)}")
