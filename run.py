#!/usr/bin/env python3
"""Minimal entry point: fetch data, build features, train transformer, evaluate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml
from dotenv import load_dotenv

from stock_transformer.data import fetch_universe, align_universe
from stock_transformer.features import build_features
from stock_transformer.model import CandleTransformer
from stock_transformer.train import (
    evaluate,
    get_device,
    seed_everything,
    split_data,
    train_model,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    load_dotenv()

    config_path = "configs/default.yaml"
    if "-c" in sys.argv:
        idx = sys.argv.index("-c")
        config_path = sys.argv[idx + 1]

    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    device = get_device()

    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Symbols: {cfg['symbols']}")
    print(f"Target: {cfg['target_symbol']}")
    print(f"Lookback: {cfg['lookback']} days")
    print()

    print("Fetching data...")
    candles = fetch_universe(cfg["symbols"], cache_dir=cfg["cache_dir"])

    print("Aligning...")
    aligned = align_universe(candles)

    print("Building features...")
    X, y, symbols = build_features(aligned, cfg["target_symbol"], cfg["lookback"])

    print("Splitting data...")
    train_data, val_data, test_data = split_data(X, y, cfg["train_pct"], cfg["val_pct"])

    target_idx = symbols.index(cfg["target_symbol"].upper())
    model = CandleTransformer(
        n_symbols=len(symbols),
        lookback=cfg["lookback"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        target_symbol_idx=target_idx,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")

    print("Training...")
    model = train_model(
        model,
        train_data,
        val_data,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        device=device,
    )

    evaluate(model, test_data, device)


if __name__ == "__main__":
    main()
