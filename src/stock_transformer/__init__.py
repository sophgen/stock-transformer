"""Leakage-safe candle sequences, transformer direction model, walk-forward evaluation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stock-transformer")
except PackageNotFoundError:  # editable install / loose path
    __version__ = "0.0.0"
