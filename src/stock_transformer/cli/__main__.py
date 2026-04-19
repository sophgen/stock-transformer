"""Allow ``python -m stock_transformer.cli`` (used in tests and docs).

Mirrors the console-script entry point so subprocess tests exercise the same import
graph as ``stx`` without requiring a venv ``bin`` directory on ``PATH``.
"""

from __future__ import annotations

from stock_transformer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
