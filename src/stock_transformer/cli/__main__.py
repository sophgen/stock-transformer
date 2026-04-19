"""Allow ``python -m stock_transformer.cli`` (used in tests and docs)."""

from __future__ import annotations

from stock_transformer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
