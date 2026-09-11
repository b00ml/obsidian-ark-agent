"""Compatibility entrypoint for ``agentlab serve``."""
from __future__ import annotations

from packages._paths import ensure_legacy_importable

ensure_legacy_importable("agentlab")

from agentlab.runtime.serve import Serve, main  # noqa: E402,F401

__all__ = ["Serve", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
