"""Compatibility CLI entrypoint for the Agent Hub service."""
from __future__ import annotations

from packages._paths import ensure_legacy_importable

ensure_legacy_importable("agentlab")

from agentlab.runtime.cli import main  # noqa: E402,F401

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
