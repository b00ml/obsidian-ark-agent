"""Compatibility entrypoint for Agent Hub serve lifecycle management."""
from __future__ import annotations

from packages._paths import ensure_legacy_importable

ensure_legacy_importable("agentlab")

from agentlab.runtime.serve_manage import (  # noqa: E402,F401
    cmd_start,
    cmd_status,
    cmd_stop,
    main,
)

__all__ = ["cmd_start", "cmd_status", "cmd_stop", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
