"""Agent Hub service facade.

``agentlab/agentlab`` remains the implementation owner.  This facade is a
stable import boundary for future relocation and does not change CLI startup.
"""
from __future__ import annotations

from packages._paths import ensure_legacy_importable

ensure_legacy_importable("agentlab")

from agentlab.runtime.cli import main  # noqa: E402,F401
from agentlab.runtime.config import Config, load_config  # noqa: E402,F401
from agentlab.runtime.serve import Serve, serve_config  # noqa: E402,F401
from agentlab.runtime.serve_manage import (  # noqa: E402,F401
    cmd_start,
    cmd_status,
    cmd_stop,
)

__all__ = [
    "Config",
    "Serve",
    "cmd_start",
    "cmd_status",
    "cmd_stop",
    "load_config",
    "main",
    "serve_config",
]
