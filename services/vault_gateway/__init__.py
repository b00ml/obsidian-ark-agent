"""Vault Gateway service facade.

The governed write implementation remains in
``obsidian_agent_brain.vault_gateway``; this package only stabilizes its import
boundary for Agent Hub and future service extraction.
"""
from __future__ import annotations

from packages._paths import ensure_legacy_importable

ensure_legacy_importable("obsidian_agent_brain")

from vault_gateway import VaultConflictError, VaultGateway  # noqa: E402,F401

__all__ = ["VaultConflictError", "VaultGateway"]
