"""Ark product boundary; TypeScript source remains in the legacy ``ark/`` owner."""
from __future__ import annotations

from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "ark"

__all__ = ["SOURCE_ROOT"]
