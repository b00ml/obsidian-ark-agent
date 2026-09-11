"""Repository path helpers used by compatibility facades.

This module is the only place where a facade adjusts ``sys.path``.  The legacy
directories remain the implementation owners during the migration period.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_legacy_importable(directory: str) -> Path:
    """Make one legacy top-level directory importable and return its path."""
    path = (REPO_ROOT / directory).resolve()
    if not path.is_dir():
        raise ImportError(f"legacy component directory not found: {path}")
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    importlib.invalidate_caches()
    return path
