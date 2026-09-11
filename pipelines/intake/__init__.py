"""Inbox intake pipeline facade.

The SQLite queue and Agent Mail adapter remain owned by ``inbox_collector``.
"""
from __future__ import annotations

from packages._paths import ensure_legacy_importable

ensure_legacy_importable("inbox_collector")

from inbox_poll import (  # noqa: E402,F401
    classify,
    extract_bvid,
    extract_urls,
    fingerprint,
    normalize_url,
    poll,
)
from queue_store import InboxQueueStore  # noqa: E402,F401

__all__ = [
    "InboxQueueStore",
    "classify",
    "extract_bvid",
    "extract_urls",
    "fingerprint",
    "normalize_url",
    "poll",
]
