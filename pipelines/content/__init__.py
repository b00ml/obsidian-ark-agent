"""Content pipeline facade.

The existing Bilibili and article modules remain the implementation owners.
Only selected public functions are exposed here; CLI behavior is unchanged.
"""
from __future__ import annotations

from packages._paths import ensure_legacy_importable

ensure_legacy_importable("bili_summarizer")

from article_summarizer import (  # noqa: E402,F401
    build_note,
    extract_article,
    fetch_article,
    summarize_article,
)
from bili_transcript import (  # noqa: E402,F401
    enqueue_bili_batch,
    enumerate_channel,
    extract_bvid,
    get_video_meta,
    parse_batch_file,
)

__all__ = [
    "build_note",
    "enqueue_bili_batch",
    "enumerate_channel",
    "extract_article",
    "extract_bvid",
    "fetch_article",
    "get_video_meta",
    "parse_batch_file",
    "summarize_article",
]
