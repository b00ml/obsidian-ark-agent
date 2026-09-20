"""Adapters from legacy source-specific results to shared P0 contracts.

The adapters deliberately do not call providers or touch storage. They only
normalise the small, historically different dictionaries returned by Bili,
article, inbox, RAG and memory code paths.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from agentlab.contracts import Citation, CitationStatus, ProcessStatus, SourceDocument


_KNOWN_FIELDS = {
    "id", "source_id", "source_kind", "source_ref", "ref", "url", "bvid",
    "title", "content", "transcript", "note", "body", "status", "strategy",
    "project_id", "session_id", "created_at", "updated_at", "source_hash",
    "content_hash", "kind", "type", "tags", "metadata", "results", "total",
}


def _jsonable(value: Any) -> Any:
    """Keep adapter metadata serialisable without silently dropping values."""

    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_jsonable(item) for item in value]
        return str(value)


def _payload(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"value": value}


def _status(value: Any, *, default: ProcessStatus = ProcessStatus.ACCEPTED) -> ProcessStatus:
    text = str(value or "").strip().lower()
    aliases = {
        "ok": ProcessStatus.COMPLETED,
        "success": ProcessStatus.COMPLETED,
        "done": ProcessStatus.COMPLETED,
        "error": ProcessStatus.FAILED,
        "failed": ProcessStatus.FAILED,
        "pending": ProcessStatus.ACCEPTED,
        "processing": ProcessStatus.FETCHED,
        "dead_letter": ProcessStatus.DEAD_LETTER,
        "dead-letter": ProcessStatus.DEAD_LETTER,
    }
    if text in aliases:
        return aliases[text]
    try:
        return ProcessStatus(text)
    except ValueError:
        return default


def _content(data: Mapping[str, Any]) -> str:
    for key in ("content", "transcript", "note", "body"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), Mapping) else {}
    for key, value in data.items():
        if key not in _KNOWN_FIELDS:
            metadata[str(key)] = _jsonable(value)
    return _jsonable(metadata)


def adapt_source_document(
    value: Mapping[str, Any] | Any,
    *,
    source_kind: str,
    source_ref: str = "",
    project_id: str = "",
) -> SourceDocument:
    """Adapt one legacy result while retaining unknown fields in metadata."""

    data = _payload(value)
    ref = source_ref or str(
        data.get("source_ref") or data.get("ref") or data.get("url")
        or data.get("bvid") or data.get("id") or source_kind
    )
    source_id = str(data.get("source_id") or data.get("id") or ref)
    title = str(data.get("title") or data.get("name") or ref)
    content_hash = str(data.get("content_hash") or data.get("source_hash") or "")
    if not content_hash:
        content_hash = hashlib.sha256(_content(data).encode("utf-8")).hexdigest()
    return SourceDocument(
        id=source_id,
        source_kind=source_kind,
        source_ref=ref,
        title=title,
        content_hash=content_hash,
        project_id=project_id or str(data.get("project_id") or ""),
        status=_status(data.get("status")),
        metadata=_metadata(data),
    )


def from_bili_result(value: Mapping[str, Any] | Any, *, project_id: str = "") -> SourceDocument:
    return adapt_source_document(value, source_kind="bilibili", project_id=project_id)


def from_article_result(value: Mapping[str, Any] | Any, *, url: str = "",
                        project_id: str = "") -> SourceDocument:
    return adapt_source_document(
        value, source_kind="article", source_ref=url, project_id=project_id
    )


def from_inbox_task(value: Mapping[str, Any] | Any, *, project_id: str = "") -> SourceDocument:
    return adapt_source_document(value, source_kind="inbox", project_id=project_id)


def from_rag_item(value: Mapping[str, Any] | Any, *, project_id: str = "",
                  session_id: str = "") -> SourceDocument:
    document = adapt_source_document(value, source_kind="rag", project_id=project_id)
    if session_id:
        document.metadata["session_id"] = session_id
    return document


def from_memory_entry(value: Mapping[str, Any] | Any, *, project_id: str = "",
                      session_id: str = "") -> SourceDocument:
    document = adapt_source_document(value, source_kind="memory", project_id=project_id)
    if session_id:
        document.metadata["session_id"] = session_id
    return document


def citation_from_result(value: Mapping[str, Any] | Any, *, source_kind: str = "vault",
                         project_id: str = "", session_id: str = "") -> Citation:
    data = _payload(value)
    ref = str(data.get("ref") or data.get("source_ref") or data.get("url") or data.get("id") or "")
    status = str(data.get("status") or "active").strip().lower()
    try:
        citation_status = CitationStatus(status)
    except ValueError:
        citation_status = CitationStatus.UNVERIFIED
    return Citation(
        ref=ref,
        source_kind=str(data.get("source_kind") or data.get("source") or source_kind),
        source_hash=str(data.get("source_hash") or data.get("content_hash") or ""),
        project_id=project_id or str(data.get("project_id") or ""),
        session_id=session_id or str(data.get("session_id") or ""),
        status=citation_status,
        metadata=_metadata(data),
    )


__all__ = [
    "adapt_source_document", "from_bili_result", "from_article_result",
    "from_inbox_task", "from_rag_item", "from_memory_entry", "citation_from_result",
]
