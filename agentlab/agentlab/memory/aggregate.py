"""Deterministic profile/topic views for atomic long-term memory.

The Markdown files remain the source of truth.  This module only builds a
bounded, rebuildable projection that groups active memories by semantic type,
scope and tags.  It deliberately performs no LLM summary or merge: each item
keeps its stable memory id and source ref so a reviewer can trace it back.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agentlab.memory.governance import DEFAULT_READ_STATUSES, content_hash, status_matches


SCHEMA = "memory-aggregate-v1"
TYPE_GROUPS = {
    "profile": frozenset({"core", "context"}),
    "decision": frozenset({"decisions"}),
    "procedure": frozenset({"procedures"}),
    "episodic": frozenset({"sessions"}),
}


def _group_for(mem_type: str) -> str:
    value = str(mem_type or "").strip().lower()
    for group, types in TYPE_GROUPS.items():
        if value in types:
            return group
    return "other"


def _safe_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        values = value.replace("，", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))[:8]


def _memory_rows(store, *, project_id: str | None = None,
                 include_archive: bool = False) -> list[dict[str, Any]]:
    root = Path(store.memory_root)
    rows: list[dict[str, Any]] = []
    allowed = set(DEFAULT_READ_STATUSES)
    if include_archive:
        allowed.update({"archived", "superseded", "revoked", "expired"})
    for mem_type in getattr(store, "MEMORY_TYPES", ("core", "context", "procedures", "decisions", "sessions")):
        if mem_type == "archive" and not include_archive:
            continue
        folder = root / mem_type
        if not folder.exists():
            continue
        for path in folder.rglob("*.md"):
            try:
                memory = store._parse_memory_file(path)
            except Exception:
                continue
            status = str(memory.get("status") or "active").strip().lower()
            if status not in allowed:
                continue
            effective_allowed, _ = status_matches(
                memory, allowed, include_archive=include_archive,
            )
            if not effective_allowed:
                continue
            row_project = str(memory.get("project_id") or "default")
            if project_id and row_project not in {project_id, "default"}:
                continue
            content = str(memory.get("content") or "").strip()
            if not content:
                continue
            rel = path.relative_to(store.vault_root).as_posix()
            rows.append({
                "id": str(memory.get("id") or ""),
                "type": str(memory.get("type") or mem_type),
                "group": _group_for(memory.get("type") or mem_type),
                "project_id": row_project,
                "scope": memory.get("scope") if isinstance(memory.get("scope"), Mapping) else {},
                "status": status,
                "importance": int(memory.get("importance") or 0),
                "confidence": memory.get("confidence", 1.0),
                "tags": _safe_tags(memory.get("tags")),
                "source_ref": rel,
                "content_hash": str(memory.get("content_hash") or content_hash(content)),
                "content": content[:400],
                "updated_at": str(memory.get("updated_at") or ""),
            })
    return rows


def build_memory_aggregate(store, *, project_id: str | None = None,
                           include_archive: bool = False,
                           max_items_per_topic: int = 20) -> dict[str, Any]:
    """Build a deterministic profile/topic projection without mutating source."""
    rows = _memory_rows(store, project_id=project_id, include_archive=include_archive)
    try:
        cap = max(1, min(100, int(max_items_per_topic)))
    except (TypeError, ValueError, OverflowError):
        cap = 20
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        tags = row["tags"] or [row["type"]]
        for tag in tags[:3]:
            groups[row["group"]][tag].append(row)
    topics: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group, by_tag in groups.items():
        topics[group] = {}
        for tag, items in by_tag.items():
            ordered = sorted(
                items,
                key=lambda item: (-int(item.get("importance") or 0),
                                  str(item.get("updated_at") or ""),
                                  str(item.get("id") or "")),
            )
            topics[group][tag] = [
                {key: item[key] for key in (
                    "id", "type", "project_id", "status", "importance", "confidence",
                    "tags", "source_ref", "content_hash", "content", "updated_at",
                )}
                for item in ordered[:cap]
            ]
    manifest = hashlib.sha256(json.dumps(
        [(row["id"], row["content_hash"], row["status"]) for row in rows],
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id or "",
        "include_archive": bool(include_archive),
        "items": len(rows),
        "manifest": manifest,
        "groups": {group: dict(by_tag) for group, by_tag in sorted(topics.items())},
    }


def write_memory_aggregate(store, path: str | Path | None = None, **kwargs) -> Path:
    """Write the hidden rebuildable projection atomically; never writes Markdown."""
    target = Path(path) if path else Path(store.vault_root) / ".agent-brain" / "memory" / "aggregate-v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_memory_aggregate(store, **kwargs), ensure_ascii=False, indent=2) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temp_name, target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return target


__all__ = ["SCHEMA", "TYPE_GROUPS", "build_memory_aggregate", "write_memory_aggregate"]
