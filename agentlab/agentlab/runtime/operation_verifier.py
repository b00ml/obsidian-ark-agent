"""Deterministic external-state checks for recovered TaskState operations.

The verifier is intentionally conservative: it only settles an ``unknown``
operation when a durable local artifact matches the operation's safe,
pre-recorded fingerprint.  A missing adapter or a mismatch remains unknown and
therefore cannot trigger an automatic replay.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any, Mapping


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_vault_path(value: str) -> str:
    path = str(value or "").replace("\\", "/").strip().lstrip("./")
    if "/" not in path:
        path = f"Inbox/{path}"
    if not path.lower().endswith(".md"):
        path += ".md"
    return path


def build_verification_metadata(tool_name: str, arguments: str | Mapping[str, Any],
                                *, vault_root: str | Path | None = None,
                                operation_id: str = "") -> dict[str, str]:
    """Build bounded, non-secret metadata before a tool is dispatched."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        args = {}
    name = str(tool_name or "")
    remote_operation_id = str(args.get("remote_operation_id") or "").strip()
    if remote_operation_id:
        return {"kind": "remote_operation",
                "remote_operation_id": remote_operation_id[:256]}
    if name in {"vault_write", "vault_patch"}:
        result = {"kind": "vault_file", "ref": _normalise_vault_path(args.get("path", ""))}
        if name == "vault_write":
            result["content_hash"] = _sha256(str(args.get("content", "")))
        else:
            result["kind"] = "vault_patch"
            result["new_hash"] = _sha256(str(args.get("new", "")))
            if vault_root:
                target = Path(vault_root).resolve() / result["ref"]
                try:
                    current = target.read_text(encoding="utf-8")
                    old, new = str(args.get("old", "")), str(args.get("new", ""))
                    if old and current.count(old) == 1:
                        result["result_hash"] = _sha256(current.replace(old, new))
                except (OSError, UnicodeError):
                    pass
        return result
    if name == "memory_commit":
        return {"kind": "memory_content", "content_hash": _sha256(str(args.get("content", "")))}
    if name == "memory_correct":
        return {"kind": "memory_correct", "ref": str(args.get("mem_id", ""))[:256],
                "content_hash": _sha256(str(args.get("content", "")))}
    if name in {"memory_revoke", "memory_delete", "memory_restore"}:
        return {"kind": f"memory_{name.removeprefix('memory_')}",
                "ref": str(args.get("mem_id", ""))[:256]}
    if name == "memory_review":
        return {"kind": "memory_review", "ref": str(args.get("mem_id", ""))[:256],
                "content_hash": str(args.get("expected_content_hash", ""))[:64],
                "decision": str(args.get("decision", ""))[:32],
                "defer_until": str(args.get("defer_until", ""))[:64]}
    if name in {"bili_screenshot", "bili_visual"}:
        return {"kind": "bili_artifact", "ref": str(args.get("bvid", ""))[:128]}
    if name == "article_summarize":
        # Strip query/fragment so tracking tokens never enter the ledger while
        # retaining a deterministic source reference for local artifact scans.
        parsed = urlsplit(str(args.get("url", "")))
        source_ref = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"[:512]
        return {"kind": "article_artifact", "source_ref": _sha256(source_ref)}
    if name == "inbox_collect":
        remote_id = str(args.get("remote_operation_id") or "").strip()
        if remote_id:
            return {"kind": "remote_operation", "remote_operation_id": remote_id[:256]}
        return {"kind": "inbox_queue", "operation_id": str(operation_id or "")[:256]}
    return {"kind": "unsupported"}


def _frontmatter(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    if not text.startswith("---"):
        return {}
    head = text.split("---", 2)
    if len(head) < 3:
        return {}
    values: dict[str, str] = {}
    for line in head[1].splitlines():
        match = re.match(r"^([A-Za-z0-9_]+):\s*[\"']?([^\"']*)[\"']?\s*$", line.strip())
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def _memory_rows(vault_root: Path):
    root = vault_root / "ark" / "memory"
    if not root.exists():
        return
    for path in root.rglob("*.md"):
        meta = _frontmatter(path)
        if meta:
            yield path, meta


def _result(status: str, source: str, evidence_ref: str, *, result_ref: str = "",
            error_code: str = "") -> dict[str, str]:
    value = {"status": status, "source": source, "evidence_ref": evidence_ref}
    if result_ref:
        value["result_ref"] = result_ref[:256]
    if error_code:
        value["error_code"] = error_code[:128]
    return value


def verify_operation(operation: Mapping[str, Any], *, vault_root: str | Path | None = None,
                     project_root: str | Path | None = None,
                     remote_config: Mapping[str, Any] | None = None) -> dict[str, str] | None:
    """Return reconcile evidence, or ``None`` when evidence is insufficient."""
    verification = operation.get("verification")
    if not isinstance(verification, Mapping):
        return None
    kind = str(verification.get("kind") or "")
    ref = str(verification.get("ref") or "")
    if kind == "remote_operation":
        operation_id = str(verification.get("remote_operation_id") or "").strip()
        if not operation_id:
            return None
        try:
            from agentlab.runtime.remote_operations import query_remote_operation
            return query_remote_operation(operation_id, remote_config)
        except (ImportError, OSError, RuntimeError, ValueError):
            return None

    root = Path(vault_root).resolve() if vault_root else None
    if root is None:
        return None

    if kind in {"vault_file", "vault_patch"}:
        relative = _normalise_vault_path(ref)
        target = (root / relative).resolve()
        if root not in target.parents and target != root:
            return None
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
        expected = str(verification.get("result_hash") or verification.get("content_hash") or "")
        if kind == "vault_patch" and not verification.get("result_hash"):
            return None
        if expected and _sha256(content) == expected:
            return _result("succeeded", "vault-filesystem", f"vault:{relative}",
                           result_ref=f"vault:{relative}:{expected[:16]}")
        return None

    if kind == "memory_content":
        expected = str(verification.get("content_hash") or "")
        for path, meta in _memory_rows(root) or ():
            if expected and meta.get("content_hash") == expected:
                return _result("succeeded", "memory-markdown", f"memory:{meta.get('id', path.name)}",
                               result_ref=f"vault:{path.relative_to(root).as_posix()}")
        return None

    if kind == "memory_correct":
        expected = str(verification.get("content_hash") or "")
        old_seen = False
        for path, meta in _memory_rows(root) or ():
            if meta.get("id") == ref and meta.get("status") in {"superseded", "revoked", "archived"}:
                old_seen = True
            if expected and meta.get("content_hash") == expected and meta.get("correction_of") == ref:
                return _result("succeeded", "memory-markdown", f"memory:{meta.get('id', path.name)}",
                               result_ref=f"vault:{path.relative_to(root).as_posix()}")
        return None if old_seen else None

    if kind.startswith("memory_") and kind in {"memory_revoke", "memory_delete", "memory_restore"}:
        found = None
        for path, meta in _memory_rows(root) or ():
            if meta.get("id") == ref:
                found = (path, meta)
                break
        action = kind.removeprefix("memory_")
        if action == "delete" and found is None:
            return _result("succeeded", "memory-markdown", f"memory:{ref}:absent")
        if found is None:
            return None
        path, meta = found
        status = str(meta.get("status") or "active").lower()
        if action == "revoke" and status in {"revoked", "superseded", "archived"}:
            return _result("succeeded", "memory-markdown", f"memory:{ref}", result_ref=f"vault:{path.relative_to(root).as_posix()}")
        if action == "restore" and status == "active":
            return _result("succeeded", "memory-markdown", f"memory:{ref}", result_ref=f"vault:{path.relative_to(root).as_posix()}")
        return None

    if kind == "memory_review":
        expected = str(verification.get("content_hash") or "").lower()
        path = next((path for path, meta in _memory_rows(root) or ()
                     if meta.get("id") == ref), None)
        if path is None:
            return None
        meta = _frontmatter(path)
        if str(meta.get("status") or "active").lower() != "active":
            return None
        # Confirmed reviews clear review_due_at; deferred reviews retain a
        # future timestamp. Both states are deterministic local evidence.
        due = str(meta.get("review_due_at") or "")
        content = path.read_text(encoding="utf-8", errors="ignore")
        body = content.split("---", 2)[-1].strip() if content.startswith("---") else content
        if expected and str(meta.get("content_hash") or "").lower() != expected:
            return None
        decision = str(verification.get("decision") or "")
        if decision == "confirm" and due:
            return None
        if decision == "defer" and not due:
            return None
        expected_due = str(verification.get("defer_until") or "")
        if decision == "defer" and expected_due and due != expected_due:
            return None
        return _result("succeeded", "memory-markdown", f"memory:{ref}",
                       result_ref=f"vault:{path.relative_to(root).as_posix()}")

    if kind == "bili_artifact" and ref:
        inbox = root / "Inbox"
        if inbox.exists():
            for path in inbox.rglob("*.md"):
                try:
                    if ref in path.read_text(encoding="utf-8", errors="ignore"):
                        return _result("succeeded", "vault-inbox", f"bilibili:{ref}",
                                       result_ref=f"vault:{path.relative_to(root).as_posix()}")
                except OSError:
                    continue
    if kind == "article_artifact" and str(verification.get("source_ref") or ""):
        source_hash = str(verification.get("source_ref"))
        inbox = root / "Inbox"
        if inbox.exists():
            for path in inbox.rglob("*.md"):
                try:
                    meta = _frontmatter(path)
                    parsed = urlsplit(str(meta.get("source") or ""))
                    canonical = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"[:512]
                    if canonical and _sha256(canonical) == source_hash:
                        return _result("succeeded", "vault-inbox", f"article:{source_hash[:16]}",
                                       result_ref=f"vault:{path.relative_to(root).as_posix()}")
                except OSError:
                    continue
    if kind == "inbox_queue":
        operation_id = str(verification.get("operation_id") or "").strip()
        if not operation_id:
            return None
        project = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
        db_path = project / "inbox" / "queue.db"
        try:
            import sys
            queue_dir = str(project / "inbox_collector")
            if queue_dir not in sys.path:
                sys.path.insert(0, queue_dir)
            from queue_store import InboxQueueStore
            row = InboxQueueStore(db_path).collection_operation(operation_id)
        except (ImportError, OSError, RuntimeError, sqlite3.Error):
            return None
        if not row or str(row.get("status") or "") != "succeeded":
            return None
        return _result(
            "succeeded", "inbox-queue", f"inbox-operation:{operation_id}",
            result_ref=f"queue:{operation_id}:{str(row.get('after_hash') or '')[:16]}",
        )
    # Unknown remote operations deliberately remain a manual recovery barrier;
    # the verifier never guesses from queue counts or a partial response.
    return None


def reconcile_unknown_operations(store, task_id: str, *, vault_root: str | Path | None = None,
                                 project_root: str | Path | None = None,
                                 remote_config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Settle only operations with deterministic evidence; never dispatch tools."""
    settled: list[dict[str, Any]] = []
    for operation in store.pending_operations(task_id):
        if operation.get("status") != "unknown":
            continue
        evidence = verify_operation(
            operation, vault_root=vault_root, project_root=project_root,
            remote_config=remote_config,
        )
        if evidence is None:
            continue
        current = store.get(task_id)
        if current is None:
            break
        try:
            state = store.reconcile_operation(
                task_id, str(operation.get("operation_id")),
                external_result=evidence, expected_version=current.state_version,
            )
        except Exception:
            continue
        settled.append(next(
            row for row in state.pending_tools
            if row.get("operation_id") == operation.get("operation_id")
        ))
    return settled


__all__ = ["build_verification_metadata", "verify_operation", "reconcile_unknown_operations"]
