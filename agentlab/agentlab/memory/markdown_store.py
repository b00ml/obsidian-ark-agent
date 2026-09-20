"""
Markdown-based memory store for Agent Hub.

Stores memories as individual Markdown files in vault's ark/memory/ directory,
replacing SQLite-based MemoryStore. Memories are user-visible, editable in Obsidian,
and automatically indexed by RAG.

Directory structure:
    ark/memory/
        core/          # User profile, preferences, long-term goals
        context/       # Project context, environment configs
        procedures/    # Workflows, operational steps
        decisions/     # Important decisions, trade-offs
        sessions/      # Recent conversation summaries
        archive/       # Low-value, expired, superseded memories
"""

import re
import os
import threading
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import yaml

from agentlab.memory.governance import (
    DEFAULT_READ_STATUSES,
    SCHEMA_VERSION,
    ReadAudit,
    content_hash,
    decide_write,
    is_expired,
    is_not_yet_valid,
    is_review_due,
    normalise_subject,
    parse_datetime,
    memory_scope_matches,
    normalise_scope,
    status_matches,
    utc_now,
)


_BUCKET_LOCK_GUARD = threading.Lock()
_BUCKET_LOCKS: dict = {}


def _atomic_write(path: Path, content: str) -> None:
    """Write a memory Markdown file atomically and release the temp handle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


class MemoryMarkdownStore:
    """Markdown file-based memory storage."""
    
    MEMORY_TYPES = ["core", "context", "procedures", "decisions", "sessions", "archive"]
    
    def __init__(self, vault_root: str, *, create_dirs: bool = True,
                 audit_path: str | Path | None = None):
        """
        Initialize Markdown memory store.
        
        Args:
            vault_root: Absolute path to Obsidian vault root (e.g., E:/peik1_books)
            create_dirs: Create the standard memory directories when missing.
                Read-only callers such as evaluators should pass ``False``.
        """
        self.vault_root = Path(vault_root)
        self.memory_root = self.vault_root / "ark" / "memory"
        self.audit_path = Path(audit_path) if audit_path else None
        self.last_query_audit = ReadAudit()
        
        if create_dirs:
            # Runtime stores keep the historical behavior of eagerly ensuring
            # the standard layout. Read-only evaluators opt out explicitly.
            for mem_type in self.MEMORY_TYPES:
                (self.memory_root / mem_type).mkdir(parents=True, exist_ok=True)
    
    def commit(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        mem_type: str = "context",
        project_id: str = "default",
        importance: int = 5,
        confidence: float = 1.0,
        source_session: Optional[str] = None,
        supersedes: Optional[list[str]] = None,
        *,
        status: str | None = None,
        source: str = "user",
        source_ref: str = "",
        scope: dict | str | None = None,
        session_id: str = "",
        valid_from: str | None = None,
        valid_until: str | None = None,
        review_due_at: str | None = None,
        subject: str = "",
        correction_of: str | None = None,
        candidate_first: bool = False,
        explicit_confirmation: bool = False,
        quarantine_external: bool = True,
    ) -> str:
        """
        Commit a new memory to Markdown file.
        
        Args:
            content: Memory content (main text)
            tags: List of tags for categorization
            mem_type: Memory type (core/context/procedures/decisions/sessions)
            project_id: Project identifier for isolation
            importance: Importance score 1-10
            confidence: Confidence level 0.0-1.0 or 'hypothesis'
            source_session: Source session ID for traceability
            supersedes: List of memory IDs this memory supersedes
            
        Returns:
            Memory ID (mem-{uuid})
        """
        if mem_type not in self.MEMORY_TYPES:
            raise ValueError(f"Invalid mem_type: {mem_type}, must be one of {self.MEMORY_TYPES}")
        
        if not 1 <= importance <= 10:
            raise ValueError(f"importance must be 1-10, got {importance}")
        
        # OPT-230：hypothesis 先行判断——字符串与数值链式比较会 TypeError
        if confidence == "hypothesis":
            pass
        elif not (0.0 <= confidence <= 1.0):
            confidence = "hypothesis" if confidence < 0.5 else min(1.0, confidence)
        
        decision = decide_write(
            requested_status=status, mem_type=mem_type, confidence=confidence,
            source=source, content=content, candidate_first=candidate_first,
            explicit_confirmation=explicit_confirmation,
            quarantine_external=quarantine_external,
        )
        mem_id = f"mem-{uuid.uuid4().hex[:12]}"
        now = utc_now().isoformat()
        memory_scope = normalise_scope(project_id, session_id or source_session, scope)
        subject_key = normalise_subject(subject)

        # OPT-230 B7：低置信（hypothesis）记忆禁止替代已确立结论——
        # 保护语义在 supersedes 必经点（commit）强制。
        if supersedes and confidence == "hypothesis":
            raise ValueError(
                "MEMORY_HYPOTHESIS_NO_SUPERSEDE: confidence=hypothesis 的低置信记忆"
                "不得替代已确立结论（OPT-230）；请先补证据后以正常置信提交。")

        # Normalize tags
        tags = self._normalize_tags(tags or [])
        
        # Build frontmatter
        frontmatter = {
            "schema_version": SCHEMA_VERSION,
            "id": mem_id,
            "type": mem_type,
            "project_id": memory_scope["project_id"],
            "scope": memory_scope,
            "importance": importance,
            "confidence": confidence,
            "created_at": now,
            "updated_at": now,
            "last_accessed_at": now,
            "access_count": 0,
            "tags": tags,
            "status": decision.status,
            "source": source or "unknown",
            "source_ref": source_ref or "",
            "content_hash": content_hash(content),
            "correction_of": correction_of or "",
            "subject": subject_key,
        }
        
        if source_session:
            frontmatter["source_session"] = source_session
        if session_id:
            frontmatter["session_id"] = session_id
        if valid_until:
            frontmatter["valid_until"] = valid_until
        if valid_from:
            if parse_datetime(valid_from) is None:
                raise ValueError("valid_from must be an ISO timestamp")
            frontmatter["valid_from"] = valid_from
        if review_due_at:
            frontmatter["review_due_at"] = review_due_at
        if decision.reasons:
            frontmatter["governance_reasons"] = list(decision.reasons)

        conflicts = self._find_subject_conflicts(
            mem_type=mem_type, scope=memory_scope, subject=subject_key,
            content_digest=frontmatter["content_hash"], exclude_ids=set(supersedes or ()),
        )
        if conflicts:
            frontmatter["status"] = "conflict"
            frontmatter["conflicts_with"] = conflicts
            reasons = list(frontmatter.get("governance_reasons") or [])
            frontmatter["governance_reasons"] = list(dict.fromkeys([*reasons, "subject_conflict"]))
        
        if supersedes:
            frontmatter["supersedes"] = supersedes
            # Mark superseded memories
            for old_id in supersedes:
                self._mark_superseded(old_id, mem_id)
        
        # Determine file path
        if mem_type == "sessions":
            # sessions/YYYY-MM/session-{id}.md
            month_dir = self.memory_root / "sessions" / datetime.now().strftime("%Y-%m")
            month_dir.mkdir(parents=True, exist_ok=True)
            file_path = month_dir / f"{mem_id}.md"
        elif mem_type == "context" and memory_scope["project_id"] != "default":
            # context/{project_id}/{mem_id}.md
            project_dir = self.memory_root / "context" / memory_scope["project_id"]
            project_dir.mkdir(parents=True, exist_ok=True)
            file_path = project_dir / f"{mem_id}.md"
        else:
            # {type}/{语义标题}-{id6}.md（OPT-225：用户可读文件名；uuid 兜底）
            file_path = self.memory_root / mem_type / self._slug_filename(content, mem_id)
        
        # Render Markdown
        markdown_content = self._render_memory_file(frontmatter, content)
        
        # Write file (atomic via temp + replace, following F5-003 Vault Gateway pattern)
        _atomic_write(file_path, markdown_content)
        
        return mem_id

    def _find_subject_conflicts(self, *, mem_type: str, scope: dict[str, str],
                                subject: str, content_digest: str,
                                exclude_ids: set[str] | None = None) -> list[str]:
        """Find explicit same-subject contradictions without semantic guessing."""
        if not subject:
            return []
        excluded = {str(item) for item in (exclude_ids or set())}
        type_dir = self.memory_root / mem_type
        if not type_dir.exists():
            return []
        conflicts: list[str] = []
        for path in type_dir.rglob("*.md"):
            try:
                memory = self._parse_memory_file(path)
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
            memory_id = str(memory.get("id") or "")
            status = str(memory.get("status") or "active").strip().lower()
            if (not memory_id or memory_id in excluded or status in {"revoked", "archived", "superseded", "expired"}
                    or normalise_subject(memory.get("subject")) != subject):
                continue
            existing_scope = normalise_scope(
                memory.get("project_id"), memory.get("session_id"), memory.get("scope"),
            )
            existing_digest = str(memory.get("content_hash") or content_hash(memory.get("content", "")))
            if existing_scope == scope and existing_digest != content_digest:
                conflicts.append(memory_id)
        return sorted(dict.fromkeys(conflicts))
    
    def query(
        self,
        query_text: str,
        limit: int = 5,
        project_id: Optional[str] = None,
        mem_types: Optional[list[str]] = None,
        *,
        track_access: bool = True,
        session_id: Optional[str] = None,
        statuses: Optional[list[str] | set[str]] = None,
        include_archive: bool = False,
        min_confidence: float = 0.0,
        allow_default_shared: bool = True,
    ) -> list[dict]:
        """
        Query memories by text search (file system scan + keyword matching).
        
        Args:
            query_text: Query string
            limit: Maximum results to return
            project_id: Filter by project (None = all projects)
            mem_types: Filter by memory types (None = all except archive)
            track_access: Update access_count/last_accessed_at for matches.
                Evaluators should disable this to keep source files unchanged.
            
        Returns:
            List of memory dicts sorted by relevance score
        """
        if mem_types is None:
            mem_types = [t for t in self.MEMORY_TYPES
                         if include_archive or t != "archive"]
        if statuses is None:
            statuses_set = set(DEFAULT_READ_STATUSES)
            if include_archive:
                statuses_set.update({"archived", "superseded", "revoked", "expired"})
        else:
            statuses_set = {str(s).strip().lower() for s in statuses if str(s).strip()}
        try:
            min_confidence = max(0.0, min(1.0, float(min_confidence)))
        except (TypeError, ValueError):
            min_confidence = 0.0
        self.last_query_audit = ReadAudit()
        
        # Extract search terms
        terms = self._extract_search_terms(query_text)
        if not terms:
            return []
        
        results = []
        
        for mem_type in mem_types:
            type_dir = self.memory_root / mem_type
            if not type_dir.exists():
                continue
            
            # Recursively scan all .md files
            for md_file in type_dir.rglob("*.md"):
                try:
                    candidates = self._load_entries(md_file)
                    for memory in candidates:
                        # Filter by project/session scope before scoring.  ``default``
                        # remains an explicit shared layer, never an accidental
                        # cross-project wildcard.
                        if not memory_scope_matches(
                            memory, project_id=project_id, session_id=session_id,
                            allow_default_shared=allow_default_shared,
                        ):
                            self.last_query_audit.reject("scope")
                            continue
                        allowed, reason = status_matches(
                            memory, statuses_set, include_archive=include_archive,
                        )
                        if not allowed:
                            self.last_query_audit.reject(reason or "status")
                            continue
                        try:
                            confidence = memory.get("confidence", 1.0)
                            if confidence == "hypothesis":
                                confidence_value = 0.0
                            else:
                                confidence_value = float(confidence)
                        except (TypeError, ValueError):
                            confidence_value = 0.0
                        if confidence_value < min_confidence:
                            self.last_query_audit.reject("confidence")
                            continue
                        
                        # Calculate relevance score
                        score = self._calculate_relevance(memory, terms)
                        if score > 0:
                            memory["_score"] = score
                            results.append(memory)
                            
                            # Update access tracking（桶内条目不回写，见 _load_entries）
                            if track_access:
                                self._update_access(memory["id"])
                except Exception:
                    # Silently skip malformed files
                    continue
        
        # Sort by score desc, then by updated_at desc
        results.sort(
            key=lambda m: (m["_score"], m.get("updated_at", "")),
            reverse=True
        )
        
        return results[:limit]

    def query_with_audit(self, query_text: str, limit: int = 5, **kwargs) -> dict:
        """Read-gated query plus non-sensitive filtering counters for trace."""
        results = self.query(query_text, limit=limit, **kwargs)
        return {"results": results, "total": len(results),
                "filtered_reasons": dict(self.last_query_audit.filtered_reasons)}

    def review_due(self, *, project_id: str | None = None,
                   session_id: str | None = None, limit: int = 100,
                   now: datetime | None = None) -> list[dict]:
        """List active memories due for explicit review without mutating them.

        The result intentionally contains metadata only.  Callers can present
        the ids to a reviewer, then use ``promote``/lifecycle APIs explicitly.
        """
        current = now or utc_now()
        rows: list[dict] = []
        for mem_type in self.MEMORY_TYPES:
            if mem_type == "archive":
                continue
            type_dir = self.memory_root / mem_type
            if not type_dir.exists():
                continue
            for md_file in type_dir.rglob("*.md"):
                try:
                    memory = self._parse_memory_file(md_file)
                except (OSError, ValueError, TypeError, yaml.YAMLError):
                    continue
                if str(memory.get("status") or "active").strip().lower() != "active":
                    continue
                if is_not_yet_valid(memory.get("valid_from"), now=current):
                    continue
                if not is_review_due(memory.get("review_due_at"), now=current):
                    continue
                if not memory_scope_matches(
                    memory, project_id=project_id, session_id=session_id,
                ):
                    continue
                rows.append({
                    "id": memory.get("id", ""),
                    "type": memory.get("type", mem_type),
                    "project_id": memory.get("project_id", "default"),
                    "session_id": memory.get("session_id", ""),
                    "source": memory.get("source", ""),
                    "source_ref": memory.get("source_ref", ""),
                    "content_hash": memory.get("content_hash", content_hash(memory.get("content", ""))),
                    "review_due_at": memory.get("review_due_at", ""),
                    "path": str(md_file.relative_to(self.vault_root).as_posix()),
                })
        rows.sort(key=lambda row: (str(row.get("review_due_at", "")), str(row.get("id", ""))))
        return rows[:max(1, min(int(limit), 1000))]

    def list_conflicts(self, *, project_id: str | None = None,
                       session_id: str | None = None, limit: int = 100) -> list[dict]:
        """List unresolved, explicit subject conflicts as metadata-only rows."""
        rows: list[dict] = []
        for mem_type in self.MEMORY_TYPES:
            if mem_type == "archive":
                continue
            type_dir = self.memory_root / mem_type
            if not type_dir.exists():
                continue
            for path in type_dir.rglob("*.md"):
                try:
                    memory = self._parse_memory_file(path)
                except (OSError, TypeError, ValueError, yaml.YAMLError):
                    continue
                if str(memory.get("status") or "").strip().lower() != "conflict":
                    continue
                if not memory_scope_matches(memory, project_id=project_id, session_id=session_id):
                    continue
                rows.append({
                    "id": str(memory.get("id") or ""),
                    "type": str(memory.get("type") or mem_type),
                    "project_id": str(memory.get("project_id") or "default"),
                    "session_id": str(memory.get("session_id") or ""),
                    "subject": normalise_subject(memory.get("subject")),
                    "conflicts_with": [str(item) for item in memory.get("conflicts_with", [])],
                    "source": str(memory.get("source") or ""),
                    "source_ref": str(memory.get("source_ref") or ""),
                    "confidence": memory.get("confidence", 1.0),
                    "created_at": str(memory.get("created_at") or ""),
                    "content_hash": str(memory.get("content_hash") or ""),
                    "path": str(path.relative_to(self.vault_root).as_posix()),
                })
        rows.sort(key=lambda row: (row["created_at"], row["id"]), reverse=True)
        return rows[:max(1, min(int(limit), 1000))]

    def lifecycle(self, *, project_id: str | None = None,
                  session_id: str | None = None, limit: int = 500,
                  now: datetime | None = None) -> list[dict]:
        """List effective lifecycle metadata from Markdown without mutation."""
        current = now or utc_now()
        rows: list[dict] = []
        for mem_type in self.MEMORY_TYPES:
            type_dir = self.memory_root / mem_type
            if not type_dir.exists():
                continue
            for path in type_dir.rglob("*.md"):
                try:
                    entries = self._load_entries(path)
                except (OSError, TypeError, ValueError, yaml.YAMLError):
                    continue
                for memory in entries:
                    if not memory_scope_matches(memory, project_id=project_id, session_id=session_id):
                        continue
                    stored = str(memory.get("status") or "active").strip().lower()
                    status = "not_yet_valid" if stored == "active" and is_not_yet_valid(memory.get("valid_from"), now=current) else stored
                    if status == "active" and is_expired(memory.get("valid_until"), now=current):
                        status = "expired"
                    if status == "active" and is_review_due(memory.get("review_due_at"), now=current):
                        status = "review_due"
                    content = str(memory.get("content") or "").replace("\n", " ").strip()
                    rows.append({"id": str(memory.get("id") or ""), "status": status,
                                 "stored_status": stored, "type": str(memory.get("type") or mem_type),
                                 "project_id": str(memory.get("project_id") or "default"),
                                 "session_id": str(memory.get("session_id") or ""),
                                 "subject": str(memory.get("subject") or ""),
                                 "source": str(memory.get("source") or ""),
                                 "source_ref": str(memory.get("source_ref") or ""),
                                 "confidence": memory.get("confidence", 1.0),
                                 "content_hash": str(memory.get("content_hash") or content_hash(content)),
                                 "valid_from": str(memory.get("valid_from") or ""),
                                 "valid_until": str(memory.get("valid_until") or ""),
                                 "review_due_at": str(memory.get("review_due_at") or ""),
                                 "updated_at": str(memory.get("updated_at") or ""),
                                 "path": str(path.relative_to(self.vault_root).as_posix()),
                                 "content_preview": content[:240]})
        rows.sort(key=lambda row: (row["status"], row["updated_at"], row["id"]), reverse=True)
        return rows[:max(1, min(int(limit), 2000))]
    
    def update(
        self,
        mem_id: str,
        content: Optional[str] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[int] = None,
        confidence: Optional[float] = None,
        supersedes: Optional[list[str]] = None
    ) -> bool:
        """
        Update an existing memory (write-side consolidation).
        
        Args:
            mem_id: Memory ID to update
            content: New content (None = keep existing)
            tags: New tags (None = keep existing)
            importance: New importance (None = keep existing)
            confidence: New confidence (None = keep existing)
            supersedes: Additional memories this supersedes
            
        Returns:
            True if updated, False if not found
        """
        file_path = self._find_memory_file(mem_id)
        if not file_path:
            return False
        
        memory = self._parse_memory_file(file_path)
        
        # Update fields
        if content is not None:
            memory["content"] = content
        
        if tags is not None:
            memory["tags"] = self._normalize_tags(tags)
        
        if importance is not None:
            if not 1 <= importance <= 10:
                raise ValueError(f"importance must be 1-10, got {importance}")
            memory["importance"] = importance
        
        if confidence is not None:
            memory["confidence"] = confidence
        
        if supersedes:
            existing = memory.get("supersedes", [])
            memory["supersedes"] = list(set(existing + supersedes))
            for old_id in supersedes:
                self._mark_superseded(old_id, mem_id)
        
        memory["updated_at"] = datetime.now(timezone.utc).isoformat()
        memory["content_hash"] = content_hash(memory.get("content", ""))
        
        # Re-render and write
        markdown_content = self._render_memory_file(memory, memory["content"])
        _atomic_write(file_path, markdown_content)
        
        return True

    def promote(self, mem_id: str, *, reason: str = "user_confirmed") -> bool:
        """Promote a candidate to active after an explicit review/confirmation."""
        file_path = self._find_memory_file(mem_id)
        if not file_path:
            return False
        memory = self._parse_memory_file(file_path)
        current_status = memory.get("status")
        if current_status not in {"candidate", "quarantine"} and not (
                current_status == "active" and memory.get("review_due_at")):
            return memory.get("status") == "active"
        old_status = memory.get("status", "candidate")
        memory["status"] = "active"
        memory.pop("review_due_at", None)
        memory["promoted_at"] = utc_now().isoformat()
        memory["promotion_reason"] = reason
        memory["updated_at"] = utc_now().isoformat()
        _atomic_write(file_path, self._render_memory_file(memory, memory["content"]))
        self._audit_event("promote", memory, old_status=old_status, reason=reason)
        return True

    def lifecycle_target(self, mem_id: str, *, expected_content_hash: str,
                         project_id: str = "", session_id: str = "") -> dict:
        """Read and validate one human-reviewed lifecycle target.

        Lifecycle UI and CLI callers must pass the hash they rendered.  This
        keeps a stale modal or queue row from applying a state change to a
        newer Markdown fact.  Scope is exact when supplied: a user who viewed
        one project/session must not mutate an identically named item in
        another scope.
        """
        expected = str(expected_content_hash or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("expected_content_hash must be a sha256 hex digest")
        path = self._find_memory_file(mem_id)
        if path is None:
            raise FileNotFoundError(f"memory not found: {mem_id}")
        memory = self._parse_memory_file(path)
        actual = str(memory.get("content_hash") or content_hash(memory.get("content", ""))).lower()
        if actual != expected:
            raise ValueError("memory content changed since lifecycle view was generated")
        requested_project = str(project_id or "").strip()
        requested_session = str(session_id or "").strip()
        actual_project = str(memory.get("project_id") or "default").strip() or "default"
        actual_session = str(memory.get("session_id") or memory.get("source_session") or "").strip()
        if requested_project and actual_project != requested_project:
            raise PermissionError("memory lifecycle project scope denied")
        if requested_session and actual_session != requested_session:
            raise PermissionError("memory lifecycle session scope denied")
        return memory

    def promote_checked(self, mem_id: str, *, expected_content_hash: str,
                        reviewer: str, reason: str, project_id: str = "",
                        session_id: str = "") -> dict:
        """Promote only a reviewed candidate/quarantine record."""
        reviewer = str(reviewer or "").strip()
        reason = str(reason or "").strip()
        if not reviewer or len(reviewer) > 128:
            raise ValueError("reviewer must be non-empty and <=128 chars")
        if not reason or len(reason) > 500:
            raise ValueError("lifecycle reason must be non-empty and <=500 chars")
        memory = self.lifecycle_target(
            mem_id, expected_content_hash=expected_content_hash,
            project_id=project_id, session_id=session_id,
        )
        current = str(memory.get("status") or "active").strip().lower()
        if current not in {"candidate", "quarantine"}:
            raise ValueError(f"memory status cannot be promoted: {current}")
        if not self.promote(mem_id, reason=reason):
            raise FileNotFoundError(f"memory not found: {mem_id}")
        promoted = self.get(mem_id) or {}
        self._audit_event("lifecycle_confirm", promoted, old_status=current, reason=reason,
                          extra={"action": "promote", "reviewer": reviewer,
                                 "expected_content_hash": expected_content_hash})
        return {
            "id": mem_id, "action": "promote", "status": promoted.get("status", ""),
            "content_hash": promoted.get("content_hash", ""), "reviewer": reviewer,
        }

    def correct_checked(self, mem_id: str, content: str, *, expected_content_hash: str,
                        reviewer: str, reason: str, project_id: str = "",
                        session_id: str = "", tags: Optional[list[str]] = None) -> str:
        """Create a successor only from the reviewed active/conflict record."""
        reviewer = str(reviewer or "").strip()
        reason = str(reason or "").strip()
        if not reviewer or len(reviewer) > 128:
            raise ValueError("reviewer must be non-empty and <=128 chars")
        if not reason or len(reason) > 500:
            raise ValueError("lifecycle reason must be non-empty and <=500 chars")
        memory = self.lifecycle_target(
            mem_id, expected_content_hash=expected_content_hash,
            project_id=project_id, session_id=session_id,
        )
        current = str(memory.get("status") or "active").strip().lower()
        if current not in {"active", "conflict"}:
            raise ValueError(f"memory status cannot be corrected: {current}")
        successor = self.correct(mem_id, content, tags=tags, reason=reason)
        if not successor:
            raise FileNotFoundError(f"memory not found: {mem_id}")
        successor_memory = self.get(successor) or {"id": successor}
        self._audit_event("lifecycle_confirm", successor_memory, old_status=current, reason=reason,
                          extra={"action": "correct", "reviewer": reviewer,
                                 "expected_content_hash": expected_content_hash,
                                 "correction_of": mem_id})
        return successor

    def revoke_checked(self, mem_id: str, *, expected_content_hash: str,
                       reviewer: str, reason: str, project_id: str = "",
                       session_id: str = "") -> bool:
        """Revoke only the currently reviewed active/conflict record."""
        reviewer = str(reviewer or "").strip()
        reason = str(reason or "").strip()
        if not reviewer or len(reviewer) > 128:
            raise ValueError("reviewer must be non-empty and <=128 chars")
        if not reason or len(reason) > 500:
            raise ValueError("lifecycle reason must be non-empty and <=500 chars")
        memory = self.lifecycle_target(
            mem_id, expected_content_hash=expected_content_hash,
            project_id=project_id, session_id=session_id,
        )
        current = str(memory.get("status") or "active").strip().lower()
        if current not in {"active", "conflict"}:
            raise ValueError(f"memory status cannot be revoked: {current}")
        changed = self.revoke(mem_id, reason=reason)
        if changed:
            revoked = self.get(mem_id) or memory
            self._audit_event("lifecycle_confirm", revoked, old_status=current, reason=reason,
                              extra={"action": "revoke", "reviewer": reviewer,
                                     "expected_content_hash": expected_content_hash})
        return changed

    def review(self, mem_id: str, *, decision: str, reviewer: str,
               expected_content_hash: str, reason: str,
               defer_until: str | None = None) -> dict:
        """Apply one explicit, hash-bound review decision.

        This is the write-side counterpart of :meth:`review_due`.  It never
        trusts a stale queue row: the caller must provide the content hash that
        was reviewed.  ``confirm`` promotes candidates or clears a due date;
        ``defer`` schedules a future review while keeping the memory active.
        """
        decision = str(decision or "").strip().lower()
        if decision not in {"confirm", "defer"}:
            raise ValueError("review decision must be confirm or defer")
        reviewer = str(reviewer or "").strip()
        reason = str(reason or "").strip()
        expected = str(expected_content_hash or "").strip().lower()
        if not reviewer or len(reviewer) > 128:
            raise ValueError("reviewer must be non-empty and <=128 chars")
        if not reason or len(reason) > 500:
            raise ValueError("review reason must be non-empty and <=500 chars")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("expected_content_hash must be a sha256 hex digest")
        if decision == "defer":
            if not defer_until or parse_datetime(defer_until) is None or is_review_due(defer_until):
                raise ValueError("defer_until must be a future ISO timestamp")
        file_path = self._find_memory_file(mem_id)
        if not file_path:
            raise FileNotFoundError(f"memory not found: {mem_id}")
        memory = self._parse_memory_file(file_path)
        actual = str(memory.get("content_hash") or content_hash(memory.get("content", ""))).lower()
        if actual != expected:
            raise ValueError("memory content changed since review queue was generated")
        old_status = str(memory.get("status") or "active")
        if decision == "confirm":
            if old_status not in {"candidate", "quarantine", "active"}:
                raise ValueError(f"memory status cannot be confirmed: {old_status}")
            memory["status"] = "active"
            memory.pop("review_due_at", None)
            memory["reviewed_at"] = utc_now().isoformat()
        else:
            if old_status != "active":
                raise ValueError(f"memory status cannot be deferred: {old_status}")
            memory["review_due_at"] = str(defer_until)
            memory["reviewed_at"] = utc_now().isoformat()
        memory["reviewer"] = reviewer
        memory["review_reason"] = reason
        memory["updated_at"] = utc_now().isoformat()
        _atomic_write(file_path, self._render_memory_file(memory, memory["content"]))
        self._audit_event("review", memory, old_status=old_status, reason=reason,
                          extra={"decision": decision, "reviewer": reviewer,
                                 "expected_content_hash": expected})
        return {
            "id": mem_id, "decision": decision, "reviewer": reviewer,
            "status": memory["status"],
            "review_due_at": memory.get("review_due_at", ""),
            "content_hash": actual,
        }

    def correct(self, mem_id: str, content: str, *, tags: Optional[list[str]] = None,
                reason: str = "user_correction", project_id: Optional[str] = None,
                source_session: Optional[str] = None) -> Optional[str]:
        """Create a corrected successor and immediately hide the old version."""
        old = self.get(mem_id)
        if not old:
            return None
        # An explicit human correction is also the only supported way to
        # resolve a recorded subject conflict.  The successor therefore
        # supersedes both the reviewed conflict row and the exact rows it was
        # declared to conflict with; otherwise it would immediately be
        # reclassified as conflict against the still-active predecessor.
        superseded_ids = [mem_id]
        if str(old.get("status") or "").strip().lower() == "conflict":
            superseded_ids.extend(
                str(item) for item in old.get("conflicts_with", []) if str(item).strip()
            )
        superseded_ids = list(dict.fromkeys(superseded_ids))
        successor = self.commit(
            content=content,
            tags=tags if tags is not None else old.get("tags", []),
            mem_type=old.get("type", "context"),
            project_id=project_id or old.get("project_id", "default"),
            importance=int(old.get("importance", 5) or 5),
            confidence=old.get("confidence", 1.0),
            source_session=source_session or old.get("source_session", ""),
            supersedes=superseded_ids,
            source="user",
            source_ref=f"correction:{mem_id}",
            correction_of=mem_id,
            subject=str(old.get("subject") or ""),
            session_id=str(old.get("session_id") or ""),
            valid_from=str(old.get("valid_from") or "") or None,
            valid_until=str(old.get("valid_until") or "") or None,
            explicit_confirmation=True,
        )
        self._audit_event("correct", {**old, "id": successor},
                          old_status=old.get("status", "active"), reason=reason,
                          extra={"correction_of": mem_id,
                                 "supersedes": ",".join(superseded_ids)})
        return successor

    def revoke(self, mem_id: str, *, reason: str = "user_revoked") -> bool:
        """Mark a memory revoked; it is excluded from all default reads."""
        file_path = self._find_memory_file(mem_id)
        if not file_path:
            return False
        memory = self._parse_memory_file(file_path)
        old_status = memory.get("status", "active")
        if old_status == "revoked":
            return True
        memory["status"] = "revoked"
        memory["revoked_at"] = utc_now().isoformat()
        memory["revocation_reason"] = reason
        memory["updated_at"] = utc_now().isoformat()
        _atomic_write(file_path, self._render_memory_file(memory, memory["content"]))
        self._audit_event("revoke", memory, old_status=old_status, reason=reason)
        return True

    def delete(self, mem_id: str, *, reason: str = "user_delete", hard: bool = True) -> bool:
        """Delete a record with a minimal audit event (no content is logged).

        ``hard=False`` is the recoverable revoke path used by normal cleanup;
        hard deletion is available for an explicit personal-data request.
        """
        file_path = self._find_memory_file(mem_id)
        if not file_path:
            return False
        memory = self._parse_memory_file(file_path)
        if not hard:
            return self.revoke(mem_id, reason=reason)
        self._audit_event("delete", memory, old_status=memory.get("status", "active"), reason=reason)
        file_path.unlink()
        return True

    def restore(self, mem_id: str, *, reason: str = "user_restore") -> bool:
        """Restore an archived/revoked record unless it has a live successor."""
        file_path = self._find_memory_file(mem_id)
        if not file_path:
            return False
        memory = self._parse_memory_file(file_path)
        if memory.get("superseded_by"):
            return False
        old_status = memory.get("status", "active")
        memory["status"] = "active"
        memory.pop("archived_at", None)
        memory.pop("archive_reason", None)
        memory.pop("revoked_at", None)
        memory.pop("revocation_reason", None)
        memory["updated_at"] = utc_now().isoformat()
        target = file_path
        if file_path.parent == self.memory_root / "archive" or \
                self.memory_root.joinpath("archive") in file_path.parents:
            mem_type = memory.get("type", "context")
            if mem_type not in self.MEMORY_TYPES or mem_type == "archive":
                mem_type = "context"
            target_dir = self.memory_root / mem_type
            if mem_type == "context" and memory.get("project_id", "default") != "default":
                target_dir = target_dir / str(memory["project_id"])
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / file_path.name
        _atomic_write(target, self._render_memory_file(memory, memory["content"]))
        if target != file_path and file_path.exists():
            file_path.unlink()
        self._audit_event("restore", memory, old_status=old_status, reason=reason)
        return True

    def _audit_event(self, event: str, memory: dict, *, old_status: str = "",
                     reason: str = "", extra: Optional[dict] = None) -> None:
        """Append a bounded, redacted lifecycle event for trace/debugging."""
        try:
            path = self.audit_path
            if path is None:
                audit_dir = self.vault_root / ".agent-brain" / "memory"
                audit_dir.mkdir(parents=True, exist_ok=True)
                path = audit_dir / "events.jsonl"
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
            import json
            row = {
                "ts": utc_now().isoformat(timespec="seconds"),
                "event": event,
                "memory_id": str(memory.get("id", "")),
                "old_status": old_status,
                "new_status": str(memory.get("status", "")),
                "project_id": str(memory.get("project_id", "default")),
                "source": str(memory.get("source", "")),
                "reason": reason[:200],
            }
            if extra:
                row.update({str(k): str(v)[:200] for k, v in extra.items()})
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            lines.append(json.dumps(row, ensure_ascii=False))
            path.write_text("\n".join(lines[-500:]) + "\n", encoding="utf-8")
        except Exception:
            # Audit failure must not make a correction/revoke/delete unsafe.
            pass
    
    def archive(self, mem_id: str, reason: str = "lifecycle") -> bool:
        """
        Archive a memory (move to archive/ and mark status).
        
        Args:
            mem_id: Memory ID to archive
            reason: Archival reason (lifecycle/superseded/manual)
            
        Returns:
            True if archived, False if not found
        """
        file_path = self._find_memory_file(mem_id)
        if not file_path:
            return False
        
        memory = self._parse_memory_file(file_path)
        
        # Update metadata
        memory["status"] = "archived"
        memory["archived_at"] = datetime.now(timezone.utc).isoformat()
        memory["archive_reason"] = reason
        
        # Determine archive path: archive/YYYY/{mem_id}.md
        year = datetime.now().year
        archive_dir = self.memory_root / "archive" / str(year)
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{mem_id}.md"
        
        # Render and move
        markdown_content = self._render_memory_file(memory, memory["content"])
        _atomic_write(archive_path, markdown_content)
        file_path.unlink()
        
        return True
    
    # ------------------------------------------------------------------
    # OPT-225 桶化候选层：自动沉淀按 type/project/month 聚合，减少文件墙
    # 桶内条目为一次性过程记录：不支持单条 update/archive/supersede/访问回写，
    # 生命周期按月整桶归档（lifecycle）。
    # ------------------------------------------------------------------

    def _slug_filename(self, content: str, mem_id: str) -> str:
        """正文首行 → 语义文件名（非法字符清洗，uuid 后缀保唯一）。"""
        first = content.strip().splitlines()[0].lstrip("# ") if content.strip() else ""
        slug = re.sub(r'[\/:*?"<>|#\[\]]', "", first).strip()
        slug = re.sub(r"\s+", "-", slug)[:24].strip("-")
        return f"{slug}-{mem_id[-6:]}.md" if slug else f"{mem_id}.md"

    def _bucket_path(self, mem_type: str, project_id: str, month: str) -> "Path":
        """桶文件路径：sessions/{YYYY-MM}.md；context/{pid|default}.md。"""
        if mem_type == "sessions":
            return self.memory_root / "sessions" / f"{month}.md"
        return self.memory_root / mem_type / f"{project_id or 'default'}.md"

    def _bucket_lock(self, path: "Path") -> "threading.Lock":
        key = str(path)
        with _BUCKET_LOCK_GUARD:
            return _BUCKET_LOCKS.setdefault(key, threading.Lock())

    def commit_to_bucket(self, mem_type: str, content: str,
                         tags: Optional[list[str]] = None,
                         project_id: str = "default",
                         importance: int = 3,
                         source_session: Optional[str] = None,
                         *, status: str = "active", source: str = "assistant",
                         source_ref: str = "", confidence: float | str = 1.0,
                         candidate_first: bool = False) -> str:
        """追加一条候选记忆到月/项目桶文件（OPT-225）。

        桶 frontmatter 标 `bucket: true`；条目区块 `## mem-{id}` + meta 引用行，
        可被 query 解析、被 lifecycle 整桶归档。并发 append 经 per-bucket 锁串行化。
        """
        if mem_type not in self.MEMORY_TYPES:
            raise ValueError(f"Invalid mem_type: {mem_type}")
        mem_id = f"mem-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        month = now.strftime("%Y-%m")
        bucket = self._bucket_path(mem_type, project_id, month)
        bucket.parent.mkdir(parents=True, exist_ok=True)
        tags = self._normalize_tags(tags or [])
        decision = decide_write(
            requested_status=status, mem_type=mem_type, confidence=confidence,
            source=source, content=content, candidate_first=candidate_first,
        )
        meta = (f"> importance={importance} | tags={', '.join(tags)} | "
                f"created={now.isoformat()} | project={project_id}"
                f" | status={decision.status} | confidence={confidence}"
                + (f" | source={source_session}" if source_session else "")
                + (f" | source_kind={source}" if source else "")
                + (f" | source_ref={source_ref}" if source_ref else ""))
        block = f"\n## {mem_id}\n{meta}\n\n{content.strip()}\n"
        with self._bucket_lock(bucket):
            if bucket.exists():
                text = bucket.read_text(encoding="utf-8")
            else:
                head = (f"---\nbucket: true\ntype: {mem_type}\n"
                        f"project_id: {project_id or 'default'}\nmonth: {month}\n---\n"
                        f"# {mem_type} 候选记忆（{month}）\n")
                text = head
            _atomic_write(bucket, text.rstrip("\n") + "\n" + block)
        return mem_id

    def _parse_bucket_file(self, path: "Path") -> list[dict]:
        """解析桶文件：`## mem-{id}` 区块 → 记忆 dict 列表（不回写访问计数）。"""
        text = path.read_text(encoding="utf-8")
        entries: list[dict] = []
        blocks = re.split(r"^## (mem-[0-9a-f]+)\s*$", text, flags=re.M)
        for i in range(1, len(blocks) - 1, 2):
            mem_id, body = blocks[i], blocks[i + 1]
            meta_m = re.match(r"\s*>\s*(.+)", body)
            meta = meta_m.group(1) if meta_m else ""
            kv = dict(re.findall(r"(\w+)=([^|]*)", meta))
            lines_ = body.lstrip("\n").split("\n")
            content = ("\n".join(lines_[1:]).strip()
                       if lines_ and lines_[0].startswith(">") else body.strip())
            raw_confidence = kv.get("confidence", "1.0").strip()
            try:
                parsed_confidence = (raw_confidence if raw_confidence == "hypothesis"
                                     else float(raw_confidence))
            except (TypeError, ValueError):
                parsed_confidence = 0.0
            entries.append({
                "id": mem_id,
                "type": "sessions",
                "schema_version": SCHEMA_VERSION,
                "bucket": True,
                "project_id": kv.get("project", "default").strip(),
                "scope": normalise_scope(kv.get("project", "default").strip(), ""),
                "importance": int(kv.get("importance", "3") or 3),
                "confidence": parsed_confidence,
                "created_at": kv.get("created", "").strip(),
                "updated_at": kv.get("created", "").strip(),
                "last_accessed_at": kv.get("created", "").strip(),
                "access_count": 0,
                "tags": [t.strip() for t in kv.get("tags", "").split(",") if t.strip()],
                "status": kv.get("status", "active").strip() or "active",
                "source": kv.get("source_kind", "assistant").strip() or "assistant",
                "source_ref": kv.get("source_ref", "").strip(),
                "content_hash": content_hash(content),
                "correction_of": "",
                "source_session": kv.get("source", "").strip(),
                "content": content,
            })
        return entries

    def _load_entries(self, path: "Path") -> list[dict]:
        """单文件 → [memory]；桶文件 → [多条 memory]（统一 query 的处理单元）。"""
        text = path.read_text(encoding="utf-8")
        head = text[:200]
        if text.lstrip().startswith("---") and "bucket: true" in head:
            return self._parse_bucket_file(path)
        return [self._parse_memory_file(path)]

    def build_index(self) -> "Path":
        """生成 ark/memory/index.md（派生导航：统计 + 链接），可随时重建。

        只做导航，不成为第二事实源；index.md 位于 memory_root 根，
        不落入任何 type 子目录，query 扫描天然不含它。
        """
        lines = ["# 记忆索引（自动生成）", "",
                 "> 由 MemoryMarkdownStore.build_index() 生成；手工编辑会被下次重建覆盖。",
                 "> 事实源是各记忆文件本体。", ""]
        total = 0
        for mem_type in self.MEMORY_TYPES:
            type_dir = self.memory_root / mem_type
            if not type_dir.exists():
                continue
            entries: list[str] = []
            for md_file in sorted(type_dir.rglob("*.md")):
                rel = md_file.relative_to(self.memory_root).as_posix()
                total += 1
                entries.append(f"- [[{rel}|{md_file.stem}]]")
            if entries:
                lines.append(f"## {mem_type}（{len(entries)}）")
                lines.extend(entries)
                lines.append("")
        lines.insert(4, f"共 {total} 个文件。最近由 OPT-225 于 "
                       f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} 重建。\n")
        index = self.memory_root / "index.md"
        _atomic_write(index, "\n".join(lines))
        return index

    def get(self, mem_id: str) -> Optional[dict]:
        """Get a single memory by ID."""
        file_path = self._find_memory_file(mem_id)
        if not file_path:
            return None
        
        memory = self._parse_memory_file(file_path)
        self._update_access(mem_id)
        return memory
    
    def _parse_memory_file(self, path: Path) -> dict:
        """Parse Markdown file into memory dict."""
        text = path.read_text(encoding="utf-8")
        
        # Split frontmatter and content
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid memory file (missing frontmatter): {path}")
        
        frontmatter = yaml.safe_load(parts[1]) or {}
        content = parts[2].strip()

        # Compatibility defaults: old Markdown records remain readable and are
        # never rewritten merely by querying them.  New commits always contain
        # the complete schema-v2 fields above.
        frontmatter.setdefault("schema_version", 1)
        frontmatter.setdefault("project_id", "default")
        frontmatter.setdefault("scope", normalise_scope(frontmatter.get("project_id"),
                                                         frontmatter.get("session_id")))
        frontmatter.setdefault("status", "active")
        frontmatter.setdefault("source", "legacy")
        frontmatter.setdefault("source_ref", "")
        frontmatter.setdefault("confidence", 1.0)
        frontmatter.setdefault("subject", "")
        frontmatter.setdefault("valid_from", "")
        frontmatter.setdefault("valid_until", "")
        frontmatter.setdefault("review_due_at", "")
        frontmatter.setdefault("content_hash", content_hash(content))
        frontmatter.setdefault("correction_of", "")
        
        frontmatter["content"] = content
        frontmatter["_file_path"] = str(path)
        
        return frontmatter
    
    def _render_memory_file(self, frontmatter: dict, content: str) -> str:
        """Render memory dict to Markdown with frontmatter."""
        # Extract content if present in frontmatter
        fm_copy = {k: v for k, v in frontmatter.items() 
                   if k not in ["content", "_file_path", "_score"]}
        
        yaml_str = yaml.dump(fm_copy, allow_unicode=True, sort_keys=False)
        
        return f"---\n{yaml_str}---\n\n{content}\n"
    
    def _find_memory_file(self, mem_id: str) -> Optional[Path]:
        """Find memory file by ID (scan all subdirectories).

        OPT-225：文件名可能为语义命名（{标题}-{id6}.md），因此先精确匹配
        {mem_id}.md，再回退为"文件名含 id6 后缀"匹配；桶内条目不支持单条
        定位（候选层语义），本方法不返回桶文件。
        """
        id6 = mem_id[-6:]
        for mem_type in self.MEMORY_TYPES:
            type_dir = self.memory_root / mem_type
            if not type_dir.exists():
                continue

            for md_file in type_dir.rglob(f"{mem_id}.md"):
                return md_file

        for mem_type in self.MEMORY_TYPES:
            type_dir = self.memory_root / mem_type
            if not type_dir.exists():
                continue

            for md_file in type_dir.rglob(f"*-{id6}.md"):
                return md_file

        return None
    
    def _mark_superseded(self, old_id: str, new_id: str):
        """Mark an old memory as superseded by a new one."""
        file_path = self._find_memory_file(old_id)
        if not file_path:
            return
        
        memory = self._parse_memory_file(file_path)
        memory["superseded_by"] = new_id
        # A4/OPT-228：立即置状态——普通查询即刻排除被替代条目；
        # 物理归档仍由 lifecycle 统一处理（状态机与归档解耦）。
        memory["status"] = "superseded"
        memory["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        markdown_content = self._render_memory_file(memory, memory["content"])
        _atomic_write(file_path, markdown_content)
    
    def _update_access(self, mem_id: str):
        """Update last_accessed_at and access_count."""
        file_path = self._find_memory_file(mem_id)
        if not file_path:
            return
        
        try:
            memory = self._parse_memory_file(file_path)
            memory["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
            memory["access_count"] = memory.get("access_count", 0) + 1
            
            markdown_content = self._render_memory_file(memory, memory["content"])
            _atomic_write(file_path, markdown_content)
        except Exception:
            # Silent failure on access tracking
            pass
    
    def _normalize_tags(self, tags: list[str]) -> list[str]:
        """
        Normalize tags to list of strings.
        Handles OPT-135 fix: prevent string being split into chars.
        """
        if isinstance(tags, str):
            # Split by comma/semicolon
            tags = re.split(r'[,，、;；]', tags)
        
        normalized = []
        for tag in tags:
            tag = str(tag).strip()
            if tag and tag not in normalized:
                normalized.append(tag)
        
        return normalized[:8]  # Limit to 8 tags
    
    def _extract_search_terms(self, query_text: str) -> list[str]:
        """
        Extract search terms from query.
        OPT-135 approach: whole query + word split + CJK bigrams.
        """
        terms = [query_text.strip()]  # Full query
        
        # Word-based split (space, punctuation)
        words = re.findall(r'\w+', query_text)
        terms.extend(words)
        
        # CJK bigrams for better recall
        cjk_chars = re.findall(r'[\u4e00-\u9fff]', query_text)
        for i in range(len(cjk_chars) - 1):
            bigram = cjk_chars[i] + cjk_chars[i + 1]
            terms.append(bigram)
        
        # Deduplicate and limit
        seen = set()
        unique_terms = []
        for term in terms:
            if term and term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:12]  # Limit to 12 terms
    
    def _calculate_relevance(self, memory: dict, terms: list[str]) -> float:
        """
        Calculate relevance score for a memory given search terms.
        
        Score components:
        - Term matches in content/tags (OR logic)
        - Importance weight
        - Time decay (OPT-101 approach)
        """
        content = memory.get("content", "").lower()
        tags = [t.lower() for t in memory.get("tags", [])]
        
        # Count term hits
        hits = 0
        for term in terms:
            term_lower = term.lower()
            if term_lower in content:
                hits += 1
            if any(term_lower in tag for tag in tags):
                hits += 2  # Tag match weighted higher
        
        if hits == 0:
            return 0.0
        
        # Base score from hits
        base_score = min(10.0, hits * 2.0)
        
        # Importance weight (1-10 → 0.1-1.0)
        importance_weight = memory.get("importance", 5) / 10.0
        
        # Time decay (OPT-101 approach: 30-day half-life)
        decay_factor = self._calculate_decay(memory.get("last_accessed_at"))
        
        return base_score * importance_weight * decay_factor
    
    def _calculate_decay(self, last_accessed_str: Optional[str]) -> float:
        """
        Calculate time decay factor (OPT-101 approach).
        Half-life = 30 days.
        """
        if not last_accessed_str:
            return 1.0
        
        try:
            last_accessed = datetime.fromisoformat(last_accessed_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_ago = (now - last_accessed).total_seconds() / 86400
            
            # Exponential decay: 2^(-days/half_life)
            half_life = 30.0
            return 2 ** (-days_ago / half_life)
        except Exception:
            return 1.0
