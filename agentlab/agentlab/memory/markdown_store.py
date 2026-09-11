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
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import yaml


class MemoryMarkdownStore:
    """Markdown file-based memory storage."""

    MEMORY_TYPES = ["core", "context", "procedures", "decisions", "sessions", "archive"]

    def __init__(self, vault_root: str):
        """
        Initialize Markdown memory store.

        Args:
            vault_root: Absolute path to an Obsidian vault root
        """
        self.vault_root = Path(vault_root)
        self.memory_root = self.vault_root / "ark" / "memory"

        # Ensure directories exist
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
        supersedes: Optional[list[str]] = None
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

        if not (0.0 <= confidence <= 1.0 or confidence == "hypothesis"):
            if confidence < 0.5:
                confidence = "hypothesis"
            else:
                confidence = min(1.0, confidence)

        mem_id = f"mem-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        # Normalize tags
        tags = self._normalize_tags(tags or [])

        # Build frontmatter
        frontmatter = {
            "id": mem_id,
            "type": mem_type,
            "project_id": project_id,
            "importance": importance,
            "confidence": confidence,
            "created_at": now,
            "updated_at": now,
            "last_accessed_at": now,
            "access_count": 0,
            "tags": tags,
            "status": "active"
        }

        if source_session:
            frontmatter["source_session"] = source_session

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
        elif mem_type == "context" and project_id != "default":
            # context/{project_id}/{mem_id}.md
            project_dir = self.memory_root / "context" / project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            file_path = project_dir / f"{mem_id}.md"
        else:
            # {type}/{mem_id}.md
            file_path = self.memory_root / mem_type / f"{mem_id}.md"

        # Render Markdown
        markdown_content = self._render_memory_file(frontmatter, content)

        # Write file (atomic via temp + replace, following F5-003 Vault Gateway pattern)
        temp_path = file_path.with_suffix(".tmp")
        try:
            temp_path.write_text(markdown_content, encoding="utf-8")
            temp_path.replace(file_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return mem_id

    def query(
        self,
        query_text: str,
        limit: int = 5,
        project_id: Optional[str] = None,
        mem_types: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Query memories by text search (file system scan + keyword matching).

        Args:
            query_text: Query string
            limit: Maximum results to return
            project_id: Filter by project (None = all projects)
            mem_types: Filter by memory types (None = all except archive)

        Returns:
            List of memory dicts sorted by relevance score
        """
        if mem_types is None:
            mem_types = [t for t in self.MEMORY_TYPES if t != "archive"]

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
                    memory = self._parse_memory_file(md_file)

                    # Filter by project_id
                    if project_id and memory.get("project_id") != project_id:
                        continue

                    # Filter by status (skip archived unless explicitly requested)
                    if memory.get("status") == "archived" and "archive" not in mem_types:
                        continue

                    # Calculate relevance score
                    score = self._calculate_relevance(memory, terms)
                    if score > 0:
                        memory["_score"] = score
                        results.append(memory)

                        # Update access tracking
                        self._update_access(memory["id"])

                except Exception as e:
                    # Silently skip malformed files
                    continue

        # Sort by score desc, then by updated_at desc
        results.sort(
            key=lambda m: (m["_score"], m.get("updated_at", "")),
            reverse=True
        )

        return results[:limit]

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

        # Re-render and write
        markdown_content = self._render_memory_file(memory, memory["content"])
        file_path.write_text(markdown_content, encoding="utf-8")

        return True

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
        archive_path.write_text(markdown_content, encoding="utf-8")
        file_path.unlink()

        return True

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

        frontmatter = yaml.safe_load(parts[1])
        content = parts[2].strip()

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
        """Find memory file by ID (scan all subdirectories)."""
        for mem_type in self.MEMORY_TYPES:
            type_dir = self.memory_root / mem_type
            if not type_dir.exists():
                continue

            for md_file in type_dir.rglob(f"{mem_id}.md"):
                return md_file

        return None

    def _mark_superseded(self, old_id: str, new_id: str):
        """Mark an old memory as superseded by a new one."""
        file_path = self._find_memory_file(old_id)
        if not file_path:
            return

        memory = self._parse_memory_file(file_path)
        memory["superseded_by"] = new_id
        memory["updated_at"] = datetime.now(timezone.utc).isoformat()

        markdown_content = self._render_memory_file(memory, memory["content"])
        file_path.write_text(markdown_content, encoding="utf-8")

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
            file_path.write_text(markdown_content, encoding="utf-8")
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
