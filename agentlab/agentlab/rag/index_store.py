"""Versioned SQLite index for the P2 RAG path.

Markdown remains the source of truth.  This module stores only derived data and
can be deleted and rebuilt at any time.  It deliberately lives next to the
legacy ``VectorIndex`` instead of changing that schema, so the old path remains
an immediate rollback option while P2 is enabled or evaluated.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Sequence

import yaml

try:  # Optional acceleration; the pure-Python path remains supported.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only in minimal installs
    _np = None

from agentlab.rag.chunker import Chunk, STRATEGY_V1, chunk_markdown_v1
from agentlab.contracts import ProcessStatus
from agentlab.memory.governance import DEFAULT_READ_STATUSES, status_matches
from agentlab.runtime.stages import StageTracker

# 词法候选窗倍数（相对 k）。前 v2 重排已证明：普通查询的最终 top-k 由 posting
# 分数截断，窗口超过 k 并不改结果，只拉宽 SQL 与 token 成本；该窗口只在“单稀
# 标识符标题救回”分支有意义，故取 3× 而非 5×，靠离线等价测试锁定不回归。
LEXICAL_CANDIDATE_WINDOW_MULT = 3


INDEX_SCHEMA_VERSION = "rag-index-p2-v1"
PARSER_VERSION = STRATEGY_V1
INDEX_VERSION = "s1-p2-v1"
DEFAULT_BATCH_SIZE = 32
DEFAULT_QUERY_CACHE_SIZE = 256
SMALL_TO_BIG_DEFAULT_NEIGHBORS = 1
SMALL_TO_BIG_DEFAULT_MAX_CHARS = 2400
_ARCHIVE_REL = "ark/memory/archive/"
_MEMORY_REL = "ark/memory/"
_EXCLUDE_DIRS = {".obsidian", ".agent-brain", ".trash", ".tmp", "node_modules"}
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ASCII_RE = re.compile(r"[A-Za-z0-9_]+(?:[.:-][A-Za-z0-9_]+)*")


class IndexCompatibilityError(RuntimeError):
    """Raised when one SQLite index would mix incompatible vector metadata."""


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *[float(x) for x in vec])


def _unpack(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob[: n * 4]))


def _row_value(row, key: str, default=None):
    """Read a sqlite Row or persisted sidecar dict across schema versions."""
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        if isinstance(row, dict):
            return row.get(key, default)
        return default


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = norm_left = norm_right = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_left * norm_right)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---(?:\r?\n|\Z)", re.DOTALL)
_VOLATILE_MEMORY_FIELDS_RE = re.compile(
    r"(?mi)^(?:last_accessed_at|access_count):[^\r\n]*(?:\r?\n|\Z)"
)


def _source_hash(text: str) -> str:
    """Hash source semantics while ignoring read-side access bookkeeping.

    Markdown memory files update ``last_accessed_at`` and ``access_count`` on
    reads.  Those fields do not change chunk content or retrieval metadata;
    including them in the source hash would turn every read into a full
    re-index.  Keep all other frontmatter fields (notably status, project and
    tags) in the hash so real filter or ranking changes still invalidate the
    document.
    """
    match = _FRONTMATTER_RE.match(text or "")
    if not match:
        return _hash(text)
    head = _VOLATILE_MEMORY_FIELDS_RE.sub("", match.group(0))
    return _hash(head + text[match.end() :])


def _normalise_path(path: str | Path) -> str:
    """Normalize a Vault-relative path without stripping hidden names.

    ``lstrip("./")`` removes every leading dot, so ``.dashboard/a.md`` was
    silently stored as ``dashboard/a.md``.  Only an explicit ``./`` prefix is
    syntactic noise; a leading dot in a path component is data.
    """
    text = str(path).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def _normalise_prefixes(prefixes: Sequence[str] | None) -> tuple[str, ...]:
    values: list[str] = []
    for value in prefixes or ():
        text = str(value or "").replace("\\", "/").strip().strip("/")
        if text and text not in values:
            values.append(text)
    return tuple(sorted(values))


def _tokens(text: str) -> set[str]:
    """Return deterministic tokens suitable for Chinese and code identifiers.

    Single CJK characters keep short queries useful; bigrams make longer
    Chinese phrases match without requiring a third-party tokenizer.  ASCII
    identifiers are retained as whole tokens, including BV numbers and dotted
    symbols.  A set avoids duplicate postings inside one field.
    """
    result: set[str] = set()
    for match in _ASCII_RE.finditer(text or ""):
        token = match.group(0).lower()
        result.add(token)
        # Queries normally use a note slug without its Markdown suffix while
        # source refs necessarily carry ``.md``.  Index both forms so exact
        # title/path lookups do not depend on the caller spelling the suffix.
        for suffix in (".md", ".markdown"):
            if token.endswith(suffix) and len(token) > len(suffix):
                result.add(token[: -len(suffix)])
                break
    cjk_run: list[str] = []
    def flush() -> None:
        if not cjk_run:
            return
        run = "".join(cjk_run)
        result.update(run)
        result.update(run[i : i + 2] for i in range(max(0, len(run) - 1)))
        cjk_run.clear()
    for char in text or "":
        if _CJK_RE.match(char):
            cjk_run.append(char)
        else:
            flush()
    flush()
    return {item for item in result if item}


def _query_tokens(query: str) -> set[str]:
    return _tokens(query)


def _is_cjk_token(token: str) -> bool:
    return bool(token) and all(_CJK_RE.fullmatch(char) for char in token)


def _ranking_query_tokens(query: str) -> set[str]:
    """Drop one-character CJK noise for multi-term natural-language asks.

    The posting table intentionally keeps CJK unigrams for short lookups.  A
    long question, however, can match hundreds of unrelated chunks through
    common characters such as ``的``/``库`` and push the actual entry outside
    the candidate window.  Preserve those unigrams for one-term queries and
    exact/entity searches, but rank multi-term questions using phrase/ASCII
    tokens and CJK bigrams.
    """
    tokens = _query_tokens(query)
    cjk_terms = [token for token in tokens if len(token) > 1 and _is_cjk_token(token)]
    ascii_terms = [token for token in tokens if _ASCII_RE.fullmatch(token)]
    if len(cjk_terms) + len(ascii_terms) >= 2:
        filtered = {
            token for token in tokens
            if not (len(token) == 1 and _is_cjk_token(token))
        }
        if filtered:
            return filtered
    return tokens


def _lexical_rank_key(
    row: sqlite3.Row,
    query_tokens: set[str],
    token_idf: dict[str, float],
) -> tuple[float, float, float, float, str]:
    """Rank a broad lexical candidate set with query-aware IDF.

    The posting score is useful for exact terms but treats every CJK unigram
    equally.  A query such as ``MySQL 知识卡片`` could therefore be pushed down
    by many chunks containing the common characters ``知识``.  IDF-weighted
    coverage makes rare identifiers and title/path terms decisive while the
    original posting score remains a stable tie breaker.
    """
    ref = str(row["source_ref"] or "")
    content = str(row["content"] or "")
    path = str(row["file_path"] or "")
    source_tokens = _tokens(" ".join((ref, content, path)))
    matched = query_tokens & source_tokens
    identifiers = {
        token for token in query_tokens
        if len(token) >= 3 and _ASCII_RE.fullmatch(token)
    }
    identifier_hit = 1.0 if identifiers and identifiers.issubset(source_tokens) else 0.0
    stem_tokens = _tokens(Path(path).stem)
    # A single distinctive identifier (e.g. ``MySQL``) is safe to use as a
    # filename/title boost.  Multi-term queries often mix a broad product
    # name with an operation (``Agent hooks``), so demanding a filename match
    # there can promote an unrelated document and hurt the established score.
    title_identifier_hit = 1.0 if (
        len(identifiers) == 1
        and identifiers.issubset(stem_tokens)
        and max(token_idf.get(token, 0.0) for token in identifiers) >= 2.5
    ) else 0.0
    total_idf = sum(token_idf.values()) or 1.0
    matched_idf = sum(token_idf.get(token, 1.0) for token in matched)
    coverage = matched_idf / total_idf
    # Preserve the proven posting score as the primary ordering.  Structural
    # signals only break ties (or rescue an exact entity title), otherwise a
    # broad IDF reorder can regress ordinary paraphrase and bucket queries.
    return (
        title_identifier_hit,
        float(row["score"] or 0.0),
        coverage,
        matched_idf,
        identifier_hit,
        ref,
    )


def _embed_text(chunk: Chunk) -> str:
    heading = " / ".join(chunk.heading_path or [])
    tags = " ".join(chunk.tags or [])
    title = chunk.title or ""
    # Labels make the prefix unambiguous to embedding models while keeping it
    # deterministic.  It is never returned as user-facing content.
    return f"title: {title}\nheading: {heading}\ntags: {tags}\ncontent:\n{chunk.content}".strip()


def _entry_key(chunk: Chunk) -> str:
    """Return an entry ref that remains unique for duplicate mem-id diagnostics."""
    if chunk.diagnostic and chunk.mem_id and ":d" in chunk.anchor:
        return f"{chunk.entry_ref.split('#', 1)[0]}#{chunk.anchor}"
    return chunk.entry_ref


class RagIndexStore:
    """P2 versioned index with incremental hashing, lexical search and cache."""

    def __init__(
        self,
        db_path: str | Path,
        embedder=None,
        *,
        vault_root: str | Path | None = None,
        embedding_model: str | None = None,
        index_version: str = INDEX_VERSION,
        parser_version: str = PARSER_VERSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_attempts: int = 3,
        chunker: Callable[[str, str], object] | None = None,
        chunk_strategy_version: str | None = None,
        include_prefixes: Sequence[str] | None = None,
        query_cache_size: int = DEFAULT_QUERY_CACHE_SIZE,
    ):
        self.db_path = Path(db_path)
        self.embedder = embedder
        self.vault_root = Path(vault_root) if vault_root else None
        self.embedding_model = embedding_model or str(getattr(embedder, "model", "") or "")
        self.index_version = index_version
        self.parser_version = parser_version
        self.batch_size = max(1, int(batch_size))
        self.max_attempts = max(1, int(max_attempts))
        self.chunker = chunker or chunk_markdown_v1
        self.chunk_strategy_version = chunk_strategy_version or parser_version
        self.include_prefixes = _normalise_prefixes(include_prefixes)
        self.query_cache_size = max(0, int(query_cache_size))
        self._query_embedding_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._vector_matrix_cache: dict[tuple, tuple[object, list[sqlite3.Row]]] = {}
        # 观测口：最近一次向量矩阵的来源（memory=进程内 / cached=索引侧车 / built=本次重建）
        self.last_matrix_source: str = "unused"
        self._last_search_status: dict[str, dict[str, str]] = {}

    @property
    def last_search_status(self) -> dict[str, dict[str, str]]:
        """Return the latest route status without exposing mutable internals."""
        return {name: dict(value) for name, value in self._last_search_status.items()}

    def _set_search_status(self, route: str, status: str, reason: str = "") -> None:
        self._last_search_status[route] = {
            "status": status,
            "reason": reason,
        }

    @contextmanager
    def _conn(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_meta(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files(
                path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                source_content_hash TEXT NOT NULL,
                doc_type TEXT NOT NULL DEFAULT 'markdown',
                project_id TEXT NOT NULL DEFAULT 'default',
                status TEXT NOT NULL DEFAULT 'active',
                parser_version TEXT NOT NULL,
                index_version TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entries(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                mem_id TEXT,
                source_ref TEXT NOT NULL,
                bucket_path TEXT,
                entry_offset INTEGER NOT NULL DEFAULT 0,
                title TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                UNIQUE(file_path, source_ref)
            );
            CREATE INDEX IF NOT EXISTS idx_entries_file ON entries(file_path);
            CREATE TABLE IF NOT EXISTS chunks(
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                entry_id INTEGER,
                chunk_index INTEGER NOT NULL,
                source_ref TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_text_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL DEFAULT '',
                embedding_dimension INTEGER,
                vec BLOB,
                updated_at REAL NOT NULL,
                FOREIGN KEY(entry_id) REFERENCES entries(id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_entry ON chunks(entry_id);
            CREATE TABLE IF NOT EXISTS lexical(
                token TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                field TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(token, chunk_id, field),
                FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS embedding_cache(
                cache_key TEXT PRIMARY KEY,
                embedding_text_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                chunk_strategy_version TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                vec BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS failures(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                failure_key TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                chunk_id TEXT,
                stage TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 1,
                next_retry_at REAL,
                last_model TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_failures_status ON failures(status, next_retry_at);
            CREATE TABLE IF NOT EXISTS vector_matrix(
                scope_hash TEXT PRIMARY KEY,
                index_version TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                max_updated REAL NOT NULL,
                dimension INTEGER NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                matrix_f32 BLOB NOT NULL,
                rows_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ingest_queue(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                operation TEXT NOT NULL CHECK(operation IN ('upsert','delete')),
                source_content_hash TEXT NOT NULL DEFAULT '',
                mtime REAL,
                priority INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','running','succeeded','failed','dead')),
                index_version TEXT NOT NULL,
                last_error TEXT NOT NULL DEFAULT '',
                next_retry_at REAL,
                lease_until REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(path,index_version)
            );
            CREATE INDEX IF NOT EXISTS idx_ingest_queue_ready
                ON ingest_queue(status,next_retry_at,priority,created_at);
            CREATE TABLE IF NOT EXISTS checkpoints(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL UNIQUE,
                queue_ids TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('started','completed','aborted')),
                created_at REAL NOT NULL,
                completed_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_checkpoints_status ON checkpoints(status,created_at);
            """
        )

    def _meta(self, conn: sqlite3.Connection) -> dict[str, str]:
        return {row["key"]: row["value"] for row in conn.execute("SELECT key,value FROM index_meta")}

    def _write_base_meta(self, conn: sqlite3.Connection) -> None:
        values = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "index_version": self.index_version,
            "parser_version": self.parser_version,
            "chunk_strategy_version": self.chunk_strategy_version,
            "embedding_model": self.embedding_model,
            "normalization": "float32-cosine",
            "scope_prefixes": json.dumps(self.include_prefixes, ensure_ascii=False),
        }
        conn.executemany("INSERT OR REPLACE INTO index_meta(key,value) VALUES(?,?)", values.items())

    def _assert_version(self, conn: sqlite3.Connection, *, allow_empty: bool = True) -> dict[str, str]:
        meta = self._meta(conn)
        if not meta:
            if allow_empty:
                self._write_base_meta(conn)
                return self._meta(conn)
            raise IndexCompatibilityError("索引元数据缺失，拒绝读取空或损坏索引")
        for key, expected in (
            ("schema_version", INDEX_SCHEMA_VERSION),
            ("index_version", self.index_version),
            ("parser_version", self.parser_version),
            ("chunk_strategy_version", self.chunk_strategy_version),
        ):
            if meta.get(key) and meta[key] != expected:
                raise IndexCompatibilityError(f"索引 {key}={meta[key]} 与当前 {expected} 不兼容")
        expected_scope = json.dumps(self.include_prefixes, ensure_ascii=False)
        stored_scope = meta.get("scope_prefixes", "[]")
        if stored_scope != expected_scope:
            raise IndexCompatibilityError(
                f"索引 scope_prefixes={stored_scope} 与当前 {expected_scope} 不兼容"
            )
        stored_model = meta.get("embedding_model", "")
        if stored_model and self.embedding_model and stored_model != self.embedding_model:
            raise IndexCompatibilityError(f"embedding model 不一致: {stored_model} != {self.embedding_model}")
        return meta

    def _scan(self, root: Path) -> dict[str, tuple[float, str, str]]:
        found: dict[str, tuple[float, str, str]] = {}
        for path in root.rglob("*.md"):
            rel = path.relative_to(root).as_posix()
            if any(part in _EXCLUDE_DIRS for part in path.parts) or rel.startswith(_ARCHIVE_REL):
                continue
            if self.include_prefixes and not any(
                rel == prefix or rel.startswith(prefix + "/")
                for prefix in self.include_prefixes
            ):
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
                found[rel] = (path.stat().st_mtime, _source_hash(raw), raw)
            except OSError:
                continue
        return found

    @staticmethod
    def _frontmatter_value(parsed: dict, *keys: str, default: str) -> str:
        for key in keys:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    def _file_meta(self, parsed: dict, rel: str) -> tuple[str, str, str]:
        parts = Path(rel).parts
        doc_type = self._frontmatter_value(parsed.frontmatter, "type", "doc_type", default="markdown")
        project = self._frontmatter_value(parsed.frontmatter, "project_id", "project", default="default")
        status = self._frontmatter_value(parsed.frontmatter, "status", default="active").lower()
        if "archive" in parts:
            status = "archived"
        return doc_type, project, status

    def _memory_row_is_readable(self, row) -> bool:
        """Re-check governed memory state against its Markdown source.

        The P2 index is a rebuildable acceleration structure, not a second
        lifecycle truth.  A source status or time change can happen before an
        incremental sync removes/rebuilds the physical chunks, so a default
        RAG read must not surface stale candidate, conflict, revoked, expired
        or review-due memory rows during that interval.
        """
        rel = _normalise_path(_row_value(row, "file_path", ""))
        if not rel.startswith(_MEMORY_REL):
            return True
        if not self.vault_root:
            return False
        try:
            source = (Path(self.vault_root) / rel).resolve()
            root = Path(self.vault_root).resolve()
            if root not in source.parents or not source.is_file():
                return False
            raw = source.read_text(encoding="utf-8", errors="ignore")
            match = _FRONTMATTER_RE.match(raw)
            if not match:
                # Historical atomic-memory notes predate lifecycle frontmatter.
                # They retain the legacy active behavior until migrated; once
                # a governed header exists, every status/time field is checked
                # fail-closed below.
                return True
            parsed = yaml.safe_load(match.group(0).split("---", 2)[1]) or {}
            if not isinstance(parsed, dict):
                return False
            allowed, _ = status_matches(parsed, set(DEFAULT_READ_STATUSES))
            return bool(allowed)
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            return False

    def _filter_readable_memory_rows(self, rows: Sequence) -> list:
        return [row for row in rows if self._memory_row_is_readable(row)]

    def _embedding_model(self) -> str:
        return self.embedding_model or str(getattr(self.embedder, "model", "") or "")

    def _cache_key(self, text_hash: str) -> str:
        return f"{text_hash}:{self._embedding_model()}:{self.chunk_strategy_version}"

    def _cached_vectors(self, conn: sqlite3.Connection, hashes: Iterable[str]) -> dict[str, tuple[list[float], int]]:
        keys = [self._cache_key(item) for item in dict.fromkeys(hashes)]
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        rows = conn.execute(
            f"SELECT cache_key, vec, embedding_dimension FROM embedding_cache WHERE cache_key IN ({placeholders})",
            keys,
        ).fetchall()
        cached: dict[str, tuple[list[float], int]] = {}
        for row in rows:
            vector = _unpack(row["vec"])
            dimension = int(row["embedding_dimension"] or 0)
            # A malformed cache entry must never become an apparently valid
            # vector.  Treat it as a cache miss so a configured provider can
            # repair it; lexical indexing can still proceed without one.
            if not vector or dimension <= 0 or len(vector) != dimension:
                continue
            cached[row["cache_key"].split(":", 1)[0]] = (vector, dimension)
        return cached

    def _embed_missing(self, texts: list[str]) -> tuple[list[list[float] | None], list[tuple[int, Exception]]]:
        result: list[list[float] | None] = [None] * len(texts)
        errors: list[tuple[int, Exception]] = []
        if not texts:
            return result, errors
        if self.embedder is None or not hasattr(self.embedder, "embed"):
            # Embedding is optional in P2: lexical indexing remains useful and
            # is the normal fallback when no provider is configured.
            return result, errors

        def call(indices: list[int]) -> None:
            if not indices:
                return
            batch = [texts[index] for index in indices]
            try:
                vectors = self.embedder.embed(batch)
                if len(vectors) != len(batch):
                    raise RuntimeError(f"embedding 数量不匹配: 期望 {len(batch)}，得到 {len(vectors)}")
                for index, vector in zip(indices, vectors):
                    if not isinstance(vector, (list, tuple)) or not vector:
                        raise RuntimeError("embedding 返回空向量")
                    result[index] = [float(value) for value in vector]
            except Exception as exc:
                if len(indices) == 1:
                    errors.append((indices[0], exc))
                    return
                middle = len(indices) // 2
                call(indices[:middle])
                call(indices[middle:])

        for start in range(0, len(texts), self.batch_size):
            call(list(range(start, min(len(texts), start + self.batch_size))))
        return result, errors

    def _write_failure(
        self,
        conn: sqlite3.Connection,
        path: str,
        stage: str,
        exc: Exception,
        *,
        chunk_id: str | None = None,
    ) -> None:
        now = time.time()
        key = f"{path}|{chunk_id or ''}|{stage}"
        existing = conn.execute("SELECT attempts FROM failures WHERE failure_key=?", (key,)).fetchone()
        attempts = int(existing["attempts"]) + 1 if existing else 1
        status = "dead" if attempts >= self.max_attempts else "pending"
        next_retry = now + min(3600.0, 2.0 ** min(attempts, 10)) if status == "pending" else None
        conn.execute(
            """INSERT INTO failures(
                failure_key,path,chunk_id,stage,error_type,error_message,attempts,
                next_retry_at,last_model,status,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(failure_key) DO UPDATE SET
                error_type=excluded.error_type,error_message=excluded.error_message,
                attempts=excluded.attempts,next_retry_at=excluded.next_retry_at,
                last_model=excluded.last_model,status=excluded.status,updated_at=excluded.updated_at""",
            (key, path, chunk_id, stage, type(exc).__name__, str(exc), attempts,
             next_retry, self._embedding_model(), status, now, now),
        )

    def _delete_failures(self, conn: sqlite3.Connection, path: str) -> None:
        conn.execute("DELETE FROM failures WHERE path=?", (path,))

    def _reconcile_failures(
        self,
        conn: sqlite3.Connection,
        path: str,
        active_failed_chunks: set[str],
    ) -> None:
        """Drop resolved/stale failures while retaining current chunk failures."""
        rows = conn.execute(
            "SELECT id,chunk_id FROM failures WHERE path=?", (path,)
        ).fetchall()
        for row in rows:
            if str(row["chunk_id"] or "") not in active_failed_chunks:
                conn.execute("DELETE FROM failures WHERE id=?", (row["id"],))

    def _insert_lexical(self, conn: sqlite3.Connection, chunk: Chunk) -> None:
        fields = {
            "content": (chunk.content, 1.0),
            "title": (chunk.title, 3.0),
            "tags": (" ".join(chunk.tags), 2.0),
            "path": (chunk.source_ref, 2.0),
        }
        postings: dict[str, tuple[list[str], float]] = {}
        for field, (value, weight) in fields.items():
            for token in _tokens(value):
                names, total = postings.setdefault(token, ([], 0.0))
                names.append(field)
                postings[token] = (names, total + weight)
        conn.executemany(
            "INSERT OR REPLACE INTO lexical(token,chunk_id,field,weight) VALUES(?,?,?,?)",
            [(token, chunk.chunk_id, "|".join(names), weight)
             for token, (names, weight) in postings.items()],
        )

    def upsert_document(self, rel_path: str, mtime: float, text: str) -> dict:
        """Replace one document; transient embedding failures leave lexical data fresh."""
        rel = _normalise_path(rel_path)
        source_hash = _source_hash(text)
        parsed = self.chunker(text, rel)
        doc_type, project_id, status = self._file_meta(parsed, rel)
        chunks = list(parsed.chunks)
        # Files without a title frontmatter still get a stable title signal.
        fallback_title = Path(rel).stem
        for chunk in chunks:
            if not chunk.title:
                chunk.title = fallback_title
        now = time.time()
        with self._conn() as conn:
            self._assert_version(conn)
            hashes = [_hash(_embed_text(item)) for item in chunks]
            cached = self._cached_vectors(conn, hashes)
            missing_hashes = list(dict.fromkeys(
                hashes[index] for index in range(len(chunks)) if hashes[index] not in cached
            ))
            representative = {text_hash: hashes.index(text_hash) for text_hash in missing_hashes}
            texts = [_embed_text(chunks[representative[text_hash]]) for text_hash in missing_hashes]
            fresh, errors = self._embed_missing(texts)
            provider_errors: list[tuple[int, Exception]] = []
            failed_chunks: set[str] = set()
            for error_index, exc in errors:
                # Budget stops are deliberate run boundaries, not provider
                # faults.  Leave those chunks without a vector so a later
                # approved run can continue without marking the file dead.
                if getattr(exc, "budget_stop", False):
                    continue
                chunk = chunks[representative[missing_hashes[error_index]]]
                provider_errors.append((error_index, exc))
                failed_chunks.add(chunk.chunk_id)

            vectors: list[list[float] | None] = []
            for index, chunk in enumerate(chunks):
                if hashes[index] in cached:
                    vectors.append(cached[hashes[index]][0])
                else:
                    vector = fresh[missing_hashes.index(hashes[index])]
                    vectors.append(vector)
            dimensions = {len(vector) for vector in vectors if vector}
            if len(dimensions) > 1:
                exc = RuntimeError("同一文件 embedding 维度不一致")
                self._write_failure(conn, rel, "embedding", exc)
                return {"path": rel, "updated": False, "chunks": 0,
                        "embedded": 0, "cache_hits": len(cached), "failed": 1}
            dimension = next(iter(dimensions), 0)
            meta = self._meta(conn)
            model = self._embedding_model()
            if model and not meta.get("embedding_model"):
                conn.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES('embedding_model',?)",
                             (model,))
                meta["embedding_model"] = model
            stored_dimension = int(meta.get("embedding_dimension", "0") or 0)
            if stored_dimension and dimension and stored_dimension != dimension:
                exc = IndexCompatibilityError(
                    f"embedding dimension 不一致: {stored_dimension} != {dimension}")
                self._write_failure(conn, rel, "compatibility", exc)
                return {"path": rel, "updated": False, "chunks": 0,
                        "embedded": 0, "cache_hits": len(cached), "failed": 1}
            if dimension:
                conn.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES('embedding_dimension',?)",
                             (str(dimension),))

            conn.execute("DELETE FROM lexical WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE file_path=?)", (rel,))
            conn.execute("DELETE FROM chunks WHERE file_path=?", (rel,))
            conn.execute("DELETE FROM entries WHERE file_path=?", (rel,))
            entry_ids: dict[str, int] = {}
            for chunk in chunks:
                entry_ref = _entry_key(chunk)
                if entry_ref not in entry_ids:
                    tags = json.dumps(chunk.tags, ensure_ascii=False)
                    cursor = conn.execute(
                        "INSERT INTO entries(file_path,mem_id,source_ref,bucket_path,entry_offset,title,tags) VALUES(?,?,?,?,?,?,?)",
                        (rel, chunk.mem_id, entry_ref, rel if chunk.mem_id else None,
                         chunk.start_offset, chunk.title, tags),
                    )
                    entry_ids[entry_ref] = int(cursor.lastrowid)
            for index, (chunk, vector, text_hash) in enumerate(zip(chunks, vectors, hashes)):
                conn.execute(
                    """INSERT INTO chunks(
                        chunk_id,file_path,entry_id,chunk_index,source_ref,content,
                        embedding_text_hash,embedding_model,embedding_dimension,vec,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (chunk.chunk_id, rel, entry_ids.get(_entry_key(chunk)), chunk.chunk_index,
                     chunk.source_ref, chunk.content, text_hash, model, dimension or None,
                     _pack(vector) if vector else None, now),
                )
                self._insert_lexical(conn, chunk)
                if vector:
                    conn.execute(
                        """INSERT OR REPLACE INTO embedding_cache(
                            cache_key,embedding_text_hash,embedding_model,chunk_strategy_version,
                            embedding_dimension,vec,updated_at
                        ) VALUES(?,?,?,?,?,?,?)""",
                        (self._cache_key(text_hash), text_hash, model, self.chunk_strategy_version,
                         dimension, _pack(vector), now),
                    )
            conn.execute(
                """INSERT OR REPLACE INTO files(
                    path,mtime,source_content_hash,doc_type,project_id,status,
                    parser_version,index_version,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (rel, float(mtime), source_hash, doc_type, project_id, status,
                 self.parser_version, self.index_version, now),
            )
            if provider_errors:
                for error_index, exc in provider_errors:
                    chunk = chunks[representative[missing_hashes[error_index]]]
                    self._write_failure(conn, rel, "embedding", exc, chunk_id=chunk.chunk_id)
                self._reconcile_failures(conn, rel, failed_chunks)
            else:
                self._delete_failures(conn, rel)
            return {"path": rel, "updated": True, "chunks": len(chunks),
                    "embedded": len([item for item in fresh if item is not None]),
                    "cache_hits": len(cached), "failed": len(provider_errors)}

    def remove_document(self, rel_path: str) -> None:
        rel = _normalise_path(rel_path)
        with self._conn() as conn:
            conn.execute("DELETE FROM lexical WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE file_path=?)", (rel,))
            conn.execute("DELETE FROM chunks WHERE file_path=?", (rel,))
            conn.execute("DELETE FROM entries WHERE file_path=?", (rel,))
            conn.execute("DELETE FROM files WHERE path=?", (rel,))
            conn.execute("DELETE FROM failures WHERE path=?", (rel,))

    def sync_vault(
        self,
        vault_root: str | Path | None = None,
        max_files: int | None = None,
        *,
        run_id: str = "",
        attempt_id: str = "",
    ) -> dict:
        """Synchronise one Vault scan and return a portable stage trace.

        The index remains the source of truth for queue/retry state.  This
        trace describes this synchronous adapter invocation only, so callers
        can correlate a direct RAG sync with the surrounding request without
        introducing a second persistence model.
        """
        trace = StageTracker(
            run_id or f"rag-sync-{uuid.uuid4().hex[:16]}",
            attempt_id or uuid.uuid4().hex[:16],
        )
        trace.record("accepted", "accepted", status=ProcessStatus.ACCEPTED)
        root = Path(vault_root or self.vault_root or ".")
        scanned = self._scan(root)
        trace.record("fetched", "fetched", status=ProcessStatus.FETCHED)
        try:
            with self._conn() as conn:
                self._assert_version(conn)
                old = {row["path"]: row for row in conn.execute("SELECT * FROM files")}
                missing_vectors = {
                    row["file_path"] for row in conn.execute(
                        "SELECT file_path FROM chunks WHERE vec IS NULL GROUP BY file_path"
                    )
                }
                now = time.time()
                blocked = {
                    row["path"] for row in conn.execute(
                        "SELECT path FROM failures WHERE status='dead' "
                        "OR (status='pending' AND next_retry_at IS NOT NULL AND next_retry_at>?)",
                        (now,),
                    )
                }
        except IndexCompatibilityError as exc:
            trace.record("failed", "failed", status=ProcessStatus.FAILED,
                         error_code=type(exc).__name__)
            return {
                "total": len(scanned), "updated": 0, "unchanged": 0,
                "attempted": 0, "deferred": 0, "removed": 0, "failed": 0,
                "chunks": self.chunk_count, "new_chunks": 0,
                "embedded": 0, "cache_hits": 0, "failures": self.failure_count,
                "compatible": False, "error": str(exc), "stages": trace.to_dict(),
            }
        candidates = [path for path, (mtime, source_hash, _) in scanned.items()
                      if path not in old or old[path]["source_content_hash"] != source_hash
                      or old[path]["parser_version"] != self.parser_version
                      or old[path]["index_version"] != self.index_version
                      or (self.embedder is not None and path in missing_vectors)]
        changed = [path for path in candidates if path not in blocked]
        deferred = [path for path in candidates if path in blocked]
        changed.sort(key=lambda path: scanned[path][0], reverse=True)
        if max_files is not None and max_files >= 0:
            deferred.extend(changed[max_files:])
            changed = changed[:max_files]
        removed = [path for path in old if path not in scanned]
        trace.record("parsed", "parsed", status=ProcessStatus.PARSED)
        for path in removed:
            self.remove_document(path)
        updated = failed = chunks = embedded = cache_hits = 0
        for path in changed:
            mtime, _, text = scanned[path]
            stat = self.upsert_document(path, mtime, text)
            updated += int(stat["updated"])
            failed += int(stat["failed"])
            chunks += int(stat["chunks"])
            embedded += int(stat["embedded"])
            cache_hits += int(stat["cache_hits"])
        trace.record("chunked", "chunked", status=ProcessStatus.CHUNKED)
        trace.record("indexed", "indexed", status=ProcessStatus.INDEXED)
        trace.record("completed", "completed", status=ProcessStatus.COMPLETED)
        return {
            "total": len(scanned), "updated": updated,
            "unchanged": max(0, len(scanned) - len(changed) - len(deferred)),
            "attempted": len(changed),
            "deferred": len(deferred), "removed": len(removed), "failed": failed,
            "chunks": self.chunk_count, "new_chunks": chunks,
            "embedded": embedded, "cache_hits": cache_hits,
            "failures": self.failure_count,
            "compatible": True, "stages": trace.to_dict(),
        }

    def plan_changes(
        self,
        vault_root: str | Path | None = None,
        *,
        since: float | None = None,
        force: bool = False,
    ) -> dict:
        """Build a deterministic ingestion plan without writing the queue.

        ``source_content_hash`` is the change detector; ``mtime`` is only a
        user-facing filter for ``since`` and a stable priority hint.  Reading
        an absent index is deliberately side-effect free so CLI ``--dry-run``
        does not create a SQLite file.
        """
        root = Path(vault_root or self.vault_root or ".")
        scanned = self._scan(root) if root.exists() else {}
        old: dict[str, sqlite3.Row] = {}
        missing_vectors: set[str] = set()
        deferred_missing_vectors: set[str] = set()
        if self.db_path.exists():
            try:
                with self._conn() as conn:
                    self._assert_version(conn)
                    old = {row["path"]: row for row in conn.execute("SELECT * FROM files")}
                    if self.embedder is not None:
                        missing_vectors = {
                            str(row["file_path"])
                            for row in conn.execute(
                                "SELECT file_path FROM chunks WHERE vec IS NULL GROUP BY file_path"
                            )
                        }
                        deferred_missing_vectors = {
                            str(row["path"])
                            for row in conn.execute(
                                """SELECT path FROM failures
                                   WHERE status='dead' OR
                                         (status='pending' AND next_retry_at IS NOT NULL AND next_retry_at>?)
                                   GROUP BY path""",
                                (time.time(),),
                            )
                        }
            except IndexCompatibilityError as exc:
                return {
                    "total_files": len(scanned), "upserts": 0, "deletes": 0,
                    "unchanged": 0, "changes": [], "compatible": False,
                    "error": str(exc),
                }

        changes: list[dict] = []
        for path, (mtime, source_hash, _text) in scanned.items():
            previous = old.get(path)
            changed = (
                force or previous is None
                or previous["source_content_hash"] != source_hash
                or previous["parser_version"] != self.parser_version
                or previous["index_version"] != self.index_version
                or (
                    self.embedder is not None
                    and path in missing_vectors
                    and path not in deferred_missing_vectors
                    and previous is not None
                )
            )
            if not changed or (since is not None and float(mtime) <= since and not force):
                continue
            changes.append({
                "path": path, "operation": "upsert",
                "source_content_hash": source_hash, "mtime": float(mtime),
                "priority": 100,
                "reason": (
                    "missing_vector"
                    if path in missing_vectors and previous and path not in deferred_missing_vectors
                    else "source_changed"
                ),
            })
        # A missing path is a deletion even when ``--since`` is used: the
        # index has no reliable deletion mtime to compare with the cutoff.
        for path in sorted(set(old) - set(scanned)):
            changes.append({
                "path": path, "operation": "delete",
                "source_content_hash": "", "mtime": None, "priority": 10,
            })
        changes.sort(key=lambda item: (-int(item["priority"]), item["path"]))
        return {
            "total_files": len(scanned),
            "upserts": sum(item["operation"] == "upsert" for item in changes),
            "deletes": sum(item["operation"] == "delete" for item in changes),
            "unchanged": max(0, len(scanned) - sum(item["operation"] == "upsert" for item in changes)),
            "changes": changes,
            "compatible": True,
        }

    def enqueue_changes(
        self,
        vault_root: str | Path | None = None,
        *,
        since: float | None = None,
        force: bool = False,
    ) -> dict:
        """Persist a scan plan idempotently in ``ingest_queue``."""
        plan = self.plan_changes(vault_root, since=since, force=force)
        if not plan.get("compatible", True):
            return {**plan, "enqueued": 0, "skipped": 0}
        if not plan["changes"]:
            return {**plan, "enqueued": 0, "skipped": 0}
        now = time.time()
        enqueued = skipped = 0
        with self._conn() as conn:
            self._assert_version(conn)
            for item in plan["changes"]:
                row = conn.execute(
                    "SELECT * FROM ingest_queue WHERE path=? AND index_version=?",
                    (item["path"], self.index_version),
                ).fetchone()
                if row is not None:
                    same = (
                        row["operation"] == item["operation"]
                        and row["source_content_hash"] == item["source_content_hash"]
                    )
                    # Never take a live lease away from another writer.  A
                    # later scan can enqueue a fresh task after that lease.
                    if same and not force and row["status"] in {"pending", "running", "succeeded", "failed", "dead"}:
                        consistent = True
                        if row["status"] == "succeeded":
                            indexed = conn.execute(
                                "SELECT source_content_hash FROM files WHERE path=?",
                                (item["path"],),
                            ).fetchone()
                            if item["operation"] == "upsert":
                                consistent = bool(
                                    indexed
                                    and indexed["source_content_hash"] == item["source_content_hash"]
                                )
                            else:
                                consistent = indexed is None
                        if consistent:
                            skipped += 1
                            continue
                    if row["status"] == "running":
                        skipped += 1
                        continue
                    conn.execute(
                        """UPDATE ingest_queue SET operation=?,source_content_hash=?,mtime=?,
                           priority=?,attempts=0,max_attempts=?,status='pending',last_error='',
                           next_retry_at=NULL,lease_until=NULL,updated_at=? WHERE id=?""",
                        (item["operation"], item["source_content_hash"], item["mtime"],
                         item["priority"], self.max_attempts, now, row["id"]),
                    )
                    enqueued += 1
                    continue
                conn.execute(
                    """INSERT INTO ingest_queue(
                       path,operation,source_content_hash,mtime,priority,attempts,max_attempts,
                       status,index_version,last_error,next_retry_at,lease_until,created_at,updated_at
                    ) VALUES(?,?,?,?,?,0,?,'pending',?,?,NULL,NULL,?,?)""",
                    (item["path"], item["operation"], item["source_content_hash"], item["mtime"],
                     item["priority"], self.max_attempts, self.index_version, "", now, now),
                )
                enqueued += 1
        return {**plan, "enqueued": enqueued, "skipped": skipped}

    def _recover_queue(self, conn: sqlite3.Connection, now: float | None = None) -> int:
        """Return expired leases to pending so a crashed worker is recoverable."""
        now = time.time() if now is None else float(now)
        expired = conn.execute(
            """SELECT id FROM ingest_queue WHERE status='running'
               AND (lease_until IS NULL OR lease_until<=?)""", (now,)
        ).fetchall()
        expired_ids = {int(row["id"]) for row in expired}
        if not expired_ids:
            return 0
        cursor = conn.execute(
            """UPDATE ingest_queue SET status='pending',lease_until=NULL,
               attempts=CASE WHEN attempts>0 THEN attempts-1 ELSE 0 END,
               next_retry_at=NULL,updated_at=?
               WHERE status='running' AND (lease_until IS NULL OR lease_until<=?)""",
            (now, now),
        )
        # A worker may have died between claim and item completion.  Mark the
        # abandoned batch explicitly; its rows are now pending and can be
        # claimed by the next worker.
        for checkpoint in conn.execute(
            "SELECT batch_id,queue_ids FROM checkpoints WHERE status='started'"
        ).fetchall():
            try:
                queue_ids = {int(item) for item in json.loads(checkpoint["queue_ids"])}
            except (TypeError, ValueError, json.JSONDecodeError):
                queue_ids = set()
            if expired_ids & queue_ids:
                conn.execute(
                    "UPDATE checkpoints SET status='aborted',completed_at=? WHERE batch_id=?",
                    (now, checkpoint["batch_id"]),
                )
        return int(cursor.rowcount)

    def _claim_queue_batch(self, limit: int, lease_seconds: float) -> tuple[str | None, list[dict]]:
        if limit <= 0:
            return None, []
        now = time.time()
        with self._conn() as conn:
            self._assert_version(conn)
            self._recover_queue(conn, now)
            rows = conn.execute(
                """SELECT * FROM ingest_queue
                   WHERE index_version=? AND attempts<max_attempts AND
                     (status='pending' OR (status='failed' AND
                       (next_retry_at IS NULL OR next_retry_at<=?)))
                   ORDER BY priority DESC,created_at,id LIMIT ?""",
                (self.index_version, now, int(limit)),
            ).fetchall()
            if not rows:
                return None, []
            ids = [int(row["id"]) for row in rows]
            placeholders = ",".join("?" for _ in ids)
            lease_until = now + max(1.0, float(lease_seconds))
            conn.execute(
                f"""UPDATE ingest_queue SET status='running',attempts=attempts+1,
                   lease_until=?,updated_at=? WHERE id IN ({placeholders})""",
                [lease_until, now, *ids],
            )
            batch_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO checkpoints(batch_id,queue_ids,status,created_at,completed_at)
                   VALUES(?,?, 'started',?,NULL)""",
                (batch_id, json.dumps(ids), now),
            )
            claimed = [dict(row) for row in rows]
            for row in claimed:
                row["attempts"] = int(row["attempts"]) + 1
                row["lease_until"] = lease_until
            return batch_id, claimed

    def _finish_queue_item(self, queue_id: int, *, success: bool, error: str = "") -> str:
        now = time.time()
        with self._conn() as conn:
            row = conn.execute("SELECT attempts,max_attempts FROM ingest_queue WHERE id=?", (queue_id,)).fetchone()
            if row is None:
                return "missing"
            if success:
                status = "succeeded"
                next_retry = None
                error = ""
            else:
                status = "dead" if int(row["attempts"]) >= int(row["max_attempts"]) else "failed"
                next_retry = None if status == "dead" else now + min(3600.0, 2.0 ** min(int(row["attempts"]), 10))
            conn.execute(
                """UPDATE ingest_queue SET status=?,last_error=?,next_retry_at=?,lease_until=NULL,updated_at=?
                   WHERE id=?""",
                (status, error[:4000], next_retry, now, queue_id),
            )
            return status

    def _finish_checkpoint(self, batch_id: str, status: str = "completed") -> None:
        if not batch_id:
            return
        with self._conn() as conn:
            conn.execute(
                "UPDATE checkpoints SET status=?,completed_at=? WHERE batch_id=? AND status='started'",
                (status, time.time(), batch_id),
            )

    def process_queue(
        self,
        vault_root: str | Path | None = None,
        *,
        limit: int | None = None,
        lease_seconds: float = 300.0,
    ) -> dict:
        """Process ready queue rows in bounded, checkpointed batches."""
        root = Path(vault_root or self.vault_root or ".")
        started = time.perf_counter()
        remaining = None if limit is None or int(limit) <= 0 else int(limit)
        stats = {
            "claimed": 0, "processed": 0, "succeeded": 0, "failed": 0,
            "dead": 0, "updated": 0, "deleted": 0, "chunks": 0,
            "embedded": 0, "cache_hits": 0, "vector_failed": 0, "batches": 0,
        }
        while remaining is None or remaining > 0:
            batch_limit = self.batch_size if remaining is None else min(self.batch_size, remaining)
            batch_id, rows = self._claim_queue_batch(batch_limit, lease_seconds)
            if not rows:
                break
            stats["batches"] += 1
            stats["claimed"] += len(rows)
            try:
                for row in rows:
                    success = False
                    try:
                        path = root / row["path"]
                        if row["operation"] == "delete":
                            self.remove_document(row["path"])
                            stats["deleted"] += 1
                        elif path.exists() and path.is_file():
                            text = path.read_text(encoding="utf-8", errors="ignore")
                            result = self.upsert_document(row["path"], path.stat().st_mtime, text)
                            # Provider failures may be partial: lexical data is
                            # committed and the missing vector is retried via
                            # the failures table. Only an atomic failure should
                            # fail the ingest queue item.
                            if int(result.get("failed", 0)) and not int(result.get("updated", 0)):
                                raise RuntimeError(f"文档写入失败: {row['path']}")
                            stats["updated"] += int(result.get("updated", 0))
                            stats["chunks"] += int(result.get("chunks", 0))
                            stats["embedded"] += int(result.get("embedded", 0))
                            stats["cache_hits"] += int(result.get("cache_hits", 0))
                            stats["vector_failed"] += int(result.get("failed", 0))
                        else:
                            # A file removed after planning is a successful,
                            # idempotent delete regardless of operation label.
                            self.remove_document(row["path"])
                            stats["deleted"] += 1
                        status = self._finish_queue_item(int(row["id"]), success=True)
                        success = status == "succeeded"
                    except Exception as exc:
                        status = self._finish_queue_item(
                            int(row["id"]), success=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        stats["failed"] += 1
                        stats["dead"] += int(status == "dead")
                    stats["processed"] += 1
                    stats["succeeded"] += int(success)
                    if remaining is not None:
                        remaining -= 1
            except BaseException:
                # Leave queue leases and the checkpoint visible on interruption;
                # the next worker recovers expired rows instead of claiming a
                # partially processed batch was complete.
                self._finish_checkpoint(batch_id, "aborted")
                raise
            else:
                # Individual row failures are already represented by failed /
                # dead queue states, so the batch itself completed normally.
                self._finish_checkpoint(batch_id, "completed")
        stats["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        stats["remaining_ready"] = self.queue_status(root).get("ready", 0)
        return stats

    def retry_queue(
        self,
        vault_root: str | Path | None = None,
        *,
        limit: int = 20,
        process: bool = True,
    ) -> dict:
        """Reset failed/dead ingestion rows and optionally process them now."""
        if limit <= 0 or not self.db_path.exists():
            return {"reset": 0, "processed": 0, "succeeded": 0, "failed": 0}
        root = Path(vault_root or self.vault_root or ".")
        scanned = self._scan(root) if root.exists() else {}
        now = time.time()
        reset = 0
        with self._conn() as conn:
            self._assert_version(conn)
            rows = conn.execute(
                """SELECT id,path FROM ingest_queue WHERE index_version=?
                   AND status IN ('failed','dead') ORDER BY updated_at,id LIMIT ?""",
                (self.index_version, int(limit)),
            ).fetchall()
            for row in rows:
                item = scanned.get(row["path"])
                operation = "upsert" if item else "delete"
                source_hash = item[1] if item else ""
                mtime = item[0] if item else None
                conn.execute(
                    """UPDATE ingest_queue SET operation=?,source_content_hash=?,mtime=?,
                       attempts=0,status='pending',last_error='',next_retry_at=NULL,
                       lease_until=NULL,updated_at=? WHERE id=?""",
                    (operation, source_hash, mtime, now, row["id"]),
                )
                reset += 1
        result = {"reset": reset, "processed": 0, "succeeded": 0, "failed": 0}
        if process and reset:
            result.update(self.process_queue(root, limit=reset))
        return result

    def queue_status(self, vault_root: str | Path | None = None) -> dict:
        """Return queue/checkpoint health plus index coverage metadata."""
        root = Path(vault_root or self.vault_root or ".")
        if not self.db_path.exists():
            expected = len(self._scan(root)) if root.exists() else 0
            return {
                "db_exists": False, "indexed_files": 0, "expected_files": expected,
                "fresh_files": 0, "stale_files": expected, "coverage": 0.0 if expected else 1.0,
                "chunks": 0, "failures": 0, "ready": 0, "pending": 0,
                "running": 0, "succeeded": 0, "failed": 0, "dead": 0,
                "oldest_ready_at": None, "oldest_deferred_at": None,
                "last_success_at": None, "queue_age_seconds": 0.0,
                "checkpoints": {"started": 0, "completed": 0, "aborted": 0},
            }
        with self._conn() as conn:
            counts = {row["status"]: int(row["count"]) for row in conn.execute(
                "SELECT status,COUNT(*) AS count FROM ingest_queue WHERE index_version=? GROUP BY status",
                (self.index_version,),
            )}
            oldest = conn.execute(
                """SELECT MIN(created_at) FROM ingest_queue WHERE index_version=?
                   AND (status='pending' OR (status='failed' AND
                        (next_retry_at IS NULL OR next_retry_at<=?)))""",
                (self.index_version, time.time()),
            ).fetchone()[0]
            oldest_deferred = conn.execute(
                """SELECT MIN(created_at) FROM ingest_queue WHERE index_version=?
                   AND status IN ('pending','failed')""", (self.index_version,),
            ).fetchone()[0]
            last_success = conn.execute(
                """SELECT MAX(updated_at) FROM ingest_queue WHERE index_version=?
                   AND status='succeeded'""", (self.index_version,),
            ).fetchone()[0]
            ready = conn.execute(
                """SELECT COUNT(*) FROM ingest_queue WHERE index_version=?
                   AND (status='pending' OR (status='failed' AND
                        (next_retry_at IS NULL OR next_retry_at<=?)))""",
                (self.index_version, time.time()),
            ).fetchone()[0]
            checkpoints = {row["status"]: int(row["count"]) for row in conn.execute(
                "SELECT status,COUNT(*) AS count FROM checkpoints GROUP BY status")}
        status = self.index_status(root)
        now = time.time()
        status.update({
            "db_exists": True,
            "pending": counts.get("pending", 0),
            "running": counts.get("running", 0),
            "succeeded": counts.get("succeeded", 0),
            "failed": counts.get("failed", 0),
            "dead": counts.get("dead", 0),
            "ready": int(ready),
            "oldest_ready_at": oldest,
            "oldest_deferred_at": oldest_deferred,
            "last_success_at": last_success,
            "queue_age_seconds": round(max(0.0, now - float(oldest)), 3) if oldest else 0.0,
            "checkpoints": {
                "started": checkpoints.get("started", 0),
                "completed": checkpoints.get("completed", 0),
                "aborted": checkpoints.get("aborted", 0),
            },
        })
        return status

    def list_queue_failures(self, limit: int = 100) -> list[dict]:
        """List failed/dead queue rows without changing their retry state."""
        if limit <= 0 or not self.db_path.exists():
            return []
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM ingest_queue WHERE index_version=?
                   AND status IN ('failed','dead') ORDER BY updated_at,id LIMIT ?""",
                (self.index_version, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    @property
    def chunk_count(self) -> int:
        if not self.db_path.exists():
            return 0
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    @property
    def failure_count(self) -> int:
        if not self.db_path.exists():
            return 0
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM failures WHERE status IN ('pending','dead')").fetchone()[0])

    def list_failures(self, limit: int = 100) -> list[dict]:
        if limit <= 0 or not self.db_path.exists():
            return []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM failures WHERE status IN ('pending','dead') ORDER BY updated_at LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def retry_failures(self, vault_root: str | Path | None = None, limit: int = 20) -> dict:
        root = Path(vault_root or self.vault_root or ".")
        pending = self.list_failures(limit)
        retried = succeeded = 0
        for failure in pending:
            path = root / failure["path"]
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                stat = self.upsert_document(failure["path"], path.stat().st_mtime, text)
                retried += 1
                succeeded += int(stat["updated"])
            except (OSError, RuntimeError):
                retried += 1
        return {"retried": retried, "succeeded": succeeded,
                "failed": max(0, retried - succeeded), "remaining": self.failure_count}

    def index_status(self, vault_root: str | Path | None = None) -> dict:
        root = Path(vault_root or self.vault_root or ".")
        scanned = self._scan(root) if root.exists() else {}
        with self._conn() as conn:
            meta = self._meta(conn)
            file_rows = conn.execute("SELECT path,source_content_hash FROM files").fetchall()
            indexed = len(file_rows)
            entries = int(conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
            chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            vector_chunks = int(conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE vec IS NOT NULL"
            ).fetchone()[0])
            failures = int(conn.execute("SELECT COUNT(*) FROM failures WHERE status IN ('pending','dead')").fetchone()[0])
            oldest = conn.execute(
                "SELECT path, updated_at FROM failures WHERE status IN ('pending','dead') ORDER BY updated_at LIMIT 1"
            ).fetchone()
        expected = len(scanned) if root.exists() else indexed
        fresh = sum(1 for row in file_rows
                    if row["path"] in scanned and row["source_content_hash"] == scanned[row["path"]][1])
        return {
            "metadata_valid": bool(meta) and meta.get("schema_version") == INDEX_SCHEMA_VERSION,
            "schema_version": meta.get("schema_version", INDEX_SCHEMA_VERSION),
            "index_version": meta.get("index_version", self.index_version),
            "parser_version": meta.get("parser_version", self.parser_version),
            "chunk_strategy_version": meta.get(
                "chunk_strategy_version", self.chunk_strategy_version
            ),
            "embedding_model": meta.get("embedding_model", self._embedding_model()),
            "embedding_dimension": int(meta.get("embedding_dimension", "0") or 0),
            "indexed_files": indexed, "expected_files": expected,
            "fresh_files": fresh, "stale_files": max(0, expected - fresh),
            "coverage": round(fresh / expected, 3) if expected else 1.0,
            "entries": entries, "chunks": chunks, "vector_chunks": vector_chunks,
            "failures": failures,
            "oldest_failure": dict(oldest) if oldest else None,
            "db_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }

    @staticmethod
    def _allowed_sql(statuses: Sequence[str] | None,
                     include_archive: bool = False) -> tuple[str, list[str]]:
        if statuses is not None:
            allowed = [str(item) for item in statuses]
            if not allowed:
                return " AND 1=0", []
            return " AND f.status IN (" + ",".join("?" for _ in allowed) + ")", allowed
        if include_archive:
            return "", []
        return " AND f.status NOT IN ('archived','superseded')", []

    @staticmethod
    def _path_prefix_sql(path_prefixes: Sequence[str] | None) -> tuple[str, list[str]]:
        """Build a bounded source-path filter for typed retrieval routes.

        Memory is a distinct corpus with shorter, entry-shaped records.  A
        caller may request that corpus before ranking so unrelated Vault
        chunks cannot consume the lexical candidate window.  This remains a
        retrieval filter, not an access-control boundary; project/status
        filters are still applied separately.
        """
        prefixes = []
        for value in path_prefixes or ():
            prefix = str(value or "").replace("\\", "/").strip().lstrip("/")
            if prefix and prefix not in prefixes:
                prefixes.append(prefix.rstrip("/") + "/")
        if not prefixes:
            return "", []
        return " AND (" + " OR ".join("f.path LIKE ?" for _ in prefixes) + ")", [
            prefix + "%" for prefix in prefixes
        ]

    def search_lexical(
        self,
        query: str,
        k: int = 20,
        *,
        project_id: str | None = None,
        statuses: Sequence[str] | None = None,
        include_archive: bool = False,
        path_prefixes: Sequence[str] | None = None,
    ) -> list[dict]:
        tokens = _ranking_query_tokens(query)
        if k <= 0:
            self._set_search_status("lexical", "unavailable", "invalid_k")
            return []
        if not self.db_path.exists():
            self._set_search_status("lexical", "unavailable", "index_missing")
            return []
        if not tokens:
            self._set_search_status("lexical", "available", "empty_query_tokens")
            return []
        with self._conn() as conn:
            try:
                self._assert_version(conn, allow_empty=False)
            except IndexCompatibilityError as exc:
                self._set_search_status("lexical", "unavailable", str(exc))
                return []
            where, params = self._allowed_sql(statuses, include_archive)
            project_clause = " AND f.project_id=?" if project_id else ""
            if project_id:
                params.append(project_id)
            path_clause, path_params = self._path_prefix_sql(path_prefixes)
            params.extend(path_params)
            where_with_path = where + project_clause + path_clause
            placeholders = ",".join("?" for _ in tokens)
            filter_params = list(params)
            # Keep the established candidate window for ordinary queries.
            # The v2 reranker only changes ordering for the narrow rare-
            # identifier case below, preserving historical lexical behavior.
            candidate_limit = max(50, int(k) * LEXICAL_CANDIDATE_WINDOW_MULT)
            rows = conn.execute(
                f"""SELECT c.chunk_id,c.source_ref,c.content,c.file_path,c.entry_id,c.chunk_index,
                    COALESCE(SUM(l.weight),0) AS score
                    FROM lexical l JOIN chunks c ON c.chunk_id=l.chunk_id
                    JOIN files f ON f.path=c.file_path
                    WHERE l.token IN ({placeholders}){where_with_path}
                    GROUP BY c.chunk_id ORDER BY score DESC,c.chunk_id LIMIT ?""",
                list(tokens) + filter_params + [candidate_limit],
            ).fetchall()
            # A rare identifier can be drowned out by common CJK postings in
            # the initial score-ordered window.  Pull a second bounded window
            # for long ASCII identifiers and merge by chunk id before the
            # query-aware reranker runs.
            identifier_tokens = sorted(
                token for token in tokens
                if len(token) >= 3 and _ASCII_RE.fullmatch(token)
            )
            total_chunks_for_rescue = max(
                1, int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            )
            rare_identifiers = []
            for token in identifier_tokens:
                df = int(conn.execute(
                    "SELECT COUNT(DISTINCT chunk_id) FROM lexical WHERE token=?",
                    (token,),
                ).fetchone()[0])
                if math.log((total_chunks_for_rescue + 1) / (df + 1)) + 1.0 >= 2.5:
                    rare_identifiers.append(token)
            if rare_identifiers:
                rescue_limit = candidate_limit
                id_placeholders = ",".join("?" for _ in rare_identifiers)
                id_rows = conn.execute(
                    f"""SELECT c.chunk_id,c.source_ref,c.content,c.file_path,c.entry_id,c.chunk_index,
                        COALESCE(SUM(l.weight),0) AS score
                        FROM lexical l JOIN chunks c ON c.chunk_id=l.chunk_id
                        JOIN files f ON f.path=c.file_path
                        WHERE l.token IN ({id_placeholders}){where_with_path}
                        GROUP BY c.chunk_id ORDER BY score DESC,c.chunk_id LIMIT ?""",
                    rare_identifiers + filter_params + [rescue_limit],
                ).fetchall()
                title_rescue_rows = [
                    row for row in id_rows
                    if any(
                        token in _tokens(Path(str(row["file_path"])).stem)
                        for token in rare_identifiers
                    )
                ]
                if title_rescue_rows:
                    merged = {str(row["chunk_id"]): row for row in rows}
                    merged.update({str(row["chunk_id"]): row for row in title_rescue_rows})
                    rows = list(merged.values())
            rows = self._filter_readable_memory_rows(rows)
            # Compute document frequency only for tokens in this query.  This
            # keeps reranking bounded even as the lexical table grows.
            idf_rows = conn.execute(
                f"""SELECT token,COUNT(DISTINCT chunk_id) AS df
                    FROM lexical WHERE token IN ({placeholders}) GROUP BY token""",
                list(tokens),
            ).fetchall()
            total_chunks = max(1, int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]))
            token_idf = {
                str(item["token"]): math.log((total_chunks + 1) / (int(item["df"]) + 1)) + 1.0
                for item in idf_rows
            }
            for token in tokens:
                token_idf.setdefault(token, math.log(total_chunks + 1) + 1.0)
            if len(rare_identifiers) == 1 and any(
                rare_identifiers[0] in _tokens(Path(str(row["file_path"])).stem)
                for row in rows
            ):
                # Entity/title rescue is deliberately narrow.  For ordinary
                # natural-language and bucket queries, preserve the proven
                # posting order; global IDF reranking regresses those queries.
                rows = sorted(
                    rows,
                    key=lambda row: _lexical_rank_key(row, tokens, token_idf),
                    reverse=True,
                )[:max(1, k)]
            else:
                rows = sorted(
                    rows,
                    key=lambda row: (-float(row["score"] or 0.0), str(row["chunk_id"])),
                )[:max(1, k)]
        self._set_search_status("lexical", "available")
        return [{"title": row["file_path"], "content": row["content"],
                 "ref": row["source_ref"], "source": "lexical",
                 "score": round(float(row["score"]), 4),
                 "chunk_id": row["chunk_id"], "file_path": row["file_path"],
                 "entry_id": _row_value(row, "entry_id"), "chunk_index": row["chunk_index"]} for row in rows]

    def search_parent_entries(
        self,
        query: str,
        k: int = 20,
        *,
        project_id: str | None = None,
        statuses: Sequence[str] | None = None,
        include_archive: bool = False,
        max_chars: int = 2400,
        path_prefixes: Sequence[str] | None = None,
    ) -> list[dict]:
        """Return entry-level lexical representatives for topic-style asks.

        ``search_lexical`` ranks chunks, so a monthly memory bucket can use
        the whole candidate window before another entry gets a chance.  This
        read-only companion aggregates each matched posting once per
        ``entries`` row, then renders the entry's first chunks behind its
        stable ``path#mem-id`` (or file) anchor.  It is intentionally opt-in
        at the caller so ordinary lexical ranking and the vector/answer gates
        remain unchanged.
        """
        tokens = _ranking_query_tokens(query)
        if k <= 0 or not tokens or not self.db_path.exists():
            return []
        with self._conn() as conn:
            try:
                self._assert_version(conn, allow_empty=False)
            except IndexCompatibilityError:
                return []
            where, params = self._allowed_sql(statuses, include_archive)
            project_clause = " AND f.project_id=?" if project_id else ""
            if project_id:
                params.append(project_id)
            path_clause, path_params = self._path_prefix_sql(path_prefixes)
            params.extend(path_params)
            where_with_path = where + project_clause + path_clause
            placeholders = ",".join("?" for _ in tokens)
            # Group by entry and token first: tags are copied to every child
            # chunk, so SUM(weight) over raw postings would reward long entries
            # rather than topic coverage.
            rows = conn.execute(
                f"""SELECT e.id,e.file_path,e.mem_id,e.source_ref,e.title,e.tags,
                           l.token,MAX(l.weight) AS token_weight,
                           l.field
                    FROM lexical l
                    JOIN chunks c ON c.chunk_id=l.chunk_id
                    JOIN entries e ON e.id=c.entry_id
                    JOIN files f ON f.path=c.file_path
                    WHERE l.token IN ({placeholders}){where_with_path}
                    GROUP BY e.id,l.token""",
                list(tokens) + params,
            ).fetchall()
            rows = self._filter_readable_memory_rows(rows)
            entry_total = max(
                1, int(conn.execute(
                    "SELECT COUNT(*) FROM entries e JOIN files f ON f.path=e.file_path"
                    + where_with_path,
                    params,
                ).fetchone()[0] or 1),
            )
            token_entries: dict[str, set[int]] = {}
            grouped: dict[int, dict] = {}
            for row in rows:
                entry_id = int(row["id"])
                token = str(row["token"])
                token_entries.setdefault(token, set()).add(entry_id)
                item = grouped.setdefault(entry_id, {
                    "entry_id": entry_id,
                    "file_path": str(row["file_path"] or ""),
                    "mem_id": str(row["mem_id"] or ""),
                    "ref": str(row["source_ref"] or ""),
                    "title": str(row["title"] or row["file_path"] or ""),
                    "tags": row["tags"] or "[]",
                    "matched_tokens": set(),
                    "score": 0.0,
                })
                item["matched_tokens"].add(token)
                item.setdefault("token_weights", {})[token] = max(
                    float(item.setdefault("token_weights", {}).get(token, 0.0)),
                    float(row["token_weight"] or 0.0),
                )
            token_idf = {
                token: math.log((entry_total + 1) / (len(entry_ids) + 1)) + 1.0
                for token, entry_ids in token_entries.items()
            }
            for item in grouped.values():
                item["score"] = sum(
                    weight * token_idf.get(token, 1.0)
                    for token, weight in item.get("token_weights", {}).items()
                )
            ranked = sorted(
                grouped.values(),
                key=lambda item: (-item["score"], -len(item["matched_tokens"]), item["ref"]),
            )[:max(1, int(k))]
            output: list[dict] = []
            budget = max(0, int(max_chars))
            for item in ranked:
                chunks = conn.execute(
                    "SELECT content FROM chunks WHERE entry_id=? ORDER BY chunk_index,chunk_id",
                    (item["entry_id"],),
                ).fetchall()
                content_parts: list[str] = []
                used = 0
                for chunk in chunks:
                    text = str(chunk["content"] or "")
                    if not text or (budget and used >= budget):
                        break
                    if budget:
                        text = text[:max(0, budget - used)]
                    if text:
                        content_parts.append(text)
                        used += len(text)
                if not content_parts:
                    continue
                try:
                    tags = json.loads(str(item["tags"] or "[]"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    tags = []
                output.append({
                    "title": item["title"],
                    "content": "\n\n".join(content_parts),
                    "ref": item["ref"],
                    "source": "lexical",
                    "score": round(float(item["score"]), 4),
                    "entry_id": item["entry_id"],
                    "file_path": item["file_path"],
                    "mem_id": item["mem_id"],
                    "tags": [str(tag) for tag in tags if str(tag).strip()],
                    "parent_entry": True,
                })
        return output

    def _vector_ready(self, conn: sqlite3.Connection) -> tuple[dict[str, str], int] | None:
        try:
            meta = self._assert_version(conn, allow_empty=False)
        except IndexCompatibilityError:
            return None
        dimension = int(meta.get("embedding_dimension", "0") or 0)
        model = meta.get("embedding_model", "")
        if not dimension or (model and self._embedding_model() and model != self._embedding_model()):
            return None
        return meta, dimension

    def _matrix_scope_hash(self, signature: tuple, dimension: int) -> str:
        """侧车键：签名（row_count/max_updated/维度/project/statuses）+ 索引版本 + 模型。"""
        parts = (
            self.index_version, self.chunk_strategy_version,
            self._embedding_model() or "", dimension,
            signature[0], signature[1],
            signature[3], signature[4], bool(signature[5]) if len(signature) > 5 else False,
            signature[6] if len(signature) > 6 else (),
        )
        raw = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _matrix_persist_load(self, conn: sqlite3.Connection, scope_hash: str):
        """从索引内置侧车读归一化矩阵；签名不符或损坏返回 None（调用方重建）。"""
        row = conn.execute(
            "SELECT row_count, dimension, matrix_f32, rows_json FROM vector_matrix WHERE scope_hash=?",
            (scope_hash,),
        ).fetchone()
        if row is None:
            return None
        try:
            rows = json.loads(row["rows_json"])
            matrix = _np.frombuffer(bytes(row["matrix_f32"]), dtype=_np.float32)
            expected = int(row["row_count"]) * int(row["dimension"])
            if matrix.size != expected or len(rows) != int(row["row_count"]):
                return None
            matrix = matrix.reshape((int(row["row_count"]), int(row["dimension"])))
            return matrix, rows
        except (ValueError, TypeError, KeyError):
            return None

    def _matrix_persist_save(self, conn: sqlite3.Connection, scope_hash: str,
                             matrix, rows, row_count: int, max_updated: float,
                             dimension: int) -> None:
        """把本次构建的归一化矩阵与行元数据回写侧车（幂等 upsert）。"""
        row_dicts = [{key: row[key] for key in
                     ("chunk_id", "source_ref", "content", "file_path", "chunk_index")}
                    for row in rows]
        conn.execute(
            "INSERT OR REPLACE INTO vector_matrix"
            "(scope_hash,index_version,row_count,max_updated,dimension,model,matrix_f32,rows_json,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (scope_hash, self.index_version, row_count, max_updated, dimension,
             self._embedding_model() or "", matrix.tobytes(),
             json.dumps(row_dicts, ensure_ascii=False), time.time()),
        )
        # 有界淘汰：只保留最近 8 个 scope 的侧车，避免索引长期变更后表无限膨胀。
        conn.execute(
            "DELETE FROM vector_matrix WHERE scope_hash NOT IN "
            "(SELECT scope_hash FROM vector_matrix ORDER BY created_at DESC LIMIT 8)"
        )

    def search_vector(
        self,
        query: str,
        k: int = 20,
        *,
        project_id: str | None = None,
        statuses: Sequence[str] | None = None,
        include_archive: bool = False,
        path_prefixes: Sequence[str] | None = None,
    ) -> list[dict]:
        if k <= 0:
            self._set_search_status("vector", "unavailable", "invalid_k")
            return []
        if not self.db_path.exists():
            self._set_search_status("vector", "unavailable", "index_missing")
            return []
        if self.embedder is None:
            self._set_search_status("vector", "unavailable", "provider_missing")
            return []
        with self._conn() as conn:
            ready = self._vector_ready(conn)
            if ready is None:
                self._set_search_status("vector", "unavailable", "vector_metadata_unavailable")
                return []
        cache_key = (self._embedding_model(), str(query))
        qvec = self._query_embedding_cache.get(cache_key)
        if qvec is not None:
            self._query_embedding_cache.move_to_end(cache_key)
            qvec = list(qvec)
        else:
            try:
                query_vectors = self.embedder.embed([query])
                qvec = query_vectors[0]
            except Exception as exc:
                self._set_search_status("vector", "unavailable", f"query_embedding_error: {type(exc).__name__}")
                return []
        if len(qvec) != ready[1]:
            self._set_search_status("vector", "unavailable", "query_dimension_mismatch")
            return []
        if cache_key not in self._query_embedding_cache and self.query_cache_size > 0:
            self._query_embedding_cache[cache_key] = list(qvec)
            self._query_embedding_cache.move_to_end(cache_key)
            while len(self._query_embedding_cache) > self.query_cache_size:
                self._query_embedding_cache.popitem(last=False)
        with self._conn() as conn:
            where, params = self._allowed_sql(statuses, include_archive)
            project_clause = " AND f.project_id=?" if project_id else ""
            if project_id:
                params.append(project_id)
            path_clause, path_params = self._path_prefix_sql(path_prefixes)
            params.extend(path_params)
            scope_params = list(params)
            scope = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(c.updated_at), 0) "
                "FROM chunks c JOIN files f ON f.path=c.file_path "
                "WHERE c.vec IS NOT NULL" + where + project_clause + path_clause,
                scope_params,
            ).fetchone()
            path_scope = tuple(
                str(value or "").replace("\\", "/").strip().lstrip("/").rstrip("/") + "/"
                for value in (path_prefixes or ()) if str(value or "").strip()
            )
            row_count, max_updated = int(scope[0] or 0), float(scope[1] or 0.0)
            signature = (
                row_count, max_updated, ready[1], project_id,
                tuple(statuses) if statuses is not None else None,
                bool(include_archive),
                path_scope,
            )
            cached_matrix = self._vector_matrix_cache.get(signature) if _np is not None else None
            if cached_matrix is not None:
                self.last_matrix_source = "memory"
            if _np is not None and row_count:
                # 优先复用进程内归一化矩阵；miss 时先读索引内置侧车（跨进程），
                # 仍 miss 才解码向量 BLOB 并回写侧车。侧车有效性由 scope 签名
                # 决定，任何 upsert/delete（updated_at 变化）都会换新签名失效。
                if cached_matrix is None:
                    scope_hash = self._matrix_scope_hash(signature, ready[1])
                    cached_matrix = self._matrix_persist_load(conn, scope_hash)
                    self.last_matrix_source = "cached" if cached_matrix is not None else "built"
                if cached_matrix is None:
                    rows = conn.execute(
                        "SELECT c.chunk_id,c.source_ref,c.content,c.file_path,c.entry_id,c.chunk_index,c.vec "
                        "FROM chunks c JOIN files f ON f.path=c.file_path WHERE c.vec IS NOT NULL"
                        + where + project_clause + path_clause,
                        scope_params,
                    ).fetchall()
                    matrix = _np.frombuffer(
                        b"".join(bytes(row["vec"]) for row in rows), dtype=_np.float32,
                    )
                    if matrix.size != len(rows) * ready[1]:
                        self._set_search_status("vector", "unavailable", "stored_vector_dimension_mismatch")
                        return []
                    matrix = matrix.reshape((len(rows), ready[1]))
                    norms = _np.linalg.norm(matrix, axis=1)
                    norms[norms == 0] = 1.0
                    matrix = matrix / norms[:, None]
                    cached_matrix = (matrix, list(rows))
                    try:
                        self._matrix_persist_save(
                            conn, scope_hash, matrix, rows, row_count, max_updated, ready[1],
                        )
                    except (sqlite3.Error, TypeError, ValueError):
                        # 侧车持久化失败只损失跨进程预热，不阻断本次检索。
                        pass
                # Keep the normalized matrix hot for subsequent queries in
                # this process too.  Bound entries by the same scope policy
                # used by the SQLite sidecar so long-lived servers do not
                # retain every historical project/status combination.
                self._vector_matrix_cache[signature] = cached_matrix
                while len(self._vector_matrix_cache) > 8:
                    self._vector_matrix_cache.pop(next(iter(self._vector_matrix_cache)))
                matrix, rows = cached_matrix
                allowed_indexes = [index for index, row in enumerate(rows)
                                   if self._memory_row_is_readable(row)]
                if len(allowed_indexes) != len(rows):
                    matrix = matrix[allowed_indexes]
                    rows = [rows[index] for index in allowed_indexes]
                if not rows:
                    self._set_search_status("vector", "available")
                    return []
                query_array = _np.asarray(qvec, dtype=_np.float32)
                query_norm = float(_np.linalg.norm(query_array))
                if query_norm <= 0:
                    self._set_search_status("vector", "available", "zero_query_vector")
                    return []
                scores = matrix.dot(query_array / query_norm)
                ranked = sorted(
                    ((float(score), row) for score, row in zip(scores, rows) if score > 0),
                    key=lambda item: (-item[0], item[1]["chunk_id"]),
                )
                self._set_search_status("vector", "available")
                return [{"title": row["file_path"], "content": row["content"],
                         "ref": row["source_ref"], "source": "vector",
                         "score": round(float(score), 4),
                         "chunk_id": row["chunk_id"], "file_path": row["file_path"],
                         "entry_id": _row_value(row, "entry_id"), "chunk_index": row["chunk_index"]}
                        for score, row in ranked[: max(1, k)]]
            rows = conn.execute(
                "SELECT c.chunk_id,c.source_ref,c.content,c.file_path,c.entry_id,c.chunk_index,c.vec "
                "FROM chunks c JOIN files f ON f.path=c.file_path WHERE c.vec IS NOT NULL"
                + where + project_clause + path_clause,
                scope_params,
            ).fetchall()
        rows = self._filter_readable_memory_rows(rows)
        scored = []
        for row in rows:
            vec = _unpack(row["vec"])
            if len(vec) != len(qvec):
                return []
            score = _cosine(qvec, vec)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        self._set_search_status("vector", "available")
        return [{"title": row["file_path"], "content": row["content"],
                 "ref": row["source_ref"], "source": "vector",
                 "score": round(float(score), 4),
                 "chunk_id": row["chunk_id"], "file_path": row["file_path"],
                 "entry_id": _row_value(row, "entry_id"), "chunk_index": row["chunk_index"]} for score, row in scored[: max(1, k)]]

    @staticmethod
    def _context_key(ref: str) -> str:
        """Return the file/entry/heading identity without the chunk suffix."""
        value = str(ref or "")
        value = re.sub(r":ch[0-9a-f]{8}(?:-\d+)?$", "", value, flags=re.IGNORECASE)
        value = re.sub(r":c\d+$", "", value, flags=re.IGNORECASE)
        return value

    def expand_context(
        self,
        rows: Sequence[dict],
        *,
        neighbor_chunks: int = SMALL_TO_BIG_DEFAULT_NEIGHBORS,
        max_chars: int = SMALL_TO_BIG_DEFAULT_MAX_CHARS,
        project_id: str | None = None,
        statuses: Sequence[str] | None = None,
        include_archive: bool = False,
    ) -> tuple[list[dict], dict[str, object]]:
        """Expand final hits with bounded same-heading neighboring chunks.

        The current row/ref/rank is left intact.  Only ``content`` and the
        optional ``context_of`` list are added, so callers can run this in
        shadow without changing RRF or answer citations.
        """
        started = time.perf_counter()
        source = [dict(row) for row in rows]
        stats: dict[str, object] = {
            "status": "disabled" if neighbor_chunks <= 0 or max_chars <= 0 else "available",
            "requested": len(source), "expanded": 0, "neighbors": 0,
            "truncated": 0,
        }
        if not source or neighbor_chunks <= 0 or max_chars <= 0 or not self.db_path.exists():
            stats["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return source, stats
        try:
            with self._conn() as conn:
                self._assert_version(conn, allow_empty=False)
                where, base_params = self._allowed_sql(statuses, include_archive)
                project_clause = " AND f.project_id=?" if project_id else ""

                def scoped_rows(file_path: str, entry_id: int | None) -> list[sqlite3.Row]:
                    sql = (
                        "SELECT c.chunk_id,c.source_ref,c.content,c.file_path,c.chunk_index,c.entry_id "
                        "FROM chunks c JOIN files f ON f.path=c.file_path WHERE c.file_path=?"
                    ) + where + project_clause
                    params: list[object] = [file_path, *base_params]
                    if project_id:
                        params.append(project_id)
                    if entry_id is not None:
                        sql += " AND c.entry_id=?"
                        params.append(entry_id)
                    return conn.execute(sql + " ORDER BY c.chunk_index,c.chunk_id", params).fetchall()

                for target in source:
                    chunk_id = str(target.get("chunk_id") or "")
                    ref = str(target.get("ref") or target.get("source_ref") or "")
                    anchor: sqlite3.Row | None = None
                    if chunk_id:
                        anchor = conn.execute(
                            "SELECT c.chunk_id,c.source_ref,c.content,c.file_path,c.chunk_index,c.entry_id "
                            "FROM chunks c JOIN files f ON f.path=c.file_path WHERE c.chunk_id=?"
                            + where + project_clause,
                            [chunk_id, *base_params, *([project_id] if project_id else [])],
                        ).fetchone()
                    if anchor is None and ref:
                        anchor = conn.execute(
                            "SELECT c.chunk_id,c.source_ref,c.content,c.file_path,c.chunk_index,c.entry_id "
                            "FROM chunks c JOIN files f ON f.path=c.file_path WHERE c.source_ref=?"
                            + where + project_clause,
                            [ref, *base_params, *([project_id] if project_id else [])],
                        ).fetchone()
                    if anchor is None:
                        continue
                    all_same_entry = scoped_rows(str(anchor["file_path"]), anchor["entry_id"])
                    context_key = self._context_key(str(anchor["source_ref"]))
                    same_heading = [
                        row for row in all_same_entry
                        if self._context_key(str(row["source_ref"])) == context_key
                    ]
                    # Bucket/legacy rows may not have a stable entry id.  In
                    # that case the file-level query is still constrained to
                    # the same heading key before neighbors are selected.
                    if len(same_heading) <= 1 and anchor["entry_id"] is None:
                        same_file = scoped_rows(str(anchor["file_path"]), None)
                        same_heading = [
                            row for row in same_file
                            if self._context_key(str(row["source_ref"])) == context_key
                        ]
                    if len(same_heading) <= 1:
                        continue
                    current_index = next(
                        (index for index, row in enumerate(same_heading)
                         if str(row["chunk_id"]) == str(anchor["chunk_id"])),
                        None,
                    )
                    if current_index is None:
                        continue
                    lo = max(0, current_index - int(neighbor_chunks))
                    hi = min(len(same_heading), current_index + int(neighbor_chunks) + 1)
                    before_rows = same_heading[lo:current_index]
                    after_rows = same_heading[current_index + 1:hi]
                    neighbors = before_rows + after_rows
                    if not neighbors:
                        continue
                    current = str(target.get("content") or anchor["content"] or "")
                    # Previous chunks first, then following chunks, while the
                    # hit content remains the primary bounded payload.
                    ordered = [
                        (str(row["source_ref"]), str(row["content"] or ""))
                        for row in neighbors
                    ]
                    rendered = current[:max_chars]
                    was_truncated = len(current) > max_chars
                    remaining = max_chars - len(rendered)
                    used: list[str] = []
                    for neighbor_ref, content in ordered:
                        if remaining <= 0:
                            break
                        marker = f"\n\n[context_of:{neighbor_ref}]\n"
                        available = remaining - len(marker)
                        if available <= 0:
                            break
                        text = content[:available]
                        rendered += marker + text
                        remaining -= len(marker) + len(text)
                        used.append(neighbor_ref)
                        if len(text) < len(content):
                            was_truncated = True
                            break
                    if not used:
                        continue
                    target["content"] = rendered
                    target["context_of"] = used
                    stats["expanded"] = int(stats["expanded"]) + 1
                    stats["neighbors"] = int(stats["neighbors"]) + len(used)
                    if was_truncated:
                        stats["truncated"] = int(stats["truncated"]) + 1
        except (sqlite3.Error, IndexCompatibilityError, ValueError, TypeError):
            stats["status"] = "unavailable"
        stats["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return source, stats

    def lexical_confidence(
        self,
        query: str,
        rows: Sequence[dict],
    ) -> float:
        """Return a conservative lexical evidence ratio for abstention.

        This is not a relevance score.  It only asks whether the returned
        candidates cover enough query tokens to justify injecting any context.
        Exact identifiers require their full token; natural-language queries
        need at least two non-trivial terms (or 40% of their token set).
        """
        query_tokens = _query_tokens(query)
        if not query_tokens or not rows:
            return 0.0
        joined = " ".join(
            " ".join((str(row.get("title", "")), str(row.get("ref", "")), str(row.get("content", ""))))
            for row in rows[:3]
        )
        covered = query_tokens & _tokens(joined)
        identifiers = {
            token for token in query_tokens
            if len(token) >= 3 and _ASCII_RE.fullmatch(token)
        }
        if identifiers and not identifiers.issubset(covered):
            return 0.0
        return len(covered) / len(query_tokens)


RAGIndexStore = RagIndexStore

__all__ = [
    "RagIndexStore", "RAGIndexStore", "IndexCompatibilityError",
    "INDEX_VERSION", "PARSER_VERSION",
]
