"""本地向量索引（P0-1/OPT-105）：SQLite BLOB 存储 + 余弦检索，语义召回一路。

- **零硬新增依赖**：sqlite-vec 可导入时用其 `vec_distance_cosine` 标量函数（C 计算），
  否则纯 Python 余弦（个人库规模足量快）。同一套 BLOB schema，装不装都跑。
- **chunking**：空行分段 → 相邻段合并到 ≤chunk_chars；超长单段按句号硬切，
  避免"整文件一向量"稀释语义（对齐 2.0 §3.4 L 系列 chunking 边界判据精神）。
- **断点约定**：索引文件不存在 / 未配 embedder / 维度不一致 → search 返回空、
  sync 由调用方提示，不抛错中断关键词路（对齐"任一路降级"惯例）。
"""
from __future__ import annotations

import math
import sqlite3
import struct
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

try:  # 可选加速：C 实现的余弦距离（纯扫描，无 ANN 表，schema 与降级路径一致）
    import sqlite_vec as _sqlite_vec  # type: ignore

    def _load_functions(conn: sqlite3.Connection) -> None:
        conn.enable_load_extension(True)
        _sqlite_vec.load(conn)
        conn.enable_load_extension(False)
except Exception:  # pragma: no cover - 未安装时走纯 Python
    def _load_functions(conn: sqlite3.Connection) -> None:
        return None

    _sqlite_vec = None

_EXCLUDE_DIRS = {".obsidian", ".agent-brain", ".trash", ".tmp", "node_modules"}


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """空行分段 → 相邻段贪心合并到 ≤max_chars；超长单段按句号硬切。"""
    paras = [p.strip() for p in (text or "").replace("\r\n", "\n").split("\n\n")]
    paras = [p for p in paras if p]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        while len(p) > max_chars:  # 超长段按句切
            cut = p.rfind("。", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(p[:cut])
            p = p[cut:].lstrip()
        if not p:
            continue
        if buf and len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n{p}"
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob[:n * 4]))


class VectorIndex:
    """Vault 分块向量索引（单 SQLite 文件）。embedder 由构造注入（测试可换 Fake）。

    vault_root + auto_sync_limit 齐备时，检索前可调 `auto_sync()` 做有界增量
    自愈（文档常改 → 索引不陈旧）：每次最多嵌 limit 个最新变更文件，其余顺延。
    """

    def __init__(self, db_path: Path, embedder: Callable, chunk_chars: int = 600,
                 vault_root: Path | None = None, auto_sync_limit: int = 8):
        self.db_path = Path(db_path)
        self.embedder = embedder
        self.chunk_chars = max(120, chunk_chars)
        self._vault_root = Path(vault_root) if vault_root else None
        self._auto_sync_limit = max(1, auto_sync_limit)
        self._sync_lock = threading.Lock()  # serve 并发查询下防同批文件重复嵌入
        self._last_scan = 0.0  # auto_sync 扫描节流：高频查询不反复 rglob 大库（秒）

    # —— 基础 ——
    @contextmanager
    def _conn(self):
        """连接上下文：建表 → yield → **必须 close**（sqlite3 的 with 只管事务；
        不 close 会在 Windows 上锁住文件，测试临时目录都清不掉）。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            _load_functions(conn)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS files("
                "path TEXT PRIMARY KEY, mtime REAL NOT NULL, chunks INT NOT NULL DEFAULT 0)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS chunks("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL, idx INT NOT NULL,"
                "title TEXT NOT NULL, content TEXT NOT NULL, vec BLOB NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
            # L11/OPT-111 会话区段档案：独立表（不与 vault 同表，sync_vault 永不清除）
            conn.execute(
                "CREATE TABLE IF NOT EXISTS ranges("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
                "seq INT NOT NULL, idx INT NOT NULL, title TEXT NOT NULL,"
                "content TEXT NOT NULL, vec BLOB NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ranges_sid ON ranges(session_id)")
            yield conn
        finally:
            conn.commit()  # sqlite3 的 with 语义不会自动生效：显式提交后关闭
            conn.close()

    @property
    def chunk_count(self) -> int:
        if not self.db_path.exists():
            return 0
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # —— 写入 / 删除 ——
    def upsert(self, rel_path: str, mtime: float, text: str) -> int:
        """单文件重建分块：删除旧块 → chunk → embed → 写入。返回新块数。"""
        rel = rel_path.replace("\\", "/")
        chunks = chunk_text(text, self.chunk_chars)
        title = Path(rel).stem
        vecs = self.embedder.embed(chunks) if chunks else []
        if len(vecs) != len(chunks):
            raise RuntimeError(f"embed 数量不匹配: {rel}")
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE path=?", (rel,))
            conn.execute("INSERT OR REPLACE INTO files(path, mtime, chunks) VALUES(?,?,?)",
                         (rel, mtime, len(chunks)))
            conn.executemany(
                "INSERT INTO chunks(path, idx, title, content, vec) VALUES(?,?,?,?,?)",
                [(rel, i, title, c, _pack(v)) for i, (c, v) in enumerate(zip(chunks, vecs))])
        return len(chunks)

    def remove(self, rel_path: str) -> None:
        rel = rel_path.replace("\\", "/")
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE path=?", (rel,))
            conn.execute("DELETE FROM files WHERE path=?", (rel,))

    # —— 会话区段档案（L11/OPT-111）：窗口外会话内容的语义召回路 ——
    def upsert_range(self, session_id: str, seq: int, text: str,
                     chunk_chars: int | None = None, max_chunks: int = 200) -> int:
        """单区段重建分块：删旧 (session_id, seq) → chunk → embed → 写入。

        区段文本是对话体，用比 vault 更粗的分块（默认 2000 字符）；超 max_chunks
        时只索引前 max_chunks 块并追加标记块（jsonl 仍全量，关键词路可兜底）。
        返回新块数。
        """
        chunks = chunk_text(text, chunk_chars or self.chunk_chars)
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks] + [f"…[区段超长，仅索引前 {max_chunks} 块]"]
        vecs = self.embedder.embed(chunks) if chunks else []
        if len(vecs) != len(chunks):
            raise RuntimeError(f"embed 数量不匹配: session={session_id} r{seq}")
        with self._conn() as conn:
            conn.execute("DELETE FROM ranges WHERE session_id=? AND seq=?",
                         (session_id, seq))
            conn.executemany(
                "INSERT INTO ranges(session_id, seq, idx, title, content, vec)"
                " VALUES(?,?,?,?,?,?)",
                [(session_id, seq, i, f"会话区段 r{seq}", c, _pack(v))
                 for i, (c, v) in enumerate(zip(chunks, vecs))])
        return len(chunks)

    def remove_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM ranges WHERE session_id=?", (session_id,))

    def search_ranges(self, query: str, k: int = 6,
                      session_id: str | None = None) -> list[dict]:
        """区段余弦 top-k（可 scoped 到单会话）。空表/无索引/维度不一致 → 空列表。

        降级语义与 search() 一致：sqlite-vec 距离下推优先，未装/报错退纯 Python。
        """
        if not self.db_path.exists():
            return []
        where, params = "", []
        if session_id:
            where, params = " WHERE session_id=?", [session_id]
        with self._conn() as conn:
            n = conn.execute(f"SELECT COUNT(*) FROM ranges{where}", params).fetchone()[0]
        if n == 0:
            return []
        qvec = self.embedder.embed([query])[0]
        if _sqlite_vec is not None:
            try:
                with self._conn() as conn:
                    rows = conn.execute(
                        "SELECT session_id, seq, idx, content,"
                        " vec_distance_cosine(?, vec) AS d FROM ranges"
                        f"{where} ORDER BY d LIMIT ?",
                        [_pack(qvec)] + params + [k]).fetchall()
                return self._range_hits(
                    sorted(((1.0 - float(d), s, q_, i, c)
                            for s, q_, i, c, d in rows), reverse=True), k)
            except sqlite3.Error:
                pass  # 维度不符等 → 走 Python 路径统一判定
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT session_id, seq, idx, content, vec FROM ranges{where}",
                params).fetchall()
        scored: list[tuple] = []
        for sid, seq, idx_, content, blob in rows:
            vec = _unpack(blob)
            if len(vec) != len(qvec):
                return []  # 维度不一致 = 换了 embedding 模型，需重建
            scored.append((_cosine(qvec, vec), sid, seq, idx_, content))
        scored.sort(reverse=True)
        return self._range_hits(scored[:k], k)

    @staticmethod
    def _range_hits(scored: list[tuple], k: int) -> list[dict]:
        out: list[dict] = []
        for s, sid, seq, idx_, content in scored:
            if s <= 0:
                continue
            out.append({"title": f"会话区段 r{seq}",
                        "content": content,
                        "ref": f"session/{sid}#r{seq}c{idx_}",
                        "source": "session", "score": round(float(s), 4),
                        "session_id": sid, "seq": seq})
        return out[:k]

    # —— 增量同步 ——
    def sync_vault(self, vault_root: Path | None = None, max_files: int | None = None) -> dict:
        """扫描 vault：mtime 变更的文件重嵌入、消失的文件清除。返回统计。

        max_files（有界自愈）：只嵌变更集中最新 N 个（按 mtime 降序），其余顺延——
        未处理的文件 mtime 保持旧值，下次同步仍视为变更，最终收敛。
        """
        vault_root = Path(vault_root or self._vault_root)
        entries: dict[str, float] = {}
        for p in vault_root.rglob("*.md"):
            if any(part in _EXCLUDE_DIRS for part in p.parts):
                continue
            try:
                entries[p.relative_to(vault_root).as_posix()] = p.stat().st_mtime
            except OSError:
                continue
        with self._conn() as conn:
            old = {r[0]: r[1] for r in conn.execute("SELECT path, mtime FROM files")}
        changed = [p for p, mt in entries.items() if old.get(p) != mt]
        if max_files is not None and len(changed) > max_files:
            changed.sort(key=lambda p: entries[p], reverse=True)  # 最新优先
            deferred = changed[max_files:]
            changed = changed[:max_files]
        else:
            deferred = []
        removed = [p for p in old if p not in entries]
        n_chunks = 0
        for rel in removed:
            self.remove(rel)
        for rel in changed:
            try:
                text = (vault_root / rel).read_text(encoding="utf-8", errors="ignore")
                n_chunks += self.upsert(rel, entries[rel], text)
            except (OSError, RuntimeError):
                continue  # 单文件失败不阻断整体同步（下轮 mtime 未变会跳过）
        return {"total": len(entries), "updated": len(changed), "removed": len(removed),
                "deferred": len(deferred),
                "chunks": self.chunk_count, "new_chunks": n_chunks}

    def auto_sync(self) -> dict:
        """检索前的顺手自愈（有界）：毫秒级 mtime 扫描，通常零变更零开销；
        有变更时每次最多嵌 auto_sync_limit 个最新文件。失败静默（不碍检索）。
        2 秒扫描节流：高频查询不反复 rglob 大库（规模化为先）。"""
        if not self._vault_root:
            return {}
        now = time.monotonic()
        if now - self._last_scan < 2.0:
            return {}
        self._last_scan = now
        with self._sync_lock:
            try:
                return self.sync_vault(max_files=self._auto_sync_limit)
            except Exception:
                return {}

    # —— 检索 ——
    def search(self, query: str, k: int = 6) -> list[dict]:
        """embed 查询 → 余弦 top-k。索引空/无 embedder/维度不一致 → 空列表。

        规模化主路径（P0-1 三期）：sqlite-vec 可用时距离下推 SQL——C 计算且
        LIMIT 在库侧截断，不全量回传 Python；未装/报错退纯 Python 全扫
        （个人库规模足够）。维度不一致两路同样空降级。
        """
        if not self.db_path.exists() or self.chunk_count == 0:
            return []
        qvec = self.embedder.embed([query])[0]
        if _sqlite_vec is not None:
            try:
                with self._conn() as conn:
                    rows = conn.execute(
                        "SELECT path, idx, title, content, vec_distance_cosine(?, vec) AS d"
                        " FROM chunks ORDER BY d LIMIT ?", (_pack(qvec), k)).fetchall()
                return self._to_hits(
                    sorted(((1.0 - float(d), p, i, t, c)
                            for p, i, t, c, d in rows), reverse=True), k)
            except sqlite3.Error:
                pass  # 维度不符等 → 走 Python 路径统一判定
        with self._conn() as conn:
            rows = conn.execute("SELECT path, idx, title, content, vec FROM chunks").fetchall()
        scored: list[tuple] = []
        for path, idx_, title, content, blob in rows:
            vec = _unpack(blob)
            if len(vec) != len(qvec):
                return []  # 维度不一致 = 换了 embedding 模型，需重建
            scored.append((_cosine(qvec, vec), path, idx_, title, content))
        scored.sort(reverse=True)
        return self._to_hits(scored[:k], k)

    @staticmethod
    def _to_hits(scored: list[tuple], k: int) -> list[dict]:
        out: list[dict] = []
        for s, path, idx_, title, content in scored:
            if s <= 0:
                continue
            out.append({"title": f"{title}（{path}#c{idx_}）", "content": content,
                        "ref": f"{path}#c{idx_}", "source": "vector",
                        "score": round(float(s), 4)})
        return out[:k]


def make_vector_recaller(index: VectorIndex, k: int = 6) -> Callable:
    """包装成 Recaller（query -> list[dict]），供 RAGRecall.extra 注入。

    检索前先 `auto_sync()` 有界自愈：文档更新后索引自动跟上（OPT-105 二期），
    不再依赖人工触发 rag_reindex；同步失败静默降级为纯检索。
    """

    def vector_recall(q: str) -> list[dict]:
        index.auto_sync()
        return index.search(q, k=k)

    return vector_recall
