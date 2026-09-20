"""可逆区段档案（L11/OPT-111，学 billion-context decompress）。

压缩（WorkingMemory.condense）把 [anchor, cut) 折叠成摘要后，原文此前只活在
当轮内存——服务端 persist_delta 只落压缩后列表，折叠原文随进程即失。本模块把
折叠区段原文落盘 `{session_id}.ranges.jsonl`（与会话 jsonl 同目录），并同步
嵌入向量索引（VectorIndex ranges 表），让"窗口外内容"可经 rag_retrieve 的
session 路召回，而非不可逆丢失。

- 事实源是 jsonl：索引失败/滞后不影响完整性（关键词兜底路直接扫 jsonl）。
- 治理对齐 govern_recall 精神：落盘单条 content 超上限做 head+tail 截断；
  召回侧沿用 rag_retrieve 既有 Tier-1/Tier-2 注入治理，不在此重复。
- 会话隔离：写入由 RangeRecorder 显式绑 session_id；读侧（召回）经 ContextVar
  绑当前会话——serve 并发请求互不串话，未绑定时会话路返回空（降级）。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Sequence

from agentlab.core.message import Message
from agentlab.rag.recall import split_keywords

_log = logging.getLogger(__name__)

# 会话 id 白名单（同 project.py 防穿越思路）：字母数字开头，仅限 [A-Za-z0-9_.-]
_SID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# 单条召回正文的展示预算（head+tail），与 recall._PER_ITEM_CHARS 同量级
_EXCERPT_CHARS = 1200
_TRUNC_TAG = "…[truncated {} chars]…"


def render_range(record: dict) -> str:
    """把一条区段记录渲染成可检索文本（"role: content" 行，与摘要侧渲染一致）。"""
    lines = []
    for m in record.get("messages", []):
        c = str(m.get("content") or "")
        if c:
            lines.append(f"{m.get('role', '?')}: {c}")
    return "\n".join(lines)


def _excerpt(text: str, cap: int = _EXCERPT_CHARS) -> str:
    if cap <= 0 or len(text) <= cap:
        return text
    tail = cap // 5
    return text[: cap - tail] + _TRUNC_TAG.format(len(text) - cap) + text[-tail:]


class RangeArchive:
    """折叠区段 jsonl 档案（append-only，每会话一文件，半行容忍——同 session_store）。"""

    def __init__(self, root: str | Path, max_msg_chars: int = 20000):
        self.root = Path(root)
        self.max_msg_chars = max_msg_chars
        self._seq: dict[str, int] = {}  # 进程内 seq 缓存；首触从文件尾恢复

    def path(self, session_id: str) -> Path:
        if not _SID_RE.match(session_id or ""):
            raise ValueError(f"非法 session_id: {session_id!r}")
        return self.root / f"{session_id}.ranges.jsonl"

    def _records(self, session_id: str) -> list[dict]:
        p = self.path(session_id)
        if not p.exists():
            return []
        out: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                break  # 追加中途崩溃残留的半行：丢弃（与 session_store 同款）
        return out

    def _sanitize(self, m: Message) -> dict:
        d = m.model_dump()
        c = d.get("content") or ""
        cap = self.max_msg_chars
        if cap and len(c) > cap:  # 单条超限：head+tail 保留（govern_recall 同级精神）
            tail = max(1, cap // 5)
            d["content"] = (c[: cap - tail]
                            + _TRUNC_TAG.format(len(c) - cap) + c[-tail:])
        return d

    def append(self, session_id: str, messages: Sequence[Message]) -> dict:
        """落盘一次折叠事件的原文；返回记录（seq 从 1 自增）。空区段拒绝。"""
        msgs = [self._sanitize(m) for m in (messages or [])]
        if not msgs:
            raise ValueError("空区段不归档")
        p = self.path(session_id)
        seq = self._seq.get(session_id)
        if seq is None:  # 跨实例续编：以文件中最后一条为准（坏行不计，与 _records 一致）
            recs = self._records(session_id)
            seq = int(recs[-1].get("seq", 0)) if recs else 0
        rec = {"session_id": session_id, "seq": seq + 1, "ts": time.time(),
               "messages": msgs}
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("ab") as f:
            f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        self._seq[session_id] = seq + 1
        return rec

    def read(self, session_id: str, seq: int) -> list[Message]:
        """取回某区段原文（解压回窗 API；L11 读侧闭环预留）。无则空列表。"""
        for rec in self._records(session_id):
            if rec.get("seq") == seq:
                return [Message.model_validate(d) for d in rec.get("messages", [])]
        return []

    def remove(self, session_id: str) -> bool:
        """删除整会话区段档案（#8/OPT-127：ark 删除按钮接线）；返回是否存在过。"""
        p = self.path(session_id)
        if not p.exists():
            return False
        p.unlink()
        self._seq.pop(session_id, None)
        return True

    def search(self, session_id: str, query: str, limit: int = 5) -> list[dict]:
        """关键词兜底路：直接扫 jsonl（无 embedder/索引滞后时的降级召回）。"""
        terms = [t.lower() for t in split_keywords(query)]
        if not terms:
            return []
        scored: list[tuple[int, dict]] = []
        for rec in self._records(session_id):
            text = render_range(rec)
            low = text.lower()
            score = sum(1 for t in terms if t in low)
            if score:
                scored.append((score, rec))
        scored.sort(key=lambda x: (-x[0], x[1].get("seq", 0)))
        return [
            {"title": f"会话区段 r{rec.get('seq', '?')}",
             "content": _excerpt(render_range(rec)),
             "ref": f"session/{session_id}#r{rec.get('seq', '?')}",
             "source": "session", "score": float(s),
             "session_id": session_id, "seq": rec.get("seq")}
            for s, rec in scored[:limit]
        ]


class RangeRecorder:
    """写入口（绑定单 session）：loop 折叠区段 → jsonl 落盘 + 向量索引同步。

    jsonl 是事实源，append 失败上抛（loop 留痕不中断）；索引嵌入失败仅记日志
    ——该区段仍可经关键词兜底路召回，不丢内容。
    """

    def __init__(self, archive: RangeArchive, session_id: str, index=None,
                 chunk_chars: int = 2000, max_chunks: int = 200):
        self._archive = archive
        self._sid = session_id
        self._index = index
        self._chunk_chars = chunk_chars
        self._max_chunks = max_chunks

    def archive(self, messages: Sequence[Message]) -> dict:
        rec = self._archive.append(self._sid, messages)
        if self._index is not None:
            try:
                self._index.upsert_range(self._sid, rec["seq"], render_range(rec),
                                         chunk_chars=self._chunk_chars,
                                         max_chunks=self._max_chunks)
            except Exception:
                _log.exception("区段索引嵌入失败（session=%s seq=%s）——jsonl 已落，"
                               "关键词兜底路可用", self._sid, rec["seq"])
        return rec


# 读侧当前会话（serve 每请求 bind/reset；asyncio 同任务链路内传递，并发不串话）
_current_sid: ContextVar[str | None] = ContextVar("agentlab_ranges_sid", default=None)


class RangeGateway:
    """进程级网关（serve 构建一次）：写侧 recorder(sid) 显式绑定，读侧 recaller()
    闭包在工具调用时读 ContextVar 决定当前会话。"""

    def __init__(self, archive: RangeArchive, index=None, k: int = 5):
        self._archive = archive
        self._index = index
        self._k = max(1, k)

    def bind(self, session_id: str):
        return _current_sid.set(session_id)

    @staticmethod
    def reset(token) -> None:
        _current_sid.reset(token)

    def recorder(self, session_id: str, **kw) -> RangeRecorder:
        return RangeRecorder(self._archive, session_id, index=self._index, **kw)

    def remove_session(self, session_id: str) -> dict:
        """删除会话的区段档案与向量索引行（#8/OPT-127）；jsonl 事实源删除为准。"""
        out = {"ranges": False, "index": False}
        out["ranges"] = self._archive.remove(session_id)
        if self._index is not None:
            try:
                self._index.remove_session(session_id)
                out["index"] = True
            except Exception:  # noqa: BLE001 —— 索引清理失败不阻断，jsonl 已删
                _log.exception("区段索引清理失败（session=%s）", session_id)
        return out

    def recaller(self):
        """session 召回路（RAGRecall.extra 形状）：向量优先，空则 jsonl 关键词兜底。"""

        def session_recall(q: str, scope=None) -> list[dict]:
            # Prefer the explicit S0 request scope; ContextVar remains the
            # compatibility fallback for callers that bind a session directly.
            sid = getattr(scope, "session_id", "") or _current_sid.get()
            if not sid:
                return []
            hits: list[dict] = []
            if self._index is not None:
                try:
                    hits = self._index.search_ranges(q, k=self._k, session_id=sid)
                except Exception:
                    _log.exception("区段向量召回失败（session=%s）——走关键词兜底", sid)
                    hits = []
            if not hits:
                hits = self._archive.search(sid, q, limit=self._k)
            return hits

        return session_recall
