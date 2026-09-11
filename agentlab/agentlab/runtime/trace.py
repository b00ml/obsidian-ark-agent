"""trace 落盘（docs/03 §6、05 §2）。对 Authorization 头脱敏。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from agentlab.core.message import Message

_HEADER_RE = re.compile(r"(?i)(authorization|api-key|x-api-key)[\"':=]+\s*([^\s\",}]+)")
_TOKEN_RE = re.compile(r"(?i)(sk-[a-z0-9-_]+)")
_BEARER_RE = re.compile(r"(?i)(bearer|key)\s+([a-z0-9._-]{8,})")


def _redact_str(s: str) -> str:
    s = _HEADER_RE.sub(r"\1: [REDACTED]", s)
    s = _TOKEN_RE.sub("[REDACTED]", s)
    s = _BEARER_RE.sub(r"\1 [REDACTED]", s)
    return s


class Tracer:
    def __init__(self, trace_dir: str | Path):
        self.dir = Path(trace_dir)
        self.trace_id = ""

    def new_session(self) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        trace_id = uuid.uuid4().hex[:12]
        self.trace_id = trace_id
        self._fp = self.dir / f"{trace_id}.jsonl"
        self._fp.touch(exist_ok=True)
        return trace_id

    def record(self, obj: dict) -> None:
        if not hasattr(self, "_fp"):
            return
        line = json.dumps(self._redact(obj), ensure_ascii=False)
        with self._fp.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def record_step(self, step: int, resp) -> None:
        self.record({
            "t": time.time(), "step": step, "type": "llm",
            "content": resp.content,
            "tool_calls": [c.function.name for c in (resp.tool_calls or [])],
            "stop_reason": resp.stop_reason,
            "usage": resp.usage.model_dump() if resp.usage else None,
        })

    def record_tool(self, name: str, phase: str, **kw) -> None:
        self.record({"t": time.time(), "type": "tool", "name": name, "phase": phase, **kw})

    def record_run(self, **fields) -> None:
        """run 级摘要（#6/OPT-126：serve 侧 run 概览数据源；此前仅 CLI 接 trace）。"""
        self.record({"t": time.time(), "type": "run", **fields})

    @staticmethod
    def _redact(obj):
        # 递归脱敏字符串里的敏感值（sk- 密钥 / Bearer / 响应头）
        if isinstance(obj, dict):
            return {k: Tracer._redact(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Tracer._redact(v) for v in obj]
        if isinstance(obj, str):
            return _redact_str(obj)
        return obj


def load_run_summaries(trace_dir: str | Path, limit: int = 20) -> list[dict]:
    """最近 N 次 run 概览（#6/OPT-126，GET /v1/runs 数据源）。

    扫描 {trace_id}.jsonl：有 run 摘要行的才算一次 run（CLI 旧 trace 无 run 行，
    暂不纳入）；llm 步数/工具次数/兜底 token 由聚合补齐。按文件 mtime 新→旧，
    损坏行跳过；只回概览字段，不带正文（输入在写入侧已截断+脱敏）。
    """
    from datetime import datetime

    root = Path(trace_dir)
    if not root.is_dir():
        return []
    try:
        files = sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime,
                       reverse=True)[: max(1, limit) * 3]
    except OSError:
        return []
    out: list[dict] = []
    for p in files:
        try:
            run_rec: dict | None = None
            steps = tools = 0
            stop = ""
            last_usage: dict | None = None
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # 崩溃半行容忍
                    t = rec.get("type")
                    if t == "run":
                        run_rec = rec
                    elif t == "llm":
                        steps += 1
                        stop = rec.get("stop_reason") or stop
                        if isinstance(rec.get("usage"), dict):
                            last_usage = rec["usage"]
                    elif t == "tool" and rec.get("phase") == "end":
                        tools += 1
            if run_rec is None:
                continue
            tokens = run_rec.get("tokens")
            if tokens is None and isinstance(last_usage, dict):
                tokens = (last_usage.get("input_tokens") or 0) \
                    + (last_usage.get("output_tokens") or 0)
            out.append({
                "trace_id": p.stem,
                "run_id": run_rec.get("run_id", ""),
                "session_id": run_rec.get("session_id", ""),
                "project_id": run_rec.get("project_id", ""),
                "time": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                "input": run_rec.get("input", ""),
                "stop_reason": run_rec.get("stop_reason") or stop,
                "tokens": tokens,
                "steps": steps,
                "tools": tools,
                "error": run_rec.get("error", ""),
                "cancel_reason": run_rec.get("cancel_reason", ""),
            })
            if len(out) >= limit:
                break
        except OSError:
            continue
    return out


def load_run_detail(trace_dir: str | Path, trace_id: str) -> dict | None:
    """读取单次 run 的诊断摘要；只保留事件元数据，不返回正文/工具结果。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", trace_id):
        return None
    path = Path(trace_dir) / f"{trace_id}.jsonl"
    if not path.is_file():
        return None
    run: dict = {}
    events: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") == "run":
                    run = rec
                elif rec.get("type") in {"tool", "llm"}:
                    event = {"type": rec.get("type"), "time": rec.get("t")}
                    if rec.get("type") == "tool":
                        event.update({"name": rec.get("name", ""), "phase": rec.get("phase", ""),
                                      "result_chars": len(str(rec.get("result", "")))})
                    else:
                        event.update({"step": rec.get("step"), "stop_reason": rec.get("stop_reason", ""),
                                      "tool_calls": rec.get("tool_calls", []), "usage": rec.get("usage")})
                    events.append(event)
    except OSError:
        return None
    return {"trace_id": trace_id, "run": run, "events": events[-200:]} if run else None


def messages_to_traces(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content, "usage": m.usage.model_dump() if m.usage else None}
            for m in messages]
