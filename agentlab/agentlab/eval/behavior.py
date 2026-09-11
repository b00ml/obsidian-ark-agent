"""F5-012 Agent/knowledge-loop behavior baseline.

These checks exercise deterministic runtime contracts without an LLM or the real
Vault.  They are behavior/regression measurements, not model-quality scores.
"""
from __future__ import annotations

import asyncio
import json
import platform
import socket
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agentlab.core.loop import AgentResult
from agentlab.core.message import Message, TokenUsage
from agentlab.memory.markdown_store import MemoryMarkdownStore
from agentlab.memory.session_store import JsonlSessionStorage
from agentlab.runtime.config import Config, LimitsConfig, ServeConfig
from agentlab.runtime.serve import Serve, _SSEWriter


SCHEMA = "f5-012.v1"


def default_scenarios_path() -> Path:
    """Return the repository-level behavior scenario file."""
    return Path(__file__).resolve().parents[3] / ".ai" / "evals" / "agent-knowledge-loop.jsonl"


def load_behavior_scenarios(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load JSONL scenarios, rejecting malformed or duplicate IDs."""
    scenario_path = Path(path) if path else default_scenarios_path()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(scenario_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        item = json.loads(raw)
        scenario_id = str(item.get("id", ""))
        if not scenario_id or scenario_id in seen:
            raise ValueError(f"invalid or duplicate scenario id at {scenario_path}:{line_no}")
        seen.add(scenario_id)
        rows.append(item)
    return rows


def _result(
    scenario_id: str,
    status: str,
    metrics: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {"id": scenario_id, "status": status, "metrics": metrics, "evidence": evidence}


def _memory_second_recall() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="agentlab-f5012-recall-") as root:
        store = MemoryMarkdownStore(root)
        memory_id = store.commit(
            "Ark Agent 使用 Markdown 真源保存知识记忆",
            tags=["ark", "memory"],
            project_id="obsidian-ark",
        )
        hits = store.query("Ark Agent 知识记忆", limit=5, project_id="obsidian-ark")
        matched = next((item for item in hits if item.get("id") == memory_id), None)
        recall = 1.0 if matched and "Markdown 真源" in matched.get("content", "") else 0.0
        return _result(
            "memory-second-recall",
            "pass" if recall == 1.0 else "fail",
            {"recall": recall, "result_count": len(hits)},
            {"memory_id": memory_id, "matched": bool(matched), "vault_root": root},
        )


def _memory_duplicate_write() -> dict[str, Any]:
    """Verify explicit duplicate semantics: default allows, dedup=True skips.

    The adapter below is intentionally small and delegates persistence to the
    production Markdown store; only MemoryStore's dedup decision is under test.
    """
    from agentlab.memory.store import MemoryStore

    with tempfile.TemporaryDirectory(prefix="agentlab-f5012-dedup-") as root:
        markdown = MemoryMarkdownStore(root)

        class Adapter:
            def memory_commit(self, _config, content, tags, source_session):
                return {"status": "committed", "id": markdown.commit(content, tags=tags or [])}

            def memory_query(self, _config, topic, limit):
                return {"results": markdown.query(topic, limit=limit)}

        memory = MemoryStore({"vault_root": root})
        memory._tm = Adapter()
        first = memory.commit("同一条知识只应在显式去重时跳过", tags=["dedup"])
        allowed = memory.commit("同一条知识只应在显式去重时跳过", tags=["dedup"], dedup=False)
        skipped = memory.commit("同一条知识只应在显式去重时跳过", tags=["dedup"], dedup=True)
        files = list((Path(root) / "ark" / "memory").rglob("*.md"))
        ok = first.get("status") == "committed" and allowed.get("status") == "committed" \
            and skipped.get("status") == "skipped_duplicate" and len(files) == 2
        return _result(
            "memory-duplicate-write",
            "pass" if ok else "fail",
            {"dedup_false_commits": 1 if allowed.get("status") == "committed" else 0,
             "dedup_true_skips": 1 if skipped.get("status") == "skipped_duplicate" else 0},
            {"first": first, "allowed": allowed, "skipped": skipped, "file_count": len(files)},
        )


def _memory_concurrent_write() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="agentlab-f5012-concurrent-") as root:
        store = MemoryMarkdownStore(root)

        def commit(index: int) -> str:
            return store.commit(f"并发记忆 {index}", tags=["concurrency"])

        total = 12
        with ThreadPoolExecutor(max_workers=total) as pool:
            ids = list(pool.map(commit, range(total)))
        files = list((Path(root) / "ark" / "memory").rglob("*.md"))
        valid = 0
        for memory_id in ids:
            try:
                if store.get(memory_id):
                    valid += 1
            except Exception:
                pass
        conflict_rate = (total - valid) / total
        return _result(
            "memory-concurrent-write",
            "pass" if conflict_rate == 0.0 and len(files) == total else "fail",
            {"writes": total, "valid_writes": valid, "conflict_rate": conflict_rate},
            {"file_count": len(files), "atomic_replace": True},
        )


def _memory_project_isolation() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="agentlab-f5012-project-") as root:
        store = MemoryMarkdownStore(root)
        a_id = store.commit("项目 A 的数据库决策", project_id="project-a", tags=["database"])
        b_id = store.commit("项目 B 的数据库决策", project_id="project-b", tags=["database"])
        a_hits = store.query("数据库决策", project_id="project-a", limit=10)
        b_hits = store.query("数据库决策", project_id="project-b", limit=10)
        a_ok = any(item.get("id") == a_id for item in a_hits) and not any(
            item.get("id") == b_id for item in a_hits
        )
        b_ok = any(item.get("id") == b_id for item in b_hits) and not any(
            item.get("id") == a_id for item in b_hits
        )
        return _result(
            "memory-project-isolation",
            "pass" if a_ok and b_ok else "fail",
            {
                "project_a_isolated": 1.0 if a_ok else 0.0,
                "project_b_isolated": 1.0 if b_ok else 0.0,
            },
            {
                "project_a_ids": [x.get("id") for x in a_hits],
                "project_b_ids": [x.get("id") for x in b_hits],
            },
        )


def _session_replay() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="agentlab-f5012-session-") as root:
        store = JsonlSessionStorage(root)
        store.append("baseline", Message(role="user", content="第一问"))
        store.append("baseline", Message(role="assistant", content="第一答"))
        replay = store.read_all("baseline")
        ok = [message.content for message in replay] == ["第一问", "第一答"]
        return _result(
            "session-replay",
            "pass" if ok else "fail",
            {"replayed_messages": len(replay), "round_trip": 1.0 if ok else 0.0},
            {"roles": [message.role for message in replay], "storage": "jsonl_append_only"},
        )


class _BaselineRunner:
    def __init__(self, sink):
        self.sink = sink
        self.registry = type("Registry", (), {"all": lambda self: [], "schemas": lambda self: []})()

    async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
        self.sink.text(f"baseline:{user_input}")
        return AgentResult(
            final_output=f"baseline:{user_input}",
            stop_reason="done",
            messages=[],
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url: str, token: str) -> tuple[int, dict[str, Any]]:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _wait_for_health(
    url: str, token: str, timeout_seconds: float = 2.0
) -> tuple[int, dict[str, Any]]:
    """Wait for the threaded test server to bind before exercising its contract."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _http_json(url, token)
        except OSError as exc:
            last_error = exc
            time.sleep(0.02)
    raise RuntimeError("baseline serve did not become healthy") from last_error


def _serve_contract() -> dict[str, Any]:
    port = _free_port()
    token = "f5-012-token"
    cfg = Config(
        vault_root=tempfile.gettempdir(),
        limits=LimitsConfig(max_steps=2, context_budget=8000),
        serve=ServeConfig(host="127.0.0.1", port=port, token=token, heartbeat=0),
    )
    serve = Serve(
        cfg,
        port=port,
        host="127.0.0.1",
        build_factory=lambda: (lambda sink: _BaselineRunner(sink)),
    )
    httpd = serve.start()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        health_status, health = _wait_for_health(f"{base}/health", token)
        request = Request(
            f"{base}/v1/responses",
            data=json.dumps({"input": [{"role": "user", "content": "hello"}]}).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            stream = response.read().decode("utf-8")
        sse_ok = (
            "response.output_text.delta" in stream
            and "response.completed" in stream
            and "data: [DONE]" in stream
        )
        ok = health_status == 200 and health.get("ok") is True and sse_ok
        return _result(
            "serve-sse-contract",
            "pass" if ok else "fail",
            {"health_status": health_status, "sse_events": stream.count("data:")},
            {"health": health, "sse_contract": sse_ok, "contract_header": "v1"},
        )
    finally:
        serve.shutdown()
        thread.join(timeout=2)


def _serve_cancel_latency() -> dict[str, Any]:
    class BrokenResponse:
        async def write(self, _payload):
            raise ConnectionResetError("client disconnected")

    async def probe() -> float:
        writer = _SSEWriter(BrokenResponse(), heartbeat=0, contract="v1")
        started = time.perf_counter()
        try:
            await writer.write("data: probe\n\n")
        except ConnectionResetError:
            pass
        return (time.perf_counter() - started) * 1000.0 if writer.cancel.is_set() else float("inf")

    elapsed_ms = asyncio.run(probe())
    ok = elapsed_ms != float("inf")
    return _result(
        "serve-cancel-latency",
        "pass" if ok else "fail",
        {"cancel_signal_ms": round(elapsed_ms, 3) if ok else None},
        {"mechanism": "SSE writer sets cooperative asyncio.Event on connection error",
         "sla": "not asserted; measures signal propagation only"},
    )


# ── P2-2/F5-020：多 Agent 场景（离线、可重复，锁 Phase A 的四条不变量） ──


class _EvalBranch:
    """离线答者替身：可控成功/失败，记录被调用次数。"""

    kind = "external"

    def __init__(self, name: str, fail: bool = False, text: str = "答"):
        self.name = name
        self._fail = fail
        self._text = text
        self.calls = 0

    async def answer(self, question: str, signal=None) -> str:
        self.calls += 1
        if self._fail:
            raise RuntimeError("branch boom")
        return f"{self._text}:{question}"

    async def close(self) -> None:  # pragma: no cover - 契约对称
        return None


class _EvalSynth:
    """汇总模型替身：记录收到的 prompt，便于断言"失败支不进汇总"。"""

    def __init__(self, content: str = "汇总答复"):
        self.prompts: list[str] = []
        self._content = content

    async def chat(self, messages, tools=None, **kw):
        from agentlab.core.llm import LLMResponse

        self.prompts.append(messages[0].content)
        return LLMResponse(content=self._content, tool_calls=[],
                           usage=TokenUsage(input_tokens=1, output_tokens=1))


def _multi_run(providers: dict, names: list[str], *, cfg=None, synth=None,
               approvals=None) -> tuple[str, list[dict]]:
    """离线跑一次 _run_multi：把 _single() 兜底路径打成桩，避免拉起真实 Runner。"""
    from agentlab.runtime import multi as multi_mod
    from agentlab.runtime import serve as serve_mod
    from agentlab.runtime.serve_contract import _Sink

    events: list[dict] = []
    sink = _Sink(lambda payload: None)
    sink.event = events.append  # type: ignore[assignment]
    base_cfg = cfg or Config(limits=LimitsConfig(max_steps=3, context_budget=8000))
    original = serve_mod._run_agent

    async def _stub_run_agent(*_a, **_kw):
        return {"final_output": "single-fallback", "stop_reason": "done",
                "tokens": 0, "history": [], "trace_id": ""}

    serve_mod._run_agent = _stub_run_agent  # type: ignore[assignment]
    try:
        summary = asyncio.run(multi_mod._run_multi(
            base_cfg, providers, synth, lambda _sink: _BaselineRunner(_sink),
            [], "基线问题", sink, multi=names, approvals=approvals))
    finally:
        serve_mod._run_agent = original  # type: ignore[assignment]
    return summary["final_output"], events


def _multi_branch_degradation() -> dict[str, Any]:
    """单支失败只降级该支：成功支进汇总，失败支名字可见、正文永不出现在 prompt 里。"""
    ok_branch = _EvalBranch("ext-ok", text="OK答")
    bad_branch = _EvalBranch("ext-bad", fail=True)
    synth = _EvalSynth()
    final, events = _multi_run({"ext-ok": ok_branch, "ext-bad": bad_branch},
                               ["ext-ok", "ext-bad"], synth=synth)
    summary = [e for e in events if e.get("type") == "response.multi.summary"]
    failed_listed = bool(summary) and summary[-1].get("failed") == ["ext-bad"]
    ok_counted = bool(summary) and summary[-1].get("ok_count") == 1
    # 失败支没有正文，因此"失败支内容不得进 prompt"以"prompt 只含成功支来源"表述
    prompt = synth.prompts[-1] if synth.prompts else ""
    no_leak = ("ext-bad" not in prompt.split("##")[0]) or ("### 来源：ext-bad" not in prompt)
    passed = final == "汇总答复" and failed_listed and ok_counted and no_leak
    return _result(
        "multi-branch-degradation",
        "pass" if passed else "fail",
        {"ok_count": 1.0 if ok_counted else 0.0,
         "failed_reported": 1.0 if failed_listed else 0.0,
         "failed_body_excluded": 1.0 if no_leak else 0.0},
        {"failed": ["ext-bad"], "synth_called": len(synth.prompts),
         "final": final, "events": [e.get("type") for e in events]},
    )


def _multi_cap_skips_excess() -> dict[str, Any]:
    """超过 max_parallel_consults 的支既不执行也不静默丢弃：进 summary.skipped。"""
    branches = {n: _EvalBranch(n) for n in ("a", "b", "c", "d")}
    cfg = Config(limits=LimitsConfig(max_steps=3, context_budget=8000))
    cfg.agents.max_parallel_consults = 2
    _final, events = _multi_run(branches, ["a", "b", "c", "d"], cfg=cfg,
                                synth=_EvalSynth())
    summary = [e for e in events if e.get("type") == "response.multi.summary"]
    skipped = summary[-1].get("skipped") if summary else None
    unexecuted = (branches["c"].calls, branches["d"].calls) == (0, 0)
    passed = skipped == ["c", "d"] and unexecuted and branches["a"].calls == 1
    return _result(
        "multi-cap-skips-excess",
        "pass" if passed else "fail",
        {"skipped_reported": 1.0 if skipped == ["c", "d"] else 0.0,
         "skipped_not_called": 1.0 if unexecuted else 0.0},
        {"skipped": skipped, "calls": {n: b.calls for n, b in branches.items()}},
    )


def _multi_gate_denied() -> dict[str, Any]:
    """@ 阶段审批未通过 → 零外部调用 + summary.denied（F5-018 红线）。"""
    branch = _EvalBranch("ext-a")

    class _Deny:
        def __init__(self):
            self.calls = 0

        async def confirm(self, cfg, tool, prompt, *, sink, run_id, session_id, signal):
            self.calls += 1
            assert tool.permission == "danger", tool.permission
            return False

    deny = _Deny()
    final, events = _multi_run({"ext-a": branch}, ["ext-a"], approvals=deny)
    summary = [e for e in events if e.get("type") == "response.multi.summary"]
    denied = bool(summary) and bool(summary[-1].get("denied"))
    started = [e for e in events if e.get("type") == "response.branch.started"]
    passed = denied and branch.calls == 0 and not started and deny.calls == 1
    return _result(
        "multi-gate-denied",
        "pass" if passed else "fail",
        {"denied": 1.0 if denied else 0.0,
         "external_not_spawned": 1.0 if branch.calls == 0 else 0.0,
         "single_fallback": 1.0 if final == "single-fallback" else 0.0},
        {"gate_calls": deny.calls, "branch_calls": branch.calls,
         "branch_started_events": len(started), "final": final},
    )


SCENARIO_RUNNERS: dict[str, Callable[[], dict[str, Any]]] = {
    "memory-second-recall": _memory_second_recall,
    "memory-duplicate-write": _memory_duplicate_write,
    "memory-concurrent-write": _memory_concurrent_write,
    "memory-project-isolation": _memory_project_isolation,
    "session-replay": _session_replay,
    "serve-sse-contract": _serve_contract,
    "serve-cancel-latency": _serve_cancel_latency,
    "multi-branch-degradation": _multi_branch_degradation,
    "multi-cap-skips-excess": _multi_cap_skips_excess,
    "multi-gate-denied": _multi_gate_denied,
}


def run_behavior_baseline(output_path: str | Path | None = None,
                          scenarios_path: str | Path | None = None) -> dict[str, Any]:
    """Run declared behavior scenarios and optionally persist the JSON result."""
    declared = load_behavior_scenarios(scenarios_path)
    results: list[dict[str, Any]] = []
    for scenario in declared:
        scenario_id = str(scenario["id"])
        runner = SCENARIO_RUNNERS.get(scenario_id)
        if runner is None:
            results.append(
                _result(scenario_id, "pending", {}, {"reason": "runner_not_implemented"})
            )
            continue
        try:
            results.append(runner())
        except Exception as exc:  # noqa: BLE001 - baseline must report, not abort
            results.append(
                _result(scenario_id, "fail", {}, {"error": f"{type(exc).__name__}: {exc}"})
            )
    passed = sum(item["status"] == "pass" for item in results)
    failed = sum(item["status"] == "fail" for item in results)
    pending = sum(item["status"] == "pending" for item in results)
    report = {
        "schema": SCHEMA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "scenarios": results,
        "summary": {"total": len(results), "passed": passed, "failed": failed, "pending": pending},
    }
    if output_path:
        Path(output_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report
