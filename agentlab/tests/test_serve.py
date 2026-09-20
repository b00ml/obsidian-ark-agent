"""AgentlabServe 单测：health/auth/SSE 事件协议 + 历史拆分。

不依赖真实 LLM：通过 build_factory 注入 fake Runner，仅验证 HTTP 层与
Hermes 兼容事件翻译是否正确。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error

from agentlab.runtime.serve import (
    Serve, _Sink, _history, _run_agent, _sse, _serve_confirm, _persist_delta,
    _finish_task_state,
)
from agentlab.memory.session_store import InMemorySessionStorage, JsonlSessionStorage
from agentlab.core.agent import Agent
from agentlab.core.loop import AgentResult
from agentlab.core.events import ToolEnd
from agentlab.core.message import TokenUsage, Message
from agentlab.core.llm import LLMProvider, LLMResponse


class FakeRunner:
    def __init__(self, sink):
        self.sink = sink
        self.registry = _FakeRegistry()
        self._handlers: dict = {}

    def on(self, ev, cb):
        # 事件订阅（对齐真实 runner）：测试用它验证 serve 侧 trace 挂钩（OPT-216）
        self._handlers[ev] = cb

    def _emit(self, ev, payload):
        cb = self._handlers.get(ev)
        if cb:
            cb(payload)

    async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
        from agentlab.core.events import Ev, ToolStart
        self.sink.tool_start("vault_search")
        self._emit(Ev.TOOL_START, ToolStart(name="vault_search",
                                            arguments='{"query": "测试"}'))
        self.sink.tool_end("vault_search", '{"total": 2}')
        self._emit(Ev.TOOL_END, ToolEnd(name="vault_search", result='{"total": 2}'))
        self.sink.text(f"已检索，回复：{user_input}")
        return AgentResult(
            final_output=f"已检索，回复：{user_input}",
            stop_reason="done",
            messages=[],
            usage=TokenUsage(input_tokens=10, output_tokens=20),
            run_trace={"counters": {"rounds": 2, "llm_calls": 2,
                                     "tool_calls": 1, "retries": 0,
                                     "plan_steps": 0}},
        )


class FakeSlowRunner(FakeRunner):
    """慢 Runner：模拟长工具阻塞，验证进度心跳沿 SSE 流周期性下发。"""

    async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
        await asyncio.sleep(1.6)
        self.sink.text("完成")
        return AgentResult(
            final_output="完成",
            stop_reason="done",
            messages=[],
            usage=TokenUsage(input_tokens=5, output_tokens=5),
            run_trace={"counters": {"rounds": 1, "llm_calls": 1,
                                     "tool_calls": 0, "retries": 0,
                                     "plan_steps": 0}},
        )


class _FakeTool:
    name = "vault_search"
    description = "检索 Vault"


class _FakeRegistry:
    def all(self):
        return [_FakeTool()]

    def schemas(self):
        return []


def _make_serve(port=18731):
    cfg = _tmp_cfg()
    factory = lambda: (lambda sink: FakeRunner(sink))  # noqa: E731
    s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory,
              session_store=InMemorySessionStorage())
    httpd = s.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return s, httpd, port


def _tmp_cfg():
    """测试用 Config：收敛为真实 pydantic 模型（对齐 C5「配置收敛到 pydantic」）。"""

    from agentlab.runtime.config import Config, LimitsConfig, ServeConfig

    return Config(
        limits=LimitsConfig(max_steps=5, context_budget=8000, max_tool_result_chars=4000, timeout=None),
        skills_dir=None,
        serve=ServeConfig(host="127.0.0.1", port=18731, token="agentlab-dev", heartbeat=10),
    )


class TestFailClosed(unittest.TestCase):
    def test_missing_token_refuses_to_start(self):
        cfg = _tmp_cfg()
        cfg.serve.token = ""
        factory = lambda: (lambda sink: FakeRunner(sink))  # noqa: E731
        s = Serve(cfg, port=9999, host="127.0.0.1", build_factory=factory)
        with self.assertRaises(RuntimeError):
            s.start()
        self.assertIsNone(getattr(s, "_served", None))  # 未启动、无监听


def _post(url, body, token="agentlab-dev"):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _options(url):
    req = urllib.request.Request(
        url, method="OPTIONS",
        headers={"Origin": "app://obsidian.md",
                 "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "authorization, content-type"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.getheader("Access-Control-Allow-Headers", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _post_hdr(url, body, token="agentlab-dev"):
    """POST 并额外返回幂等重放头 X-Agentlab-Replay。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8"), r.getheader("X-Agentlab-Replay")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), None


class TestServeProto(unittest.TestCase):
    def test_history_strips_system_and_takes_latest_user(self):
        hist, latest = _history([
            {"role": "system", "content": "你是舰长"},
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "回答一"},
            {"role": "user", "content": "第二问"},
        ])
        self.assertEqual([m.role for m in hist], ["user", "assistant"])
        self.assertEqual(hist[0].content, "第一问")
        self.assertEqual(latest, "第二问")

    def test_sse_formatting(self):
        self.assertEqual(_sse({"a": 1}), 'data: {"a": 1}\n\n')

    def test_sink_translates_to_hermes_events(self):
        out: list[str] = []
        sink = _Sink(out.append)
        sink.tool_start("rag_retrieve")
        sink.tool_end("rag_retrieve", '{"total":3}')
        sink.text("你好")
        self.assertEqual(json.loads(out[0].removeprefix("data: "))["type"],
                         "response.output_item.added")
        self.assertEqual(json.loads(out[1].removeprefix("data: "))["item"]["type"],
                         "function_call_output")
        self.assertEqual(json.loads(out[2].removeprefix("data: "))["type"],
                         "response.output_text.delta")

    def test_tool_start_defaults_to_empty_object(self):
        """F5-016：未传参的旧调用点仍产出合法 JSON 对象，不破坏既有前端解析。"""
        out: list[str] = []
        _Sink(out.append).tool_start("vault_search")
        self.assertEqual(
            json.loads(out[0].removeprefix("data: "))["item"]["arguments"], "{}")

    def test_tool_start_forwards_real_arguments(self):
        """F5-016：`_build_backend` 注册的 TOOL_START 必须把模型真实入参透传到 SSE。

        此前 lambda 只取 p.name，`arguments` 恒为 "{}"——前端无法显示
        「正在咨询 @agentX：问题…」。此行是 agent_consult 可观测性的唯一通路。
        """
        from unittest import mock

        from agentlab.core.events import ToolStart
        from agentlab.core.loop import Ev
        from agentlab.runtime import serve as serve_mod

        class FakeRunner:
            registry = None

            def __init__(self):
                self.handlers = {}

            def on(self, ev, cb):
                self.handlers[ev] = cb

        fake = FakeRunner()
        out: list[str] = []
        with mock.patch("agentlab.runtime.cli._build_registry",
                        return_value=(mock.Mock(all=lambda: []), 0)), \
                mock.patch("agentlab.runtime.cli._build_runner", return_value=fake), \
                mock.patch("agentlab.tools.rag_tools.build_vector_index",
                           return_value=None):
            build, _n_brain, _gateway = serve_mod._build_backend(_tmp_cfg())
            self.assertIs(build(_Sink(out.append)), fake)

        payload = '{"agent": "claude-code", "question": "问题X"}'
        fake.handlers[Ev.TOOL_START](
            ToolStart(name="agent_consult", arguments=payload))
        item = json.loads(out[0].removeprefix("data: "))["item"]
        self.assertEqual(item["name"], "agent_consult")
        self.assertEqual(item["arguments"], payload,
                         "工具入参必须原样透传，而不是空对象")

    def test_build_backend_forwards_rag_llm_to_registry(self):
        """答案 probe 必须给 rag_assess 注入 provider，而非走无 LLM 降级。"""
        from unittest import mock

        from agentlab.runtime import serve as serve_mod

        registry = mock.Mock(all=lambda: [])
        fake_runner = mock.Mock(registry=registry)
        rag_llm = object()
        with mock.patch("agentlab.runtime.cli._build_registry",
                        return_value=(registry, 0)) as build_registry, \
                mock.patch("agentlab.runtime.cli._build_runner", return_value=fake_runner), \
                mock.patch("agentlab.tools.rag_tools.build_p2_store", return_value=None), \
                mock.patch("agentlab.tools.rag_tools.build_vector_index", return_value=None):
            serve_mod._build_backend(_tmp_cfg(), rag_llm=rag_llm)

        self.assertIs(build_registry.call_args.kwargs["rag_llm"], rag_llm)



class TestServeConfirm(unittest.TestCase):
    """F4/F5：serve 链路审批策略与工具门禁。"""

    def _cfg(self, allowlist=None):
        cfg = _tmp_cfg()
        if allowlist is not None:
            cfg.danger_allowlist = allowlist
        return cfg

    def _tool(self, name, permission):
        return type("T", (), {"name": name, "permission": permission})()

    def test_danger_not_in_allowlist_rejected(self):
        cfg = self._cfg(allowlist=["allowed_tool"])
        self.assertFalse(_serve_confirm(cfg, self._tool("danger_tool", "danger"), "p"))

    def test_danger_in_allowlist_allowed(self):
        cfg = self._cfg(allowlist=["danger_tool"])
        self.assertTrue(_serve_confirm(cfg, self._tool("danger_tool", "danger"), "p"))

    def test_write_unsanctioned_rejected_by_default(self):
        # 未被认可的非 read 工具 → 默认拒绝（fail-closed 防越权写库）
        self.assertFalse(_serve_confirm(self._cfg(), self._tool("do_write", "write"), "p"))
        # danger_allowlist 只放行 danger 权限；列入白名单的 write 仍不因此放行
        cfg = self._cfg(allowlist=["do_write"])
        self.assertFalse(_serve_confirm(cfg, self._tool("do_write", "write"), "p"))

    def test_write_sanctioned_vault_writer_allowed(self):
        # 被认可的 Vault/memory 写入器放行（路径边界由 brain wrapper raw/templates 守卫兜底）
        self.assertTrue(_serve_confirm(self._cfg(), self._tool("vault_write", "write"), "p"))
        self.assertTrue(_serve_confirm(self._cfg(), self._tool("bili_visual", "write"), "p"))
        self.assertTrue(_serve_confirm(self._cfg(), self._tool("memory_commit", "write"), "p"))

    def test_write_configured_allowlist_narrows(self):
        # 显式 write_allowlist 覆盖默认集（如只留 vault_write）
        cfg = self._cfg()
        cfg.write_allowlist = ["vault_write"]
        self.assertTrue(_serve_confirm(cfg, self._tool("vault_write", "write"), "p"))
        cfg.write_allowlist = []
        self.assertFalse(_serve_confirm(cfg, self._tool("vault_write", "write"), "p"))

    def test_allow_all_policy_approves_registered_permissions(self):
        cfg = self._cfg()
        cfg.approval_mode = "allow_all"
        self.assertTrue(_serve_confirm(cfg, self._tool("read_tool", "read"), "p"))
        self.assertTrue(_serve_confirm(cfg, self._tool("write_tool", "write"), "p"))
        self.assertTrue(_serve_confirm(cfg, self._tool("danger_tool", "danger"), "p"))

    def test_invalid_approval_mode_fails_closed(self):
        cfg = self._cfg()
        cfg.approval_mode = "typo"
        self.assertFalse(_serve_confirm(cfg, self._tool("danger_tool", "danger"), "p"))


class TestSessionStore(unittest.TestCase):
    """S2 会话持久化：JSONL append-only 往返、半行崩溃容错、InMemory 替身、delta 增量。"""

    def _tmp_jsonl(self):
        import tempfile
        return JsonlSessionStorage(tempfile.mkdtemp())

    def test_jsonl_roundtrip(self):
        store = self._tmp_jsonl()
        store.append("s1", Message(role="user", content="第一问"))
        store.append("s1", Message(role="assistant", content="第一答"))
        out = store.read_all("s1")
        self.assertEqual([m.role for m in out], ["user", "assistant"])
        self.assertEqual(out[1].content, "第一答")

    def test_jsonl_partial_line_tolerated(self):
        # 模拟追加中途崩溃：末尾写一行不完整 JSON，read 必须丢弃而非封死会话
        import tempfile
        from pathlib import Path
        root = tempfile.mkdtemp()
        store = JsonlSessionStorage(root)
        store.append("s2", Message(role="user", content="ok"))
        with (Path(root) / "s2.jsonl").open("a", encoding="utf-8") as f:
            f.write('{"role": "assist')  # 半行：追加中途崩溃的残尾
        out = store.read_all("s2")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].content, "ok")

    def test_in_memory_substitute(self):
        store = InMemorySessionStorage()
        store.append("s", Message(role="user", content="q"))
        self.assertEqual(store.read_all("s")[0].content, "q")

    def test_persist_delta_skips_system_and_history(self):
        store = InMemorySessionStorage()
        # 生产 flow：hist = 服务端重放的完整历史（旧问+旧答），恰好等于 messages 中 system 之后那一段
        hist = [Message(role="user", content="旧问"),
                Message(role="assistant", content="旧答")]
        messages = [
            Message(role="system", content="指令"),
            *hist,
            Message(role="user", content="新问"),
            Message(role="assistant", content="新答"),
        ]
        _persist_delta(store, "s", hist, messages)
        saved = store.read_all("s")
        # 仅新增的两条被持久化，system 与已重放的历史不入库
        self.assertEqual([m.content for m in saved], ["新问", "新答"])

    def test_persist_delta_without_system(self):
        store = InMemorySessionStorage()
        messages = [Message(role="user", content="u"), Message(role="assistant", content="a")]
        _persist_delta(store, "s", [], messages)
        self.assertEqual([m.content for m in store.read_all("s")], ["u", "a"])


class _RecordingGateway:
    """假区段网关（L11）：记录 bind/recorder/reset 调用序列。"""

    def __init__(self):
        self.bound = []
        self.recorder_for = []
        self.resets = []

    def bind(self, sid):
        self.bound.append(sid)
        return ("tok", sid)

    def reset(self, tok):
        self.resets.append(tok)

    def recorder(self, sid, **kw):
        self.recorder_for.append(sid)
        return object()


class TestRunAgentRangeGateway(unittest.TestCase):
    """L11/OPT-111：_run_agent 每请求注入折叠归档器 + bind 当前会话，结束即 reset。"""

    def test_binds_and_injects_recorder_then_resets(self):
        captured = {}

        class CapRunner(FakeRunner):
            async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
                captured["recorder"] = getattr(cfg, "range_recorder", "MISSING")
                return await super().run(agent, user_input, ctx=ctx, cfg=cfg, hooks=hooks)

        gw = _RecordingGateway()
        res = asyncio.run(_run_agent(
            _tmp_cfg(), lambda sink: CapRunner(sink), [], "你好", _Sink(lambda chunk: None),
            session_id="s-1", range_gateway=gw))
        self.assertEqual(res["stop_reason"], "done")
        self.assertEqual(gw.bound, ["s-1"])
        self.assertEqual(gw.recorder_for, ["s-1"])
        self.assertIsNotNone(captured.get("recorder"), "折叠归档器应注入 RunConfig")
        self.assertEqual(len(gw.resets), 1, "run 结束（含异常路径）必须 reset 防串话")

    def test_no_gateway_no_binding(self):
        captured = {}

        class CapRunner(FakeRunner):
            async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
                captured["recorder"] = getattr(cfg, "range_recorder", None)
                return await super().run(agent, user_input, ctx=ctx, cfg=cfg, hooks=hooks)

        asyncio.run(_run_agent(
            _tmp_cfg(), lambda sink: CapRunner(sink), [], "你好", _Sink(lambda chunk: None),
            session_id="s-1"))
        self.assertIsNone(captured.get("recorder"))

    def test_binds_request_retrieval_scope_and_restores_it(self):
        captured = {}

        class CapRunner(FakeRunner):
            async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
                from agentlab.contracts import current_retrieval_scope
                scope = current_retrieval_scope()
                captured["scope"] = (scope.project_id, scope.session_id)
                return await super().run(agent, user_input, ctx=ctx, cfg=cfg, hooks=hooks)

        from agentlab.contracts import current_retrieval_scope
        self.assertEqual((current_retrieval_scope().project_id,
                          current_retrieval_scope().session_id), ("", ""))
        asyncio.run(_run_agent(
            _tmp_cfg(), lambda sink: CapRunner(sink), [], "问题", _Sink(lambda _: None),
            session_id="s-1", project_id="p-1",
        ))
        self.assertEqual(captured["scope"], ("p-1", "s-1"))
        self.assertEqual((current_retrieval_scope().project_id,
                          current_retrieval_scope().session_id), ("", ""))

    def test_optional_task_state_checkpoint_records_request_lifecycle(self):
        cfg = _tmp_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.context.task_state_path = os.path.join(tmp, "task-state.db")
            asyncio.run(_run_agent(
                cfg, lambda sink: FakeRunner(sink), [], "任务目标", _Sink(lambda _: None),
                session_id="s-state", project_id="p-state", run_id="r-state",
            ))
            from agentlab.runtime.task_state import TaskStateStore
            state = TaskStateStore(cfg.context.task_state_path).get("r-state")
            self.assertEqual(state.phase, "DONE")
            self.assertEqual(state.session_id, "s-state")
            self.assertEqual(state.project_id, "p-state")
            self.assertEqual(state.core_intent["goal"], "任务目标")

    def test_finish_refreshes_task_state_after_tool_ledger_updates(self):
        from agentlab.runtime.task_state import TaskStateStore

        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStateStore(os.path.join(tmp, "task-state.db"))
            state = store.ensure("r-ledger")
            state = store.transition("r-ledger", "PLANNING")
            state = store.transition("r-ledger", "EXECUTING")
            stale_snapshot = state
            store.plan_tool(
                "r-ledger", operation_id="op-1", tool_name="vault_write",
                permission="write", side_effects="write", idempotent=True,
            )
            store.update_tool("r-ledger", "op-1", "running")
            store.update_tool("r-ledger", "op-1", "succeeded")

            saved = _finish_task_state(store, stale_snapshot)
            self.assertEqual(saved.phase, "DONE")
            self.assertEqual(store.get("r-ledger").phase, "DONE")

    def test_run_agent_returns_answer_gate_summary(self):
        class GateRunner(FakeRunner):
            async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
                return AgentResult(
                    final_output="有证据的回答",
                    stop_reason="done",
                    messages=[],
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                    answer_gate={"mode": "shadow", "reasons": []},
                )

        result = asyncio.run(_run_agent(
            _tmp_cfg(), lambda sink: GateRunner(sink), [], "问题",
            _Sink(lambda _: None), session_id="s-gate",
        ))
        self.assertEqual(result["answer_gate"], {"mode": "shadow", "reasons": []})


class TestCancelPropagation(unittest.TestCase):
    """S3 取消传播（asyncio 版）：_SSEWriter 写流异常（客户端断连）置位 cancel 并上抛。"""

    def test_writer_sets_cancel_on_pipe_error(self):
        from agentlab.runtime.serve import _SSEWriter

        class _BrokenResp:
            async def write(self, b):
                raise ConnectionResetError("client gone")

        async def scenario() -> bool:
            w = _SSEWriter(_BrokenResp(), heartbeat=0, contract="v1")
            try:
                await w.write("data: x\n\n")
            except ConnectionResetError:
                pass
            return w.cancel.is_set()

        self.assertTrue(asyncio.run(scenario()))  # 断连已置位 → loop 将中止

    def test_writer_serializes_concurrent_writes(self):
        # 并发写字块（主流程 + 心跳）经 asyncio.Lock 串行，不等字节交错
        from agentlab.runtime.serve import _SSEWriter

        class _Resp:
            def __init__(self):
                self.writes: list[str] = []

            async def write(self, b):
                self.writes.append(b.decode("utf-8"))

        async def scenario() -> int:
            w = _SSEWriter(_Resp(), heartbeat=0, contract="v1")
            await asyncio.gather(*(w.write(f"{i}") for i in range(10)))
            return len(w._resp.writes)

        self.assertEqual(asyncio.run(scenario()), 10)


class TestServeIdempotencyCache(unittest.TestCase):
    """S6 幂等键：_IdempotencyCache 的 begin/commit/snapshot/abandon/过期/淘汰。"""

    def _cache(self):
        from agentlab.runtime.serve_idem import _IdempotencyCache
        return _IdempotencyCache(size=3, ttl=300)

    def test_done_replayable(self):
        c = self._cache()
        self.assertTrue(c.begin("r1"))
        c.commit("r1", ["a", "b"])
        self.assertEqual(c.snapshot("r1"), ("done", ["a", "b"]))  # 可逐条重放

    def test_pending_duplicate_rejected_then_release(self):
        c = self._cache()
        self.assertTrue(c.begin("r1"))
        self.assertFalse(c.begin("r1"))            # pending 期间并发重复被拒
        self.assertEqual(c.snapshot("r1"), ("pending", None))
        c.abandon("r1")                            # 首请求失败/中断 → 释放
        self.assertTrue(c.begin("r1"))             # 重试可重新跑

    def test_ttl_expiry(self):
        from agentlab.runtime.serve_idem import _IdempotencyCache
        c = _IdempotencyCache(size=3, ttl=0.01)
        self.assertTrue(c.begin("r1")); c.commit("r1", ["x"])
        time.sleep(0.02)
        self.assertIsNone(c.snapshot("r1"))        # 过期即丢

    def test_eviction_oldest_when_full(self):
        from agentlab.runtime.serve_idem import _IdempotencyCache
        c = _IdempotencyCache(size=2, ttl=300)
        c.begin("r1"); c.commit("r1", ["a"])
        c.begin("r2"); c.commit("r2", ["b"])
        c.begin("r3"); c.commit("r3", ["c"])       # size=2 → r1（最旧）被淘汰
        self.assertIsNone(c.snapshot("r1"))
        self.assertEqual(c.snapshot("r2"), ("done", ["b"]))
        self.assertEqual(c.snapshot("r3"), ("done", ["c"]))


class TestServeIdempotentReplay(unittest.TestCase):
    """S6 幂等 HTTP 层：同 request_id 重复提交 → 重放首次结果，不重复跑 agent。"""

    def test_duplicate_request_id_replays_without_rerun(self):
        port = 18733
        counters: list[str] = []

        class _CountingRunner(FakeRunner):
            async def run(self, *a, **k):
                counters.append("run")
                return await super().run(*a, **k)

        cfg = _tmp_cfg(); cfg.serve.port = port
        factory = lambda: (lambda sink: _CountingRunner(sink))  # noqa: E731
        # 固定 request_id 用例必须用独立幂等库：默认持久背板（OPT-214）会让上一轮
        # 运行的 "idem-aa" 命中重放，破坏测试隔离
        import tempfile as _tf
        idem_dir = _tf.mkdtemp(prefix="idem-test-")
        s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory,
                  session_store=InMemorySessionStorage(),
                  idem_store_path=os.path.join(idem_dir, "idem.db"))
        httpd = s.start()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            time.sleep(0.15)
            payload = {"request_id": "idem-aa", "input": [{"role": "user", "content": "hi"}]}
            url = f"http://127.0.0.1:{port}/v1/responses"
            code, body = _post(url, payload)
            self.assertEqual(code, 200)
            self.assertEqual(counters, ["run"])           # 首次恰好跑一次
            code2, body2, hdr = _post_hdr(url, payload)
            self.assertEqual(code2, 200)
            self.assertEqual(hdr, "1")                    # 命中重放标记
            self.assertEqual(body, body2)                 # SSE 字节级一致
            self.assertEqual(counters, ["run"])           # 未重复跑 → 幂等成立
            # 不同 request_id 正常新跑
            code3, _ = _post(url, {"request_id": "idem-bb", "input": [{"role": "user", "content": "yo"}]})
            self.assertEqual(code3, 200)
            self.assertEqual(len(counters), 2)
        finally:
            httpd.shutdown(); httpd.server_close()


class TestServeHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, cls.httpd, cls.port = _make_serve()

    @classmethod
    def tearDownClass(cls):
        cls.s.shutdown()

    def test_health(self):
        code, body = _get(f"http://127.0.0.1:{self.port}/health")
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(body)["ok"])

    def test_unauthorized(self):
        code, _ = _post(f"http://127.0.0.1:{self.port}/v1/responses",
                        {"input": [{"role": "user", "content": "x"}]}, token="wrong")
        self.assertEqual(code, 401)

    def test_cors_preflight_allows_post(self):
        # 浏览器先行 OPTIONS 预检；此前 501 导致 Obsidian fetch 报 "Failed to fetch"（回归）
        code, allow_headers = _options(f"http://127.0.0.1:{self.port}/v1/responses")
        self.assertEqual(code, 204)
        self.assertIn("authorization", allow_headers.lower())
        self.assertIn("content-type", allow_headers.lower())

    def test_cors_preflight_covers_all_endpoints(self):
        """F5-021 回归：带 Authorization 的 GET 也会预检，未注册 OPTIONS 就报 Failed to fetch。

        此前只有 /v1/responses 与 /v1/approvals 注册了 OPTIONS，
        /v1/runs 与 /v1/agents 预检是 405 —— Ark 侧表现为"无法连接 Agent 内核"。
        """
        for path in ("/v1/agents", "/v1/runs", "/v1/runs/abc", "/v1/sessions/s-1",
                     "/v1/context-status", "/v1/tasks/task-1/operations",
                     "/v1/tasks/task-1/operations/op-1/reconcile"):
            code, allow_headers = _options(f"http://127.0.0.1:{self.port}{path}")
            self.assertEqual(code, 204, f"{path} 预检未通过")
            self.assertIn("authorization", allow_headers.lower())

    def test_tool_end_not_clipped_to_1000(self):
        # 回归：SSE 工具结果不再硬编码截到 1000（曾致长转写只露开头 → agent 反复重拉卡顿）
        from agentlab.runtime.serve import _Sink
        out: list[str] = []
        _Sink(out.append).tool_end("bili_transcribe", "字" * 3000)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(json.loads(out[0][len("data: "):-2])["item"]["output"]), 3000)

    def test_transcribe_cache_reuses_result(self):
        # 回归：同 bvid 重复调 bili_transcribe 走缓存，不再重跑 Whisper（分钟级）
        from agentlab.tools.connectors import brain_tools as bt
        calls = []
        def fake(config, bvid=None, **kw):
            calls.append(bvid)
            return {"bvid": bvid, "transcript": "x" * 100, "strategy": 3}
        cached = bt._with_transcribe_cache(fake)
        bt._TRANSCRIPT_CACHE.clear()
        r1 = cached({}, "BV01")
        self.assertEqual(calls, ["BV01"])
        r2 = cached({}, "BV01")
        # 命中缓存：底层不再被调用，返回打上 cached 标记
        self.assertEqual(calls, ["BV01"])
        self.assertTrue(r2.get("cached"))
        self.assertEqual(r2["transcript"], r1["transcript"])
        bt._TRANSCRIPT_CACHE.clear()

    def test_visual_cache_reuses_result(self):
        # OPT-218 真机回归：bili_visual 含截帧+视觉模型约 4 分钟，serve 断连重试后
        # 同 bvid 重跑 = 全部重做（实测第一次 run 237s 白费后重跑又 239s）。缓存命中。
        from agentlab.tools.connectors import brain_tools as bt
        calls = []
        def fake(config, bvid=None, **kw):
            calls.append(bvid)
            return {"bvid": bvid, "note": "Inbox/x-visual.md", "grids": 3}
        cached = bt._with_key_cache(fake, bt._VISUAL_CACHE, "bvid")
        bt._VISUAL_CACHE.clear()
        r1 = cached({}, "BV01")
        r2 = cached({}, "BV01")
        self.assertEqual(calls, ["BV01"])  # 第二次命中缓存，底层不再执行
        self.assertTrue(r2.get("cached"))
        self.assertEqual(r2["note"], r1["note"])
        bt._VISUAL_CACHE.clear()


    def test_article_cache_by_url(self):
        # OPT-222 真机回归："处理收件箱"重发循环中同 URL 的 article_summarize
        # 重复抓取+总结烧钱；url 参数名与 bvid 不同，key 必须正确提取
        from agentlab.tools.connectors import brain_tools as bt
        calls = []
        def fake(config, url=None, **kw):
            calls.append(url)
            return {"url": url, "note": "Inbox/a-文章.md"}
        cached = bt._with_key_cache(fake, bt._ARTICLE_CACHE, "url")
        bt._ARTICLE_CACHE.clear()
        r1 = cached({}, "https://mp.weixin.qq.com/s/abc")
        r2 = cached({}, "https://mp.weixin.qq.com/s/abc")
        r3 = cached({}, "https://mp.weixin.qq.com/s/other")
        self.assertEqual(calls, ["https://mp.weixin.qq.com/s/abc",
                                 "https://mp.weixin.qq.com/s/other"])
        self.assertTrue(r2.get("cached"))
        self.assertFalse(r3.get("cached"))
        bt._ARTICLE_CACHE.clear()

    def test_stream_events_and_done(self):
        code, body = _post(f"http://127.0.0.1:{self.port}/v1/responses",
                           {"input": [{"role": "user", "content": "查知识库"}, {"role": "assistant", "content": "先行"}],
                            "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("response.output_item.added", body)
        self.assertIn("function_call_output", body)
        self.assertIn("response.output_text.delta", body)
        self.assertIn("[DONE]", body)
        self.assertIn("response.completed", body)

    def test_empty_input_rejected(self):
        code, _ = _post(f"http://127.0.0.1:{self.port}/v1/responses", {"input": [{"role": "system", "content": "s"}]})
        self.assertEqual(code, 400)

    def test_invalid_payload_rejected_400(self):
        # S4 契约加固：非法载荷（input 非消息数组）→ 400，而非绕过校验
        code, body = _post(f"http://127.0.0.1:{self.port}/v1/responses", {"input": "not-a-list"})
        self.assertEqual(code, 400)
        self.assertIn("bad payload", body)

    def test_contract_header_on_sse(self):
        # S4 契约版本号：SSE 响应带 X-Agentlab-Contract，前端可据此识别不匹配
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/responses",
            data=json.dumps({"input": [{"role": "user", "content": "x"}]}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer agentlab-dev"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            self.assertEqual(r.getheader("X-Agentlab-Contract"), "v1")
            r.read()  # 消费完整 SSE 体再关闭，避免服务端写未完成即被断连

    def test_responses_request_model(self):
        # S4：schema 关键字段类型校验，额外字段放行（兼容前端 model/tools）
        from agentlab.runtime.serve import ResponsesRequest
        req = ResponsesRequest.model_validate(
            {"model": "agentlab-demo", "input": [{"role": "user", "content": "u"}], "stream": True})
        self.assertEqual(req.input[0]["role"], "user")
        self.assertTrue(req.stream)
        with self.assertRaises(Exception):
            ResponsesRequest.model_validate({"stream": "not-a-bool"})

    def test_responses_request_request_id(self):
        # S6 幂等键：request_id 为字符串字段，非法类型 400（复用 pydantic）
        from agentlab.runtime.serve import ResponsesRequest
        req = ResponsesRequest.model_validate({"request_id": "k-1", "input": []})
        self.assertEqual(req.request_id, "k-1")
        with self.assertRaises(Exception):
            ResponsesRequest.model_validate({"request_id": 123, "input": []})

    def test_heartbeat_emitted_on_long_task(self):
        # 进度心跳：长任务期间沿 SSE 流推 response.heartbeat，且不污染最终 delta 正文
        os.environ["AGENTLAB_SERVE_HEARTBEAT"] = "1"
        port = 18732
        try:
            cfg = _tmp_cfg(); cfg.serve.port = port
            factory = lambda: (lambda sink: FakeSlowRunner(sink))  # noqa: E731
            s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory)
            httpd = s.start()
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            try:
                time.sleep(0.1)
                code, body = _post(f"http://127.0.0.1:{port}/v1/responses",
                                   {"input": [{"role": "user", "content": "long"}]})
                self.assertEqual(code, 200)
                self.assertIn("response.heartbeat", body)
                self.assertIn("response.completed", body)
            finally:
                httpd.shutdown(); httpd.server_close()
        finally:
            os.environ.pop("AGENTLAB_SERVE_HEARTBEAT", None)


if __name__ == "__main__":
    unittest.main()

class TestProjectContext(unittest.TestCase):
    """P0-2/OPT-107：Project 长期任务空间——id 白名单、规则/背景加载、system 注入。"""

    def test_sanitize_rejects_traversal_and_empty(self):
        from agentlab.runtime.project import sanitize_project_id
        self.assertEqual(sanitize_project_id("../../etc"), "")
        self.assertEqual(sanitize_project_id("a/b"), "")
        self.assertEqual(sanitize_project_id("a\b"), "")
        self.assertEqual(sanitize_project_id(".."), "")
        self.assertEqual(sanitize_project_id(""), "")
        self.assertEqual(sanitize_project_id(None), "")
        self.assertEqual(sanitize_project_id(" my-proj_1 "), "my-proj_1")
        self.assertEqual(sanitize_project_id("中文项目"), "中文项目")

    def test_project_context_loads_rules_and_background(self):
        import tempfile
        from pathlib import Path
        from agentlab.runtime.project import project_context, apply_project_context
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "ark" / "projects" / "p1"
            d.mkdir(parents=True)
            (d / "AGENTS.md").write_text("规则：笔记按内容命名", encoding="utf-8")
            (d / "project.md").write_text("背景：视频入库流水线", encoding="utf-8")
            block = project_context(tmp, "p1")
            self.assertIn("按内容命名", block)
            self.assertIn("优先于全局规范", block)
            self.assertIn("视频入库流水线", block)
            self.assertEqual(project_context(tmp, "ghost"), "")   # 未注册项目 → 空降级
            self.assertEqual(project_context(tmp, None), "")
            self.assertEqual(project_context(tmp, "../../etc"), "")
            # 注入：有项目 → 追加块；无项目 → 原样
            self.assertIn("优先于全局规范", apply_project_context("SYS", tmp, "p1"))
            self.assertEqual(apply_project_context("SYS", tmp, "ghost"), "SYS")

    def test_partial_project_only_project_md(self):
        import tempfile
        from pathlib import Path
        from agentlab.runtime.project import project_context
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "ark" / "projects" / "p2"
            d.mkdir(parents=True)
            (d / "project.md").write_text("只有背景", encoding="utf-8")
            block = project_context(tmp, "p2")
            self.assertIn("只有背景", block)
            self.assertNotIn("优先于全局规范", block)


class TestServeProjectHTTP(unittest.TestCase):
    """P0-2 集成：project_id 进契约，未知项目静默降级为全局行为，请求正常完成。"""

    @classmethod
    def setUpClass(cls):
        cls.s, cls.httpd, cls.port = _make_serve()

    @classmethod
    def tearDownClass(cls):
        cls.s.shutdown()

    def test_responds_with_project_id_degrades_silently(self):
        code, body = _post(
            f"http://127.0.0.1:{self.port}/v1/responses",
            {"input": [{"role": "user", "content": "hi"}],
             "project_id": "ghost", "request_id": "proj-smoke-1"})
        self.assertEqual(code, 200)
        self.assertIn("response.completed", body)

    def test_project_rules_reach_agent_instructions(self):
        """回归：请求 project_id 后，规则必须进入实际 Agent system instructions。"""
        import tempfile
        from pathlib import Path

        root = tempfile.mkdtemp()
        project = Path(root) / "ark" / "projects" / "p1"
        project.mkdir(parents=True)
        (project / "AGENTS.md").write_text("RULE-P1", encoding="utf-8")
        (project / "project.md").write_text("BACKGROUND-P1", encoding="utf-8")
        cfg = _tmp_cfg()
        cfg.vault_root = root
        captured = {}

        class CaptureRunner(FakeRunner):
            async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
                captured["instructions"] = agent.instructions
                return await super().run(agent, user_input, ctx=ctx, cfg=cfg, hooks=hooks)

        asyncio.run(_run_agent(
            cfg, lambda sink: CaptureRunner(sink), [], "问题", _Sink(lambda _: None),
            project_id="p1"))
        self.assertIn("RULE-P1", captured["instructions"])
        self.assertIn("BACKGROUND-P1", captured["instructions"])


class TestContextStatusAndHistory(unittest.TestCase):
    """W2/OPT-110：context-status 计算端点 + completed 事件携带历史快照。"""

    def test_context_status_endpoint(self):
        s, httpd, port = _make_serve()
        try:
            code, body = _post(f"http://127.0.0.1:{port}/v1/context-status",
                               {"messages": [
                                   {"role": "system", "content": "s" * 40},
                                   {"role": "user", "content": "u" * 40},
                               ]})
            self.assertEqual(code, 200)
            data = json.loads(body)
            self.assertTrue(data["ok"])
            self.assertGreater(data["tokens"], 0)
            self.assertEqual(data["budget"], int(s.cfg.limits.context_budget))
            self.assertEqual(data["effective_budget"], data["budget"])
            self.assertEqual(data["scope"], "client_history_estimate")
            self.assertGreater(data["usage_pct"], 0)
            self.assertEqual(data["messages"], 2)
        finally:
            httpd.shutdown(); httpd.server_close()


    def test_context_status_unauthorized(self):
        s, httpd, port = _make_serve()
        try:
            code, _ = _post(f"http://127.0.0.1:{port}/v1/context-status",
                            {"messages": []}, token="wrong")
            self.assertEqual(code, 401)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_completed_carries_history_snapshot(self):
        # FakeRunner.run 返回 messages=[] → history=[] 也应出现在 completed 事件（契约在位）
        s, httpd, port = _make_serve()
        try:
            code, body = _post(f"http://127.0.0.1:{port}/v1/responses",
                               {"input": [{"role": "user", "content": "hi"}],
                                "request_id": "hist-1"})
            self.assertEqual(code, 200)
            self.assertIn('"history"', body)
        finally:
            httpd.shutdown(); httpd.server_close()


class TestTaskStateHTTP(unittest.TestCase):
    """P2-02：恢复账本只允许外部证据结算 unknown，不触发工具重放。"""

    def setUp(self):
        from agentlab.runtime.task_state import TaskStateStore

        self.tmp = tempfile.TemporaryDirectory()
        cfg = _tmp_cfg()
        cfg.context.task_state_path = os.path.join(self.tmp.name, "task-state.db")
        self.store = TaskStateStore(cfg.context.task_state_path)
        self.store.ensure("task-reconcile")
        self.store.plan_tool(
            "task-reconcile", operation_id="op-1", tool_name="external_write",
            permission="danger", side_effects="external", idempotent=False,
        )
        self.store.update_tool("task-reconcile", "op-1", "unknown")
        factory = lambda: (lambda sink: FakeRunner(sink))  # noqa: E731
        self.serve = Serve(
            cfg, port=18790, host="127.0.0.1", build_factory=factory,
            session_store=InMemorySessionStorage(),
        )
        self.httpd = self.serve.start()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:18790/v1/tasks/task-reconcile/operations"

    def tearDown(self):
        self.serve.shutdown()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def _get_auth(self, token="agentlab-dev"):
        req = urllib.request.Request(
            self.url, headers={"Authorization": f"Bearer {token}"}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8")

    def test_operations_require_auth_and_list_unknown(self):
        code, _ = self._get_auth(token="wrong")
        self.assertEqual(code, 401)
        code, body = self._get_auth()
        self.assertEqual(code, 200)
        payload = json.loads(body)
        self.assertEqual(payload["task_id"], "task-reconcile")
        self.assertEqual(payload["operations"][0]["operation_id"], "op-1")
        self.assertEqual(payload["operations"][0]["status"], "unknown")

    def test_reconcile_requires_external_evidence_and_is_terminal(self):
        code, _ = _post(self.url + "/op-1/reconcile", {
            "status": "succeeded", "source": "remote", "evidence_ref": "op:1",
            "unexpected": True,
        })
        self.assertEqual(code, 400)
        self.assertEqual(self.store.pending_operations("task-reconcile")[0]["status"], "unknown")

        version = self.store.get("task-reconcile").state_version
        code, body = _post(self.url + "/op-1/reconcile", {
            "status": "succeeded", "source": "remote-api",
            "evidence_ref": "remote:operations/op-1", "result_ref": "artifact:op-1",
            "expected_version": version,
        })
        self.assertEqual(code, 200)
        payload = json.loads(body)
        self.assertEqual(payload["operation"]["status"], "succeeded")
        self.assertEqual(payload["operation"]["reconciliation"]["source"], "remote-api")
        self.assertEqual(self.store.pending_operations("task-reconcile"), [])

        code, _ = _post(self.url + "/op-1/reconcile", {
            "status": "failed", "source": "remote-api", "evidence_ref": "remote:op-1",
        })
        self.assertEqual(code, 409)

# ── P2-2/OPT-121：multi-agent 并行作答 + 主 Agent 汇总 ──


class FakeBranch:
    """注入 Serve.multi_providers 的假答者：文本/失败/延迟可控。"""

    def __init__(self, name, kind="external", text="答", fail=False, delay=0.0):
        self.name = name
        self.kind = kind
        self._text = text
        self._fail = fail
        self._delay = delay
        self.calls = 0
        self.closed = 0

    async def answer(self, question, signal=None):
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError("boom")
        return f"{self._text}:{question}"

    async def close(self):
        """P2-2/F5-015：BranchProvider 收尾钩子（Serve.aclose 统一调用）。"""
        self.closed += 1


class FakeSynth(LLMProvider):
    """汇总模型替身：记录调用次数，返回固定综合答复。"""

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools=None, **kw):
        self.calls += 1
        return LLMResponse(content="综合答复", tool_calls=[],
                           usage=TokenUsage(input_tokens=1, output_tokens=1))


class TestMultiAgent(unittest.TestCase):
    def test_approval_mode_env_override(self):
        """F5-021：审批策略可由环境变量下发（Ark 设置 → serve 启动时注入）。"""
        from unittest import mock

        def _boot(port):
            cfg = _tmp_cfg()
            cfg.approval_mode = "risk_based"
            factory = lambda: (lambda sink: FakeRunner(sink))  # noqa: E731
            s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory,
                      session_store=InMemorySessionStorage())
            httpd = s.start()
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            return s

        s1 = _boot(18761)
        try:
            self.assertEqual(s1.cfg.approval_mode, "risk_based", "无环境变量时用 config 值")
        finally:
            s1.shutdown()

        with mock.patch.dict(os.environ, {"AGENTLAB_APPROVAL_MODE": "allow_all"}):
            s2 = _boot(18762)
            try:
                self.assertEqual(s2.cfg.approval_mode, "allow_all")
            finally:
                s2.shutdown()

        # 非法值 fail-closed 回落 risk_based，不能因拼错而静默全放行
        with mock.patch.dict(os.environ, {"AGENTLAB_APPROVAL_MODE": "ALLOW"}):
            s3 = _boot(18763)
            try:
                self.assertEqual(s3.cfg.approval_mode, "risk_based")
            finally:
                s3.shutdown()

    def test_agents_endpoint_lists_builtin_and_external(self):
        """F5-019：GET /v1/agents 是 Ark @ 补全与名单校验的唯一数据源。"""
        from agentlab.runtime.config import AgentsConfig, Config, ExternalAgentConfig

        cfg = _tmp_cfg()
        cfg.agents = AgentsConfig(
            external=[ExternalAgentConfig(name="claude-code", command="claude-code-acp"),
                      ExternalAgentConfig(name="disabled", command="")],
            max_parallel_consults=2, consult_result_max_chars=4000)
        factory = lambda: (lambda sink: FakeRunner(sink))  # noqa: E731
        s = Serve(cfg, port=18757, host="127.0.0.1", build_factory=factory,
                  session_store=InMemorySessionStorage())
        httpd = s.start()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:18757/v1/agents",
                headers={"Authorization": "Bearer agentlab-dev"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
                body = json.loads(r.read().decode("utf-8"))
            names = [a["name"] for a in body["agents"]]
            self.assertEqual(names, ["agentlab", "claude-code"],
                             "内置恒在；command 为空的未启用项不得出现")
            self.assertEqual(body["agents"][0]["kind"], "internal")
            self.assertEqual(body["agents"][1]["kind"], "external")
            self.assertEqual(body["max_parallel_consults"], 2)
            self.assertEqual(body["consult_result_max_chars"], 4000)
            # 未鉴权 fail-closed（与 /v1/runs 同口径）
            code, _body = _get("http://127.0.0.1:18757/v1/agents")
            self.assertEqual(code, 401)
        finally:
            s.shutdown()

    def test_synth_prompt_carries_failure_face_and_three_questions(self):
        """F5-020：汇总模板必须显式给出失败面，并包含冲突/缺口/再派三问。"""
        from agentlab.runtime.multi import BranchOutcome, synthesize

        captured = {}

        class Synth:
            async def chat(self, messages, tools=None, **kw):
                captured["prompt"] = messages[0].content
                return LLMResponse(content="汇总", tool_calls=[],
                                   usage=TokenUsage(input_tokens=1, output_tokens=1))

        outcomes = [BranchOutcome(name="ext-ok", kind="external", ok=True, text="成功支正文"),
                    BranchOutcome(name="ext-bad", kind="external", ok=False, error="boom")]
        text = asyncio.run(synthesize(Synth(), "原始问题X", outcomes))
        self.assertEqual(text, "汇总")
        prompt = captured["prompt"]
        self.assertIn("ext-bad", prompt, "失败面必须点名未响应来源")
        self.assertIn("成功支正文", prompt)
        for kw in ("冲突", "缺口", "是否值得再派一轮"):
            self.assertIn(kw, prompt, f"三问缺项：{kw}")

    class _AllowAllApprovals:
        """默认放行的审批替身：编排类用例不测 HITL，避免门禁等待真实超时。"""

        async def confirm(self, *a, **kw):
            return True

    def _make(self, providers, synth=None, port=18741):
        cfg = _tmp_cfg()
        factory = lambda: (lambda sink: FakeRunner(sink))  # noqa: E731
        s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory,
                  session_store=InMemorySessionStorage(),
                  multi_providers=providers, synth_provider=synth)
        httpd = s.start()
        # 必须在 start() 之后注入：start 会按 serve 配置重建默认 ApprovalManager
        s.approvals = self._AllowAllApprovals()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return s, httpd, port

    def _make_with_cfg(self, providers, cfg, synth=None, port=18741):
        factory = lambda: (lambda sink: FakeRunner(sink))  # noqa: E731
        s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory,
                  session_store=InMemorySessionStorage(),
                  multi_providers=providers, synth_provider=synth)
        httpd = s.start()
        s.approvals = self._AllowAllApprovals()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return s, httpd, port

    @staticmethod
    def _events(body):
        out = []
        for chunk in body.split("\n\n"):
            chunk = chunk.strip()
            if chunk.startswith("data: ") and chunk != "data: [DONE]":
                out.append(json.loads(chunk[len("data: "):]))
        return out

    def _post_multi(self, port, multi, question="问题Q"):
        return _post(f"http://127.0.0.1:{port}/v1/responses",
                     {"input": [{"role": "user", "content": question}],
                      "multi": multi})

    def test_aclose_closes_providers_and_is_idempotent(self):
        """F5-015：Serve 收尾必须关掉长驻外部 agent，且重复调用不炸。"""
        a = FakeBranch("ext-a")
        b = FakeBranch("ext-b")
        s, _httpd, _port = self._make({"ext-a": a, "ext-b": b}, port=18749)
        try:
            asyncio.run(s.aclose())
            self.assertEqual((a.closed, b.closed), (1, 1))
            self.assertEqual(s.multi_registry, {}, "收尾后不应再持有 provider 引用")
            asyncio.run(s.aclose())  # 幂等：注册表已空，不抛错
            self.assertEqual((a.closed, b.closed), (1, 1), "重复收尾不得二次 close")
        finally:
            s.shutdown()

    def test_aclose_survives_provider_failure(self):
        """单个 provider 收尾失败不得阻断其余 provider 的收尾。"""
        class Boom(FakeBranch):
            async def close(self):
                raise RuntimeError("close boom")

        bad, good = Boom("bad"), FakeBranch("good")
        s, _httpd, _port = self._make({"bad": bad, "good": good}, port=18750)
        try:
            asyncio.run(s.aclose())
            self.assertEqual(good.closed, 1)
            self.assertEqual(s.multi_registry, {})
        finally:
            s.shutdown()

    def test_external_gate_denied_degrades_to_single_agent(self):
        """F5-018：审批拒绝 → 零外部调用，降级单 agent 直答并标 denied。"""
        class DenyApprovals:
            def __init__(self):
                self.calls = []

            async def confirm(self, cfg, tool, prompt, *, sink, run_id,
                              session_id, signal):
                self.calls.append((tool.name, tool.permission, prompt))
                return False

        a = FakeBranch("ext-a")
        approvals = DenyApprovals()
        s, _httpd, port = self._make({"ext-a": a}, port=18753)
        s.approvals = approvals
        try:
            status, body = self._post_multi(port, ["ext-a"])
            self.assertEqual(status, 200)
            evs = self._events(body)
            types = [e["type"] for e in evs]
            self.assertEqual(a.calls, 0, "拒绝后不得 spawn 外部支")
            self.assertNotIn("response.branch.started", types)
            ms = [e for e in evs if e["type"] == "response.multi.summary"][0]
            self.assertTrue(ms["denied"])
            self.assertEqual(ms["denied_agents"], ["ext-a"])
            self.assertIn("已检索", body, "拒绝后应降级为单 agent 直答")
            # 整轮只确认一次，且走 danger 语义
            self.assertEqual(len(approvals.calls), 1)
            self.assertEqual(approvals.calls[0][:2], ("multi_consult", "danger"))
        finally:
            s.shutdown()

    def test_external_gate_approved_then_spawns(self):
        class AllowApprovals:
            async def confirm(self, *a, **kw):
                return True

        a = FakeBranch("ext-a")
        s, _httpd, port = self._make({"ext-a": a}, port=18754)
        s.approvals = AllowApprovals()
        try:
            _status, body = self._post_multi(port, ["ext-a"])
            self.assertEqual(a.calls, 1)
            self.assertIn("response.branch.started", [e["type"] for e in self._events(body)])
        finally:
            s.shutdown()

    def test_allow_all_skips_card_but_still_gates(self):
        """allow_all：不弹卡，但仍走 serve_confirm 策略并留审批审计事件。"""
        from agentlab.runtime.approvals import ApprovalManager

        cfg = _tmp_cfg()
        cfg.approval_mode = "allow_all"
        a = FakeBranch("ext-a")
        s, _httpd, port = self._make_with_cfg({"ext-a": a}, cfg, port=18755)
        s.approvals = ApprovalManager(1.0)
        try:
            _status, body = self._post_multi(port, ["ext-a"])
            types = [e["type"] for e in self._events(body)]
            self.assertNotIn("approval.requested", types, "allow_all 不应弹卡")
            self.assertIn("approval.resolved", types, "应留下策略放行审计")
            self.assertEqual(a.calls, 1)
        finally:
            s.shutdown()

    def test_risk_based_asks_before_spawning(self):
        """risk_based：确认卡必须出现在分支发起之前；无人应答则超时拒绝且零调用。"""
        from agentlab.runtime.approvals import ApprovalManager

        a = FakeBranch("ext-a")
        s, _httpd, port = self._make({"ext-a": a}, port=18756)
        s.approvals = ApprovalManager(1.0)  # 最小 1s，无人工 resolve → 超时拒绝
        try:
            _status, body = self._post_multi(port, ["ext-a"])
            evs = self._events(body)
            self.assertIn("approval.requested", [e["type"] for e in evs])
            self.assertEqual(a.calls, 0, "卡未决前不得 spawn 外部支")
            ms = [e for e in evs if e["type"] == "response.multi.summary"][0]
            self.assertTrue(ms["denied"])
        finally:
            s.shutdown()

    def test_branch_cap_skips_excess_and_reports(self):
        """F5-017：超过 max_parallel_consults 的名单进 skipped，不静默丢弃也不执行。"""
        cfg = _tmp_cfg()
        cfg.agents.max_parallel_consults = 2
        a, b, c = (FakeBranch("a"), FakeBranch("b"), FakeBranch("c"))
        s, _httpd, port = self._make_with_cfg({"a": a, "b": b, "c": c}, cfg, port=18751)
        try:
            status, body = self._post_multi(port, ["a", "b", "c"])
            self.assertEqual(status, 200)
            evs = self._events(body)
            ms = [e for e in evs if e["type"] == "response.multi.summary"][0]
            self.assertEqual(ms["skipped"], ["c"], f"events={evs}")
            self.assertEqual((a.calls, b.calls, c.calls), (1, 1, 0),
                             "被 skip 的支不得发起调用")
        finally:
            s.shutdown()

    def test_long_branch_truncated_and_archived(self):
        """F5-017：超限正文截断 + 全文归档 session range，事件带 truncated/ref。"""
        from agentlab.memory.ranges import RangeArchive, RangeGateway

        cfg = _tmp_cfg()
        cfg.agents.consult_result_max_chars = 100
        long_text = "答" * 500
        s, _httpd, port = self._make_with_cfg(
            {"long": FakeBranch("long", text=long_text, delay=0.0)}, cfg, port=18752)
        tmp = tempfile.mkdtemp()
        s.range_gateway = RangeGateway(RangeArchive(tmp))
        try:
            status, body = _post(
                f"http://127.0.0.1:{port}/v1/responses",
                {"input": [{"role": "user", "content": "Q"}], "multi": ["long"],
                 "previous_response_id": "s-multi-1"})
            self.assertEqual(status, 200)
            evs = self._events(body)
            done = [e for e in evs if e["type"] == "response.branch.done"][0]
            self.assertEqual(done["chars"], 100, "SSE 正文必须已按上限截断")
            self.assertTrue(done["truncated"])
            self.assertTrue(done["ref"].startswith("session/"), done.get("ref"))
            # 归档的是**全文**（不是截断后的 head），且 jsonl 可读回
            sid, seq = done["ref"].split("#")
            archive = RangeArchive(tmp)
            rec = archive.read(sid.removeprefix("session/"), int(seq))
            self.assertEqual(len(rec), 1)
            self.assertEqual(rec[0].content, f"{long_text}:Q",
                             "归档必须是完整原文（含被截掉的尾部），而非 head")
            self.assertGreater(len(rec[0].content), len(done["text"]),
                               "归档内容必须长于 SSE 里截断后的正文")
        finally:
            s.shutdown()

    def test_two_branches_then_synth(self):
        synth = FakeSynth()
        s, httpd, port = self._make(
            {"ext-a": FakeBranch("ext-a", text="A答"),
             "ext-b": FakeBranch("ext-b", text="B答")}, synth=synth, port=18742)
        try:
            status, body = self._post_multi(port, ["ext-a", "ext-b"])
            self.assertEqual(status, 200)
            evs = self._events(body)
            types = [e["type"] for e in evs]
            self.assertEqual(types.count("response.branch.started"), 2)
            done = [e for e in evs if e["type"] == "response.branch.done"]
            self.assertEqual({e["agent"] for e in done}, {"ext-a", "ext-b"})
            self.assertIn("response.multi.summary", types)
            deltas = [e for e in evs if e["type"] == "response.output_text.delta"]
            self.assertTrue(any(e["delta"] == "综合答复" for e in deltas))
            summary = [e for e in evs if e["type"] == "response.completed"][0]["response"]["summary"]
            self.assertEqual(summary["stop_reason"], "multi-synth")
            self.assertEqual(summary["multi"]["ok"], 2)
            self.assertEqual(summary["multi"]["failed"], [])
            self.assertEqual(synth.calls, 1)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_one_branch_failure_excluded_from_synth(self):
        s, httpd, port = self._make(
            {"ext-a": FakeBranch("ext-a", text="A答"),
             "ext-b": FakeBranch("ext-b", fail=True)}, synth=FakeSynth(), port=18743)
        try:
            status, body = self._post_multi(port, ["ext-a", "ext-b"])
            self.assertEqual(status, 200)
            evs = self._events(body)
            failed = [e for e in evs if e["type"] == "response.branch.failed"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["agent"], "ext-b")
            self.assertIn("boom", failed[0]["error"])
            summary = [e for e in evs if e["type"] == "response.completed"][0]["response"]["summary"]
            self.assertEqual(summary["multi"]["failed"], ["ext-b"])
            self.assertEqual(summary["multi"]["ok"], 1)
            self.assertEqual(summary["stop_reason"], "multi-synth")
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_all_external_failed_degrades_to_single_agent(self):
        synth = FakeSynth()
        s, httpd, port = self._make(
            {"ext-a": FakeBranch("ext-a", fail=True),
             "ext-b": FakeBranch("ext-b", fail=True)}, synth=synth, port=18744)
        try:
            status, body = self._post_multi(port, ["ext-a", "ext-b"])
            self.assertEqual(status, 200)
            evs = self._events(body)
            ms = [e for e in evs if e["type"] == "response.multi.summary"][0]
            self.assertEqual(ms["ok_count"], 0)
            self.assertTrue(ms["degraded"])
            self.assertEqual(synth.calls, 0)  # 无成功分支不烧汇总调用
            summary = [e for e in evs if e["type"] == "response.completed"][0]["response"]["summary"]
            self.assertIn("问题Q", summary["final_output"])  # 单 agent（FakeRunner）直答
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_unknown_agent_degrades(self):
        s, httpd, port = self._make({}, port=18745)
        try:
            status, body = self._post_multi(port, ["nope"])
            self.assertEqual(status, 200)
            evs = self._events(body)
            failed = [e for e in evs if e["type"] == "response.branch.failed"]
            self.assertEqual(failed[0]["error"], "unknown agent: nope")
            self.assertIn("response.multi.summary", [e["type"] for e in evs])
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_no_synth_falls_back_to_first_ok_branch(self):
        s, httpd, port = self._make(
            {"ext-a": FakeBranch("ext-a", text="A答"),
             "ext-b": FakeBranch("ext-b", text="B答")}, synth=None, port=18746)
        try:
            status, body = self._post_multi(port, ["ext-a", "ext-b"])
            summary = [e for e in self._events(body) if e["type"] == "response.completed"][0]["response"]["summary"]
            self.assertEqual(summary["stop_reason"], "multi-fallback")
            self.assertTrue(summary["final_output"].startswith("A答:"))
            self.assertFalse(summary["multi"]["synthesized"])
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_agentlab_in_multi_runs_internal_branch(self):
        # @agentlab + 外部：本地 agent（FakeRunner）也作为一支参与，答案进 branch.done
        s, httpd, port = self._make(
            {"ext-a": FakeBranch("ext-a", text="A答")}, synth=FakeSynth(), port=18747)
        try:
            status, body = self._post_multi(port, ["agentlab", "ext-a"])
            self.assertEqual(status, 200)
            evs = self._events(body)
            kinds = {e["agent"]: e["kind"]
                     for e in evs if e["type"] == "response.branch.done"}
            self.assertEqual(kinds.get("agentlab"), "internal")
            self.assertEqual(kinds.get("ext-a"), "external")
            summary = [e for e in evs if e["type"] == "response.completed"][0]["response"]["summary"]
            self.assertEqual(summary["multi"]["ok"], 2)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_only_agentlab_equals_single_agent(self):
        # 只 @ 本地 agent：无分支事件、无汇总事件，走原单 agent 语义
        s, httpd, port = self._make({}, port=18748)
        try:
            status, body = self._post_multi(port, ["agentlab"])
            self.assertEqual(status, 200)
            types = [e["type"] for e in self._events(body)]
            self.assertNotIn("response.branch.started", types)
            self.assertNotIn("response.multi.summary", types)
            self.assertIn("response.completed", types)
        finally:
            httpd.shutdown(); httpd.server_close()


class TestMemoryInjection(unittest.TestCase):
    def test_run_agent_injects_memory_block_and_policy(self):
        # #10①/OPT-123：召回块 + 记忆政策应进入 system instructions（patch 掉真实 brain 仓库）
        from unittest.mock import patch

        captured = {}

        class CapRunner(FakeRunner):
            async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
                captured["instructions"] = agent.instructions
                return await super().run(agent, user_input, ctx=ctx, cfg=cfg, hooks=hooks)

        cfg = _tmp_cfg()
        port = 18749
        factory = lambda: (lambda sink: CapRunner(sink))  # noqa: E731
        s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory,
                  session_store=InMemorySessionStorage())
        httpd = s.start()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with patch("agentlab.memory.recall.memory_block_for",
                       return_value="- [pref] 记住用户喜欢深色主题"):
                code, body = _post(f"http://127.0.0.1:{port}/v1/responses",
                                   {"input": [{"role": "user", "content": "q"}]})
            self.assertEqual(code, 200)
            self.assertIn("记住用户喜欢深色主题", captured["instructions"])
            self.assertIn("记忆政策", captured["instructions"])
            self.assertNotIn("{{memory}}", captured["instructions"])  # 占位符必须被渲染
        finally:
            httpd.shutdown(); httpd.server_close()


class TestRunsEndpoint(unittest.TestCase):
    """#6/OPT-126：GET /v1/runs 最近 run 概览。"""

    def _serve_with_trace(self, tmp_trace, port=18751):
        cfg = _tmp_cfg()
        cfg.trace_dir = tmp_trace
        factory = lambda: (lambda sink: FakeRunner(sink))  # noqa: E731
        s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory,
                  session_store=InMemorySessionStorage())
        httpd = s.start()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return s, httpd, port

    @staticmethod
    def _get_auth(url, token="agentlab-dev"):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    def test_runs_lists_serve_runs_with_summary_fields(self):
        import tempfile

        from agentlab.runtime.trace import Tracer

        tmp = tempfile.mkdtemp()
        tr = Tracer(tmp)
        tr.new_session()
        tr.record_run(input="历史run", stop_reason="done", tokens=7)
        s, httpd, port = self._serve_with_trace(tmp)
        try:
            code, _ = _post(f"http://127.0.0.1:{port}/v1/responses",
                            {"input": [{"role": "user", "content": "hi"}]})
            self.assertEqual(code, 200)
            code, body = self._get_auth(f"http://127.0.0.1:{port}/v1/runs?limit=10")
            self.assertEqual(code, 200)
            data = json.loads(body)
            self.assertTrue(data["ok"])
            by_input = {r["input"]: r for r in data["runs"]}
            self.assertIn("hi", by_input)          # 刚才那次 serve run
            self.assertIn("历史run", by_input)      # 预置的 run 行
            mine = by_input["hi"]
            self.assertEqual(mine["stop_reason"], "done")
            self.assertEqual(mine["tokens"], 30)   # FakeRunner usage 10+20
            self.assertIn("trace_id", mine)
            self.assertIn("time", mine)
            self.assertIn("steps", mine)
            self.assertEqual(mine["steps"], 2)
            self.assertEqual(mine["llm_calls"], 2)
            self.assertEqual(mine["tool_calls"], 1)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_runs_unauthorized(self):
        import tempfile

        s, httpd, port = self._serve_with_trace(tempfile.mkdtemp(), port=18752)
        try:
            code, _ = self._get_auth(f"http://127.0.0.1:{port}/v1/runs", token="wrong")
            self.assertEqual(code, 401)
        finally:
            httpd.shutdown(); httpd.server_close()


class TestDeleteSessionEndpoint(unittest.TestCase):
    """#8/OPT-127：DELETE /v1/sessions/{sid} 会话删除（连同 ranges/索引清理）。"""

    def _serve(self, port=18753):
        cfg = _tmp_cfg()
        factory = lambda: (lambda sink: FakeRunner(sink))  # noqa: E731
        s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory,
                  session_store=InMemorySessionStorage())
        httpd = s.start()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return s, httpd, port

    @staticmethod
    def _delete(url, token="agentlab-dev"):
        req = urllib.request.Request(url, method="DELETE",
                                     headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    def test_delete_removes_session_history(self):
        s, httpd, port = self._serve()
        try:
            # 直接 seed（FakeRunner 的 messages 为空，POST 不产生持久化增量）
            s.session_store.append("sess-del-1", Message(role="user", content="x"))
            self.assertTrue(s.session_store.read_all("sess-del-1"))
            code, body = self._delete(f"http://127.0.0.1:{port}/v1/sessions/sess-del-1")
            self.assertEqual(code, 200)
            self.assertTrue(json.loads(body)["deleted"]["session"])
            self.assertEqual(s.session_store.read_all("sess-del-1"), [])
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_delete_invalid_sid_400(self):
        s, httpd, port = self._serve(port=18754)
        try:
            code, _ = self._delete(f"http://127.0.0.1:{port}/v1/sessions/bad%20sid")
            self.assertEqual(code, 400)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_delete_unauthorized(self):
        s, httpd, port = self._serve(port=18755)
        try:
            code, _ = self._delete(f"http://127.0.0.1:{port}/v1/sessions/abc",
                                   token="wrong")
            self.assertEqual(code, 401)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_jsonl_storage_delete_removes_files(self):
        import tempfile
        from pathlib import Path

        d = tempfile.mkdtemp()
        st = JsonlSessionStorage(d)
        st.append("s1", Message(role="user", content="x"))
        self.assertTrue(st.delete("s1"))
        self.assertFalse(st.delete("s1"))
        self.assertEqual(list(Path(d).glob("*s1*")), [])


class TestServeTraceP102(unittest.TestCase):
    """OPT-216 P1-02：run trace 补 duration_ms/model/error_code，tool start 记参数指纹。"""

    def _serve(self, factory, port, trace_dir):
        cfg = _tmp_cfg()
        cfg.trace_dir = trace_dir
        s = Serve(cfg, port=port, host="127.0.0.1", build_factory=factory,
                  session_store=InMemorySessionStorage())
        httpd = s.start()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return s, httpd

    def _read_events(self, trace_dir):
        files = os.listdir(trace_dir)
        assert files, "trace 目录为空"
        with open(os.path.join(trace_dir, files[0]), encoding="utf-8") as f:
            return [json.loads(ln) for ln in f.read().splitlines() if ln.strip()]

    def test_run_trace_has_duration_model_and_tool_args_hash(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        s, httpd = self._serve(lambda: (lambda sink: FakeRunner(sink)), 18781, tmp)
        try:
            time.sleep(0.15)
            code, _ = _post(f"http://127.0.0.1:18781/v1/responses",
                            {"input": [{"role": "user", "content": "查一下"}]})
            self.assertEqual(code, 200)
            events = self._read_events(tmp)
            run = next(e for e in events if e.get("type") == "run")
            self.assertIn("duration_ms", run)
            self.assertIsInstance(run["duration_ms"], int)
            self.assertIn("model", run)
            self.assertEqual(run["stop_reason"], "done")
            start = next(e for e in events
                         if e.get("type") == "tool" and e.get("phase") == "start")
            self.assertEqual(start["arguments_hash"],
                             hashlib.sha1('{"query": "测试"}'.encode("utf-8")).hexdigest()[:12])
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_run_trace_error_code_on_failure(self):
        import tempfile

        from agentlab.core.errors import AgentError

        class FailingRunner:
            def __init__(self, sink):
                self.sink = sink
                self.registry = _FakeRegistry()

            def on(self, ev, cb):
                pass

            async def run(self, agent, user_input, ctx=None, cfg=None, hooks=None):
                raise AgentError("AGENT_TOOL_PERMISSION", "拒绝执行 vault_write")

        tmp = tempfile.mkdtemp()
        s, httpd = self._serve(lambda: (lambda sink: FailingRunner(sink)), 18782, tmp)
        try:
            time.sleep(0.15)
            try:
                _post(f"http://127.0.0.1:18782/v1/responses",
                      {"input": [{"role": "user", "content": "写一下"}]})
            except urllib.error.HTTPError:
                pass  # 500/错误响应均可——重点是 trace 留 error_code
            events = self._read_events(tmp)
            run = next(e for e in events if e.get("type") == "run")
            self.assertEqual(run["stop_reason"], "error")
            self.assertEqual(run["error_code"], "AGENT_TOOL_PERMISSION")
            self.assertIn("duration_ms", run)
        finally:
            httpd.shutdown(); httpd.server_close()


class TestVisualToolGating(unittest.TestCase):
    """OPT-219：视觉工具按需启用——用户明确提出视觉意图才进工具面。"""

    def test_no_visual_intent_hides_visual_tools(self):
        from agentlab.runtime.serve import _visual_tool_filter
        f = _visual_tool_filter("BV1AdokBNENj，生成笔记")
        self.assertIsNotNone(f)

        class T:
            def __init__(self, name):
                self.name = name

        self.assertFalse(f(T("bili_visual")))
        self.assertFalse(f(T("bili_screenshot")))
        self.assertTrue(f(T("bili_transcribe")))
        self.assertTrue(f(T("vault_search")))

    def test_visual_intent_enables_visual_tools(self):
        from agentlab.runtime.serve import _visual_tool_filter
        for text in ("BV1xx 做视觉分析", "帮我截图关键画面", "截几张图", "Do a visual analysis"):
            self.assertIsNone(_visual_tool_filter(text), f"应识别视觉意图: {text}")

    def test_empty_input_hides_visual_tools(self):
        from agentlab.runtime.serve import _visual_tool_filter
        self.assertIsNotNone(_visual_tool_filter(""))
        self.assertIsNotNone(_visual_tool_filter(None))
