"""P2-1/OPT-112 ACP 客户端单测：与假 agent 进程（stdin/stdout JSON-RPC）真实往返。

覆盖：initialize/session 往返与正文收集、tool_call 进度透传、非 JSON banner 容错、
反向请求 fail-closed（fs 写拒绝 + 权限自动 reject）、超时、启动即崩、契约不符、
ExternalAgent 崩溃自愈与跨事件循环自愈、工具注册与降级。
不依赖任何真实外部 CLI agent。
"""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from agentlab.connectors.acp_agent import ExternalAgent
from agentlab.connectors.acp_client import AcpClient, AcpError
from agentlab.runtime.config import AgentsConfig, Config, ExternalAgentConfig
from agentlab.tools.acp_tools import build_acp_tools

PY = sys.executable

# 假 ACP agent：按 argv[1] 模式行为分叉；断言收到的反向响应不符即崩溃（测试即失败）
FAKE_AGENT = r'''
import json, os, sys, time

def send(msg):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def recv():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)

mode = sys.argv[1] if len(sys.argv) > 1 else "ok"

def note(kind, **kw):
    """trace 模式：把协议里程碑追加写入 argv[2] 指定的 JSONL，供并发断言。"""
    if mode == "trace":
        with open(sys.argv[2], "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "t": time.time(), **kw},
                               ensure_ascii=False) + "\n")

if mode == "banner":
    sys.stdout.write("== 欢迎横幅（非 JSON 噪声行）==\n")
    sys.stdout.flush()
if mode == "die":
    os._exit(1)  # 启动即崩：initialize 无响应
while True:
    msg = recv()
    m, rid = msg.get("method"), msg.get("id")
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result":
              {"protocolVersion": 1, "agentCapabilities": {}, "authMethods": []}})
    elif m == "session/new":
        note("new_session")
        if mode == "bad_session":
            send({"jsonrpc": "2.0", "id": rid, "result": {"nope": True}})
        else:
            send({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": "sess-1"}})
    elif m == "session/prompt":
        if mode == "trace":
            # 回显问题并留出重叠窗口：并发未被串行化时 prompt_start 会交错出现
            q = "".join(p.get("text", "") for p in (msg.get("params") or {}).get("prompt") or [])
            note("prompt_start", q=q)
            time.sleep(0.2)
            note("prompt_end", q=q)
            send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "sess-1", "update": {"sessionUpdate": "agent_message_chunk",
                                                  "content": {"type": "text", "text": q}}}})
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
        elif mode == "ok":
            send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "sess-1", "update": {"sessionUpdate": "agent_message_chunk",
                                                  "content": {"type": "text", "text": "你好"}}}})
            send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "sess-1", "update": {"sessionUpdate": "tool_call",
                                                  "title": "搜索", "kind": "search"}}})
            send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "sess-1", "update": {"sessionUpdate": "agent_message_chunk",
                                                  "content": {"type": "text", "text": "，世界"}}}})
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
        elif mode == "fs_request":
            # 反向请求 fs 写文件：客户端必须 fail-closed 拒绝（协议错误响应）
            send({"jsonrpc": "2.0", "id": 900, "method": "fs/write_text_file",
                  "params": {"path": "evil.txt", "content": "x"}})
            reply = recv()
            assert reply.get("id") == 900 and "error" in reply, reply
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
        elif mode == "permission":
            # 权限请求：客户端必须在有 reject 项时自动选 reject
            send({"jsonrpc": "2.0", "id": 901, "method": "session/request_permission",
                  "params": {"options": [
                      {"optionId": "allow", "name": "允许", "kind": "allow_once"},
                      {"optionId": "reject", "name": "拒绝", "kind": "reject_once"}]}})
            reply = recv()
            assert reply.get("id") == 901 and \
                reply["result"]["outcome"]["optionId"] == "reject", reply
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
        elif mode == "banner":
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
        elif mode == "silent":
            time.sleep(60)  # 等客户端超时
        else:
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
'''


class _FakeAgent:
    def __init__(self, tmp: Path, mode: str):
        self.script = tmp / f"fake_agent_{mode}.py"
        self.script.write_text(FAKE_AGENT, encoding="utf-8")
        self.mode = mode

    def client(self, timeout: float = 15.0, on_update=None) -> AcpClient:
        return AcpClient(PY, [str(self.script), self.mode], timeout=timeout,
                         env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                         on_update=on_update)

    def external(self, timeout: float = 15.0) -> ExternalAgent:
        return ExternalAgent("fake", PY, args=[str(self.script), self.mode],
                             env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                             timeout=timeout)


class TestAcpClient(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = _FakeAgent(Path(self.tmp.name), "ok")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_collects_text_and_tool_calls(self):
        async def go():
            updates = []
            c = self.fake.client(on_update=updates.append)
            try:
                await c.start()
                self.assertTrue(c.is_alive)
                sid = await c.new_session()
                self.assertEqual(sid, "sess-1")
                r = await c.prompt(sid, "打招呼")
                return r, updates, c
            finally:
                await c.close()

        r, updates, _ = asyncio.run(go())
        self.assertEqual(r["text"], "你好，世界", "分片消息应拼接为完整正文")
        self.assertEqual(r["stopReason"], "end_turn")
        self.assertEqual(len(r["tool_calls"]), 1)
        self.assertEqual(len(updates), 3, "on_update 透传全部 session/update")

    def test_banner_line_tolerated(self):
        async def go():
            c = _FakeAgent(Path(self.tmp.name), "banner").client()
            try:
                await c.start()
                sid = await c.new_session()
                r = await c.prompt(sid, "问")
                return r
            finally:
                await c.close()

        r = asyncio.run(go())
        self.assertEqual(r["text"], "", "横幅行被忽略，正文为空不炸")
        self.assertEqual(r["stopReason"], "end_turn")

    def test_fs_write_reverse_request_denied(self):
        async def go():
            c = _FakeAgent(Path(self.tmp.name), "fs_request").client()
            try:
                await c.start()
                sid = await c.new_session()
                return await c.prompt(sid, "写个文件")
            finally:
                await c.close()

        # 假 agent 断言"收到的是错误响应"，断言不过会崩 → prompt 抛 AcpError → 本测失败
        r = asyncio.run(go())
        self.assertEqual(r["stopReason"], "end_turn")

    def test_permission_request_auto_reject(self):
        async def go():
            c = _FakeAgent(Path(self.tmp.name), "permission").client()
            try:
                await c.start()
                sid = await c.new_session()
                return await c.prompt(sid, "要权限")
            finally:
                await c.close()

        r = asyncio.run(go())
        self.assertEqual(r["stopReason"], "end_turn")

    def test_prompt_timeout_raises(self):
        async def go():
            c = _FakeAgent(Path(self.tmp.name), "silent").client(timeout=1.5)
            try:
                await c.start()
                sid = await c.new_session()
                return await c.prompt(sid, "装死")
            finally:
                await c.close()

        with self.assertRaises(AcpError):
            asyncio.run(go())

    def test_start_crash_raises_acp_error(self):
        async def go():
            c = _FakeAgent(Path(self.tmp.name), "die").client()
            return await c.start()

        with self.assertRaises(AcpError):
            asyncio.run(go())

    def test_bad_session_contract(self):
        async def go():
            c = _FakeAgent(Path(self.tmp.name), "bad_session").client()
            try:
                await c.start()
                return await c.new_session()
            finally:
                await c.close()

        with self.assertRaises(AcpError):
            asyncio.run(go())

    def test_start_command_missing_raises(self):
        async def go():
            c = AcpClient("definitely-not-exist-cmd-xyz", timeout=5.0)
            return await c.start()

        with self.assertRaises(AcpError):
            asyncio.run(go())

    def test_resolve_spawn_windows_cmd_shim(self):
        # Windows：npm 全局包只有 .cmd shim → which 定位后必须经 cmd /c 起进程
        from agentlab.connectors import acp_client as m
        c = AcpClient("some-adapter", ["--flag"])
        orig_os, orig_which = m.os.name, m.shutil.which
        m.os.name, m.shutil.which = "nt", lambda s: "C:/npm/some-adapter.cmd"
        try:
            exe, args = c._resolve_spawn()
        finally:
            m.os.name, m.shutil.which = orig_os, orig_which
        self.assertEqual(exe, "cmd")
        self.assertEqual(args[:2], ["/c", "C:/npm/some-adapter.cmd"])
        self.assertIn("--flag", args)

    def test_resolve_spawn_passthrough(self):
        # 非 Windows / 显式带扩展名路径 / which 落空 → 原样透传
        from agentlab.connectors import acp_client as m
        c = AcpClient(PY, ["-c", "1"])
        orig_os, orig_which = m.os.name, m.shutil.which
        m.os.name, m.shutil.which = "nt", (lambda s: None)
        try:
            exe, args = c._resolve_spawn()
        finally:
            m.os.name, m.shutil.which = orig_os, orig_which
        self.assertEqual((exe, args), (PY, ["-c", "1"]))


class TestExternalAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = _FakeAgent(Path(self.tmp.name), "ok")

    def tearDown(self):
        self.tmp.cleanup()

    def test_consult_and_crash_self_heal(self):
        ext = self.fake.external()

        async def first():
            return await ext.consult("第一问")

        try:
            self.assertEqual(asyncio.run(first()), "你好，世界")
            # 模拟进程被外力杀死 → 下次 consult 自动重建成功
            proc = ext._client._proc
            # asyncio.subprocess.Process.wait() 是 coroutine 且绑定旧事件循环；
            # 此处在同步测试体中通过底层 Popen 杀死并回收，避免未 await 警告。
            raw = getattr(getattr(proc, "_transport", None), "_proc", None)
            self.assertIsNotNone(raw)
            raw.kill()
            raw.wait(timeout=5)

            async def second():
                return await ext.consult("第二问")

            self.assertEqual(asyncio.run(second()), "你好，世界")
        finally:
            asyncio.run(ext.close())

    def test_consult_across_event_loops(self):
        ext = self.fake.external()
        try:
            r1 = asyncio.run(ext.consult("一"))
            old_raw = ext._client._raw_proc
            r2 = asyncio.run(ext.consult("二"))
            self.assertIsNotNone(old_raw)
            self.assertIsNotNone(old_raw.poll(), f"old ACP process still alive: {old_raw.pid}")
            self.assertEqual((r1, r2), ("你好，世界", "你好，世界"),
                             "跨 asyncio.run（CLI repl 形态）应检测旧 loop 失效并重建")
        finally:
            asyncio.run(ext.close())

    def test_concurrent_consults_serialize_same_agent(self):
        """P2-2/F5-015：同一 agent 名并发 consult 必须排队串行。

        反例后果（未加锁时）：① `_ensure_started` 竞态 → 同一实例 spawn 两个子进程，
        泄漏一个；② `new_session` 竞态 → 两个问题落进同一 ACP session，会话态跨问泄漏。
        """
        trace = Path(self.tmp.name) / "trace.jsonl"
        ext = ExternalAgent(
            "fake", PY, args=[str(self.fake.script), "trace", str(trace)],
            env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}, timeout=15.0)

        async def go():
            try:
                return await asyncio.gather(ext.consult("问题A"), ext.consult("问题B"))
            finally:
                await ext.close()

        r1, r2 = asyncio.run(go())
        # ① 答案不串：回显问题保证归因可验证
        self.assertEqual(r1, "问题A")
        self.assertEqual(r2, "问题B")

        events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
        kinds = [e["kind"] for e in events]
        # ② 只创建一个会话（未加锁时会 spawn 两次 / 建两个 session）
        self.assertEqual(kinds.count("new_session"), 1, f"events={events}")
        # ③ 两次提问不重叠：prompt_start/end 严格成对交替
        self.assertEqual(kinds, ["new_session", "prompt_start", "prompt_end",
                                 "prompt_start", "prompt_end"], f"events={events}")

    def test_lock_rebuilt_across_event_loops(self):
        """换事件循环后锁必须重建，否则 asyncio.Lock 会因绑定死循环而报错。"""
        ext = self.fake.external()
        try:
            asyncio.run(ext.consult("一"))
            first_lock = ext._lock
            first_loop = ext._lock_loop
            asyncio.run(ext.consult("二"))
            self.assertIsNotNone(ext._lock)
            self.assertIsNot(ext._lock, first_lock, "跨 loop 应重建锁对象")
            self.assertIsNot(ext._lock_loop, first_loop)
        finally:
            asyncio.run(ext.close())


class TestAcpTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = _FakeAgent(Path(self.tmp.name), "ok")
        self._agents = []

    def tearDown(self):
        async def _close_all():
            for a in self._agents:
                await a.close()

        asyncio.run(_close_all())  # 先收子进程（其 cwd 占着 tmp，Windows 删不掉）
        self.tmp.cleanup()

    def _factory(self):
        def factory(spec, cwd):
            ext = ExternalAgent(spec.name, spec.command, args=spec.args, cwd=cwd,
                                env=spec.env, timeout=spec.timeout)
            self._agents.append(ext)
            return ext
        return factory

    def test_empty_config_returns_no_tools(self):
        self.assertEqual(build_acp_tools(AgentsConfig(), vault_root="."), [])
        self.assertEqual(build_acp_tools(None), [])

    def test_tool_registered_and_consult_roundtrip(self):
        cfg = AgentsConfig(external=[ExternalAgentConfig(
            name="fake", command=PY, args=[str(self.fake.script), "ok"],
            env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            timeout=15.0)])
        tools = build_acp_tools(cfg, vault_root=self.tmp.name,
                                factory=self._factory())
        self.assertEqual([t.name for t in tools], ["agent_consult"])
        t = tools[0]
        self.assertEqual(t.permission, "danger", "spawn 外部 agent 默认 HITL")
        out = asyncio.run(t.fn(agent="fake", question="咨询"))
        self.assertEqual(out, "你好，世界")

    def test_consult_close_releases_windows_temp_cwd(self):
        """Closing a real ACP child must release its cwd before TempDir cleanup."""
        cfg = AgentsConfig(external=[ExternalAgentConfig(
            name="fake", command=PY, args=[str(self.fake.script), "ok"],
            env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            timeout=15.0)])
        tools = build_acp_tools(cfg, vault_root=self.tmp.name,
                                factory=self._factory())
        self.assertEqual(asyncio.run(tools[0].fn(agent="fake", question="cleanup")), "你好，世界")
        # tearDown performs the actual TemporaryDirectory cleanup. This test
        # exists to keep the lifecycle contract explicit under Windows.

    def test_unknown_agent_reports_available_names(self):
        cfg = AgentsConfig(external=[ExternalAgentConfig(
            name="fake", command=PY, args=[str(self.fake.script), "ok"])])
        t = build_acp_tools(cfg, vault_root=self.tmp.name,
                            factory=self._factory())[0]
        out = asyncio.run(t.fn(agent="不存在的", question="x"))
        self.assertIn("[未配置的外部 agent]", out)
        self.assertIn("fake", out)

    def test_agent_down_wrapped_as_readable_result(self):
        cfg = AgentsConfig(external=[ExternalAgentConfig(
            name="dead", command=PY, args=[str(self.fake.script), "die"],
            timeout=5.0)])
        t = build_acp_tools(cfg, vault_root=self.tmp.name,
                            factory=self._factory())[0]
        out = asyncio.run(t.fn(agent="dead", question="x"))
        self.assertTrue(out.startswith("[外部 agent 不可用]"), out)


class TestAgentsConfigWiring(unittest.TestCase):
    def test_default_config_has_no_external_agents(self):
        cfg = Config()
        self.assertEqual(cfg.agents.external, [], "默认零外部 agent，行为不变")

    def test_config_parses_agents_section(self):
        cfg = Config.model_validate({"agents": {"external": [
            {"name": "cc", "command": "claude-code-acp", "env": {"X": "1"}}]}})
        self.assertEqual(cfg.agents.external[0].name, "cc")
        self.assertEqual(cfg.agents.external[0].env["X"], "1")

    def test_multi_limits_defaults_and_fail_closed(self):
        """F5-017：并发上限与结果限额有默认值，非法值 fail-closed 回落默认。"""
        cfg = Config()
        self.assertEqual(cfg.agents.max_parallel_consults, 3)
        self.assertEqual(cfg.agents.consult_result_max_chars, 6000)
        parsed = Config.model_validate({"agents": {
            "max_parallel_consults": 5, "consult_result_max_chars": 9000}})
        self.assertEqual(parsed.agents.max_parallel_consults, 5)
        self.assertEqual(parsed.agents.consult_result_max_chars, 9000)
        # 0 / 负数会让"全部 skipped"或"无上限"，一律回落默认而不是放行
        bad = Config.model_validate({"agents": {
            "max_parallel_consults": 0, "consult_result_max_chars": -1}})
        self.assertEqual(bad.agents.max_parallel_consults, 3)
        self.assertEqual(bad.agents.consult_result_max_chars, 6000)



if __name__ == "__main__":
    unittest.main()
