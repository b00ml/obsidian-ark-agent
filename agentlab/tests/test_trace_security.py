"""trace 脱敏 + web_search 解析 + 权限分权 + raw 守卫的单元测试。"""
import os
import shutil
import tempfile
import unittest

from agentlab.core.errors import AgentError
from agentlab.runtime.trace import Tracer, load_run_detail, load_run_summaries
from agentlab.tools.web_search import _parse_results

RESULT_HTML = (
    '<a rel="nofollow" class="result__a" href="https://ex.com/a">Alpha</a>'
    '<a class="result__snippet" href="https://ex.com/a">第一段 <b>加粗</b> 摘要</a>'
    '<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2Fb">Beta</a>'
    '<a class="result__snippet">第二段摘要</a>'
)


class TestTrace(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_write_and_redact(self):
        t = Tracer(self.dir)
        t.new_session()
        t.record({"headers": {"Authorization": "Bearer sk-secret123", "X-Custom": 1}})
        t.record({"type": "llm", "content": "[STATE] DONE ok"})
        fp = os.path.join(self.dir, f"{t.trace_id}.jsonl")
        with open(fp, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertNotIn("sk-secret123", lines[0])
        self.assertIn("[REDACTED]", lines[0])
        self.assertIn("DONE", lines[1])

    def test_run_diagnostics_are_correlated_and_redacted(self):
        t = Tracer(self.dir)
        t.new_session()
        t.record_tool("vault_search", "end", result="secret body")
        t.record_run(input="查资料", run_id="req-1", session_id="sess-1",
                     project_id="proj-1", stop_reason="done", tokens=9)
        rows = load_run_summaries(self.dir, 5)
        self.assertEqual(rows[0]["run_id"], "req-1")
        self.assertEqual(rows[0]["session_id"], "sess-1")
        detail = load_run_detail(self.dir, t.trace_id)
        self.assertEqual(detail["run"]["project_id"], "proj-1")
        self.assertEqual(detail["events"][0]["result_chars"], len("secret body"))
        self.assertNotIn("secret body", str(detail))

    def test_run_detail_rejects_path_traversal(self):
        self.assertIsNone(load_run_detail(self.dir, "../secret"))


class TestWebSearchParse(unittest.TestCase):
    def test_parse_results(self):
        res = _parse_results(RESULT_HTML, limit=5)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["title"], "Alpha")
        self.assertEqual(res[0]["snippet"], "第一段 加粗 摘要")
        # uddg 跳转还原直链
        self.assertEqual(res[1]["url"], "https://ex.com/b")

    def test_limit(self):
        self.assertEqual(len(_parse_results(RESULT_HTML, limit=1)), 1)


class TestPermissionConfirm(unittest.TestCase):
    def test_write_confirm_policy(self):
        from agentlab.tools.base import tool

        @tool(name="fake_write", description="测试", permission="write")
        def f(path: str):
            return path

        calls = {"n": 0}

        def confirm(t, prompt):
            calls["n"] += 1
            return True

        from agentlab.core.message import ToolCall, ToolCallFunction
        from agentlab.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register(f)
        import asyncio
        r = asyncio.run(reg.execute(
            ToolCall(id="1", type="function",
                     function=ToolCallFunction(name="fake_write", arguments='{"path":"Inbox/a.md"}')),
            confirm=confirm,
        ))
        self.assertEqual(calls["n"], 1)


class TestRawGuard(unittest.TestCase):
    def test_brain_write_wrap_rejects_raw(self):
        from agentlab.tools.connectors.brain_tools import _wrap

        def fake_write(config, path, content="..."):
            return {"ok": True}

        wrapped = _wrap(fake_write, {}, "write")
        with self.assertRaises(AgentError) as ctx:
            wrapped(path="raw/immutable.md")
        self.assertEqual(ctx.exception.code, "AGENT_TOOL_PERMISSION")
        # 正常路径放行
        self.assertEqual(wrapped(path="Inbox/a.md", content="x"), {"ok": True})


if __name__ == "__main__":
    unittest.main()
