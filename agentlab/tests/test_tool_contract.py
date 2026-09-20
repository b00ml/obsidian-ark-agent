"""OPT-212 工具契约测试：side_effects/idempotent 必须显式声明且被装配链消费。

- validate_contract：write/danger 工具缺声明 → 违规清单非空（注册/测试期失败，不静默默认）
- build_brain_tools：从 BrainToolSpec 映射契约字段，装配后全量校验通过
- registry._authorize：write 工具无 confirm 拒绝（fail-closed），read 不误拦
"""
import asyncio
import tempfile
import unittest
from pathlib import Path

from agentlab.core.errors import AgentError
from agentlab.tools.base import Tool, tool, validate_contract
from agentlab.tools.registry import ToolRegistry


def _mk_tool(name: str, permission: str, side_effects=None, idempotent=None) -> Tool:
    def fn(path: str) -> str:  # noqa
        return path

    return Tool(name=name, description="t", permission=permission,  # type: ignore[arg-type]
                side_effects=side_effects, idempotent=idempotent,
                schema={}, fn=fn)


class TestValidateContract(unittest.TestCase):
    def test_write_tool_missing_declarations_rejected(self):
        errs = validate_contract([_mk_tool("w1", "write")])
        self.assertEqual(len(errs), 2)
        self.assertTrue(any("side_effects" in e for e in errs))
        self.assertTrue(any("idempotent" in e for e in errs))

    def test_danger_tool_missing_declarations_rejected(self):
        errs = validate_contract([_mk_tool("d1", "danger")])
        self.assertEqual(len(errs), 2)

    def test_explicit_declarations_pass(self):
        self.assertEqual(validate_contract([
            _mk_tool("w2", "write", side_effects="write", idempotent=True),
            _mk_tool("r1", "read"),
        ]), [])

    def test_side_effects_value_controlled(self):
        errs = validate_contract([_mk_tool("x", "read", side_effects="explode")])
        self.assertEqual(len(errs), 1)
        self.assertIn("不在", errs[0])


class TestBrainToolsContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agentlab.tools.connectors.brain_tools import build_brain_tools, brain_available
        if not brain_available():
            raise unittest.SkipTest("brain 目录不存在")
        cls.tmp = tempfile.TemporaryDirectory()
        Path(cls.tmp.name, "Inbox").mkdir()
        cls.tools = build_brain_tools(cls.tmp.name, {})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_tools_have_complete_contract(self):
        errs = validate_contract(self.tools)
        self.assertEqual(errs, [], f"brain 工具契约不完整: {errs}")

    def test_write_tools_declared_as_expected(self):
        by_name = {t.name: t for t in self.tools}
        self.assertEqual(by_name["vault_write"].permission, "write")
        self.assertEqual(by_name["vault_write"].side_effects, "write")
        self.assertTrue(by_name["vault_write"].idempotent)  # 同内容重写状态收敛
        self.assertFalse(by_name["memory_commit"].idempotent)  # 重复沉淀会产生新条目
        self.assertFalse(by_name["bili_visual"].idempotent)

    def test_read_tools_with_hidden_side_effects_are_marked(self):
        # Index rebuild and inbox collection both mutate local state and must
        # consume the runtime confirmation gate.
        by_name = {t.name: t for t in self.tools}
        self.assertEqual(by_name["brain_reindex"].permission, "write")
        self.assertEqual(by_name["inbox_collect"].permission, "write")
        self.assertEqual(by_name["brain_reindex"].side_effects, "index")
        self.assertEqual(by_name["inbox_collect"].side_effects, "write")
        self.assertEqual(by_name["bili_transcribe"].side_effects, "cache")


class TestAuthorizeConsumesPermission(unittest.TestCase):
    def test_write_without_confirm_rejected_fail_closed(self):
        reg = ToolRegistry()
        t = _mk_tool("w3", "write", side_effects="write", idempotent=True)
        with self.assertRaisesRegex(AgentError, "AGENT_TOOL_PERMISSION"):
            asyncio.run(reg._authorize(t, None))

    def test_write_confirm_rejected_by_user(self):
        reg = ToolRegistry()
        t = _mk_tool("w4", "write", side_effects="write", idempotent=True)
        with self.assertRaisesRegex(AgentError, "拒绝"):
            asyncio.run(reg._authorize(t, lambda tool, msg: False))

    def test_read_tool_not_blocked(self):
        reg = ToolRegistry()
        t = _mk_tool("r2", "read")
        asyncio.run(reg._authorize(t, None))  # 不抛即通过


if __name__ == "__main__":
    unittest.main()
