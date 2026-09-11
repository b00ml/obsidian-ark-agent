import unittest
from agentlab.core.message import (
    Message, ToolCall, ToolCallFunction, ToolResult, TokenUsage, tool_result,
)


class TestMessage(unittest.TestCase):
    def test_roundtrip(self):
        m = Message(role="assistant", content="hi", tool_calls=[], usage=TokenUsage(input_tokens=5))
        d = m.model_dump()
        m2 = Message.model_validate(d)
        self.assertEqual(m2.role, "assistant")
        self.assertEqual(m2.usage.input_tokens, 5)

    def test_toolcall_pair(self):
        tc = ToolCall(id="call_1", function=ToolCallFunction(name="add", arguments='{"a":1}'))
        m = Message(
            role="assistant", content=None,
            tool_calls=[tc],
            stop_reason="tool_calls",
            usage=TokenUsage(output_tokens=2),
        )
        prov = m.to_provider_dict()
        self.assertEqual(prov["tool_calls"][0]["function"]["name"], "add")

    def test_tool_result_fields(self):
        r = tool_result("call_1", "ok", terminate=True, details={"files": ["a.md"]})
        self.assertTrue(r.terminate)
        self.assertEqual(r.details["files"], ["a.md"])

    def test_usage_total(self):
        u = TokenUsage(input_tokens=10, output_tokens=5, cache_read_tokens=3, cache_write_tokens=2)
        self.assertEqual(u.total(), 20)


if __name__ == "__main__":
    unittest.main()