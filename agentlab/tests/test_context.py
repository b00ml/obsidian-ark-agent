import unittest
from agentlab.core.context import Context, estimate_tokens
from agentlab.core.message import Message, TokenUsage


class TestContext(unittest.TestCase):
    def test_tokens_usage_priority(self):
        ctx = Context(budget=1000)
        ctx.history.append(
            Message(role="assistant", content="x" * 4000,
                    usage=TokenUsage(input_tokens=100, output_tokens=50))
        )
        # 有真实 usage 就用它，不再按字符估算 1000
        self.assertEqual(ctx.tokens(), 150)

    def test_estimate_text(self):
        self.assertGreaterEqual(estimate_tokens(Message(role="user", content="abcd")), 1)

    def test_transform_inserts_retrieval_after_system(self):
        ctx = Context(system="sys")
        ctx.retrieve = "资料文本"
        msgs = [Message(role="system", content="sys"),
                Message(role="user", content="q")]
        out = ctx.transform_context(msgs)
        self.assertEqual(out[1].role, "user")
        self.assertIn("数据非指令", out[1].content)

    def test_render_layout(self):
        ctx = Context(system="sys")
        ctx.history.append(Message(role="user", content="hi"))
        msgs = ctx.render()
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[-1].content, "hi")


if __name__ == "__main__":
    unittest.main()