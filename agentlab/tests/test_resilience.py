"""熔断 + 重试的单元测试（docs/03 工程化实践）。"""
import asyncio
import unittest

from agentlab.core.errors import AgentError
from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import TokenUsage
from agentlab.core.resilience import CircuitBreaker, ResilientLLM


def _ok(provider, calls):
    """让 provider.chat 直接成功。"""


class _FakeProvider(LLMProvider):
    def __init__(self, outcomes):
        # outcomes: 依次消费的错误/成功事件；"ok" 表示成功，AgentError 抛错
        self.outcomes = list(outcomes)
        self.calls = []

    async def chat(self, messages, tools=None, **kw):
        self.calls.append(1)
        out = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(out, AgentError):
            raise out
        return LLMResponse(
            content="hi", tool_calls=[], stop_reason="stop",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def advance(self, dt):
        self.t += dt

    def __call__(self):
        return self.t


class TestCircuitBreaker(unittest.TestCase):
    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1000)
        self.assertTrue(cb.allow())
        for _ in range(3):
            cb.on_failure()
        self.assertEqual(cb.state, "open")
        self.assertFalse(cb.allow())

    def test_recovers_to_half_open_then_closed(self):
        clock = _FakeClock()
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.001, clock=clock)
        cb.on_failure(); cb.on_failure()
        self.assertEqual(cb.state, "open")
        clock.advance(0.01)  # 越恢复期
        self.assertEqual(cb.state, "half_open")
        self.assertTrue(cb.allow())
        cb.on_success()
        self.assertEqual(cb.state, "closed")

    def test_half_open_limits_probes(self):
        clock = _FakeClock()
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.001, half_open_max=1,
                            clock=clock)
        cb.on_failure()
        clock.advance(0.01)
        self.assertTrue(cb.allow())
        self.assertFalse(cb.allow())  # half_open 已发出一探测


class TestResilientLLM(unittest.TestCase):
    def test_retries_then_succeeds(self):
        prov = _FakeProvider([AgentError("AGENT_LLM_RATE", "限流"), "ok"])
        rl = ResilientLLM(prov, max_retries=2, backoff=0.0)
        resp = asyncio.run(rl.chat([]))
        self.assertEqual(resp.content, "hi")
        self.assertEqual(len(prov.calls), 2)

    def test_non_retryable_raises_immediately(self):
        prov = _FakeProvider([AgentError("AGENT_TOOL_PERMISSION", "拒绝")])
        rl = ResilientLLM(prov, max_retries=2, backoff=0.0)
        with self.assertRaisesRegex(AgentError, "拒绝"):
            asyncio.run(rl.chat([]))
        self.assertEqual(len(prov.calls), 1)

    def test_auth_failure_is_not_retried(self):
        prov = _FakeProvider([AgentError("AGENT_LLM_AUTH", "配置错误"), "ok"])
        rl = ResilientLLM(prov, max_retries=2, backoff=0.0)
        with self.assertRaisesRegex(AgentError, "配置错误"):
            asyncio.run(rl.chat([]))
        self.assertEqual(len(prov.calls), 1)

    def test_provider_schema_or_logic_exception_is_not_retried(self):
        class BrokenProvider(LLMProvider):
            def __init__(self):
                self.calls = 0

            async def chat(self, messages, tools=None, **kw):
                self.calls += 1
                raise ValueError("malformed response")

        prov = BrokenProvider()
        rl = ResilientLLM(prov, max_retries=2, backoff=0.0)
        with self.assertRaisesRegex(AgentError, "AGENT_LLM_PROVIDER"):
            asyncio.run(rl.chat([]))
        self.assertEqual(prov.calls, 1)

    def test_circuit_open_blocks_calls(self):
        prov = _FakeProvider([AgentError("AGENT_LLM_RATE", "限流") for _ in range(10)])
        rl = ResilientLLM(
            prov, max_retries=0,
            breaker=CircuitBreaker(failure_threshold=2, recovery_timeout=1000),
        )
        # 触发熔断
        for _ in range(3):
            with self.assertRaises(AgentError):
                asyncio.run(rl.chat([]))
        # 熔断态直接拒绝（不触网）
        prov.calls.clear()
        with self.assertRaisesRegex(AgentError, "AGENT_LLM_CIRCUIT_OPEN"):
            asyncio.run(rl.chat([]))
        self.assertEqual(len(prov.calls), 0)


if __name__ == "__main__":
    unittest.main()
