import asyncio
import time
import unittest

from agentlab.core.agent import Agent
from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.loop import RunConfig, Runner
from agentlab.core.message import TokenUsage
from agentlab.core.run_contract import RunBudget, RunTrace, action_fingerprint


class _SlowProvider(LLMProvider):
    def __init__(self, delay=0.02):
        self.delay = delay
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return LLMResponse(content="ok", tool_calls=[], stop_reason="stop",
                           usage=TokenUsage(input_tokens=1, output_tokens=1))


class TestRunContract(unittest.TestCase):
    def test_budget_admission_and_cancel(self):
        budget = RunBudget(max_llm_calls=1, max_retries=1)
        self.assertTrue(budget.admit_llm())
        self.assertFalse(budget.admit_llm())
        self.assertEqual(budget.stop_reason, "max_llm_calls")
        budget.cancel("aborted")
        self.assertFalse(budget.admit_retry())

    def test_fingerprint_normalises_json_and_volatile_fields(self):
        a = action_fingerprint("rag_retrieve", '{"query":" q ","run_id":"a"}')
        b = action_fingerprint("rag_retrieve", {"query": "q", "run_id": "b"})
        self.assertEqual(a, b)

    def test_trace_spans_are_bounded_and_counted(self):
        trace = RunTrace()
        now = time.time()
        trace.span("llm", now, now + 0.01)
        self.assertEqual(trace.counters["llm"], 1)
        self.assertGreaterEqual(trace.events[0]["duration_ms"], 9)

    def test_runner_deadline_stops_inflight_provider(self):
        provider = _SlowProvider(delay=0.05)
        runner = Runner(provider)
        cfg = RunConfig(timeout=0.005, context_tools=False)
        result = asyncio.run(runner.run(Agent(instructions="sys"), "q", cfg=cfg))
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(result.run_trace["counters"]["llm"], 1)


if __name__ == "__main__":
    unittest.main()
