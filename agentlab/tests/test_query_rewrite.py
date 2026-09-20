import asyncio
import json
import unittest

from agentlab.core.llm import LLMResponse
from agentlab.core.message import TokenUsage
from agentlab.rag.rewrite import rewrite_query


class FakeProvider:
    def __init__(self, content=None, error=None, delay=0):
        self.content = content
        self.error = error
        self.delay = delay
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return LLMResponse(content=self.content, tool_calls=[], usage=TokenUsage())


class TestQueryRewrite(unittest.IsolatedAsyncioTestCase):
    async def test_off_mode_does_not_call_provider(self):
        provider = FakeProvider(json.dumps({"query": "继续解释 RRF", "preserved_entities": ["RRF"]}))
        result = await rewrite_query(provider, "继续解释 RRF", recent_context="上一轮", mode="off")
        self.assertFalse(result.applied)
        self.assertEqual(result.query, "继续解释 RRF")
        self.assertEqual(provider.calls, 0)

    async def test_on_mode_rewrites_followup_and_keeps_entity(self):
        provider = FakeProvider(json.dumps({"query": "解释上一轮讨论的 RRF 检索融合", "preserved_entities": ["RRF"], "reason": "补全指代"}))
        result = await rewrite_query(provider, "继续解释这个 RRF", recent_context="上一轮讨论了 RRF", mode="on")
        self.assertTrue(result.applied)
        self.assertIn("RRF", result.query)
        self.assertEqual(provider.calls, 1)

    async def test_invalid_output_and_timeout_fall_back(self):
        bad = FakeProvider(json.dumps({"query": "", "preserved_entities": []}))
        result = await rewrite_query(bad, "继续解释这个主题", recent_context="上一轮讨论了主题", mode="on")
        self.assertFalse(result.applied)
        self.assertEqual(result.error, "empty_or_oversized")
        invented = FakeProvider(json.dumps({"query": "继续解释主题 RRF", "preserved_entities": ["RRF"]}))
        result = await rewrite_query(invented, "继续解释这个主题", recent_context="上一轮讨论了主题", mode="on")
        self.assertFalse(result.applied)
        self.assertEqual(result.error, "invented_entity")
        slow = FakeProvider(json.dumps({"query": "anything"}), delay=0.05)
        result = await rewrite_query(slow, "继续这个", recent_context="上一轮", mode="on", deadline_ms=1)
        self.assertFalse(result.applied)
        self.assertEqual(result.error, "timeout")

    async def test_shadow_returns_original_but_keeps_candidate_reason(self):
        provider = FakeProvider(json.dumps({"query": "解释 RRF 检索融合", "preserved_entities": ["RRF"]}))
        result = await rewrite_query(provider, "继续解释这个 RRF", recent_context="RRF", mode="shadow")
        self.assertFalse(result.applied)
        self.assertEqual(result.query, "继续解释这个 RRF")
        self.assertEqual(result.candidate_query, "解释 RRF 检索融合")
        self.assertEqual(result.error, "")
