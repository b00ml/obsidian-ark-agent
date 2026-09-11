"""#10①/OPT-123 LLM 记忆提取单测：解析/打分过滤/沉淀编排。

FakeProvider/FakeDepo 全程替身，不出网不落库。
"""
from __future__ import annotations

import json
import unittest

from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import Message, TokenUsage
from agentlab.memory.extract import (
    MemoryExtractor,
    _parse_drafts,
    deposit_via_extractor,
)


class FakeProvider(LLMProvider):
    def __init__(self, content):
        self._content = content
        self.last_prompt = None

    async def chat(self, messages, tools=None, **kw):
        self.last_prompt = messages[-1].content
        return LLMResponse(content=self._content, tool_calls=[],
                           usage=TokenUsage(input_tokens=1, output_tokens=1))


class FakeDepo:
    def __init__(self):
        self.commits = []

    def commit(self, content, tags=None, source_session="", dedup=False):
        self.commits.append({"content": content, "tags": tags,
                             "source_session": source_session, "dedup": dedup})
        return {"status": "committed"}


class TestParseDrafts(unittest.TestCase):
    def test_plain_json_filters_low_importance_and_empty(self):
        raw = json.dumps([
            {"content": "用户偏好 Markdown", "tags": ["pref"], "importance": 7},
            {"content": "低分条目", "tags": [], "importance": 2},
            {"content": "", "importance": 9},
        ], ensure_ascii=False)
        drafts = _parse_drafts(raw, max_items=6, min_importance=4, per_item_chars=2000)
        self.assertEqual([d["content"] for d in drafts], ["用户偏好 Markdown"])
        self.assertEqual(drafts[0]["tags"], ["pref"])

    def test_fenced_json_and_max_items_cap(self):
        raw = ("```json\n"
               + json.dumps([{"content": f"c{i}", "importance": 5} for i in range(10)])
               + "\n```")
        drafts = _parse_drafts(raw, max_items=3, min_importance=4, per_item_chars=2000)
        self.assertEqual(len(drafts), 3)

    def test_per_item_chars_truncates(self):
        drafts = _parse_drafts('[{"content": "' + "x" * 500 + '", "importance": 6}]',
                               max_items=6, min_importance=4, per_item_chars=100)
        self.assertEqual(len(drafts[0]["content"]), 100)

    def test_invalid_output_returns_empty(self):
        self.assertEqual(_parse_drafts("不是json", 6, 4, 2000), [])
        self.assertEqual(_parse_drafts("[{broken]", 6, 4, 2000), [])
        self.assertEqual(_parse_drafts("", 6, 4, 2000), [])


class TestExtractorAndDeposit(unittest.IsolatedAsyncioTestCase):
    async def test_extract_renders_user_assistant_only(self):
        p = FakeProvider(json.dumps(
            [{"content": "事实", "tags": ["t"], "importance": 6}], ensure_ascii=False))
        ex = MemoryExtractor(p)
        drafts = await ex.extract([
            Message(role="system", content="sys"),
            Message(role="user", content="问题"),
            Message(role="assistant", content="回答"),
            Message(role="tool", content="tool-out"),
        ])
        self.assertIn("user: 问题", p.last_prompt)
        self.assertIn("assistant: 回答", p.last_prompt)
        self.assertNotIn("tool-out", p.last_prompt)
        self.assertNotIn("sys", p.last_prompt)
        self.assertEqual(drafts[0]["content"], "事实")

    async def test_empty_transcript_skips_llm(self):
        p = FakeProvider("[]")
        ex = MemoryExtractor(p)
        self.assertEqual(await ex.extract([Message(role="system", content="s")]), [])
        self.assertIsNone(p.last_prompt)  # 零素材不烧 LLM

    async def test_deposit_via_extractor_commits_and_counts(self):
        p = FakeProvider(json.dumps([
            {"content": "A", "importance": 8, "tags": ["x"]},
            {"content": "B", "importance": 3},
        ], ensure_ascii=False))
        depo = FakeDepo()
        n = await deposit_via_extractor(depo, MemoryExtractor(p),
                                        [Message(role="user", content="hi")],
                                        source_session="sess-1")
        self.assertEqual(n, 1)
        self.assertEqual(depo.commits[0]["content"], "A")
        self.assertEqual(depo.commits[0]["source_session"], "sess-1")
        self.assertTrue(depo.commits[0]["dedup"])

    async def test_extractor_boom_returns_zero(self):
        class Boom:
            async def extract(self, entries):
                raise RuntimeError("boom")

        self.assertEqual(await deposit_via_extractor(
            FakeDepo(), Boom(), [Message(role="user", content="x")]), 0)


if __name__ == "__main__":
    unittest.main()
