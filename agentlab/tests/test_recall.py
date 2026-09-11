"""#10①/OPT-123 记忆召回注入单测：build_memory_block / memory_block_for。

不碰真实 brain：open_memory_store 以 patch 替身注入，query 结果形状对齐
MemoryStore.query（dict 条目 content/tags/created_at）。
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agentlab.memory.recall import build_memory_block, memory_block_for


class _FakeStore:
    def __init__(self, results):
        self._results = results
        self.available = True
        self.topic = None
        self.limit = None

    def query(self, topic, limit=10, **kw):
        self.topic = topic
        self.limit = limit
        return {"topic": topic, "total": len(self._results), "results": self._results}


class TestBuildMemoryBlock(unittest.TestCase):
    def test_formats_dict_items_with_tag_and_date(self):
        block = build_memory_block([
            {"content": "用户偏好深色主题", "tags": "preference,ui",
             "created_at": "2026-09-01 10:00:00"},
            {"content": "x" * 600, "tags": [], "created_at": ""},
        ])
        self.assertIn("- [preference] 用户偏好深色主题（2026-09-01）", block)
        self.assertIn("…[truncated", block)  # 单条超注入预算 → head+tail 截断
        self.assertNotIn("（）", block)  # 空日期不带空括号

    def test_accepts_attr_items_and_skips_empty(self):
        block = build_memory_block([
            SimpleNamespace(content="  ", tags=[], created_at=""),
            SimpleNamespace(content="事实A", tags=["t"], created_at="2026-09-02"),
        ])
        self.assertEqual(block, "- [t] 事实A（2026-09-02）")

    def test_empty_results_return_empty_string(self):
        self.assertEqual(build_memory_block([]), "")
        self.assertEqual(build_memory_block(None), "")


class TestMemoryBlockFor(unittest.TestCase):
    def test_injects_query_results_with_topic_and_topk(self):
        store = _FakeStore([{"content": "记忆甲", "tags": ["a"],
                             "created_at": "2026-09-03"}])
        cfg = SimpleNamespace(limits=SimpleNamespace(memory_inject_topk=5))
        with patch("agentlab.memory.recall.open_memory_store", return_value=store):
            block = memory_block_for(cfg, "怎么配置whisper", topk=5)
        self.assertEqual(block, "- [a] 记忆甲（2026-09-03）")
        self.assertEqual(store.topic, "怎么配置whisper")
        self.assertEqual(store.limit, 5)

    def test_degrades_silently(self):
        cfg = SimpleNamespace(limits=SimpleNamespace(memory_inject_topk=5))
        with patch("agentlab.memory.recall.open_memory_store", return_value=None):
            self.assertEqual(memory_block_for(cfg, "q"), "")
        # 空主题 / topk=0：不触仓直接空串
        with patch("agentlab.memory.recall.open_memory_store") as m:
            self.assertEqual(memory_block_for(cfg, "", topk=5), "")
            self.assertEqual(memory_block_for(cfg, "q", topk=0), "")
            m.assert_not_called()


if __name__ == "__main__":
    unittest.main()
