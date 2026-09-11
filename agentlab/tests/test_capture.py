"""记忆触发/去重注入单测（docs/07 §4.3 open-note 借鉴点 A）。"""
from __future__ import annotations

import unittest

from agentlab.memory import capture as cap
from agentlab.memory.store import MemoryStore, _similar


class TestCapturePolicy(unittest.TestCase):
    def test_should_capture_on_trigger(self):
        self.assertTrue(cap.should_capture("这块值得记：线程池防阻塞"))
        self.assertTrue(cap.should_capture("复盘结论是：先拿真实数据再作答"))

    def test_govern_capture_truncates_long_body_keeps_head_tail(self):
        # OPT-090 / tau Tier-1：超长记忆正文 head+tail 截断留标记
        long = "要" * 5000
        out = cap.govern_capture(long, per_item_chars=1000)
        self.assertLess(len(out), 1500)
        self.assertIn("truncated", out)
        self.assertTrue(out.startswith("要"))
        self.assertTrue(out.endswith("要"))

    def test_govern_capture_keeps_short_unchanged(self):
        self.assertEqual(cap.govern_capture("短记忆", 1000), "短记忆")
        self.assertEqual(cap.govern_capture(""), "")

    def test_should_capture_ignores_bland_text(self):
        self.assertFalse(cap.should_capture("好的，我来查一下知识库。"))

    def test_period_every_5(self):
        self.assertTrue(cap.period(5))
        self.assertTrue(cap.period(10))
        self.assertFalse(cap.period(3))

    def test_dedup_context_builds_injection(self):
        existing = [
            {"content": "长任务用线程池防阻塞"},
            {"content": "先查证再作答"},
            {"content": ""},  # 空内容跳过
        ]
        ctx = cap.dedup_context(existing, limit=10)
        self.assertIn("避免重复沉淀", ctx)
        self.assertIn("线程池", ctx)
        self.assertNotIn("\n\n- ", ctx)  # 空串被过滤

    def test_dedup_context_empty_when_none(self):
        self.assertEqual(cap.dedup_context([]), "")
        self.assertEqual(cap.dedup_context(None), "")


class TestExtractSnippets(unittest.TestCase):
    """S6 记忆调度：extract_snippets 从回合消息抽触发词命中片段（纯函数）。"""

    def _msg(self, role, content):
        from agentlab.core.message import Message
        return Message(role=role, content=content)

    def test_extracts_trigger_hits_only(self):
        entries = [
            self._msg("user", "这块值得记：线程池防阻塞"),
            self._msg("assistant", "补充到记忆：先查证再作答"),
            self._msg("user", "好的，我去查知识库。"),  # 无触发词 → 跳过
            self._msg("tool", "值不值得记：工具结果不算"),  # tool 角色 → 跳过
        ]
        got = cap.extract_snippets(entries)
        self.assertEqual(len(got), 2)
        self.assertIn("线程池", got[0])
        self.assertIn("先查证", got[1])

    def test_dedup_and_limit(self):
        entries = [
            self._msg("user", "复盘结论是：X"),
            self._msg("assistant", "复盘结论是：X"),  # 内容去重
            self._msg("user", "规律：一二三"),
            self._msg("user", "教训：坏"),
            self._msg("user", "重点：重要"),
        ]
        got = cap.extract_snippets(entries, limit=2)
        self.assertEqual(len(got), 2)  # limit 生效
        self.assertEqual(got.count("复盘结论是：X"), 1)  # 原文去重保序

    def test_skips_bland_or_empty(self):
        self.assertEqual(cap.extract_snippets([self._msg("user", "嗯嗯")]), [])
        self.assertEqual(cap.extract_snippets([]), [])


class TestStoreDedup(unittest.TestCase):
    def test_similar_overlap(self):
        self.assertGreaterEqual(_similar("长任务用线程池防阻塞", "长任务用线程池减少阻塞"), 0.72)

    def test_commit_skips_near_duplicate_when_dedup_on(self):
        store = MemoryStore(None)
        fake = _FakeMemory([{"content": "长任务用线程池来防止阻塞"}])
        store._tm = fake
        r = store.commit("长任务用线程池防阻塞", tags=["agentlab"], dedup=True)
        self.assertEqual(r["status"], "skipped_duplicate")
        self.assertEqual(fake.commits, 0)  # 未真正写入

    def test_commit_proceeds_when_not_duplicate(self):
        store = MemoryStore(None)
        fake = _FakeMemory([{"content": "检索应该先查证再作答"}])
        store._tm = fake
        r = store.commit("长任务用线程池防阻塞", tags=["agentlab"], dedup=True)
        self.assertEqual(r["status"], "committed")
        self.assertEqual(fake.commits, 1)

    def test_dedup_off_by_default_proceeds(self):
        store = MemoryStore(None)
        fake = _FakeMemory([{"content": "长任务用线程池来防止事件循环阻塞"}])
        store._tm = fake
        r = store.commit("长任务应该用线程池防阻塞", tags=["agentlab"])
        self.assertEqual(r["status"], "committed")  # 默认 dedup=False 不拦截


class _FakeMemory:
    """替身：不发真 brain，只返回预设 memory_query 结果并记录 commit。"""

    def __init__(self, results: list[dict]):
        self.query_results = results
        self.commits = 0

    def memory_query(self, cfg, topic, limit):
        return {"status": "ok", "total": len(self.query_results),
                "results": list(self.query_results)}

    def memory_commit(self, cfg, content, tags, source_session):
        self.commits += 1
        return {"status": "committed", "id": self.commits}


if __name__ == "__main__":
    unittest.main()