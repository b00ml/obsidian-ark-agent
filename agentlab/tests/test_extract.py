"""#10①/OPT-123 LLM 记忆提取单测：解析/打分过滤/沉淀编排。

FakeProvider/FakeDepo 全程替身，不出网不落库。
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest

from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import Message, TokenUsage
from agentlab.memory.extract import (
    MemoryExtractor,
    _parse_drafts,
    _queue_guard,
    deposit_failure_status,
    deposit_via_extractor,
    retry_deposit_failures,
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
        self._brain_config = {}

    def commit(self, content, tags=None, source_session="", dedup=False, importance=None,
               mem_type="context", bucket=False, confidence=None):
        self.commits.append({"content": content, "tags": tags,
                             "source_session": source_session, "dedup": dedup,
                             "importance": importance, "mem_type": mem_type,
                             "bucket": bucket, "confidence": confidence})
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
    async def test_failure_queue_cross_process_lock_reclaims_stale_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            path = Path(tmp) / ".agent-brain" / "memory" / "deposit-failures.jsonl"
            lock = path.with_suffix(path.suffix + ".lock")
            path.parent.mkdir(parents=True)
            lock.write_text("crashed worker", encoding="ascii")
            old = time.time() - 3600
            os.utime(lock, (old, old))
            with _queue_guard(path):
                self.assertTrue(lock.exists())
            self.assertFalse(lock.exists())

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


    async def test_deposit_routes_by_importance(self):
        """OPT-225：importance>=7 落 context 独立文件，<7 进 sessions 月桶。"""
        import json as _json
        p = FakeProvider(_json.dumps([
            {"content": "高价值决策：采用混合粒度存储", "importance": 8, "tags": ["决策"]},
            {"content": "过程记录：跑了一轮评测", "importance": 5},
        ], ensure_ascii=False))
        depo = FakeDepo()
        n = await deposit_via_extractor(depo, MemoryExtractor(p),
                                        [Message(role="user", content="hi")],
                                        source_session="sess-x")
        self.assertEqual(n, 2)
        by_mt = {c["mem_type"]: c for c in depo.commits}
        self.assertEqual(by_mt["context"]["bucket"], False)
        self.assertEqual(by_mt["sessions"]["bucket"], True)


    async def test_deposit_routes_by_llm_suggested_type(self):
        """OPT-230 B5：晋升时采用 LLM 建议类型（decisions 允许、core 回退 context）。"""
        p = FakeProvider(json.dumps([
            {"content": "决策A：采用桶化存储", "importance": 8, "tags": ["d"],
             "type": "decisions", "confidence": 0.9},
            {"content": "应该写进 core 的尝试", "importance": 9, "tags": ["c"],
             "type": "core", "confidence": 0.9},
            {"content": "非法类型条目", "importance": 8, "tags": ["x"],
             "type": "unknown-type", "confidence": 0.9},
        ], ensure_ascii=False))
        depo = FakeDepo()
        n = await deposit_via_extractor(depo, MemoryExtractor(p),
                                        [Message(role="user", content="hi")])
        self.assertEqual(n, 3)
        by_content = {c["content"]: {**c, "confidence": c.get("confidence")} for c in depo.commits}
        self.assertEqual(by_content["决策A：采用桶化存储"]["mem_type"], "decisions")
        self.assertEqual(by_content["决策A：采用桶化存储"]["bucket"], False)
        # core 禁止自动晋升 → 回退 context
        self.assertEqual(by_content["应该写进 core 的尝试"]["mem_type"], "context")
        # 非法类型 → 回退 context
        self.assertEqual(by_content["非法类型条目"]["mem_type"], "context")
        # confidence 透传
        self.assertEqual(by_content["决策A：采用桶化存储"]["confidence"], 0.9)

    async def test_extract_failure_leaves_audit_trail(self):
        """OPT-230 B6：提取失败写入有界留痕文件（不回退触发词路）。"""
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "Inbox"))
        depo = FakeDepo()
        depo._brain_config = {"vault_path": tmp}
        from agentlab.memory.extract import _log_deposit_failure
        _log_deposit_failure(depo, [Message(role="user", content="丢失的记忆素材XYZ")],
                             RuntimeError("boom"))
        path = os.path.join(tmp, ".agent-brain", "memory", "deposit-failures.jsonl")
        self.assertTrue(os.path.exists(path))
        body = open(path, encoding="utf-8").read()
        self.assertIn("丢失的记忆素材XYZ", body)
        self.assertIn("boom", body)

    async def test_extractor_boom_returns_zero(self):
        class Boom:
            async def extract(self, entries):
                raise RuntimeError("boom")

        self.assertEqual(await deposit_via_extractor(
            FakeDepo(), Boom(), [Message(role="user", content="x")]), 0)

    async def test_extractor_failure_is_replayable_and_marks_success(self):
        depo = FakeDepo()
        with tempfile.TemporaryDirectory() as tmp:
            depo._brain_config = {"vault_path": tmp}

            class Boom:
                async def extract(self, entries):
                    raise RuntimeError("provider down")

            self.assertEqual(await deposit_via_extractor(
                depo, Boom(), [Message(role="user", content="稳定偏好")],
                source_session="s-retry"), 0)
            self.assertEqual(deposit_failure_status(depo)["counts"]["pending"], 1)

            good = FakeProvider(json.dumps(
                [{"content": "稳定偏好", "importance": 8}], ensure_ascii=False))
            out = await retry_deposit_failures(
                depo, MemoryExtractor(good), limit=5,
            )
            self.assertEqual(out["succeeded"], 1)
            self.assertEqual(deposit_failure_status(depo)["counts"]["succeeded"], 1)
            self.assertEqual(depo.commits[0]["source_session"], "s-retry")

    async def test_retry_dead_letters_after_bounded_attempts(self):
        depo = FakeDepo()
        with tempfile.TemporaryDirectory() as tmp:
            depo._brain_config = {
                "vault_path": tmp,
                "memory_policy": {"retry_max_attempts": 1},
            }

            class Boom:
                async def extract(self, entries):
                    raise RuntimeError("still down")

            await deposit_via_extractor(
                depo, Boom(), [Message(role="user", content="待重放")],
            )
            out = await retry_deposit_failures(depo, Boom(), limit=1)
            self.assertEqual(out["dead_letter"], 1)
            self.assertEqual(deposit_failure_status(depo)["counts"]["dead_letter"], 1)

    async def test_retry_claim_is_single_consumer(self):
        """Concurrent retry workers must not replay one pending row twice."""
        depo = FakeDepo()
        with tempfile.TemporaryDirectory() as tmp:
            depo._brain_config = {"vault_path": tmp}

            class Boom:
                async def extract(self, entries):
                    raise RuntimeError("provider down")

            await deposit_via_extractor(
                depo, Boom(), [Message(role="user", content="只入队一次")],
            )

            class SlowGood(FakeProvider):
                calls = 0

                def __init__(self):
                    super().__init__(json.dumps([
                        {"content": "只入队一次", "importance": 8},
                    ], ensure_ascii=False))

                async def extract(self, entries):
                    type(self).calls += 1
                    import asyncio
                    await asyncio.sleep(0.03)
                    return json.loads(self._content)

            good = SlowGood()
            first, second = await asyncio.gather(
                retry_deposit_failures(depo, good, limit=1),
                retry_deposit_failures(depo, good, limit=1),
            )
            self.assertEqual(SlowGood.calls, 1)
            self.assertEqual(first["succeeded"] + second["succeeded"], 1)
            self.assertEqual(len(depo.commits), 1)

    async def test_failed_draft_queue_redacts_sensitive_content(self):
        class FailingDepo(FakeDepo):
            def commit(self, *args, **kwargs):
                raise RuntimeError("write rejected")

        depo = FailingDepo()
        with tempfile.TemporaryDirectory() as tmp:
            depo._brain_config = {"vault_path": tmp}
            provider = FakeProvider(json.dumps([
                {"content": "api_key=top-secret should not persist", "importance": 8},
            ]))
            self.assertEqual(await deposit_via_extractor(
                depo, MemoryExtractor(provider),
                [Message(role="user", content="写入偏好")],
            ), 0)
            path = os.path.join(tmp, ".agent-brain", "memory", "deposit-failures.jsonl")
            body = open(path, encoding="utf-8").read()
            self.assertNotIn("top-secret", body)
            self.assertIn("[REDACTED]", body)


if __name__ == "__main__":
    unittest.main()
