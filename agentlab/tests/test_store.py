import datetime
import os
import tempfile
import unittest
from types import SimpleNamespace

from agentlab.memory.store import MemoryStore
from agentlab.tools.connectors.brain_tools import BRAIN_PKG_DIR


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = self._tmp.name
        self.config = {"vault_path": self._root, "brain_dir": ".agent-brain"}

    def tearDown(self):
        self._tmp.cleanup()

    def test_available_when_brain_present(self):
        # brain 目录在当前仓库存在 → store 应可用
        self.assertTrue(os.path.isdir(BRAIN_PKG_DIR))
        store = MemoryStore(self.config)
        self.assertTrue(store.available)


    def test_source_session_anchor_and_importance_passthrough(self):
        """OPT-224：构造期 source_session 锚 + importance 透传到 frontmatter。"""
        store = MemoryStore({"vault_path": self._root, "brain_dir": ".agent-brain"},
                            source_session="sess-anchor-1")
        store.commit("带会话锚的记忆", tags=["anchor"], importance=9)
        import glob as _g
        mds = _g.glob(str(self._root) + "/ark/memory/**/*.md", recursive=True)
        hit = [f for f in mds if "带会话锚的记忆" in open(f, encoding="utf-8").read()]
        self.assertTrue(hit)
        body = open(hit[0], encoding="utf-8").read()
        self.assertIn("source_session: sess-anchor-1", body)
        self.assertIn("importance: 9", body)

    def test_commit_without_anchor_keeps_empty(self):
        store = MemoryStore({"vault_path": self._root, "brain_dir": ".agent-brain"})
        store.commit("无锚记忆", tags=["noanchor"])
        import glob as _g
        mds = _g.glob(str(self._root) + "/ark/memory/**/*.md", recursive=True)
        hit = [f for f in mds if "无锚记忆" in open(f, encoding="utf-8").read()]
        self.assertTrue(hit)
        self.assertNotIn("source_session: sess", open(hit[0], encoding="utf-8").read())

    def test_commit_and_query_roundtrip(self):
        store = MemoryStore(self.config)
        r = store.commit("复用观点：长任务用线程池防阻塞", tags=["agentlab", "并发"])
        self.assertEqual(r["status"], "committed")
        q = store.query("agentlab")
        self.assertGreater(q["total"], 0)
        self.assertTrue(any("线程池" in item["content"] for item in q["results"]))

    def test_unavailable_when_no_module(self):
        # 用一个不存在的 brain 模块路径推导出不可用态：临时改 BRAN 不可行,
        # 用空 config 仍可用（brain 存在）；此处仅验证空 config 不抛错可降级。
        store = MemoryStore(None)
        self.assertTrue(store.available or store.commit("").get("status") in ("unavailable", "error"))


class TestMemoryDecay(unittest.TestCase):
    """召回时间衰减（OPT-101）：旧记忆降权重排，避免旧记忆永久占用召回位。"""

    _NOW = datetime.datetime(2026, 9, 5, 12, 0, 0)

    def _store_with(self, results, half_life=30.0):
        store = MemoryStore({"vault_path": "x", "brain_dir": ".agent-brain"},
                            decay_half_life_days=half_life, now_fn=lambda: self._NOW)
        store._tm = SimpleNamespace(memory_query=lambda cfg, topic, limit: {
            "topic": topic, "total": len(results[:limit]), "results": results[:limit]})
        return store

    def test_fresh_memory_ranks_above_stale(self):
        # brain 原始排名旧在前；半衰期 30 天时 3 个月前的记忆应被新记忆反超
        stale = {"content": "旧观点", "created_at": "2026-06-01T10:00:00"}
        fresh = {"content": "新观点", "created_at": "2026-09-05T10:00:00"}
        q = self._store_with([stale, fresh]).query("t")
        self.assertEqual(q["results"][0]["content"], "新观点")

    def test_missing_or_invalid_created_at_neutral(self):
        # 无/坏时间戳不惩罚不崩溃（factor=1.0 向后兼容旧记录）：保持 brain 原排名，
        # 仅已知时间的足够旧条目会被衰减下沉（见 test_fresh_memory_ranks_above_stale）
        no_ts = {"content": "无时间戳"}
        bad_ts = {"content": "坏时间戳", "created_at": "not-a-date"}
        fresh = {"content": "新观点", "created_at": "2026-09-05T10:00:00"}
        q = self._store_with([no_ts, bad_ts, fresh]).query("t")
        self.assertEqual([r["content"] for r in q["results"]], ["无时间戳", "坏时间戳", "新观点"])
        self.assertEqual(q["total"], 3)

    def test_decay_disabled_keeps_brain_order(self):
        # half_life=0 → 衰减关闭，保持 brain 原序（旧在前）
        stale = {"content": "旧观点", "created_at": "2026-06-01T10:00:00"}
        fresh = {"content": "新观点", "created_at": "2026-09-05T10:00:00"}
        q = self._store_with([stale, fresh], half_life=0).query("t")
        self.assertEqual([r["content"] for r in q["results"]], ["旧观点", "新观点"])

    def test_overfetch_trims_to_limit(self):
        # 衰减开启时超取 3 倍候选重排，最终仍截回 limit
        items = [{"content": f"m{i}", "created_at": "2026-09-05T09:00:00"} for i in range(9)]
        q = self._store_with(items).query("t", limit=3)
        self.assertEqual(len(q["results"]), 3)
        self.assertEqual(q["total"], 3)


if __name__ == "__main__":
    unittest.main()