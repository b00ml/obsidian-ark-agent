"""OPT-214（P0-04）：幂等缓存持久化背板测试——重启后不重复处理已完成请求。

验收对应（执行手册 4.3）：
- 服务重启后，同一 request_id 的已完成请求仍命中重放（不重复跑 agent/烧 token）。
- 失败/中断（abandon）不缓存；pending 不跨重启（重启后重跑而非 409）。
- TTL 过期不恢复；损坏行静默跳过不拖垮加载。
"""
import os
import sqlite3
import contextlib
import tempfile
import unittest

from agentlab.runtime.serve_idem import _IdempotencyCache


class IdempotencyStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "idempotency.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_restart_replays_completed_request(self):
        first = _IdempotencyCache(store_path=self.db)
        self.assertTrue(first.begin("rq-1"))
        first.commit("rq-1", ['data: {"a": 1}\n\n', "data: [DONE]\n\n"])
        # "重启"：同库新建实例，内存清零
        second = _IdempotencyCache(store_path=self.db)
        snap = second.snapshot("rq-1")
        self.assertIsNotNone(snap)
        state, chunks = snap
        self.assertEqual(state, "done")
        self.assertEqual(chunks, ['data: {"a": 1}\n\n', "data: [DONE]\n\n"])
        # 持键判定：重启后重复提交同样拿不到执行权
        self.assertFalse(second.begin("rq-1"))

    def test_abandoned_request_not_persisted(self):
        first = _IdempotencyCache(store_path=self.db)
        self.assertTrue(first.begin("rq-x"))
        first.abandon("rq-x")
        second = _IdempotencyCache(store_path=self.db)
        self.assertIsNone(second.snapshot("rq-x"))
        self.assertTrue(second.begin("rq-x"))  # 可重新执行

    def test_pending_not_persisted(self):
        first = _IdempotencyCache(store_path=self.db)
        self.assertTrue(first.begin("rq-p"))
        second = _IdempotencyCache(store_path=self.db)
        # pending 执行本身随进程消失：重启后重跑，而非命中 409
        self.assertIsNone(second.snapshot("rq-p"))

    def test_expired_rows_not_loaded_and_pruned(self):
        first = _IdempotencyCache(ttl=1.0, store_path=self.db)
        first.begin("rq-old")
        first.commit("rq-old", ["old"])
        with contextlib.closing(sqlite3.connect(self.db)) as conn:
            conn.execute("UPDATE idempotency SET created_at=created_at-9999")
            conn.commit()
        second = _IdempotencyCache(ttl=1.0, store_path=self.db)
        self.assertIsNone(second.snapshot("rq-old"))
        with contextlib.closing(sqlite3.connect(self.db)) as conn:
            self.assertIsNone(conn.execute(
                "SELECT request_id FROM idempotency WHERE request_id='rq-old'").fetchone())

    def test_corrupt_rows_skipped(self):
        first = _IdempotencyCache(store_path=self.db)
        first.begin("rq-good")
        first.commit("rq-good", ["good"])
        with contextlib.closing(sqlite3.connect(self.db)) as conn:
            conn.execute("INSERT INTO idempotency VALUES ('rq-bad', ?, 'not-json{')",
                         (__import__("time").time(),))
            conn.commit()
        second = _IdempotencyCache(store_path=self.db)
        self.assertEqual(second.snapshot("rq-good"), ("done", ["good"]))
        self.assertIsNone(second.snapshot("rq-bad"))

    def test_memory_only_mode_unchanged(self):
        # store_path=None 时保持原语义（不建库、不报错）
        cache = _IdempotencyCache()
        self.assertTrue(cache.begin("rq-m"))
        cache.commit("rq-m", ["x"])
        self.assertEqual(cache.snapshot("rq-m"), ("done", ["x"]))


class ServeIdemStoreWiringTest(unittest.TestCase):
    def test_serve_wires_persisted_cache_by_default(self):
        from agentlab.runtime.serve import Serve
        from types import SimpleNamespace
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "idem.db")
            s = Serve(cfg=SimpleNamespace(), port=18643, build_factory=lambda: (None, 0),
                      idem_store_path=path)
            self.assertEqual(s.idem._store_path, path)
            self.assertTrue(s.idem.begin("rq-wire"))
            s.idem.commit("rq-wire", ["data: hi\n\n"])
            reopened = Serve(cfg=SimpleNamespace(), port=18643,
                             build_factory=lambda: (None, 0), idem_store_path=path)
            self.assertEqual(reopened.idem.snapshot("rq-wire"), ("done", ["data: hi\n\n"]))


if __name__ == "__main__":
    unittest.main()
