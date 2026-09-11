"""L11/OPT-111 可逆区段档案单测：jsonl 落盘/读回/关键词兜底 + 索引同步 + 网关会话隔离。

覆盖：append/read 往返（含 tool_calls）、seq 跨实例续编、半行容忍、超长治理截断、
空区段/非法 sid 拒绝、关键词兜底检索、索引失败不丢档案、gateway 未绑定降级与
bind/reset 会话隔离、向量优先+关键词兜底链。
"""
import json
import tempfile
import unittest
from pathlib import Path

from agentlab.core.message import Message
from agentlab.memory.ranges import (
    RangeArchive, RangeGateway, RangeRecorder, render_range,
)
from agentlab.rag.vector_index import VectorIndex


class FakeEmbedder:
    """词表命中向量（同 test_rag 约定）：每词一维按出现计数，确定性可断言。"""

    def __init__(self, vocab):
        self.vocab = list(vocab)

    def embed(self, texts):
        return [[float(t.count(w)) for w in self.vocab] for t in texts]


def _msgs():
    return [
        Message(role="user", content="帮我查猫的喂养计划"),
        Message(role="assistant", content=None, tool_calls=[]),
        Message(role="tool", content="猫每天喂两次", name="vault_search", tool_call_id="c1"),
    ]


class TestRangeArchive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.arch = RangeArchive(Path(self.tmp.name) / "sessions")

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_read_roundtrip(self):
        rec = self.arch.append("s-1", _msgs())
        self.assertEqual(rec["seq"], 1)
        self.assertEqual(rec["session_id"], "s-1")
        rec2 = self.arch.append("s-1", [Message(role="user", content="第二条")])
        self.assertEqual(rec2["seq"], 2, "同会话 seq 自增")
        back = self.arch.read("s-1", 1)
        self.assertEqual([m.role for m in back], ["user", "assistant", "tool"])
        self.assertEqual(back[0].content, "帮我查猫的喂养计划")
        self.assertEqual(back[2].name, "vault_search")
        self.assertEqual(back[2].tool_call_id, "c1")
        self.assertEqual(self.arch.read("s-1", 99), [], "不存在的 seq 返回空")

    def test_seq_continues_across_instances(self):
        self.arch.append("s-1", _msgs())
        fresh = RangeArchive(Path(self.tmp.name) / "sessions")
        rec = fresh.append("s-1", _msgs())
        self.assertEqual(rec["seq"], 2, "跨实例以文件尾续编")

    def test_sanitize_caps_long_content(self):
        long = "很长的工具输出" * 200  # 1200 字
        arch = RangeArchive(Path(self.tmp.name) / "s2", max_msg_chars=1000)
        arch.append("s-1", [Message(role="tool", content=long, name="t", tool_call_id="c")])
        back = arch.read("s-1", 1)[0]
        self.assertLess(len(back.content), len(long), "超限即截断")
        self.assertIn("…[truncated", back.content)
        self.assertTrue(back.content.endswith(long[-200:]), "head+tail 保留尾部（tail=cap//5）")

    def test_empty_segment_and_bad_sid_rejected(self):
        with self.assertRaises(ValueError):
            self.arch.append("s-1", [])
        with self.assertRaises(ValueError):
            self.arch.path("../escape")
        with self.assertRaises(ValueError):
            self.arch.path("a/b")

    def test_half_line_tolerated(self):
        self.arch.append("s-1", _msgs())
        with self.arch.path("s-1").open("ab") as f:  # 模拟追加中途崩溃残留半行
            f.write(b'{"session_id": "s-1", "seq": 2, "mess')
        fresh = RangeArchive(Path(self.tmp.name) / "sessions")
        rec = fresh.append("s-1", _msgs())
        self.assertEqual(rec["seq"], 2, "坏行不计入 seq")
        self.assertEqual(len(self.arch.read("s-1", 1)), 3)

    def test_search_keyword_fallback(self):
        self.arch.append("s-1", [Message(role="user", content="讨论了猫的喂养计划")])
        self.arch.append("s-1", [Message(role="user", content="部署了狗屋方案")])
        hits = self.arch.search("s-1", "喂养 猫", limit=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["source"], "session")
        self.assertTrue(hits[0]["ref"].startswith("session/s-1#r"))
        self.assertEqual(hits[0]["seq"], 1)
        self.assertIn("猫", hits[0]["content"])
        self.assertEqual(self.arch.search("s-1", "火箭"), [], "无命中返回空")


class TestRangeRecorder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.idx = VectorIndex(self.vault / ".agent-brain" / "rag-index.sqlite",
                               FakeEmbedder(["猫", "狗"]), chunk_chars=200)

    def tearDown(self):
        self.tmp.cleanup()

    def test_archive_upserts_index_and_jsonl(self):
        rec = RangeRecorder(RangeArchive(self.root / "sessions"), "s-1",
                            index=self.idx).archive(_msgs())
        self.assertTrue((self.root / "sessions" / "s-1.ranges.jsonl").exists())
        hits = self.idx.search_ranges("猫 喂养", k=3, session_id="s-1")
        self.assertTrue(hits, "归档后向量路立即可召回")
        self.assertEqual(hits[0]["session_id"], "s-1")
        self.assertEqual(hits[0]["seq"], rec["seq"])
        self.assertEqual(hits[0]["source"], "session")

    def test_index_failure_still_archives(self):
        class _Boom:  # 无 upsert_range → 嵌入必炸
            pass

        rec = RangeRecorder(RangeArchive(self.root / "sessions"), "s-1",
                            index=_Boom()).archive(_msgs())
        self.assertEqual(rec["seq"], 1, "索引失败上抛被吞，档案照常落盘")
        with (self.root / "sessions" / "s-1.ranges.jsonl").open(encoding="utf-8") as f:
            self.assertEqual(json.loads(f.readline())["messages"][0]["content"],
                             "帮我查猫的喂养计划")


class TestRangeGateway(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.arch = RangeArchive(self.root / "sessions")
        self.arch.append("s-1", [Message(role="user", content="猫的喂养计划是早晚各一次")])
        self.arch.append("s-2", [Message(role="user", content="狗的驱虫安排在每月初")])

    def tearDown(self):
        self.tmp.cleanup()

    def test_unbound_returns_empty(self):
        self.assertEqual(RangeGateway(self.arch).recaller()("猫"), [])

    def test_bind_scopes_and_reset(self):
        gw = RangeGateway(self.arch, k=3)
        tok = gw.bind("s-1")
        try:
            hits = gw.recaller()("喂养 猫")
            self.assertEqual([h["session_id"] for h in hits], ["s-1"], "只召回当前会话")
        finally:
            gw.reset(tok)
        self.assertEqual(gw.recaller()("喂养 猫"), [], "reset 后降级为空")

    def test_vector_first_keyword_fallback(self):
        vault = self.root / "vault"
        vault.mkdir()
        idx = VectorIndex(vault / "idx.sqlite", FakeEmbedder(["猫", "狗"]), chunk_chars=200)
        RangeRecorder(self.arch, "s-1", index=idx).archive(
            [Message(role="user", content="猫的喂养计划是早晚各一次")])  # 归档即入索引
        gw = RangeGateway(self.arch, index=idx, k=3)
        tok = gw.bind("s-1")
        try:
            hits = gw.recaller()("猫")
            self.assertTrue(hits and hits[0]["source"] == "session", "向量路优先命中")
            idx.embedder = FakeEmbedder(["x"] * 5)  # 维度漂移 → 向量路空
            hits = gw.recaller()("喂养 猫")
            self.assertTrue(hits and hits[0]["source"] == "session",
                            "向量路降级后走 jsonl 关键词兜底，不丢召回")
        finally:
            gw.reset(tok)

    def test_recorder_binds_session(self):
        gw = RangeGateway(self.arch)
        gw.recorder("s-9").archive([Message(role="user", content="新会话片段")])
        self.assertTrue((self.root / "sessions" / "s-9.ranges.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
