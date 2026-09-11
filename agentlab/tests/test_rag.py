import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import Message, TokenUsage
from agentlab.rag.assess import RAGAssessor, Sufficiency
from agentlab.rag.recall import RAGRecall, RecallItem, fuse, retrieval_terms, split_keywords, govern_recall
from agentlab.rag.vector_index import VectorIndex, chunk_text
from agentlab.runtime.config import RagConfig
from agentlab.tools.rag_tools import build_rag_tools


def _item(title, content="", ref="", source="vault"):
    return RecallItem(title=title, content=content, ref=ref, source=source)


class TestRecall(unittest.TestCase):
    def test_split_keywords_chinese_and_en(self):
        self.assertEqual(split_keywords("对比 X 与 Y 视频的结论"), ["对比", "X", "Y", "视频的结论"])

    def test_retrieval_terms_windows_long_chinese(self):
        # 长中文串 → 补充 2/4 字窗口；短/英文标识符保持原样
        terms = retrieval_terms("Obsidian 知识管理的核心工具")
        self.assertIn("Obsidian", terms)
        self.assertIn("知识管理", terms)
        self.assertIn("工具", terms)
        self.assertEqual(retrieval_terms("abcd"), ["abcd"])

    def test_union_recall_handles_dict_and_dedupes(self):
        from agentlab.rag.recall import _union_recall

        def fake(term, limit):
            if term == "知识管理":
                return {"status": "ok", "total": 2, "results": [
                    {"path": "a.md", "content": "x", "source": "vault"},
                    {"path": "b.md", "content": "y", "source": "vault"},
                ]}
            if term == "工具":
                return {"status": "ok", "total": 1, "results": [
                    {"path": "a.md", "content": "x2", "source": "vault"},
                ]}
            return {"status": "ok", "total": 0, "results": []}
        rows = _union_recall(fake, "知识管理工具", limit=5)
        paths = [r["path"] for r in rows]
        self.assertIn("a.md", paths)
        self.assertEqual(len(paths), 2, "跨词命中应并集，a.md 去重")

    def test_fuse_scores_and_dedupes_by_ref(self):
        cands = [
            _item("对比文", content="采用向量检索", ref="vault/A.md", source="vault"),
            _item("另一篇", content="纯关键词", ref="vault/B.md", source="vault"),
            _item("记忆观点", content="向量检索效果好", ref="memory#3", source="memory"),
            # 同 ref 低分候选（应被高分覆盖去重）
            _item("忽略", content="向量", ref="vault/A.md", source="web"),
        ]
        out = fuse("向量检索", cands, k=10)
        refs = [x.ref for x in out]
        self.assertEqual(refs.count("vault/A.md"), 1, "同 ref 应去重")
        # A（含 query 词"向量检索"）分最高
        self.assertEqual(out[0].ref, "vault/A.md")

    def test_retrieve_with_injected_mock_recaller(self):
        def rec(q):
            return [{"title": "笔记", "content": "obsidian 知识管理", "ref": "vault/n.md", "source": "vault"}]
        r = RAGRecall([rec])
        items = r.retrieve("obsidian 知识管理")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].ref, "vault/n.md")

    def test_retrieve_tolerates_recaller_error(self):
        def bad(q):
            raise RuntimeError("boom")
        def good(q):
            return [{"title": "t", "content": "c", "ref": "vault/x.md", "source": "vault"}]
        r = RAGRecall([bad, good])
        items = r.retrieve("关键词")
        self.assertEqual(len(items), 1, "一路失败应降级，不影响其余")

    def test_govern_recall_truncates_long_content_keeps_identity(self):
        # OPT-089 / tau Tier-1+2：超长正文 head+tail 截断留标记；身份锚 title/ref/source 恒存
        long = "要" * 3000
        items = govern_recall([_item("笔记", content=long, ref="vault/big.md", source="vault")],
                              per_item_chars=1000, max_items=8)
        c = items[0].content
        self.assertLess(len(c), 1500)          # 已被截短
        self.assertIn("truncated", c)          # 截断标记
        self.assertTrue(c.startswith("要"))     # head 保留
        self.assertTrue(c.endswith("要"))       # tail 保留
        self.assertEqual(items[0].title, "笔记")
        self.assertEqual(items[0].ref, "vault/big.md")
        self.assertEqual(items[0].source, "vault")

    def test_govern_recall_caps_items(self):
        items = [RecallItem(title=f"n{i}", content="x" * 100, ref=f"r{i}", source="vault")
                 for i in range(10)]
        out = govern_recall(items, per_item_chars=1000, max_items=3)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].ref, "r0")

    def test_retrieve_governed_truncates_via_injected_recaller(self):
        def rec(q):
            return [{"title": "大笔记", "content": "超" * 3000, "ref": "vault/huge.md",
                     "source": "vault"}]
        r = RAGRecall([rec])
        items = r.retrieve_governed("查询", limit=1, per_item_chars=1000)
        self.assertIn("truncated", items[0].content)
        self.assertEqual(items[0].ref, "vault/huge.md")


class _AssessProvider(LLMProvider):
    def __init__(self, payload: dict):
        self._p = payload

    async def chat(self, messages, tools=None, **kw):
        self.captured_prompt = messages[0].content
        return LLMResponse(
            content=json.dumps(self._p, ensure_ascii=False),
            tool_calls=[], usage=TokenUsage(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


class TestAssess(unittest.TestCase):
    def test_assess_answer_when_sufficient(self):
        prov = _AssessProvider({"sufficient": True, "action": "answer",
                                "reformulated_query": "", "message": "够",
                                "refs": ["vault/A.md"]})
        res = asyncio.run(RAGAssessor(prov).assess(
            "结论差异", [_item("A", "结论", "vault/A.md")]))
        self.assertTrue(res.sufficient)
        self.assertEqual(res.action, "answer")
        # 走 .st 模板（含新增/判定关键词）
        self.assertIn("充分性", prov.captured_prompt)
        self.assertIn("reformulate", prov.captured_prompt)

    def test_assess_reformulate(self):
        prov = _AssessProvider({"sufficient": False, "action": "reformulate",
                                "reformulated_query": "更精确的查询",
                                "message": "覆盖不足", "refs": []})
        res = asyncio.run(RAGAssessor(prov).assess("q", [_item("A", "a")]))
        self.assertEqual(res.action, "reformulate")
        self.assertEqual(res.reformulated_query, "更精确的查询")

    def test_assess_empty_items_is_insufficient(self):
        res = asyncio.run(RAGAssessor(_AssessProvider({})).assess("q", []))
        self.assertEqual(res.action, "insufficient")
        self.assertFalse(res.sufficient)

    def test_bad_structured_output_raises_guardrail(self):
        prov = _AssessProvider({"sufficient": "yes", "action": "answer"})  # sufficient 非 bool
        from agentlab.core.errors import AgentError
        with self.assertRaises(AgentError):
            asyncio.run(RAGAssessor(prov).assess("q", [_item("A", "a")]))


class TestRRFFuse(unittest.TestCase):
    """P0-1/OPT-105：RRF 按路内排名融合，替代旧"命中分+路权重"线性加权。"""

    def test_cross_source_rank_boost_and_dedup(self):
        # 同文档被两路命中（vault rank1 + web rank1）应击败单路 rank1；同 ref 去重
        cands = [
            _item("A", content="x", ref="A", source="vault"),
            _item("B", content="x", ref="B", source="vault"),
            _item("C", content="x", ref="C", source="memory"),
            _item("A", content="x", ref="A", source="web"),
        ]
        out = fuse("任意", cands, k=10)
        self.assertEqual(out[0].ref, "A")
        self.assertEqual([x.ref for x in out].count("A"), 1)
        self.assertGreater(out[0].score, out[1].score)

    def test_within_source_order_is_rank(self):
        # 同路内靠前者排名高（顺序即排名）；RRF 分单调递减
        cands = [_item(f"n{i}", content="x", ref=f"r{i}", source="vault") for i in range(5)]
        out = fuse("任意", cands, k=10)
        self.assertEqual([x.ref for x in out], ["r0", "r1", "r2", "r3", "r4"])
        self.assertGreater(out[0].score, out[-1].score)


class FakeEmbedder:
    """词表命中向量：vocab 每词一维按出现计数——可手工构造语义相似度，确定性可断言。"""

    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(t.count(w)) for w in self.vocab] for t in texts]


class TestChunkText(unittest.TestCase):
    def test_merge_paragraphs_and_hard_split_long(self):
        text = "第一段。\n\n第二段。\n\n第三段。"
        chunks = chunk_text(text, max_chars=100)
        self.assertEqual(len(chunks), 1)
        self.assertIn("第一段", chunks[0])
        long_para = "。".join(["句子内容"] * 60)  # 远超上限的单段
        chunks = chunk_text(long_para, max_chars=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 51 for c in chunks), "硬切不超过上限+1（句号）")


class TestVectorIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name)
        (self.vault / "cat.md").write_text("# 猫\n\n猫会抓老鼠。\n\n猫是宠物。", encoding="utf-8")
        (self.vault / "dog.md").write_text("# 狗\n\n狗会看家。", encoding="utf-8")
        self.emb = FakeEmbedder(["猫", "狗", "老鼠"])
        self.idx = VectorIndex(self.vault / ".agent-brain" / "rag-index.sqlite",
                               self.emb, chunk_chars=200)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sync_and_search(self):
        stat = self.idx.sync_vault(self.vault)
        self.assertEqual(stat["total"], 2)
        self.assertGreaterEqual(stat["new_chunks"], 2)  # 每文件段落合并后 ≈1 块
        hits = self.idx.search("猫", k=3)
        self.assertTrue(hits, "向量路应命中")
        self.assertTrue(all(h["source"] == "vector" for h in hits))
        self.assertIn("猫", hits[0]["content"])
        self.assertTrue(hits[0]["ref"].startswith("cat.md#c"))

    def test_incremental_sync_only_changed(self):
        self.idx.sync_vault(self.vault)
        before = self.emb.calls
        stat = self.idx.sync_vault(self.vault)  # 无变更 → 零嵌入
        self.assertEqual(stat["updated"], 0)
        self.assertEqual(self.emb.calls, before)
        p = self.vault / "cat.md"  # 改一个文件 → 只重嵌它
        p.write_text("# 猫\n\n猫会抓老鼠。猫爱吃鱼。", encoding="utf-8")
        st = p.stat()
        os.utime(p, (st.st_atime, st.st_mtime + 5))
        stat = self.idx.sync_vault(self.vault)
        self.assertEqual(stat["updated"], 1)
        self.assertEqual(self.emb.calls, before + 1)
        (self.vault / "dog.md").unlink()  # 删除 → 索引清除
        stat = self.idx.sync_vault(self.vault)
        self.assertEqual(stat["removed"], 1)
        hits = self.idx.search("狗", k=5)
        self.assertEqual([h for h in hits if "狗会看家" in h["content"]], [])

    def test_search_degrades_when_no_index(self):
        self.assertEqual(self.idx.search("任意"), [])  # 未建库 → 空降级不抛错

    def test_dim_mismatch_degrades(self):
        self.idx.sync_vault(self.vault)
        self.idx.embedder = FakeEmbedder(["x"] * 5)  # 换了维度的 embedding 模型
        self.assertEqual(self.idx.search("猫"), [])

    def test_sync_max_files_cap_defers_rest(self):
        # 有界自愈（OPT-105 二期）：单次最多嵌 N 个最新变更，其余顺延且最终收敛
        idx = VectorIndex(self.vault / ".agent-brain" / "rag-index.sqlite",
                          self.emb, chunk_chars=200, vault_root=self.vault)
        idx.sync_vault()  # 基线：cat/dog 入库
        for name in ("a", "b", "c"):
            (self.vault / f"{name}.md").write_text(f"{name} 内容", encoding="utf-8")
        stat = idx.sync_vault(max_files=1)
        self.assertEqual((stat["updated"], stat["deferred"]), (1, 2))
        stat = idx.sync_vault(max_files=1)
        self.assertEqual((stat["updated"], stat["deferred"]), (1, 1))
        stat = idx.sync_vault(max_files=1)
        self.assertEqual((stat["updated"], stat["deferred"]), (1, 0))
        self.assertEqual(idx.sync_vault(max_files=1)["updated"], 0)  # 收敛

    def test_recaller_auto_sync_picks_up_new_file(self):
        # 文档更新自愈：新增笔记不经手动 reindex，下一次检索即命中
        emb = FakeEmbedder(["猫", "狗", "金鱼"])  # 词表须覆盖新笔记的语义词
        idx = VectorIndex(self.vault / ".agent-brain" / "rag-index.sqlite",
                          emb, chunk_chars=200, vault_root=self.vault,
                          auto_sync_limit=8)
        idx.sync_vault()
        (self.vault / "fish.md").write_text("金鱼在水里游。", encoding="utf-8")
        from agentlab.rag.vector_index import make_vector_recaller
        rec = make_vector_recaller(idx, k=5)
        hits = rec("金鱼")
        self.assertTrue(any("fish.md" in h["ref"] for h in hits), "自动同步应让新笔记立即可检索")

    def test_auto_sync_throttled_within_window(self):
        # 规模化（P0-1 三期）：2 秒扫描节流，高频查询不反复 rglob 大库
        idx = VectorIndex(self.vault / ".agent-brain" / "rag-index.sqlite",
                          self.emb, chunk_chars=200, vault_root=self.vault,
                          auto_sync_limit=8)
        idx.sync_vault()
        (self.vault / "new2.md").write_text("猫又来了", encoding="utf-8")
        r1 = idx.auto_sync()
        r2 = idx.auto_sync()
        self.assertEqual(r1["updated"], 1)
        self.assertEqual(r2, {})  # 节流窗内跳过扫描

    # —— L11/OPT-111 会话区段档案（ranges 表） ——
    def test_ranges_upsert_search_scoping_remove(self):
        self.idx.sync_vault(self.vault)  # vault 分块与区段共存一库、互不干扰
        self.idx.upsert_range("s1", 1, "讨论了猫的喂养计划。", chunk_chars=200)
        self.idx.upsert_range("s1", 2, "部署了狗屋的搭建方案。", chunk_chars=200)
        self.idx.upsert_range("s2", 1, "老鼠实验数据汇总。", chunk_chars=200)
        hits = self.idx.search_ranges("猫 喂养", k=5, session_id="s1")
        self.assertTrue(hits, "区段向量路应命中")
        self.assertTrue(all(h["session_id"] == "s1" for h in hits), "scope 隔离其他会话")
        self.assertEqual(hits[0]["source"], "session")
        self.assertTrue(hits[0]["ref"].startswith("session/s1#r"))
        self.assertIn("seq", hits[0])
        self.assertTrue(any(h["session_id"] == "s2"
                            for h in self.idx.search_ranges("老鼠", k=5)),
                        "不带 scope 可跨会话检索")
        self.idx.remove_session("s1")
        self.assertEqual(self.idx.search_ranges("猫", session_id="s1"), [])

    def test_vault_sync_never_touches_ranges(self):
        self.idx.sync_vault(self.vault)
        self.idx.upsert_range("s1", 1, "会话区段：猫又出现了。", chunk_chars=200)
        self.idx.sync_vault(self.vault)  # vault 重扫（含 removed 清理）不得清区段表
        self.assertTrue(self.idx.search_ranges("猫", session_id="s1"))

    def test_range_max_chunks_cap(self):
        text = "。".join(["很长内容"] * 400)
        n = self.idx.upsert_range("s1", 1, text, chunk_chars=50, max_chunks=3)
        self.assertEqual(n, 4, "3 块 + 1 超界标记块")


class TestRagToolsVector(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name)
        (self.vault / "cat.md").write_text("猫会抓老鼠。\n\n猫是宠物。", encoding="utf-8")
        self.emb = FakeEmbedder(["猫", "狗", "老鼠"])
        self.cfg = RagConfig(embed_base_url="http://fake", embed_model="t-v4")

    def tearDown(self):
        self.tmp.cleanup()

    def test_vector_tools_registered_and_retrieve_hits(self):
        tools = build_rag_tools(rag_config=self.cfg, vault_root=self.vault, embedder=self.emb)
        by_name = {t.name: t for t in tools}
        self.assertIn("rag_reindex", by_name)
        stat = json.loads(by_name["rag_reindex"].fn())
        self.assertEqual(stat["total"], 1)
        out = json.loads(by_name["rag_retrieve"].fn(query="猫", limit=5))
        self.assertTrue(out, "向量路应出结果")
        self.assertTrue(any(o["source"] == "vector" and "猫" in o["content"] for o in out))

    def test_no_embed_config_degrades_to_keyword_only(self):
        tools = build_rag_tools(rag_config=RagConfig(), vault_root=self.vault,
                                embedder=self.emb)  # 未配 embed_base_url
        self.assertEqual([t.name for t in tools], ["rag_retrieve", "rag_assess"])

    def test_extra_recaller_failure_degrades(self):
        def bad(q):
            raise RuntimeError("向量路炸了")
        def good(q):
            return [{"title": "t", "content": "c", "ref": "vault/x.md", "source": "vault"}]
        r = RAGRecall([good], extra=[bad])
        items = r.retrieve("查询")
        self.assertEqual(len(items), 1, "向量路失败应降级，不影响关键词路")

    def test_session_route_via_gateway(self):
        # L11/OPT-111：注入 gateway 后 rag_retrieve 多一路 "session"（当前会话窗口外内容）
        from agentlab.memory.ranges import RangeArchive, RangeGateway
        arch = RangeArchive(self.vault / "sessions")
        arch.append("s-abc", [Message(role="user", content="秘密口令是青花瓷，藏在书房")])
        tools = build_rag_tools(rag_config=self.cfg, vault_root=self.vault,
                                embedder=self.emb, range_gateway=RangeGateway(arch, k=3))
        rag = {t.name: t for t in tools}["rag_retrieve"]
        gw = RangeGateway(arch)  # 同一 ContextVar，绑定即生效
        tok = gw.bind("s-abc")
        try:
            out = json.loads(rag.fn(query="青花瓷", limit=5))
        finally:
            gw.reset(tok)
        self.assertTrue(any(o["source"] == "session" and "青花瓷" in o["content"]
                            for o in out), "绑定会话后 session 路应出结果")
        out0 = json.loads(rag.fn(query="青花瓷", limit=5))
        self.assertFalse(any(o["source"] == "session" for o in out0),
                         "未绑定会话时 session 路为空（降级不抛错）")


if __name__ == "__main__":
    unittest.main()