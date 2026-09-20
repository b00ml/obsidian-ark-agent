import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import Message, TokenUsage
from agentlab.contracts import RetrievalScope, RetrievalStatus, RetrievalStrategy
from agentlab.rag.assess import (
    AnswerEvidence,
    RAGAssessor,
    Sufficiency,
    _item_block,
    evaluate_answer_gate,
    evaluate_generated_answer,
    sanitize_generated_answer,
)
from agentlab.rag.recall import (
    RAGRecall, RecallItem, _canonical_memory_ref, fuse, retrieval_terms,
    split_keywords, govern_recall, expand_query_variants,
)
from agentlab.rag.vector_index import VectorIndex, chunk_text
from agentlab.runtime.config import RagConfig
from agentlab.tools.rag_tools import build_p2_store, build_rag_tools
from agentlab.prompts import load_prompt


def _item(title, content="", ref="", source="vault"):
    return RecallItem(title=title, content=content, ref=ref, source=source)


class TestRecall(unittest.TestCase):
    def test_canonical_memory_ref_resolves_markdown_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "ark" / "memory" / "core"
            root.mkdir(parents=True)
            (root / "mem-shared-default.md").write_text(
                "---\nid: mem-shared-default\n---\n\nInbox task processing.\n",
                encoding="utf-8",
            )
            ref = _canonical_memory_ref(
                "mem-shared-default", {"vault_path": temp_dir},
            )
            self.assertEqual(ref, "ark/memory/core/mem-shared-default.md#mem-shared-default")

    def test_canonical_memory_ref_keeps_legacy_fallback(self):
        self.assertEqual(
            _canonical_memory_ref("mem-unknown", {"vault_path": ""}),
            "memory#mem-unknown",
        )

    def test_canonical_memory_ref_resolves_bucket_entry_anchor(self):
        from agentlab.memory.markdown_store import MemoryMarkdownStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryMarkdownStore(temp_dir)
            mem_id = store.commit_to_bucket(
                "sessions", "桶内记忆的稳定引用", tags=["引用"]
            )
            ref = _canonical_memory_ref(mem_id, {"vault_path": temp_dir})
            self.assertTrue(ref.startswith("ark/memory/sessions/"))
            self.assertTrue(ref.endswith(f"#{mem_id}"))

    def test_canonical_memory_ref_does_not_expose_out_of_scope_path(self):
        with tempfile.TemporaryDirectory() as configured, tempfile.TemporaryDirectory() as other:
            memory_dir = Path(other) / "ark" / "memory" / "core"
            memory_dir.mkdir(parents=True)
            (memory_dir / "mem-outside.md").write_text(
                "---\nid: mem-outside\n---\n\noutside\n", encoding="utf-8",
            )
            self.assertEqual(
                _canonical_memory_ref("mem-outside", {"vault_path": configured}),
                "memory#mem-outside",
            )

    def test_default_memory_route_emits_canonical_markdown_ref(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir) / "ark" / "memory" / "core"
            memory_dir.mkdir(parents=True)
            (memory_dir / "mem-shared-default.md").write_text(
                "---\nid: mem-shared-default\nproject_id: default\n---\n\n"
                "Inbox task processing.\n",
                encoding="utf-8",
            )

            class FakeStore:
                available = True

                def search(self, _query, limit=20):
                    return {"results": []}

                def query(self, _query, limit=10, project_id=None):
                    return {"results": [{
                        "id": "mem-shared-default",
                        "content": "Inbox task processing.",
                    }]}

            with patch("agentlab.memory.store.MemoryStore", return_value=FakeStore()):
                routes = RAGRecall._default_recallers({"vault_path": temp_dir})
            memory_route = next(route for route in routes
                                if getattr(route, "_rag_source", "") == "memory")
            rows = memory_route("inbox task processing", RetrievalScope(project_id="proj-a"))
            self.assertEqual(rows[0]["ref"],
                             "ark/memory/core/mem-shared-default.md#mem-shared-default")

    def test_retrieve_result_forwards_scope_and_preserves_strategy(self):
        seen = []

        def rec(query, scope=None):
            seen.append((query, scope.project_id, scope.session_id))
            return [{"title": "项目笔记", "content": "证据", "ref": "p.md", "source": "vault"}]

        result = RAGRecall([rec], strategy=RetrievalStrategy.LEXICAL_ONLY).retrieve_result(
            "证据", scope={"project_id": "p1", "session_id": "s1"}, limit=1,
        )
        self.assertEqual(result.strategy, RetrievalStrategy.LEXICAL_ONLY)
        self.assertEqual(result.status, RetrievalStatus.AVAILABLE)
        self.assertEqual(result.scope.project_id, "p1")
        self.assertEqual(result.items[0].project_id, "p1")
        self.assertEqual(seen, [("证据", "p1", "s1")])

    def test_retrieve_result_marks_route_failure_degraded(self):
        def bad(_query):
            raise RuntimeError("route down")

        result = RAGRecall([bad], strategy="lexical-only").retrieve_result("q")
        self.assertEqual(result.status, RetrievalStatus.UNAVAILABLE)
        self.assertTrue(result.warnings)

    def test_split_keywords_chinese_and_en(self):
        self.assertEqual(split_keywords("对比 X 与 Y 视频的结论"), ["对比", "X", "Y", "视频的结论"])

    def test_retrieval_terms_windows_long_chinese(self):
        # 长中文串 → 补充 2/4 字窗口；短/英文标识符保持原样
        terms = retrieval_terms("Obsidian 知识管理的核心工具")
        self.assertIn("Obsidian", terms)
        self.assertIn("知识管理", terms)
        self.assertIn("工具", terms)
        self.assertEqual(retrieval_terms("abcd"), ["abcd"])

    def test_local_query_expansion_removes_question_words_and_segments(self):
        variants = expand_query_variants("帮我处理视频任务前要注意什么")
        self.assertTrue(variants)
        self.assertTrue(any("处理视频任务" in value for value in variants))
        self.assertNotIn("帮我处理视频任务前要注意什么", variants)

    def test_local_query_expansion_does_not_promote_generic_topic_tokens(self):
        variants = expand_query_variants("Obsidian 笔记库覆盖了哪些主题")
        self.assertTrue(any("Obsidian 笔记库覆盖了哪些主题" not in value
                            for value in variants))
        self.assertNotIn("AI", variants)
        self.assertNotIn("知识", variants)

    def test_local_query_expansion_bridges_video_workflow_synonyms(self):
        from agentlab.rag.recall import expand_query_variants

        variants = expand_query_variants("帮我处理视频任务前要注意什么", max_variants=5)
        self.assertIn("B站视频处理前核对 bili_meta", variants)
        self.assertIn("收件箱 B站处理前先核对", variants)
        self.assertNotIn("主题", variants)

    def test_local_query_expansion_keeps_negative_and_identifier_queries_closed(self):
        self.assertEqual(expand_query_variants("有没有 BV1ABC123456 的记录"), [])
        self.assertEqual(expand_query_variants("BV1ABC123456"), [])

    def test_query_expansion_shadow_does_not_change_recall(self):
        calls = []

        def rec(query):
            calls.append(query)
            return [{"title": query, "content": query, "ref": f"{query}.md",
                     "source": "vault"}]

        shadow = RAGRecall([rec], query_expansion_mode="shadow")
        rows = shadow.retrieve("如何处理视频任务", limit=2)
        self.assertEqual([row.ref for row in rows], ["如何处理视频任务.md"])
        self.assertGreater(len(calls), 1)

    def test_query_expansion_on_fuses_variant_candidates(self):
        def rec(query):
            if "处理视频任务" in query and query != "如何处理视频任务":
                return [{"title": "gold", "content": "gold", "ref": "gold.md",
                         "source": "vault"}]
            return []

        rows = RAGRecall([rec], query_expansion_mode="on").retrieve(
            "如何处理视频任务", limit=2,
        )
        self.assertEqual(rows[0].ref, "gold.md")

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

    def test_p2_can_replace_only_legacy_vault_route(self):
        def legacy_vault(q):
            return [{"title": "旧 Vault", "content": "旧", "ref": "notes/a.md", "source": "vault"}]
        legacy_vault._rag_source = "vault"

        def memory(q):
            return [{"title": "记忆", "content": "辅助", "ref": "memory#1", "source": "memory"}]
        memory._rag_source = "memory"

        p2 = lambda q: [{
            "title": "P2 chunk", "content": "最新", "ref": "notes/a.md#h:ch1", "source": "lexical",
        }]
        with patch.object(RAGRecall, "_default_recallers", return_value=[legacy_vault, memory]):
            items = RAGRecall(extra=[p2], exclude_sources={"vault"}).retrieve("最新")
        refs = {item.ref for item in items}
        self.assertIn("notes/a.md#h:ch1", refs)
        self.assertIn("memory#1", refs)
        self.assertNotIn("notes/a.md", refs)

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
    def test_rag_prompts_require_exact_candidate_refs(self):
        system = load_prompt("system-user", tools="", skills="", memory="")
        assessor = load_prompt("rag-assess-user", query="q", items="[]", refs="")
        self.assertIn("精确 `ref`", system)
        self.assertIn("普通代码文本", system)
        self.assertIn("（ref: <精确 ref>）", system)
        self.assertIn("可用 ref 白名单", assessor)
        self.assertIn("不能把", assessor)

    def test_assessor_item_block_separates_ref_from_source(self):
        rendered = _item_block([
            _item("笔记", content="证据", ref="wiki/a.md#h:ch1", source="lexical")
        ])
        self.assertIn('ref="wiki/a.md#h:ch1"', rendered)
        self.assertIn('source="lexical"', rendered)
        self.assertNotIn("lexical/wiki/a.md", rendered)

    def test_sanitize_generated_answer_demotes_out_of_scope_wikilinks(self):
        answer = "依据 [[wiki/a.md]]，另见 [[wiki/related]]。"
        sanitized = sanitize_generated_answer(answer, ["wiki/a.md#h:ch1"])
        self.assertIn("[[wiki/a.md]]", sanitized)
        self.assertIn("`wiki/related`", sanitized)
        self.assertNotIn("[[wiki/related]]", sanitized)

    def test_generated_answer_does_not_parse_source_edition_as_citation(self):
        answer = (
            "保留 `wiki/source.md` 为来源版（文件夹总结也是这么标注的："
            "ref: `wiki/a.md#h:ch1`）。"
        )
        result = evaluate_generated_answer(answer, ["wiki/a.md#h:ch1"])
        self.assertTrue(result.allowed)

    def test_generated_answer_rejects_missing_and_outside_citations(self):
        missing = evaluate_generated_answer(
            "这是一个确定结论。", ["wiki/a.md"], require_citation=True,
        )
        self.assertIn("citation_missing", missing.reasons)
        outside = evaluate_generated_answer(
            "依据 wiki/b.md。", ["wiki/a.md"], require_citation=True,
        )
        self.assertIn("citation_out_of_scope", outside.reasons)

    def test_generated_answer_accepts_chunk_parent_and_explicit_abstention(self):
        accepted = evaluate_generated_answer(
            "结论见 [[wiki/a.md]]。", ["wiki/a.md#intro:ch123"],
        )
        self.assertTrue(accepted.allowed)
        abstained = evaluate_generated_answer(
            "当前资料不足，无法确认。", ["wiki/a.md"],
        )
        self.assertFalse(abstained.allowed)
        self.assertTrue(abstained.abstained)
        evidence_limited = evaluate_generated_answer(
            "现有资料未覆盖该偏好；可确认的是用户偏好深色主题（ref: wiki/a.md）。",
            ["wiki/a.md"],
        )
        self.assertFalse(evidence_limited.allowed)
        self.assertTrue(evidence_limited.abstained)

    def test_generated_answer_does_not_mark_qualified_uncertainty_as_abstention(self):
        answer = (
            "依据 [[wiki/a.md]]，当前流程可以按文档中的三步执行。"
            "其中一个未覆盖的边界条件仍无法确认，但不影响上述结论。"
        )
        result = evaluate_generated_answer(answer, ["wiki/a.md"], require_citation=True)
        self.assertTrue(result.allowed)
        self.assertFalse(result.abstained)

    def test_generated_answer_accepts_cited_non_supporting_proposition(self):
        answer = (
            "资料并不支持把两种记忆定义当作互斥选项；"
            "它们是同一治理框架的两种描述（ref: wiki/a.md）。"
        )
        result = evaluate_generated_answer(answer, ["wiki/a.md"], require_citation=True)
        self.assertTrue(result.allowed)
        self.assertFalse(result.abstained)

    def test_bounded_partial_is_opt_in_and_requires_citation_and_limitation(self):
        answer = "依据 [[wiki/a.md]]，可以确认已记录的三步；其余背景资料未覆盖。"
        blocked = evaluate_generated_answer(
            answer, ["wiki/a.md"], assessment="insufficient", require_citation=True,
        )
        self.assertFalse(blocked.allowed)
        self.assertIn("assessment_requires_abstention", blocked.reasons)
        allowed = evaluate_generated_answer(
            answer, ["wiki/a.md"], assessment="insufficient", require_citation=True,
            allow_bounded_partial=True,
        )
        self.assertTrue(allowed.allowed)
        self.assertIn("bounded_partial", allowed.reasons)
        unbounded = evaluate_generated_answer(
            "依据 [[wiki/a.md]]，结论就是这样。", ["wiki/a.md"],
            assessment="insufficient", allow_bounded_partial=True,
        )
        self.assertFalse(unbounded.allowed)

        paraphrased = evaluate_generated_answer(
            "依据 [[wiki/a.md]]，下面只复述命中的片段，不补充库外结论。",
            ["wiki/a.md"], assessment="insufficient", require_citation=True,
            allow_bounded_partial=True,
        )
        self.assertTrue(paraphrased.allowed)
        self.assertIn("bounded_partial", paraphrased.reasons)

        semantic_abstention = evaluate_generated_answer(
            "依据 [[wiki/a.md]]，下面只复述命中的片段。",
            ["wiki/a.md"], answerability="insufficient",
            assessment="insufficient", require_citation=True,
            allow_bounded_partial=True,
        )
        self.assertFalse(semantic_abstention.allowed)
        self.assertIn("answerability_insufficient", semantic_abstention.reasons)

    def test_generated_answer_accepts_wikilink_with_spaces_without_suffix_leak(self):
        ref = "wiki/notes/带 空格.md#标题:ch123"
        accepted = evaluate_generated_answer(
            f"结论见 [[{ref.split('#', 1)[0]}]]。", [ref],
        )
        self.assertTrue(accepted.allowed)

    def test_generated_answer_ignores_code_and_template_paths(self):
        answer = (
            '调用 `rag_retrieve("BV1ZZZZ99999")` 后，模板为 `Inbox/{标题}-总结.md`。'
            "没有可核验的来源。"
        )
        result = evaluate_generated_answer(answer, ["wiki/a.md"], require_citation=True)
        self.assertIn("citation_missing", result.reasons)
        self.assertNotIn("citation_out_of_scope", result.reasons)

    def test_generated_answer_accepts_candidate_plain_path_without_citation_cue(self):
        result = evaluate_generated_answer(
            "检索结果包含 wiki/a.md，可用于核对。", ["wiki/a.md#h:ch1"],
        )
        self.assertTrue(result.allowed)

    def test_generated_answer_accepts_matching_inline_code_path(self):
        result = evaluate_generated_answer(
            "来源：`wiki/a.md`。", ["wiki/a.md#h:ch1"],
        )
        self.assertTrue(result.allowed)

    def test_generated_answer_accepts_note_title_wikilink_against_vault_path(self):
        result = evaluate_generated_answer(
            "来源见 [[京东Agent开发面试全程实录-总结]]。",
            ["Inbox/京东Agent开发面试全程实录-总结.md#h:ch1"],
        )
        self.assertTrue(result.allowed)

    def test_generated_answer_ignores_placeholder_wikilinks(self):
        result = evaluate_generated_answer(
            "示例格式是 [[wikilink]] 或 [[{标题}-逐字稿]]。",
            ["wiki/a.md#h:ch1"],
        )
        self.assertIn("citation_missing", result.reasons)
        self.assertNotIn("citation_out_of_scope", result.reasons)

    def test_generated_answer_does_not_equate_same_filename_in_other_folder(self):
        result = evaluate_generated_answer(
            "依据 other/a.md。", ["wiki/a.md#h:ch1"],
        )
        self.assertIn("citation_out_of_scope", result.reasons)

    def test_answer_evidence_observes_retrieve_and_assess(self):
        evidence = AnswerEvidence()
        evidence.observe("rag_retrieve", json.dumps([
            {"ref": "wiki/a.md#intro:ch123", "status": "active"},
            {"ref": "wiki/old.md", "status": "archived"},
        ]))
        evidence.observe("rag_assess", json.dumps({
            "sufficient": True, "action": "answer", "answerability": "answerable",
        }))
        result = evidence.check("结论见 wiki/a.md#intro:ch123")
        self.assertIsNotNone(result)
        self.assertTrue(result.allowed)
        self.assertEqual(evidence.candidate_refs, {"wiki/a.md#intro:ch123"})

    def test_answer_evidence_fresh_retrieval_clears_stale_semantic_verdict(self):
        """Follow-up retrieval must be evaluated against its own candidates."""
        evidence = AnswerEvidence()
        evidence.observe("rag_retrieve", json.dumps({
            "items": [{"ref": "wiki/old.md", "status": "active"}],
        }))
        evidence.observe("rag_assess", json.dumps({
            "sufficient": False, "action": "insufficient",
            "answerability": "insufficient",
        }))
        evidence.observe("rag_retrieve", json.dumps({
            "items": [{"ref": "wiki/new.md", "status": "active"}],
        }))
        result = evidence.check("结论见 wiki/new.md")
        self.assertTrue(result.allowed)
        self.assertEqual(result.allowed_refs, ("wiki/new.md",))
        self.assertEqual(evidence.answerability, "answerable")

    def test_answer_gate_rejects_absent_and_inactive_evidence(self):
        absent = evaluate_answer_gate("不存在的问题", [_item("A", "x", "a.md")],
                                     answerability="absent")
        self.assertFalse(absent.allowed)
        inactive = evaluate_answer_gate(
            "q", [{"ref": "a.md", "status": "candidate", "project_id": "p"}],
            scope={"project_id": "p"},
        )
        self.assertFalse(inactive.allowed)
        self.assertIn("inactive:candidate", inactive.reasons)

        conflict = evaluate_answer_gate(
            "q", [{"ref": "conflict.md", "status": "conflict", "project_id": "p"}],
            scope={"project_id": "p"},
        )
        self.assertFalse(conflict.allowed)
        self.assertIn("inactive:conflict", conflict.reasons)

    def test_answer_gate_rejects_forbidden_and_cross_project_refs(self):
        result = evaluate_answer_gate(
            "q",
            [
                {"ref": "p2.md", "project_id": "p2", "status": "active"},
                {"ref": "secret.md", "project_id": "p1", "status": "active"},
            ],
            scope={"project_id": "p1"}, forbidden_refs=["secret.md"],
        )
        self.assertFalse(result.allowed)
        self.assertIn("scope_denied", result.reasons)
        self.assertIn("forbidden_ref", result.reasons)

    def test_answer_gate_requires_expected_refs(self):
        result = evaluate_answer_gate(
            "q", [_item("A", "a", "a.md")], required_refs=["b.md"],
        )
        self.assertFalse(result.allowed)
        self.assertIn("required_ref_missing", result.reasons)

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

    def test_assess_rejects_action_contract_mismatch(self):
        prov = _AssessProvider({
            "sufficient": False, "action": "answer", "refs": [],
        })
        from agentlab.core.errors import AgentError
        with self.assertRaises(AgentError):
            asyncio.run(RAGAssessor(prov).assess("q", [_item("A", "a", "a.md")]))

    def test_assess_abstains_from_citation_outside_candidates(self):
        prov = _AssessProvider({
            "sufficient": True, "action": "answer", "refs": ["other.md"],
        })
        result = asyncio.run(RAGAssessor(prov).assess(
            "q", [_item("A", "a", "a.md#h:ch12345678")],
        ))
        self.assertFalse(result.sufficient)
        self.assertEqual(result.action, "insufficient")
        self.assertIn("本次检索之外", result.message)


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
        cfg = RagConfig(embed_base_url="http://fake", embed_model="t-v4", vector_mode="on")
        tools = build_rag_tools(rag_config=cfg, vault_root=self.vault, embedder=self.emb)
        by_name = {t.name: t for t in tools}
        self.assertIn("rag_reindex", by_name)
        stat = json.loads(by_name["rag_reindex"].fn())
        self.assertEqual(stat["total"], 1)
        out = json.loads(by_name["rag_retrieve"].fn(query="猫", limit=5))
        self.assertTrue(out, "向量路应出结果")
        self.assertTrue(any(o["source"] == "vector" and "猫" in o["content"] for o in out))

    def test_shadow_vector_route_is_observed_but_not_displayed(self):
        tools = build_rag_tools(rag_config=self.cfg, vault_root=self.vault, embedder=self.emb)
        by_name = {t.name: t for t in tools}
        json.loads(next(t for t in tools if t.name == "rag_reindex").fn())
        out = json.loads(by_name["rag_retrieve"].fn(query="猫", limit=5))
        self.assertFalse(any(o["source"] == "vector" for o in out))

    def test_rag_reindex_is_write_side_tool(self):
        # OPT-218：真机 trace 显示模型在写完笔记后"顺手"重建全库索引（17s + embedding
        # 费用）——rag_reindex 从 read 收紧为 write（走审批），并显式声明契约
        tools = build_rag_tools(rag_config=self.cfg, vault_root=self.vault, embedder=self.emb)
        reindex = next(t for t in tools if t.name == "rag_reindex")
        self.assertEqual(reindex.permission, "write")
        self.assertEqual(reindex.side_effects, "index")
        self.assertTrue(reindex.idempotent)
        self.assertIn("不会每日全量嵌入", reindex.description)
        self.assertIn("hash 变化", reindex.description)


    def test_sync_excludes_memory_archive(self):
        """OPT-227 A2：ark/memory/archive/ 不入向量索引（默认不可召回历史）。"""
        (self.vault / "ark" / "memory" / "archive" / "2026").mkdir(parents=True, exist_ok=True)
        (self.vault / "ark" / "memory" / "archive" / "2026" / "archived.md").write_text(
            "归档内容：量子褶皱引擎", encoding="utf-8")
        tools = build_rag_tools(rag_config=self.cfg, vault_root=self.vault, embedder=self.emb)
        by_name = {t.name: t for t in tools}
        by_name["rag_reindex"].fn()
        import json as _json
        out = _json.loads(by_name["rag_retrieve"].fn(query="量子褶皱引擎", limit=10))
        refs = [_o.get("ref") or _o.get("path", "") for _o in out]
        self.assertTrue(all("archive" not in r for r in refs),
                        f"归档内容不应进向量召回: {refs}")

    def test_no_embed_config_degrades_to_keyword_only(self):
        tools = build_rag_tools(rag_config=RagConfig(), vault_root=self.vault,
                                embedder=self.emb)  # 未配 embed_base_url
        self.assertEqual([t.name for t in tools], ["rag_retrieve", "rag_assess"])

    def test_rag_assess_does_not_trust_model_answerability_hint(self):
        """A model-supplied stale ``insufficient`` label must not poison the gate."""
        provider = _AssessProvider({
            "sufficient": True,
            "action": "answer",
            "reformulated_query": "",
            "message": "候选足以支持结论",
            "refs": ["cat.md"],
            "answerability": "answerable",
        })
        tools = build_rag_tools(
            rag_config=RagConfig(), vault_root=self.vault, llm=provider,
        )
        assess = next(tool for tool in tools if tool.name == "rag_assess")
        result = asyncio.run(assess.fn(
            query="猫会做什么？",
            items=json.dumps([{
                "title": "猫",
                "content": "猫会抓老鼠。",
                "ref": "cat.md",
                "source": "vault",
                "status": "active",
            }], ensure_ascii=False),
            answerability="insufficient",
        ))
        parsed = json.loads(result)
        self.assertTrue(parsed["sufficient"])
        self.assertEqual(parsed["answerability"], "answerable")

        schema = assess.schema["function"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"query", "items"})

    def test_rag_assess_uses_latest_request_retrieval_when_items_are_compressed(self):
        provider = _AssessProvider({
            "sufficient": True, "action": "answer", "reformulated_query": "",
            "message": "有直接证据", "refs": ["cat.md"],
            "answerability": "answerable",
        })
        recaller = lambda _query: [{
            "title": "猫", "content": "猫会抓老鼠。", "ref": "cat.md",
            "source": "vault", "status": "active",
        }]
        with patch.object(RAGRecall, "_default_recallers", return_value=[recaller]):
            tools = build_rag_tools(rag_config=RagConfig(), vault_root=self.vault,
                                    llm=provider)
        retrieve = next(tool for tool in tools if tool.name == "rag_retrieve")
        assess = next(tool for tool in tools if tool.name == "rag_assess")
        json.loads(retrieve.fn(query="猫", limit=3))
        result = asyncio.run(assess.fn(query="猫会做什么？", items="cat.md"))
        parsed = json.loads(result)
        self.assertTrue(parsed["sufficient"])

    def test_rag_assess_does_not_reuse_unrelated_request_candidates(self):
        provider = _AssessProvider({
            "sufficient": True, "action": "answer", "reformulated_query": "",
            "message": "有直接证据", "refs": ["cat.md"],
            "answerability": "answerable",
        })
        recaller = lambda _query: [{
            "title": "猫", "content": "猫会抓老鼠。", "ref": "cat.md",
            "source": "vault", "status": "active",
        }]
        with patch.object(RAGRecall, "_default_recallers", return_value=[recaller]):
            tools = build_rag_tools(rag_config=RagConfig(), vault_root=self.vault,
                                    llm=provider)
        retrieve = next(tool for tool in tools if tool.name == "rag_retrieve")
        assess = next(tool for tool in tools if tool.name == "rag_assess")
        json.loads(retrieve.fn(query="猫", limit=3))
        result = asyncio.run(assess.fn(query="狗会做什么？", items="cat.md"))
        self.assertFalse(json.loads(result)["sufficient"])

    def test_rag_assess_rejects_injected_item_outside_current_registry(self):
        provider = _AssessProvider({
            "sufficient": True, "action": "answer", "reformulated_query": "",
            "message": "伪造候选看似充分", "refs": ["injected.md"],
            "answerability": "answerable",
        })
        recaller = lambda _query: [{
            "title": "猫", "content": "猫会抓老鼠。", "ref": "cat.md",
            "source": "vault", "status": "active",
        }]
        with patch.object(RAGRecall, "_default_recallers", return_value=[recaller]):
            tools = build_rag_tools(rag_config=RagConfig(), vault_root=self.vault, llm=provider)
        retrieve = next(tool for tool in tools if tool.name == "rag_retrieve")
        assess = next(tool for tool in tools if tool.name == "rag_assess")
        json.loads(retrieve.fn(query="猫", limit=3))
        injected = json.dumps([{
            "title": "伪造", "content": "不可信事实", "ref": "injected.md",
            "source": "vault", "status": "active",
        }], ensure_ascii=False)
        result = asyncio.run(assess.fn(query="猫会做什么？", items=injected))
        parsed = json.loads(result)
        self.assertFalse(parsed["sufficient"])
        self.assertEqual(parsed["answerability"], "scope_denied")

    def test_embedding_can_be_disabled_or_stacked_with_keyword_route(self):
        keyword_only = build_rag_tools(
            rag_config=RagConfig(vector_enabled=False),
            vault_root=self.vault,
            embedder=self.emb,
        )
        self.assertNotIn("rag_reindex", {tool.name for tool in keyword_only})

        stacked = build_rag_tools(
            rag_config=RagConfig(
                vector_enabled=True,
                embed_base_url="http://fake",
                vector_mode="on",
                lexical_mode="on",
            ),
            vault_root=self.vault,
            embedder=self.emb,
        )
        names = {tool.name for tool in stacked}
        self.assertIn("rag_reindex", names)
        reindex = next(tool for tool in stacked if tool.name == "rag_reindex")
        retrieve = next(tool for tool in stacked if tool.name == "rag_retrieve")
        json.loads(reindex.fn())
        rows = json.loads(retrieve.fn(query="猫", limit=5))
        self.assertTrue(rows)
        self.assertTrue(any(row["source"] == "vector" for row in rows))

    def test_vector_only_mode_excludes_keyword_candidates(self):
        """显式 vector-only 时不得把默认关键词路悄悄带回结果。"""
        lexical = lambda _query: [{
            "title": "关键词候选", "content": "只应在关键词路出现",
            "ref": "lexical.md", "source": "vault",
        }]
        cfg = RagConfig(
            vector_enabled=True,
            embed_base_url="http://fake",
            embed_model="t-v4",
            vector_mode="on",
            lexical_mode="off",
        )
        with patch.object(RAGRecall, "_default_recallers", return_value=[lexical]):
            tools = build_rag_tools(rag_config=cfg, vault_root=self.vault,
                                    embedder=self.emb)
        by_name = {tool.name: tool for tool in tools}
        json.loads(by_name["rag_reindex"].fn())
        rows = json.loads(by_name["rag_retrieve"].fn(query="猫", limit=5))
        self.assertTrue(rows)
        self.assertTrue(all(row["source"] == "vector" for row in rows))
        self.assertNotIn("lexical.md", {row["ref"] for row in rows})

    def test_vector_mode_off_never_builds_embedding_route(self):
        tools = build_rag_tools(
            rag_config=RagConfig(
                vector_enabled=True,
                embed_base_url="http://fake",
                vector_mode="off",
            ),
            vault_root=self.vault,
            embedder=self.emb,
        )
        self.assertNotIn("rag_reindex", {tool.name for tool in tools})

    def test_auto_backend_without_p2_file_keeps_legacy_route(self):
        """未完成 P2 reconcile 时，auto 不应把空索引切到生产召回。"""
        cfg = RagConfig(embed_base_url="http://fake", embed_model="t-v4", vector_mode="on")
        tools = build_rag_tools(rag_config=cfg, vault_root=self.vault, embedder=self.emb)
        by_name = {tool.name: tool for tool in tools}
        self.assertIn("rag_reindex", by_name, "legacy provider 路仍应可用")
        self.assertIsNone(build_p2_store(cfg, str(self.vault), embedder=self.emb))
        json.loads(by_name["rag_reindex"].fn())
        rows = json.loads(by_name["rag_retrieve"].fn(query="猫", limit=5))
        self.assertTrue(rows)
        self.assertTrue(any(row["source"] == "vector" for row in rows))

    def test_auto_backend_activates_only_current_p2_index(self):
        """auto must retain legacy fallback while a P2 build is stale."""
        p2_config = RagConfig(
            index_backend="p2", vector_enabled=False, vector_mode="off",
        )
        p2_tools = build_rag_tools(rag_config=p2_config, vault_root=self.vault)
        json.loads(next(tool for tool in p2_tools if tool.name == "rag_reindex").fn())

        auto_config = RagConfig(vector_enabled=False, vector_mode="off")
        active = build_p2_store(auto_config, str(self.vault))
        self.assertIsNotNone(active, "healthy P2 index may be selected by auto")

        (self.vault / "cat.md").write_text("猫会守护花园。", encoding="utf-8")
        self.assertIsNone(
            build_p2_store(auto_config, str(self.vault)),
            "source changes must keep auto on the legacy rollback route until reconciled",
        )

    def test_auto_backend_rejects_incompatible_p2_index(self):
        """A version mismatch must not turn an auto migration into an outage."""
        old_config = RagConfig(
            index_backend="p2", index_version="old-p2", vector_enabled=False,
            vector_mode="off",
        )
        old_tools = build_rag_tools(rag_config=old_config, vault_root=self.vault)
        json.loads(next(tool for tool in old_tools if tool.name == "rag_reindex").fn())

        auto_config = RagConfig(vector_enabled=False, vector_mode="off")
        self.assertIsNone(build_p2_store(auto_config, str(self.vault)))

    def test_explicit_p2_backend_supports_lexical_without_embedding(self):
        """P2 可先完成词法迁移，未配置 provider 时仍不影响关键词召回。"""
        cfg = RagConfig(index_backend="p2", vector_enabled=False, vector_mode="off")
        tools = build_rag_tools(rag_config=cfg, vault_root=self.vault)
        by_name = {tool.name: tool for tool in tools}
        self.assertIn("rag_reindex", by_name)
        json.loads(by_name["rag_reindex"].fn())
        rows = json.loads(by_name["rag_retrieve"].fn(query="猫", limit=5))
        self.assertTrue(rows)
        self.assertTrue(any(row["source"] == "lexical" for row in rows))

    def test_explicit_p2_backend_supports_hybrid_modes(self):
        """P2 production wiring exposes on/shadow/vector-only without legacy index."""
        cfg = RagConfig(
            index_backend="p2", embed_base_url="http://fake", embed_model="t-v4",
            vector_mode="on", lexical_mode="on",
        )
        tools = build_rag_tools(rag_config=cfg, vault_root=self.vault, embedder=self.emb)
        by_name = {tool.name: tool for tool in tools}
        json.loads(by_name["rag_reindex"].fn())
        rows = json.loads(by_name["rag_retrieve"].fn(query="猫", limit=5))
        self.assertTrue(rows)
        self.assertTrue(any(row["source"] in {"lexical", "vector"} for row in rows))

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

    def test_retrieve_envelope_contains_scope_and_strategy(self):
        cfg = RagConfig(index_backend="p2", vector_enabled=False, vector_mode="off")
        tools = build_rag_tools(rag_config=cfg, vault_root=self.vault)
        by_name = {tool.name: tool for tool in tools}
        json.loads(by_name["rag_reindex"].fn())
        from agentlab.contracts import bind_retrieval_scope, reset_retrieval_scope
        token = bind_retrieval_scope({"project_id": "default", "session_id": "s1"})
        try:
            result = json.loads(by_name["rag_retrieve"].fn(
                query="猫", limit=5, envelope=True,
            ))
        finally:
            reset_retrieval_scope(token)
        self.assertEqual(result["contract"], "ark-contract-v1")
        self.assertEqual(result["strategy"], "lexical-only")
        self.assertEqual(result["scope"]["project_id"], "default")
        self.assertTrue(result["items"])

    def test_retrieve_envelope_enforces_project_scope_in_p2(self):
        (self.vault / "p1.md").write_text(
            "---\nproject_id: p1\n---\n\n猫属于项目一。", encoding="utf-8",
        )
        (self.vault / "p2.md").write_text(
            "---\nproject_id: p2\n---\n\n猫属于项目二。", encoding="utf-8",
        )
        cfg = RagConfig(index_backend="p2", vector_enabled=False, vector_mode="off")
        tools = build_rag_tools(rag_config=cfg, vault_root=self.vault)
        by_name = {tool.name: tool for tool in tools}
        json.loads(by_name["rag_reindex"].fn())
        from agentlab.contracts import bind_retrieval_scope, reset_retrieval_scope
        token = bind_retrieval_scope({"project_id": "p1"})
        try:
            result = json.loads(by_name["rag_retrieve"].fn(
                query="猫", limit=10, envelope=True,
            ))
        finally:
            reset_retrieval_scope(token)
        refs = {item["ref"] for item in result["items"]}
        self.assertTrue(refs)
        self.assertTrue(all("p1.md" in ref for ref in refs))
        self.assertFalse(any("p2.md" in ref for ref in refs))


if __name__ == "__main__":
    unittest.main()
