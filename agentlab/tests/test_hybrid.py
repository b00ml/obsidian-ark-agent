import unittest
import json
import tempfile
import time

from agentlab.rag.hybrid import HybridRetriever, ShadowLogWriter, entry_key, hybrid_fuse
from agentlab.runtime.config import RagConfig


class StubStore:
    index_version = "s1-p2-v1"
    parser_version = "markdown-structure-v1"
    embedding_model = "fake-v1"

    def __init__(self):
        self.calls = []

    def search_lexical(self, query, k=20, **kwargs):
        self.calls.append(("lexical", query, k, kwargs))
        return [
            {"title": "桶", "content": "甲", "ref": "sessions/2026-09.md#mem-a:ch11111111", "score": 9},
            {"title": "桶", "content": "乙", "ref": "sessions/2026-09.md#mem-a:ch22222222", "score": 8},
            {"title": "普通", "content": "丙", "ref": "notes/c.md#h:ch33333333", "score": 7},
        ][:k]

    def search_vector(self, query, k=20, **kwargs):
        self.calls.append(("vector", query, k, kwargs))
        return [
            {"title": "桶", "content": "甲", "ref": "sessions/2026-09.md#mem-a:ch11111111", "score": .9},
            {"title": "向量", "content": "丁", "ref": "notes/d.md#h:ch44444444", "score": .8},
        ][:k]


class TestHybrid(unittest.TestCase):
    def test_lexical_and_vector_routes_run_concurrently(self):
        class SlowStore(StubStore):
            def search_lexical(self, query, k=20, **kwargs):
                time.sleep(0.08)
                return super().search_lexical(query, k=k, **kwargs)

            def search_vector(self, query, k=20, **kwargs):
                time.sleep(0.08)
                return super().search_vector(query, k=k, **kwargs)

        started = time.perf_counter()
        HybridRetriever(SlowStore(), vector_mode="on", candidate_k=5).retrieve("猫", limit=2)
        self.assertLess(time.perf_counter() - started, 0.14)

    def test_entry_key_removes_chunk_suffix(self):
        self.assertEqual(entry_key("a.md#mem-x:ch1234abcd"), "a.md#mem-x")
        self.assertEqual(entry_key("a.md#c12"), "a.md")
        self.assertEqual(entry_key("a.md#mem-x:ch1234abcd", "file"), "a.md")
        self.assertEqual(entry_key("a.md#mem-x:ch1234abcd", "ref"), "a.md#mem-x:ch1234abcd")
        self.assertEqual(entry_key("a.md#doc:ch1234abcd"), "a.md")

    def test_hybrid_fuse_dedupes_within_route_before_rrf(self):
        rows = hybrid_fuse({
            "lexical": [
                {"ref": "a.md#mem-x:ch11111111", "content": "first"},
                {"ref": "a.md#mem-x:ch22222222", "content": "second"},
            ],
            "vector": [{"ref": "a.md#mem-x:ch33333333", "content": "first"}],
        }, k=8)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entry_key"], "a.md#mem-x")
        self.assertEqual(rows[0]["ref"], "a.md#mem-x:ch11111111")

    def test_coverage_contract_reserves_each_required_entry(self):
        class CrowdedStore(StubStore):
            def search_lexical(self, query, k=20, **kwargs):
                return [
                    {"ref": "docs/a.md#h1:ch11111111", "content": "a1", "score": 10},
                    {"ref": "docs/a.md#h2:ch22222222", "content": "a2", "score": 9},
                    {"ref": "docs/a.md#h3:ch33333333", "content": "a3", "score": 8},
                    {"ref": "docs/b.md#h1:ch44444444", "content": "b", "score": 7},
                ][:k]

        retriever = HybridRetriever(
            CrowdedStore(), vector_mode="off", candidate_k=4,
        )
        record = retriever.shadow_record(
            "q", limit=2,
            required_groups=[["docs/a.md#h1"], ["docs/b.md#h1"]],
        )
        self.assertEqual(
            {entry_key(ref) for ref in record["hybrid_refs"]},
            {"docs/a.md#h1", "docs/b.md#h1"},
        )
        self.assertEqual(record["coverage_contract"]["satisfied_groups"], 2)

    def test_entry_dedupe_replaces_heading_only_chunk_with_body_chunk(self):
        rows = hybrid_fuse({
            "lexical": [
                {"ref": "a.md#mem-x:ch11111111", "content": "# Heading"},
                {"ref": "a.md#mem-x:ch22222222", "content": "正文事实"},
            ],
        }, k=8)
        self.assertEqual(rows[0]["ref"], "a.md#mem-x:ch22222222")
        self.assertEqual(rows[0]["content"], "正文事实")

    def test_off_and_shadow_do_not_display_vector_results(self):
        store = StubStore()
        off = HybridRetriever(store, vector_mode="off", candidate_k=5)
        off_rows = off.retrieve("猫", limit=2)
        self.assertEqual([item.source for item in off_rows], ["lexical", "lexical"])
        self.assertFalse(any(call[0] == "vector" for call in store.calls))

        store.calls.clear()
        shadow = HybridRetriever(store, vector_mode="shadow", candidate_k=5)
        shadow_rows = shadow.retrieve("猫", limit=2)
        self.assertEqual(len(shadow_rows), 2)
        self.assertTrue(any(call[0] == "vector" for call in store.calls))
        record = shadow.shadow_record("猫", limit=2, project_id="p1")
        self.assertEqual(record["schema"], "rag-hybrid-shadow-v2")
        self.assertEqual(len(record["query_hash"]), 64)
        self.assertIn("vector", record["routes"])
        self.assertEqual(record["project_id"], "p1")
        self.assertNotIn("content", str(record))

    def test_retrieve_with_shadow_record_collects_routes_once(self):
        store = StubStore()
        retriever = HybridRetriever(store, vector_mode="shadow", candidate_k=5)
        items, record = retriever.retrieve_with_shadow_record("猫", limit=2)
        self.assertEqual(len(items), 2)
        self.assertIn("vector", record["routes"])
        self.assertEqual([call[0] for call in store.calls], ["lexical", "vector"])

    def test_guarded_vector_fallback_is_semantic_only(self):
        store = StubStore()
        retriever = HybridRetriever(
            store, vector_mode="shadow", vector_fallback_mode="on", candidate_k=5,
        )
        rows, record = retriever.retrieve_with_shadow_record(
            "原始素材存放在哪里", limit=5,
        )
        self.assertIn("notes/d.md#h:ch44444444", [row.ref for row in rows])
        self.assertEqual(record["display_mode"], "guarded_hybrid_fallback")

        rows, record = retriever.retrieve_with_shadow_record("猫", limit=5)
        self.assertNotIn("notes/d.md#h:ch44444444", [row.ref for row in rows])
        self.assertEqual(record["display_mode"], "lexical")

    def test_guarded_fallback_separates_hybrid_diagnostics_from_display(self):
        store = StubStore()
        retriever = HybridRetriever(
            store, vector_mode="shadow", vector_fallback_mode="on", candidate_k=5,
        )
        _rows, record = retriever.retrieve_with_shadow_record(
            "原始素材存放在哪里", limit=3,
        )
        self.assertIn("notes/d.md#h:ch44444444", record["display_refs"])
        self.assertEqual(len(record["hybrid_refs"]), 3)

    def test_memory_focus_keeps_atomic_memory_entries_together(self):
        class MemoryStore(StubStore):
            def search_lexical(self, query, k=20, **kwargs):
                return [
                    {"ref": "ark/memory/context/pref.md", "content": "深色主题", "score": 9},
                    {"ref": "notes/unrelated.md#h:ch33333333", "content": "深色主题", "score": 8},
                ]

        store = MemoryStore()
        retriever = HybridRetriever(store, vector_mode="shadow", candidate_k=8)
        rows, record = retriever.retrieve_with_shadow_record("用户偏好深色主题", limit=5)
        self.assertTrue(all(item.ref.replace("\\", "/").startswith("ark/memory/")
                            for item in rows))
        self.assertTrue(record["memory_focused"])

    def test_memory_focus_recognizes_natural_preference_language(self):
        class MemoryStore(StubStore):
            def search_lexical(self, query, k=20, **kwargs):
                self.calls.append(("lexical", query, k, kwargs))
                return [{
                    "ref": "ark/memory/core/theme.md#mem-theme",
                    "content": "用户喜欢深色主题",
                    "score": 4,
                }]

        store = MemoryStore()
        retriever = HybridRetriever(store, vector_mode="off", candidate_k=8)
        rows, record = retriever.retrieve_with_shadow_record(
            "我喜欢深色主题", limit=5,
        )
        self.assertTrue(rows)
        self.assertEqual(record["display_mode"], "memory_focused")
        self.assertEqual(
            store.calls[0][3]["path_prefixes"],
            ("ark/memory/core/", "ark/memory/context/"),
        )

    def test_non_memory_query_does_not_add_memory_path_filter(self):
        store = StubStore()
        HybridRetriever(store, vector_mode="off", candidate_k=8).retrieve(
            "猫会做什么", limit=2,
        )
        self.assertNotIn("path_prefixes", store.calls[0][3])

    def test_cross_document_query_reserves_distinct_documents(self):
        class CrossDocStore(StubStore):
            def search_lexical(self, query, k=20, **kwargs):
                return [
                    {"ref": "docs/a.md#one:ch11111111", "content": "a1", "score": 10},
                    {"ref": "docs/a.md#two:ch22222222", "content": "a2", "score": 9},
                    {"ref": "docs/b.md#one:ch33333333", "content": "b1", "score": 8},
                ]

        rows = HybridRetriever(
            CrossDocStore(), vector_mode="off", candidate_k=5,
        ).retrieve("Agent 记忆有哪些观点和判据", limit=2)
        self.assertEqual(
            {row.ref.split("#", 1)[0] for row in rows},
            {"docs/a.md", "docs/b.md"},
        )

    def test_small_to_big_shadow_preserves_display_but_on_expands(self):
        class ContextStore(StubStore):
            def expand_context(self, rows, **kwargs):
                return [
                    {**row, "content": f"{row.get('content')}\n[context_of:neighbor]", "context_of": ["neighbor"]}
                    for row in rows
                ], {"status": "available", "requested": len(rows), "expanded": len(rows),
                    "neighbors": len(rows), "truncated": 0}

        shadow = HybridRetriever(
            ContextStore(), vector_mode="off", small_to_big_mode="shadow",
        )
        shadow_rows = shadow.retrieve("猫", limit=1)
        self.assertNotIn("context_of", shadow_rows[0].to_dict())
        _, record = shadow.retrieve_with_shadow_record("猫", limit=1)
        self.assertEqual(record["small_to_big"]["expanded"], 1)

        on = HybridRetriever(
            ContextStore(), vector_mode="off", small_to_big_mode="on",
        )
        on_rows = on.retrieve("猫", limit=1)
        self.assertIn("context_of", on_rows[0].to_dict())
        self.assertIn("[context_of:neighbor]", on_rows[0].content)

    def test_on_returns_hybrid_and_forwards_filters(self):
        store = StubStore()
        retriever = HybridRetriever(store, vector_mode="on", candidate_k=10)
        rows = retriever.retrieve("猫", limit=5, project_id="p2", statuses=["active"])
        refs = [item.ref for item in rows]
        self.assertIn("notes/d.md#h:ch44444444", refs)
        self.assertTrue(all(call[3] == {"project_id": "p2", "statuses": ["active"]}
                            for call in store.calls))

    def test_vector_score_floor_filters_candidates_before_fusion(self):
        store = StubStore()
        retriever = HybridRetriever(
            store, vector_mode="on", candidate_k=10, vector_min_score=0.85,
        )
        rows = retriever.retrieve("猫", limit=5)
        refs = [item.ref for item in rows]
        self.assertNotIn("notes/d.md#h:ch44444444", refs)
        self.assertIn("sessions/2026-09.md#mem-a:ch11111111", refs)

    def test_vector_score_floor_is_bounded(self):
        with self.assertRaises(ValueError):
            HybridRetriever(StubStore(), vector_min_score=1.1)

    def test_lexical_coverage_floor_can_abstain(self):
        class LowEvidenceStore(StubStore):
            def lexical_confidence(self, query, rows):
                return 0.1

        record = HybridRetriever(
            LowEvidenceStore(), vector_mode="off", lexical_min_coverage=0.5,
        ).shadow_record("不存在的概念", limit=5)
        self.assertEqual(record["routes"]["lexical"]["refs"], [])
        self.assertIn("coverage_below", record["routes"]["lexical"]["reason"])

    def test_unknown_identifier_with_no_lexical_hit_abstains_from_vector(self):
        class EmptyLexicalStore(StubStore):
            def search_lexical(self, query, k=20, **kwargs):
                self.calls.append(("lexical", query, k, kwargs))
                return []

        store = EmptyLexicalStore()
        retriever = HybridRetriever(store, vector_mode="on", candidate_k=10)
        record = retriever.shadow_record("BVZZzz9999", limit=5)
        self.assertEqual(record["routes"]["vector"]["refs"], [])
        self.assertEqual(record["routes"]["vector"]["reason"], "identifier_lexical_miss_skip")
        self.assertFalse(any(call[0] == "vector" for call in store.calls))

    def test_config_has_safe_hybrid_defaults_and_fail_closed_values(self):
        cfg = RagConfig()
        self.assertEqual(cfg.vector_mode, "shadow")
        self.assertEqual(cfg.hybrid_candidate_k, 40)
        self.assertEqual(cfg.vector_min_score, 0.0)
        self.assertEqual(cfg.query_embedding_cache_size, 256)
        self.assertEqual(RagConfig().index_backend, "auto")
        self.assertEqual(cfg.dedupe_by, "entry")
        self.assertEqual(cfg.small_to_big_mode, "shadow")
        self.assertEqual(cfg.small_to_big_neighbors, 1)
        self.assertEqual(cfg.small_to_big_max_chars, 2400)
        self.assertEqual(cfg.vector_fallback_mode, "off")
        self.assertEqual(cfg.query_expansion_mode, "shadow")
        self.assertEqual(cfg.query_expansion_max_variants, 5)
        self.assertEqual(cfg.query_expansion_min_candidates, 8)
        invalid = RagConfig(vector_mode="bad", lexical_mode="bad", dedupe_by="bad", hybrid_candidate_k=0)
        self.assertEqual(invalid.vector_mode, "shadow")
        self.assertEqual(invalid.lexical_mode, "on")
        self.assertEqual(invalid.dedupe_by, "entry")
        self.assertEqual(invalid.hybrid_candidate_k, 40)
        self.assertEqual(RagConfig(vector_min_score=1.5).vector_min_score, 0.0)
        self.assertEqual(RagConfig(query_embedding_cache_size=-1).query_embedding_cache_size, 256)
        self.assertEqual(RagConfig(index_backend="bad").index_backend, "auto")
        self.assertEqual(RagConfig(vector_fallback_mode="bad").vector_fallback_mode, "off")

    def test_shadow_log_writer_is_bounded_and_contains_no_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = __import__("pathlib").Path(temp_dir) / "shadow.jsonl"
            writer = ShadowLogWriter(path, max_bytes=1024)
            store = StubStore()
            retriever = HybridRetriever(store, vector_mode="shadow", shadow_logger=writer)
            retriever.shadow_record("猫", limit=2)
            self.assertTrue(path.exists())
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["schema"], "rag-hybrid-shadow-v2")
            self.assertNotIn("content", record)
            for index in range(20):
                writer({"schema": "test", "query_hash": str(index), "payload": "x" * 200})
            self.assertLessEqual(path.stat().st_size, 1024)

    def test_empty_results_are_available_when_store_reports_success(self):
        class EmptyStore(StubStore):
            def search_lexical(self, query, k=20, **kwargs):
                self.calls.append(("lexical", query, k, kwargs))
                return []

            def search_vector(self, query, k=20, **kwargs):
                self.calls.append(("vector", query, k, kwargs))
                return []

            @property
            def last_search_status(self):
                return {
                    "lexical": {"status": "available", "reason": ""},
                    "vector": {"status": "available", "reason": ""},
                }

        record = HybridRetriever(
            EmptyStore(), vector_mode="shadow",
        ).shadow_record("无结果查询")
        self.assertEqual(record["routes"]["lexical"]["status"], "available")
        self.assertEqual(record["routes"]["vector"]["status"], "available")
        self.assertEqual(record["hybrid_status"], "available")

    def test_missing_provider_is_unavailable_even_when_vector_is_empty(self):
        class NoProviderStore(StubStore):
            embedding_model = ""

            def search_vector(self, query, k=20, **kwargs):
                self.calls.append(("vector", query, k, kwargs))
                return []

            @property
            def last_search_status(self):
                return {
                    "lexical": {"status": "available", "reason": ""},
                    "vector": {"status": "unavailable", "reason": "provider_missing"},
                }

        record = HybridRetriever(
            NoProviderStore(), vector_mode="shadow",
        ).shadow_record("查询")
        self.assertEqual(record["routes"]["vector"]["status"], "unavailable")
        self.assertEqual(record["routes"]["vector"]["reason"], "provider_missing")
        self.assertEqual(record["hybrid_status"], "degraded_vector_unavailable")


if __name__ == "__main__":
    unittest.main()
