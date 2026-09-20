import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as _np
except ImportError:  # pragma: no cover - minimal install keeps pure-Python path
    _np = None

from agentlab.rag.chunker import make_chunker_v2
from agentlab.rag.index_store import RagIndexStore, _ranking_query_tokens
from agentlab.memory.markdown_store import MemoryMarkdownStore


class FakeEmbedder:
    def __init__(self, vocab, fail=False, model="fake-v1"):
        self.vocab = list(vocab)
        self.fail = fail
        self.model = model
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("provider unavailable")
        return [[float(text.lower().count(word.lower())) for word in self.vocab]
                for text in texts]


class DimensionMismatchEmbedder:
    model = "mismatch-v1"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] if index == 0 else [1.0]
                for index, _ in enumerate(texts)]


class SelectiveFailEmbedder(FakeEmbedder):
    def embed(self, texts):
        self.calls.append(list(texts))
        if any("坏文档" in text for text in texts):
            raise RuntimeError("provider unavailable")
        return [[float(text.lower().count(word.lower())) for word in self.vocab]
                for text in texts]


class TestRagIndexStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "notes").mkdir()
        self.embedder = FakeEmbedder(["猫", "狗", "Redis", "BV1abc"])
        self.store = RagIndexStore(self.root / "index.sqlite", self.embedder,
                                   vault_root=self.root, batch_size=8)

    def tearDown(self):
        self.tmp.cleanup()

    def test_multiterm_ranking_drops_cjk_unigram_noise_but_keeps_short_queries(self):
        self.assertNotIn("的", _ranking_query_tokens("知识库的主题"))
        self.assertIn("主题", _ranking_query_tokens("知识库的主题"))
        self.assertIn("猫", _ranking_query_tokens("猫"))

    def test_schema_and_structure_v1_refs(self):
        (self.root / "notes" / "bucket.md").write_text(
            "---\nbucket: true\ntags: [memory]\n---\n\n"
            "## mem-aaa\n> importance=8\n\n猫的喂养。\n\n"
            "## mem-bbb\n\n狗的训练。", encoding="utf-8")
        stat = self.store.sync_vault()
        self.assertEqual(stat["updated"], 1)
        self.assertEqual(stat["failed"], 0)
        self.assertEqual(
            [row["status"] for row in stat["stages"]],
            ["accepted", "fetched", "parsed", "chunked", "indexed", "completed"],
        )
        self.assertEqual(self.store.chunk_count, 2)
        refs = [item["ref"] for item in self.store.search_lexical("猫")]
        self.assertTrue(any("#mem-aaa:ch" in ref for ref in refs))
        conn = sqlite3.connect(self.root / "index.sqlite")
        try:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()
        self.assertTrue({"files", "entries", "chunks", "lexical", "index_meta", "failures"}.issubset(tables))

    def test_small_to_big_expands_same_heading_with_bound_and_scope(self):
        path = self.root / "notes" / "context.md"
        paragraphs = [
            f"猫命中段{i}。" + ("这是同一标题下的邻接上下文。" * 24)
            for i in range(4)
        ]
        path.write_text("# 同一标题\n\n" + "\n\n".join(paragraphs), encoding="utf-8")
        self.store.sync_vault()
        hits = self.store.search_lexical("猫命中段1", k=1)
        self.assertTrue(hits)
        original_ref = hits[0]["ref"]
        expanded, stats = self.store.expand_context(hits, neighbor_chunks=1, max_chars=900)
        self.assertEqual(expanded[0]["ref"], original_ref)
        self.assertLessEqual(len(expanded[0]["content"]), 900)
        self.assertTrue(stats["expanded"] >= 1)
        self.assertTrue(expanded[0]["context_of"])
        self.assertIn("[context_of:", expanded[0]["content"])

        denied, denied_stats = self.store.expand_context(
            hits, neighbor_chunks=1, max_chars=900, project_id="other-project",
        )
        self.assertEqual(denied[0]["content"], hits[0]["content"])
        self.assertEqual(denied_stats["expanded"], 0)

    def test_duplicate_bucket_ids_keep_distinct_entry_rows(self):
        path = self.root / "duplicate.md"
        path.write_text(
            "---\nbucket: true\n---\n\n## mem-aaa\n\n第一份。\n\n"
            "## mem-aaa\n\n第二份。", encoding="utf-8")
        self.store.sync_vault()
        conn = sqlite3.connect(self.root / "index.sqlite")
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM entries WHERE file_path='duplicate.md'").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 2)

    def test_source_hash_incremental_and_embedding_cache(self):
        path = self.root / "a.md"
        path.write_text("# 猫\n\n猫会抓老鼠。", encoding="utf-8")
        first = self.store.sync_vault()
        calls = len(self.embedder.calls)
        self.assertEqual(first["embedded"], 2)
        second = self.store.sync_vault()
        self.assertEqual(second["updated"], 0)
        self.assertEqual(len(self.embedder.calls), calls)
        path.write_text("# 猫\n\n猫会抓老鼠。\n\n# 狗\n\n狗也会看家。", encoding="utf-8")
        third = self.store.sync_vault()
        self.assertEqual(third["updated"], 1)
        self.assertEqual(third["cache_hits"], 2)
        self.assertEqual(third["embedded"], 2)

    def test_query_embedding_cache_avoids_repeated_provider_calls(self):
        path = self.root / "query-cache.md"
        path.write_text("猫的内容。", encoding="utf-8")
        self.store.sync_vault()
        before = len(self.embedder.calls)
        self.assertTrue(self.store.search_vector("猫"))
        after_first = len(self.embedder.calls)
        self.assertEqual(after_first, before + 1)
        self.assertTrue(self.store.search_vector("猫"))
        self.assertEqual(len(self.embedder.calls), after_first)

    def test_vector_matrix_memory_cache_reused_within_instance(self):
        if _np is None:
            self.skipTest("numpy unavailable")
        path = self.root / "matrix-cache.md"
        path.write_text("# 猫\n\n猫和 Redis 的内容。", encoding="utf-8")
        self.store.sync_vault()
        self.assertTrue(self.store.search_vector("猫"))
        self.assertEqual(self.store.last_matrix_source, "built")
        self.assertTrue(self.store.search_vector("Redis"))
        self.assertEqual(self.store.last_matrix_source, "memory")

    def test_access_metadata_does_not_invalidate_source_hash(self):
        path = self.root / "memory.md"
        path.write_text(
            "---\nproject_id: default\nlast_accessed_at: '2026-09-13T00:00:00Z'\n"
            "access_count: 1\n---\n\n稳定正文。", encoding="utf-8")
        first = self.store.sync_vault()
        self.assertEqual(first["updated"], 1)
        path.write_text(
            "---\nproject_id: default\nlast_accessed_at: '2026-09-13T01:00:00Z'\n"
            "access_count: 99\n---\n\n稳定正文。", encoding="utf-8")
        second = self.store.sync_vault()
        self.assertEqual(second["updated"], 0)
        self.assertEqual(second["attempted"], 0)

    def test_lexical_chinese_code_and_project_status_filter(self):
        (self.root / "redis.md").write_text(
            "---\nproject_id: p1\n---\n\nRedis 缓存使用 BV1abc。", encoding="utf-8")
        (self.root / "old.md").write_text(
            "---\nproject_id: p1\nstatus: superseded\n---\n\nRedis 旧方案。", encoding="utf-8")
        self.store.sync_vault()
        hits = self.store.search_lexical("Redis BV1abc", project_id="p1")
        self.assertTrue(hits)
        self.assertTrue(any("redis.md" in hit["ref"] for hit in hits))
        self.assertFalse(any("old.md" in hit["ref"] for hit in hits))

    def test_scope_include_archive_explicitly_allows_archived_rows(self):
        (self.root / "archived.md").write_text(
            "---\nproject_id: p1\nstatus: archived\n---\n\nRedis 归档方案。",
            encoding="utf-8",
        )
        self.store.sync_vault()
        self.assertFalse(self.store.search_lexical("Redis", project_id="p1"))
        hits = self.store.search_lexical(
            "Redis", project_id="p1", include_archive=True,
        )
        self.assertTrue(any("archived.md" in hit["ref"] for hit in hits))

    def test_memory_effective_lifecycle_gate_filters_all_p2_routes(self):
        """Physical index rows must not bypass Markdown memory read governance."""
        memory = MemoryMarkdownStore(self.root)
        active = memory.commit("active memory Redis 可用", mem_type="core")
        revoked = memory.commit("revoked memory Redis 不可用", mem_type="core")
        memory.revoke(revoked)
        expired = memory.commit(
            "expired memory Redis 不可用", mem_type="core",
            valid_until=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        )
        future = memory.commit(
            "future memory Redis 不可用", mem_type="core",
            valid_from=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        )
        due = memory.commit(
            "review due memory Redis 不可用", mem_type="core",
            review_due_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        )
        conflict_path = self.root / "ark" / "memory" / "core" / "conflict.md"
        conflict_path.write_text(
            "---\nid: mem-conflict\ntype: core\nproject_id: default\n"
            "status: conflict\ncontent_hash: x\n---\n\nconflict memory Redis 不可用\n",
            encoding="utf-8",
        )
        self.store.sync_vault()

        active_path = memory._find_memory_file(active).relative_to(self.root).as_posix()
        blocked_paths = {
            memory._find_memory_file(memory_id).relative_to(self.root).as_posix()
            for memory_id in (revoked, expired, future, due)
        }
        blocked_paths.add(conflict_path.relative_to(self.root).as_posix())
        for route, hits in {
            "lexical": self.store.search_lexical("Redis", k=20),
            "parent": self.store.search_parent_entries("Redis", k=20),
            "vector": self.store.search_vector("Redis", k=20),
        }.items():
            refs = "\n".join(str(item["ref"]) for item in hits)
            self.assertIn(active_path, refs, f"{route} lost active memory: {refs}")
            for path in blocked_paths:
                self.assertNotIn(path, refs, f"{route} leaked inactive memory {path}: {refs}")

    def test_vector_matrix_sidecar_reused_across_instances(self):
        # OPT-265：归一化矩阵持久化在索引内置侧车，跨实例（新进程）直接复用，
        # 不再逐条解码向量 BLOB；结果必须与首次重建完全一致。
        if _np is None:
            self.skipTest("numpy unavailable")
        (self.root / "notes" / "mem.md").write_text(
            "# 门户\n\n猫狗红牛与 BV1abc 的知识卡片，含 Redis。", encoding="utf-8")
        self.store.sync_vault()
        first = [(r["ref"], r["score"]) for r in self.store.search_vector("猫", k=5)]
        self.assertEqual(self.store.last_matrix_source, "built")
        conn = sqlite3.connect(self.root / "index.sqlite")
        try:
            rows = conn.execute("SELECT COUNT(*) FROM vector_matrix").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(rows, 1, "首次检索应回写一次侧车")

        second_instance = RagIndexStore(self.root / "index.sqlite", self.embedder,
                                        vault_root=self.root, batch_size=8)
        reused = [(r["ref"], r["score"]) for r in second_instance.search_vector("猫", k=5)]
        self.assertEqual(second_instance.last_matrix_source, "cached", "新实例应从侧车复用")
        self.assertEqual(reused, first, "侧车矩阵必须与重建矩阵逐位等价")

    def test_vector_matrix_sidecar_invalidated_on_vault_update(self):
        # OPT-265：向量/文档更新（max updated_at 前移）必须让侧车签名失效并重建。
        if _np is None:
            self.skipTest("numpy unavailable")
        path = self.root / "notes" / "evolve.md"
        path.write_text("# 主题\n\nRedis 第一版内容。", encoding="utf-8")
        self.store.sync_vault()
        self.store.search_vector("Redis", k=5)
        self.assertEqual(self.store.last_matrix_source, "built")
        path.write_text("# 主题\n\nRedis 第二版内容，新增要点。", encoding="utf-8")
        self.store.sync_vault()
        fresh = self.store.search_vector("Redis", k=5)
        self.assertEqual(self.store.last_matrix_source, "built", "内容更新后侧车应失效重建")
        self.assertTrue(fresh)
        conn = sqlite3.connect(self.root / "index.sqlite")
        try:
            rows = conn.execute("SELECT COUNT(*) FROM vector_matrix").fetchone()[0]
        finally:
            conn.close()
        self.assertLessEqual(rows, 2, "侧车有界：旧 scope 一条 + 当前一条，不无限累积")

    def test_lexical_candidate_window_multiple_keeps_top_k_stable(self):
        # OPT-264（P5-E 候选数收缩）：3x/5x 候选窗对普通查询的 top-k 必须一致。
        # 先刷 70 个文档避免 “max(50, k*3)” 下限掩盖倍数差异。
        import agentlab.rag.index_store as mod
        for i in range(70):
            (self.root / "notes" / f"n{i:02d}.md").write_text(
                f"# 条目 {i}\n\n内容包括 通用 词汇 卡片 主题 {i} 相关说明。",
                encoding="utf-8")
        self.store.sync_vault()
        saved = mod.LEXICAL_CANDIDATE_WINDOW_MULT
        try:
            for query in ("通用 词汇", "主题 42", "条目 07"):
                mod.LEXICAL_CANDIDATE_WINDOW_MULT = 5
                wide = [r["ref"] for r in self.store.search_lexical(query, k=20)]
                mod.LEXICAL_CANDIDATE_WINDOW_MULT = 3
                narrow = [r["ref"] for r in self.store.search_lexical(query, k=20)]
                self.assertEqual(wide, narrow, f"候选窗倍数不应改变 top-k：{query}")
        finally:
            mod.LEXICAL_CANDIDATE_WINDOW_MULT = saved

    def test_lexical_rerank_prefers_long_identifier_entity(self):
        (self.root / "MySQL.md").write_text(
            "# MySQL\n\nMySQL 是关系数据库。", encoding="utf-8"
        )
        (self.root / "noise.md").write_text(
            "知识卡片可以记录很多知识和卡片。", encoding="utf-8"
        )
        for index in range(20):
            (self.root / f"noise-{index}.md").write_text(
                "知识卡片内容。", encoding="utf-8"
            )
        store = RagIndexStore(self.root / "rerank.sqlite", None, vault_root=self.root)
        store.sync_vault()
        hits = store.search_lexical("MySQL 知识卡片", k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["ref"].split("#", 1)[0], "MySQL.md")

    def test_path_slug_matches_without_markdown_suffix(self):
        path = self.root / "test-mcp-connection.md"
        path.write_text("MCP 链路测试。", encoding="utf-8")
        self.store.sync_vault()
        hits = self.store.search_lexical("test-mcp-connection")
        self.assertTrue(any("test-mcp-connection.md" in hit["ref"] for hit in hits))

    def test_single_file_failure_is_recorded_and_retryable(self):
        path = self.root / "broken.md"
        path.write_text("猫的内容。", encoding="utf-8")
        self.embedder.fail = True
        stat = self.store.sync_vault()
        self.assertEqual(stat["updated"], 1)
        self.assertEqual(stat["failed"], 1)
        self.assertEqual(len(self.store.list_failures()), 1)
        self.assertTrue(self.store.search_lexical("猫"), "provider 故障不应阻塞最新词法索引")
        calls = len(self.embedder.calls)
        deferred = self.store.sync_vault()
        self.assertEqual(deferred["deferred"], 1)
        self.assertEqual(len(self.embedder.calls), calls,
                         "退避窗口内普通同步不应重复调用失败 provider")
        self.embedder.fail = False
        retry = self.store.retry_failures()
        self.assertEqual(retry["succeeded"], 1)
        self.assertEqual(self.store.failure_count, 0)
        self.assertTrue(self.store.search_vector("猫"))

    def test_batch_failure_does_not_block_other_documents(self):
        (self.root / "one.md").write_text("猫。", encoding="utf-8")
        (self.root / "two.md").write_text("狗。", encoding="utf-8")
        self.embedder.fail = True
        # Both failures are visible independently; lexical chunks still commit.
        stat = self.store.sync_vault()
        self.assertEqual(stat["updated"], 2)
        self.assertEqual(len(self.store.list_failures()), 2)
        self.assertTrue(self.store.search_lexical("猫"))
        self.assertTrue(self.store.search_lexical("狗"))

    def test_index_status_reports_coverage_and_metadata(self):
        (self.root / "a.md").write_text("猫。", encoding="utf-8")
        self.store.sync_vault()
        status = self.store.index_status()
        self.assertEqual(status["indexed_files"], 1)
        self.assertEqual(status["expected_files"], 1)
        self.assertEqual(status["fresh_files"], 1)
        self.assertEqual(status["stale_files"], 0)
        self.assertEqual(status["coverage"], 1.0)
        self.assertEqual(status["embedding_dimension"], 4)
        self.assertEqual(status["parser_version"], "markdown-structure-v1")
        self.assertTrue(status["metadata_valid"])

    def test_lexical_index_works_without_embedding_provider(self):
        path = self.root / "keyword.md"
        path.write_text("Redis 缓存与 BV1abc。", encoding="utf-8")
        store = RagIndexStore(self.root / "keyword.sqlite", None, vault_root=self.root)
        stat = store.sync_vault()
        self.assertEqual(stat["failed"], 0)
        self.assertTrue(store.search_lexical("Redis BV1abc"))
        self.assertEqual(store.search_vector("Redis"), [])

    def test_adding_provider_later_fills_missing_vectors(self):
        path = self.root / "late.md"
        path.write_text("猫的内容。", encoding="utf-8")
        lexical_only = RagIndexStore(self.root / "late.sqlite", None, vault_root=self.root)
        lexical_only.sync_vault()
        with_vectors = RagIndexStore(self.root / "late.sqlite", self.embedder,
                                     vault_root=self.root)
        stat = with_vectors.sync_vault()
        self.assertEqual(stat["updated"], 1)
        self.assertTrue(with_vectors.search_vector("猫"))

    def test_incompatible_index_version_fails_closed(self):
        path = self.root / "versioned.md"
        path.write_text("猫的内容。", encoding="utf-8")
        first = RagIndexStore(self.root / "versioned.sqlite", self.embedder,
                              vault_root=self.root, index_version="v1")
        first.sync_vault()
        second = RagIndexStore(self.root / "versioned.sqlite", self.embedder,
                               vault_root=self.root, index_version="v2")
        stat = second.sync_vault()
        self.assertFalse(stat["compatible"])
        self.assertEqual(second.search_lexical("猫"), [])

    def test_empty_index_fails_closed_on_read(self):
        path = self.root / "empty.sqlite"
        # Creating the SQLite file without the index metadata models an
        # interrupted/invalid build.  A query must not report an available
        # empty route, because that hides an indexing outage.
        sqlite3.connect(path).close()
        store = RagIndexStore(path, None, vault_root=self.root)
        self.assertEqual(store.search_lexical("猫"), [])
        self.assertEqual(store.last_search_status["lexical"]["status"], "unavailable")
        self.assertIn("元数据", store.last_search_status["lexical"]["reason"])
        self.assertFalse(store.index_status()["metadata_valid"])

    def test_shadow_chunker_and_strategy_version_are_injectable(self):
        path = self.root / "v2.md"
        path.write_text("# 标题\n\n这是标题对应的正文。", encoding="utf-8")
        store = RagIndexStore(
            self.root / "v2.sqlite",
            None,
            vault_root=self.root,
            index_version="s1-p4.5b-test",
            parser_version="markdown-structure-v2-min64",
            chunk_strategy_version="markdown-structure-v2-min64",
            chunker=make_chunker_v2(64),
        )
        stat = store.sync_vault()
        self.assertEqual(stat["failed"], 0)
        self.assertEqual(store.index_status()["chunk_strategy_version"],
                         "markdown-structure-v2-min64")
        self.assertTrue(any("# 标题" in hit["content"] for hit in store.search_lexical("标题")))

    def test_include_prefix_scope_is_enforced_by_scan_and_metadata(self):
        (self.root / "wiki").mkdir()
        (self.root / "wiki" / "ok.md").write_text("允许内容。", encoding="utf-8")
        (self.root / "other.md").write_text("不应入索引。", encoding="utf-8")
        scoped = RagIndexStore(
            self.root / "scoped.sqlite", None, vault_root=self.root,
            include_prefixes=["wiki"],
        )
        stat = scoped.sync_vault()
        self.assertEqual(stat["total"], 1)
        self.assertEqual(scoped.index_status(self.root)["expected_files"], 1)
        self.assertTrue(scoped.search_lexical("允许内容"))
        self.assertFalse(scoped.search_lexical("不应入索引"))

    def test_embedding_cache_key_changes_with_model_and_strategy(self):
        text_hash = "a" * 64
        first = RagIndexStore(self.root / "one.sqlite", FakeEmbedder(["猫"]),
                              chunk_strategy_version="strategy-a")
        second = RagIndexStore(self.root / "one.sqlite", FakeEmbedder(["猫"], model="fake-v2"),
                               chunk_strategy_version="strategy-b")
        self.assertNotEqual(first._cache_key(text_hash), second._cache_key(text_hash))
        self.assertIn("fake-v1", first._cache_key(text_hash))
        self.assertIn("strategy-b", second._cache_key(text_hash))

    def test_malformed_cached_vector_is_ignored(self):
        path = self.root / "cached.md"
        path.write_text("猫的内容。", encoding="utf-8")
        self.store.sync_vault()
        conn = sqlite3.connect(self.root / "index.sqlite")
        try:
            conn.execute(
                "UPDATE embedding_cache SET embedding_dimension=99 WHERE embedding_dimension=4"
            )
            conn.commit()
        finally:
            conn.close()
        replacement = FakeEmbedder(["猫", "狗", "Redis", "BV1abc"])
        store = RagIndexStore(self.root / "index.sqlite", replacement,
                              vault_root=self.root, batch_size=8)
        path.write_text("猫的内容。\n\n新增内容。", encoding="utf-8")
        stat = store.sync_vault()
        self.assertEqual(stat["failed"], 0)
        self.assertGreaterEqual(stat["embedded"], 1)

    def test_dimension_mismatch_is_atomic_and_recorded(self):
        path = self.root / "mismatch.md"
        path.write_text(
            "# 甲\n\n第一段猫。\n\n# 乙\n\n第二段狗。", encoding="utf-8"
        )
        mismatch = RagIndexStore(self.root / "mismatch.sqlite",
                                 DimensionMismatchEmbedder(), vault_root=self.root)
        stat = mismatch.sync_vault()
        self.assertEqual(stat["failed"], 1)
        self.assertEqual(mismatch.chunk_count, 0)
        self.assertEqual(mismatch.search_lexical("猫"), [])
        self.assertEqual(len(mismatch.list_failures()), 1)

    def test_one_provider_failure_does_not_block_other_documents(self):
        (self.root / "bad.md").write_text("坏文档内容。", encoding="utf-8")
        (self.root / "good.md").write_text("正常文档内容。", encoding="utf-8")
        embedder = SelectiveFailEmbedder(["正常"])
        store = RagIndexStore(self.root / "selective.sqlite", embedder,
                              vault_root=self.root)
        stat = store.sync_vault()
        self.assertEqual(stat["failed"], 1)
        self.assertEqual(stat["updated"], 2)
        self.assertTrue(store.search_lexical("正常"))
        self.assertTrue(store.search_lexical("坏文档"))
        self.assertEqual(len(store.list_failures()), 1)

    def test_edit_stays_lexically_fresh_while_vector_retry_is_pending(self):
        path = self.root / "evolving.md"
        path.write_text("初始猫内容。", encoding="utf-8")
        self.store.sync_vault()
        path.write_text("更新狗内容。", encoding="utf-8")
        self.embedder.fail = True
        stat = self.store.sync_vault()
        self.assertEqual(stat["updated"], 1)
        self.assertEqual(stat["failed"], 1)
        self.assertTrue(self.store.search_lexical("更新狗内容"))
        self.assertEqual(self.store.search_lexical("初始"), [])
        self.embedder.fail = False
        retry = self.store.retry_failures()
        self.assertEqual(retry["succeeded"], 1)
        self.assertTrue(self.store.search_vector("更新狗内容"))


if __name__ == "__main__":
    unittest.main()
