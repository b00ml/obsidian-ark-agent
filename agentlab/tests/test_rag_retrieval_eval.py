import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from agentlab.eval.run_rag_retrieval import (
    _bucket_dup_ratio,
    _expectation_sets,
    _memory_search,
    _map_at,
    _ndcg_at,
    _recall_at,
    evaluate,
)
from agentlab.eval.rag_task_schema import migrate_v1_task, validate_task, validate_tasks
from agentlab.memory.markdown_store import MemoryMarkdownStore
from agentlab.rag.hybrid import HybridRetriever
from agentlab.rag.index_store import RagIndexStore


class EvalEmbedder:
    model = "eval-fake-v1"

    def embed(self, texts):
        return [[float(text.count("猫")), float(text.count("狗"))]
                for text in texts]


class TestRagRetrievalEval(unittest.TestCase):
    def test_v2_task_migration_adds_scope_and_binary_qrels(self):
        task = migrate_v1_task({
            "id": "knowledge",
            "query": "缓存",
            "expected_refs": ["wiki/concepts/cache.md"],
        })
        self.assertEqual(task["corpus_scope"], "knowledge")
        self.assertEqual(task["qrels"], [{"ref": "wiki/concepts/cache.md", "relevance": 1}])
        self.assertEqual(task["graded_qrels"], task["qrels"])
        self.assertEqual(task["answerability"], "answerable")
        self.assertEqual(task["scope"]["project_id"], "default")
        self.assertEqual(validate_task(task), [])

    def test_v2_negative_migration_defaults_to_absent_and_validates_scope(self):
        task = migrate_v1_task({"id": "negative", "query": "未知", "expected_refs": []})
        self.assertEqual(task["answerability"], "absent")
        self.assertEqual(task["forbidden_refs"], [])
        self.assertEqual(validate_task(task), [])
        task["answerability"] = "not-a-label"
        self.assertTrue(any("answerability" in error for error in validate_task(task)))

    def test_v2_task_validation_requires_qrels_for_positive(self):
        task = migrate_v1_task({"id": "bad", "query": "x", "expected_refs": ["a.md"]})
        task["qrels"] = []
        self.assertTrue(any("every expected_ref" in error for error in validate_task(task)))

    def test_v2_audit_rejects_duplicate_ids(self):
        tasks = [migrate_v1_task({"id": "x", "query": "x", "expected_refs": []})] * 2
        report = validate_tasks(tasks)
        self.assertFalse(report["valid"])
        self.assertIn("duplicate task id", report["errors"][0])

    def test_entry_refs_keep_bucket_identity(self):
        task = {
            "expected_refs": ["ark/memory/sessions/2026-09.md#mem-a"],
        }
        expectations = _expectation_sets(task)
        self.assertEqual(_recall_at(expectations, [
            "ark/memory/sessions/2026-09.md#mem-b",
        ], 5), 0.0)
        self.assertEqual(_recall_at(expectations, [
            "ark/memory/sessions/2026-09.md#mem-a",
        ], 5), 1.0)
        self.assertEqual(_recall_at(expectations, [
            "ark/memory/sessions/2026-09.md#mem-a:chdeadbeef",
        ], 5), 1.0)
        self.assertEqual(_recall_at(expectations, [
            "ark/memory/sessions/2026-09.md#mem-ab:chdeadbeef",
        ], 5), 0.0)

    def test_any_policy_is_one_condition(self):
        task = {
            "expected_refs": ["a.md", "b.md"],
            "expected_policy": "any",
        }
        expectations = _expectation_sets(task)
        self.assertEqual(len(expectations), 1)
        self.assertEqual(_recall_at(expectations, ["b.md"], 5), 1.0)

    def test_graded_qrels_score_order_and_partial_evidence(self):
        task = {
            "qrels": [
                {"ref": "direct.md", "relevance": 2},
                {"ref": "support.md", "relevance": 1},
            ],
        }
        self.assertEqual(_ndcg_at(task, ["direct.md", "support.md"], 5), 1.0)
        self.assertLess(
            _ndcg_at(task, ["support.md", "direct.md"], 5), 1.0
        )
        self.assertEqual(_map_at(task, ["direct.md", "support.md"], 5), 1.0)

    def test_legacy_refs_get_binary_ndcg_and_map(self):
        task = {"expected_refs": ["a.md", "b.md"]}
        self.assertEqual(_ndcg_at(task, ["a.md", "b.md"], 5), 1.0)
        self.assertEqual(_map_at(task, ["a.md", "b.md"], 5), 1.0)

    def test_ndcg_deduplicates_multiple_chunks_from_one_file(self):
        task = {"expected_refs": ["a.md"]}
        self.assertEqual(_ndcg_at(task, ["a.md#h1:ch1", "a.md#h2:ch2"], 5), 1.0)

    def test_bucket_duplicate_ratio_uses_raw_candidates(self):
        refs = [
            "ark/memory/sessions/2026-09.md#mem-a",
            "ark/memory/sessions/2026-09.md#mem-b",
            "ark/memory/sessions/2026-09.md#mem-c",
        ]
        self.assertEqual(_bucket_dup_ratio(refs), 0.667)

    def test_isolation_fixture_filters_project_b(self):
        fixture = pathlib.Path(__file__).parents[2] / ".ai" / "evals" / "fixtures" / "rag_isolation"
        tasks = [{
            "id": "fixture",
            "query": "inbox task processing",
            "expected_refs": [
                "ark/memory/core/mem-shared-default.md#mem-shared-default",
                "ark/memory/context/proj-a/mem-project-a.md#mem-project-a",
            ],
            "forbidden_refs": [
                "ark/memory/context/proj-b/mem-project-b.md#mem-project-b",
            ],
            "expected_policy": "all",
            "route": "memory_only",
            "vault_root": str(fixture),
            "project_id": "proj-a",
            "filters": {"project_id": "proj-a"},
        }]
        report = evaluate({"vault_path": str(fixture)}, tasks)
        row = report["per_query"][0]
        self.assertEqual(row["memory_keyword"]["recall@5"], 1.0)
        self.assertEqual(row["forbidden_hits@5"], 0)
        self.assertEqual(row["rrf"]["status"], "not_applicable")
        self.assertIn("independent_vault_fixture", row["rrf"]["reason"])

    def test_empty_positive_result_is_available_and_scored(self):
        task = {
            "id": "empty-positive",
            "query": "missing concept",
            "expected_refs": ["missing.md"],
            "route": "memory_only",
        }
        report = evaluate({"vault_path": "."}, [task])
        row = report["per_query"][0]
        self.assertEqual(row["memory_keyword"]["status"], "available")
        self.assertEqual(row["memory_keyword"]["recall@5"], 0.0)
        self.assertEqual(report["summary"]["memory_keyword"]["status"], "available")

    def test_fixture_rrf_is_not_applicable_but_canonical_route_is_available(self):
        fixture = pathlib.Path(__file__).parents[2] / ".ai" / "evals" / "fixtures" / "rag_isolation"
        task = {
            "id": "fixture-status",
            "query": "inbox task processing",
            "expected_refs": [],
            "route": "memory_only",
            "vault_root": "fixtures/rag_isolation",
        }
        report = evaluate({"vault_path": str(fixture)}, [task])
        row = report["per_query"][0]
        self.assertEqual(row["rrf"]["status"], "not_applicable")
        self.assertEqual(report["summary"]["rrf"]["status"], "not_applicable")

    def test_memory_search_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryMarkdownStore(temp_dir)
            mem_id = store.commit("Read-only retrieval fixture", mem_type="core")
            file_path = store._find_memory_file(mem_id)
            before_text = file_path.read_text(encoding="utf-8")
            before_mtime = file_path.stat().st_mtime_ns

            refs, _ = _memory_search(
                {"vault_path": temp_dir, "project_id": "default"},
                "Read-only retrieval",
                5,
            )

            self.assertTrue(any(ref.endswith(f"#{mem_id}") for ref in refs))
            self.assertEqual(file_path.read_text(encoding="utf-8"), before_text)
            self.assertEqual(file_path.stat().st_mtime_ns, before_mtime)

    def test_lexical_path_prefix_isolates_short_memory_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            memory = root / "ark" / "memory" / "core"
            memory.mkdir(parents=True)
            (memory / "theme.md").write_text("用户喜欢猫主题。", encoding="utf-8")
            (root / "notes.md").write_text("猫主题是界面设计的一种方案。", encoding="utf-8")
            store = RagIndexStore(root / "index.sqlite", vault_root=root)
            self.assertEqual(store.sync_vault()["failed"], 0)
            rows = store.search_lexical(
                "我喜欢深色主题", k=10, path_prefixes=("ark/memory/",),
            )
            self.assertTrue(rows)
            self.assertTrue(all(
                str(row["ref"]).replace("\\", "/").startswith("ark/memory/")
                for row in rows
            ))

    def test_vector_path_prefix_uses_a_separate_matrix_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            memory = root / "ark" / "memory" / "core"
            memory.mkdir(parents=True)
            (memory / "theme.md").write_text("用户喜欢猫主题。", encoding="utf-8")
            (root / "notes.md").write_text("猫主题是界面设计的一种方案。", encoding="utf-8")
            store = RagIndexStore(
                root / "index.sqlite", EvalEmbedder(), vault_root=root,
            )
            self.assertEqual(store.sync_vault()["failed"], 0)
            rows = store.search_vector(
                "猫", k=10, path_prefixes=("ark/memory/",),
            )
            self.assertTrue(rows)
            self.assertTrue(all(
                str(row["ref"]).replace("\\", "/").startswith("ark/memory/")
                for row in rows
            ))

    def test_read_only_store_does_not_create_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            MemoryMarkdownStore(temp_dir, create_dirs=False)
            self.assertFalse((pathlib.Path(temp_dir) / "ark" / "memory").exists())

    def test_shadow_evaluator_scores_available_vector_and_rrf_routes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            note = root / "note.md"
            note.write_text("# 猫\n\n猫的缓存方案。", encoding="utf-8")
            store = RagIndexStore(root / "shadow.sqlite", EvalEmbedder(), vault_root=root)
            self.assertEqual(store.sync_vault()["failed"], 0)
            cfg = {
                "vault_path": str(root),
                "_p2_store": store,
                "_hybrid_retriever": HybridRetriever(
                    store, vector_mode="on", lexical_mode="on", candidate_k=5,
                ),
            }
            report = evaluate(cfg, [{
                "id": "vector-fixture",
                "query": "猫",
                "expected_refs": ["note.md"],
                "route": "vault_only",
            }])
            row = report["per_query"][0]
            self.assertEqual(row["route_statuses"]["p2_vector"], "available")
            self.assertEqual(row["route_statuses"]["rrf"], "available")
            self.assertEqual(row["p2_vector"]["recall@5"], 1.0)
            self.assertEqual(row["rrf"]["recall@5"], 1.0)
            self.assertIn("ndcg@5", row["p2_vector"])
            self.assertIn("map@10", report["summary"]["p2_vector"])

    def test_production_shaped_mode_does_not_pass_expected_refs_to_hybrid(self):
        class ContractAwareRetriever:
            def __init__(self, vault_root):
                self.store = SimpleNamespace(vault_root=vault_root)

            def shadow_record(self, _query, *, required_groups=None, **_kwargs):
                ref = "gold.md" if required_groups else "other.md"
                return {
                    "hybrid_status": "available",
                    "hybrid_refs": [ref],
                    "display_refs": [ref],
                    "display_mode": "hybrid",
                    "latency_ms": 1.0,
                    "coverage_contract": {
                        "required_groups": [list(group) for group in (required_groups or ())],
                    },
                    "routes": {
                        "lexical": {"status": "available", "refs": ["other.md"], "latency_ms": 0.1},
                        "vector": {"status": "available", "refs": ["gold.md"], "latency_ms": 0.1},
                    },
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "vault_path": temp_dir,
                "_hybrid_retriever": ContractAwareRetriever(temp_dir),
            }
            task = {
                "id": "coverage", "query": "question", "expected_refs": ["gold.md"],
                "route": "vault_only",
            }
            diagnostic = evaluate(cfg, [task])
            production_shaped = evaluate(
                cfg, [task], use_eval_coverage_contract=False,
            )
        self.assertEqual(diagnostic["per_query"][0]["rrf"]["refs"], ["gold.md"])
        self.assertEqual(production_shaped["per_query"][0]["rrf"]["refs"], ["other.md"])
        self.assertEqual(
            production_shaped["per_query"][0]["eval_coverage_contract"], "disabled",
        )


if __name__ == "__main__":
    unittest.main()
