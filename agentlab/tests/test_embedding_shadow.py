import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from agentlab.rag.embedding_shadow import (
    BudgetExceeded,
    EmbeddingBudget,
    build_plan,
    collect_chunks,
    main,
    select_sample,
    execute,
)


class FakeEmbedder:
    model = "fake-v1"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class TestEmbeddingShadow(unittest.TestCase):
    def test_budget_blocks_before_provider_call(self):
        inner = FakeEmbedder()
        budget = EmbeddingBudget(inner, max_items=1)
        self.assertEqual(budget.embed(["one"]), [[1.0, 0.0]])
        with self.assertRaises(BudgetExceeded):
            budget.embed(["two"])
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(budget.summary()["items"], 1)

    def test_collection_excludes_archive_and_plan_is_count_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ark" / "memory" / "archive").mkdir(parents=True)
            (root / "note.md").write_text("# 标题\n\n正文内容。", encoding="utf-8")
            (root / "ark" / "memory" / "archive" / "old.md").write_text(
                "不应发送。", encoding="utf-8"
            )
            chunks = collect_chunks(root)
            self.assertTrue(chunks)
            self.assertTrue(all(rel == "note.md" for rel, _ in chunks))
            plan = build_plan(root, sample_limit=20, model="fake-v1")
            encoded = json.dumps(plan, ensure_ascii=False)
            self.assertNotIn("正文内容", encoded)
            self.assertEqual(plan["total"]["chunks"], len(chunks))

    def test_include_prefix_allowlist_is_shared_by_manifest_and_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "wiki").mkdir()
            (root / "Inbox").mkdir()
            (root / "wiki" / "ok.md").write_text("允许内容。", encoding="utf-8")
            (root / "wikidata.md").write_text("前缀相似但不允许。", encoding="utf-8")
            (root / "Inbox" / "no.md").write_text("不允许内容。", encoding="utf-8")
            plan = build_plan(root, sample_limit=20, include_prefixes=["wiki"])
            self.assertEqual(plan["vault_manifest"]["files"], 1)
            self.assertEqual(plan["files"], 1)
            self.assertEqual(plan["include_prefixes"], ["wiki"])
            self.assertTrue(all(rel == "wiki/ok.md" for rel, _ in collect_chunks(
                root, include_prefixes=["wiki"])))

    def test_execute_requires_explicit_remote_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "note.md").write_text("猫。", encoding="utf-8")
            index = root / "shadow.sqlite"
            code = main([
                "--vault", str(root), "--index", str(index),
                "--execute", "--model", "fake-v1",
            ])
            self.assertEqual(code, 2)
            self.assertFalse(index.exists())

    def test_sample_categories_look_at_all_chunks_in_a_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mixed.md").write_text(
                "正文。\n\n- 列表项\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```text\n代码\n```",
                encoding="utf-8",
            )
            chunks = collect_chunks(root)
            sample = select_sample(chunks, limit=len(chunks))
            self.assertEqual(sample["files"], ["mixed.md"])
            self.assertTrue({"prose", "list", "table", "fence"}.issubset(sample["categories"]))

    def test_full_execution_stops_after_budget_is_exceeded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.md").write_text("文档甲。", encoding="utf-8")
            (root / "b.md").write_text("文档乙。", encoding="utf-8")
            index = root / "shadow.sqlite"
            plan = build_plan(root, sample_limit=20, model="fake-v1")
            args = Namespace(
                vault=str(root), index=str(index), model="fake-v1",
                index_version="test-shadow", min_chars=64,
                batch_size=8, phase="full", max_items=1,
                price_per_1k_tokens=0.0, max_cost=0.0,
            )
            result = execute(args, plan, FakeEmbedder())
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["run"]["attempted"], 1)
            self.assertEqual(result["budget"]["items"], 1)

    def test_budget_stop_is_not_recorded_as_retryable_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.md").write_text(
                "# 甲\n\n文档甲。\n\n# 乙\n\n第二段。", encoding="utf-8"
            )
            plan = build_plan(root, sample_limit=20, model="fake-v1")
            args = Namespace(
                vault=str(root), index=str(root / "shadow.sqlite"), model="fake-v1",
                index_version="test-shadow", min_chars=64, batch_size=8,
                phase="sample", max_items=1, price_per_1k_tokens=0.0, max_cost=0.0,
            )
            result = execute(args, plan, FakeEmbedder())
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["index"]["failures"], 0)


if __name__ == "__main__":
    unittest.main()
