import tempfile
import unittest
from pathlib import Path

from agentlab.rag.hybrid import HybridRetriever
from agentlab.rag.index_store import RagIndexStore


def _bucket(entries: list[tuple[str, str, str]]) -> str:
    lines = ["---", "bucket: true", "type: sessions", "---", "# sessions", ""]
    for mem_id, tags, body in entries:
        lines.extend([
            f"## {mem_id}",
            f"> importance=3 | tags={tags} | created=2026-09",
            "",
            body,
            "",
        ])
    return "\n".join(lines)


class TestTopicParentRecall(unittest.TestCase):
    def test_parent_search_aggregates_tags_and_returns_stable_entry_anchor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sessions.md").write_text(_bucket([
                ("mem-aaa", "主题地图, Agent", "Agent、RAG 与知识库主题总览。"),
                ("mem-bbb", "日常", "与主题无关的记录。"),
            ]), encoding="utf-8")
            store = RagIndexStore(root / "index.sqlite", vault_root=root)
            self.assertEqual(store.sync_vault()["failed"], 0)

            rows = store.search_parent_entries("这个知识库覆盖哪些知识主题", k=5)

            self.assertTrue(rows)
            self.assertEqual(rows[0]["ref"], "sessions.md#mem-aaa")
            self.assertTrue(rows[0]["parent_entry"])
            self.assertIn("主题总览", rows[0]["content"])
            self.assertNotIn(":ch", rows[0]["ref"])

    def test_hybrid_uses_parent_route_only_for_topic_style_query(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sessions.md").write_text(_bucket([
                ("mem-aaa", "主题地图, Agent", "Agent、RAG 与知识库主题总览。"),
                ("mem-bbb", "日常", "与主题无关的记录。"),
            ]), encoding="utf-8")
            store = RagIndexStore(root / "index.sqlite", vault_root=root)
            store.sync_vault()
            retriever = HybridRetriever(
                store, vector_mode="off", lexical_mode="on", candidate_k=5,
            )

            topic_rows = retriever.retrieve("这个知识库覆盖哪些知识主题", limit=2)
            exact_rows = retriever.retrieve("Agent RAG", limit=2)

            self.assertTrue(topic_rows)
            self.assertEqual(topic_rows[0].ref, "sessions.md#mem-aaa")
            self.assertTrue(exact_rows)
            self.assertTrue(any(":ch" in row.ref for row in exact_rows))


if __name__ == "__main__":
    unittest.main()
