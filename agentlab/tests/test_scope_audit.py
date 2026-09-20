import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agentlab.eval.scope_audit import audit, classify_task, scoped_tasks


class TestScopeAudit(unittest.TestCase):
    def test_classifies_out_of_scope_and_mixed_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = root / "tasks.jsonl"
            tasks.write_text("\n".join([
                json.dumps({"id": "in", "query_type": "p", "expected_refs": ["wiki/a.md"]}),
                json.dumps({"id": "out", "query_type": "p", "expected_refs": ["Inbox/a.md"]}),
                json.dumps({"id": "mix", "query_type": "p", "expected_refs": ["wiki/a.md", "Inbox/a.md"]}),
                json.dumps({"id": "neg", "query_type": "negative", "expected_refs": []}),
            ]), encoding="utf-8")
            db = root / "index.sqlite"
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE files(path TEXT PRIMARY KEY)")
                conn.execute("INSERT INTO files(path) VALUES ('wiki/a.md')")
                conn.commit()
            finally:
                conn.close()
            report = audit(tasks, db)
            self.assertEqual(report["in_scope_positive_tasks"], 1)
            self.assertEqual(report["out_of_scope_positive_tasks"], 1)
            self.assertEqual(report["mixed_scope_positive_tasks"], 1)

    def test_independent_fixture_is_not_applicable_to_canonical_index(self):
        row = classify_task(
            {
                "id": "fixture",
                "vault_root": "fixtures/rag_isolation",
                "expected_refs": ["ark/memory/core/mem.md"],
            },
            {"wiki/a.md"},
        )
        self.assertEqual(row["category"], "not_applicable")

    def test_scoped_projection_keeps_negatives_and_full_in_scope_tasks(self):
        tasks = [
            {"id": "in", "expected_refs": ["wiki/a.md"]},
            {"id": "out", "expected_refs": ["Inbox/a.md"]},
            {"id": "neg", "expected_refs": [], "negative_kind": "random_token"},
            {"id": "fixture", "vault_root": "fixtures/x", "expected_refs": ["wiki/a.md"]},
        ]
        selected, manifest = scoped_tasks(tasks, {"wiki/a.md"})
        self.assertEqual([task["id"] for task in selected], ["in", "neg"])
        self.assertEqual(manifest["selected_tasks"], 2)
        self.assertEqual(manifest["excluded_positive_tasks"], 1)
        self.assertEqual(manifest["not_applicable_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
