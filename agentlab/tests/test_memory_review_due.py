import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentlab.eval.memory_review_due import apply_review, list_review_due, main
from agentlab.memory.governance import content_hash
from agentlab.memory.markdown_store import MemoryMarkdownStore


class TestMemoryReviewDue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name) / "vault"
        self.store = MemoryMarkdownStore(str(self.vault))

    def tearDown(self):
        self.tmp.cleanup()

    def test_lists_due_active_metadata_without_mutation(self):
        due = self.store.commit(
            "待复核事实", project_id="p1", source="assistant",
            review_due_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        )
        future = self.store.commit(
            "未来事实", project_id="p1", source="assistant",
            review_due_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        )
        other = self.store.commit(
            "其他项目事实", project_id="p2", source="assistant",
            review_due_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        )
        before = self.store._parse_memory_file(self.store._find_memory_file(due))
        report = list_review_due(self.vault, project_id="p1")
        ids = {item["id"] for item in report["items"]}
        self.assertEqual(ids, {due})
        self.assertNotIn(future, ids)
        self.assertNotIn(other, ids)
        self.assertFalse(report["mutated"])
        self.assertEqual(len(report["queue_hash"]), 64)
        self.assertIn("content_hash", report["items"][0])
        self.assertNotIn("content", report["items"][0])
        after = self.store._parse_memory_file(self.store._find_memory_file(due))
        self.assertEqual(after["access_count"], before["access_count"])

    def test_cli_requires_explicit_vault_and_writes_report(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--vault", str(self.vault)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["total"], 0)

        report = self.vault.parent / "review-due.json"
        main(["--vault", str(self.vault), "--out", str(report)])
        self.assertTrue(report.exists())

    def test_apply_requires_hash_and_supports_confirm_and_defer(self):
        due = self.store.commit(
            "需确认的事实", project_id="p1", source="assistant",
            review_due_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        )
        memory = self.store._parse_memory_file(self.store._find_memory_file(due))
        confirmed = apply_review(
            self.vault, memory_id=due, decision="confirm", reviewer="alice",
            expected_content_hash=memory["content_hash"], reason="人工核对通过",
        )
        self.assertTrue(confirmed["mutated"])
        self.assertEqual(self.store.get(due)["review_due_at"], "")

        deferred = self.store.commit(
            "需延期的事实", project_id="p1", source="assistant",
            review_due_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        )
        deferred_memory = self.store._parse_memory_file(self.store._find_memory_file(deferred))
        future = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        result = apply_review(
            self.vault, memory_id=deferred, decision="defer", reviewer="alice",
            expected_content_hash=content_hash(deferred_memory["content"]),
            reason="等待项目复盘", defer_until=future,
        )
        self.assertEqual(result["result"]["decision"], "defer")
        self.assertEqual(self.store.get(deferred)["review_due_at"], future)

        with self.assertRaises(ValueError):
            apply_review(
                self.vault, memory_id=deferred, decision="confirm", reviewer="alice",
                expected_content_hash="0" * 64, reason="过期清单",
            )


if __name__ == "__main__":
    unittest.main()
