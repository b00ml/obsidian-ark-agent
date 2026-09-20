import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agentlab.eval.memory_lifecycle_action import apply_lifecycle_action, main
from agentlab.memory.markdown_store import MemoryMarkdownStore


class MemoryLifecycleActionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name) / "vault"
        self.vault.mkdir()
        self.store = MemoryMarkdownStore(self.vault)

    def tearDown(self):
        self.tmp.cleanup()

    def test_promote_correct_and_revoke_are_hash_and_scope_bound(self):
        candidate = self.store.commit(
            "候选偏好", mem_type="core", project_id="p1", session_id="s1",
            source="assistant", candidate_first=True,
        )
        before = self.store.get(candidate)
        promoted = apply_lifecycle_action(
            self.vault, memory_id=candidate, action="promote",
            expected_content_hash=before["content_hash"], reviewer="alice",
            reason="人工确认", project_id="p1", session_id="s1",
        )
        self.assertEqual(promoted["result"]["status"], "active")
        self.assertEqual(promoted["derived"]["aggregate"], "rebuilt")
        self.assertTrue((self.vault / ".agent-brain" / "memory" / "aggregate-v1.json").exists())

        active = self.store.get(candidate)
        corrected = apply_lifecycle_action(
            self.vault, memory_id=candidate, action="correct",
            expected_content_hash=active["content_hash"], reviewer="alice",
            reason="用户纠正", project_id="p1", session_id="s1",
            content="修正后的偏好",
        )
        successor = corrected["result"]["id"]
        self.assertEqual(self.store.get(candidate)["status"], "superseded")
        self.assertEqual(self.store.get(successor)["status"], "active")
        successor_row = self.store.get(successor)
        revoked = apply_lifecycle_action(
            self.vault, memory_id=successor, action="revoke",
            expected_content_hash=successor_row["content_hash"], reviewer="alice",
            reason="用户撤销", project_id="p1", session_id="s1",
        )
        self.assertEqual(revoked["result"]["status"], "revoked")
        self.assertEqual(revoked["derived"]["rag_index"], "not_configured")
        self.assertFalse(self.store.query("偏好", project_id="p1", track_access=False))

    def test_stale_hash_and_wrong_scope_fail_before_mutation(self):
        candidate = self.store.commit("候选", project_id="p1", candidate_first=True)
        row = self.store.get(candidate)
        with self.assertRaisesRegex(PermissionError, "scope denied"):
            apply_lifecycle_action(
                self.vault, memory_id=candidate, action="promote",
                expected_content_hash=row["content_hash"], reviewer="alice",
                reason="确认", project_id="p2",
            )
        with self.assertRaisesRegex(ValueError, "content changed"):
            apply_lifecycle_action(
                self.vault, memory_id=candidate, action="promote",
                expected_content_hash="0" * 64, reviewer="alice", reason="确认",
                project_id="p1",
            )
        self.assertEqual(self.store.get(candidate)["status"], "candidate")

    def test_conflict_cannot_be_promoted_but_can_be_revoked(self):
        first = self.store.commit("旧决策", mem_type="decisions", subject="storage")
        conflict = self.store.commit("新决策", mem_type="decisions", subject="storage")
        row = self.store.get(conflict)
        self.assertEqual(row["status"], "conflict")
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            apply_lifecycle_action(
                self.vault, memory_id=conflict, action="promote",
                expected_content_hash=row["content_hash"], reviewer="alice", reason="确认",
            )
        revoked = apply_lifecycle_action(
            self.vault, memory_id=conflict, action="revoke",
            expected_content_hash=row["content_hash"], reviewer="alice", reason="拒绝冲突",
        )
        self.assertEqual(revoked["result"]["status"], "revoked")
        self.assertEqual(self.store.get(first)["status"], "active")

    def test_conflict_can_be_resolved_only_by_explicit_correction(self):
        first = self.store.commit("旧决策", mem_type="decisions", subject="storage")
        conflict = self.store.commit("新决策", mem_type="decisions", subject="storage")
        row = self.store.get(conflict)
        corrected = apply_lifecycle_action(
            self.vault, memory_id=conflict, action="correct",
            expected_content_hash=row["content_hash"], reviewer="alice",
            reason="人工确认新决策", content="人工确认后的新决策",
        )
        successor = corrected["result"]["id"]
        self.assertEqual(self.store.get(conflict)["status"], "superseded")
        self.assertEqual(self.store.get(first)["status"], "superseded")
        self.assertEqual(self.store.get(successor)["status"], "active")

    def test_cli_requires_explicit_apply(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--vault", str(self.vault)])
        self.assertEqual(code, 1)
        self.assertIn("--apply", json.loads(output.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
