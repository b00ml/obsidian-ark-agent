import hashlib
import tempfile
import unittest
from pathlib import Path

from agentlab.runtime.operation_verifier import (
    build_verification_metadata,
    reconcile_unknown_operations,
    verify_operation,
)
from agentlab.runtime.task_state import TaskStateStore


class TestOperationVerifier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = TaskStateStore(self.root / "task-state.db")
        self.store.ensure("task-verify")

    def tearDown(self):
        self.tmp.cleanup()

    def test_metadata_is_bounded_and_does_not_keep_content(self):
        meta = build_verification_metadata(
            "vault_write", '{"path":"note.md","content":"secret body"}'
        )
        self.assertEqual(meta["ref"], "Inbox/note.md")
        self.assertEqual(meta["content_hash"], hashlib.sha256(b"secret body").hexdigest())
        self.assertNotIn("secret body", str(meta))

    def test_vault_write_reconciles_from_matching_artifact(self):
        content = "---\ntitle: test\n---\nbody\n"
        path = self.root / "Inbox" / "note.md"
        path.parent.mkdir()
        path.write_text(content, encoding="utf-8")
        # Build the metadata directly to model the pre-dispatch ledger entry;
        # the ledger stores only a content hash, never the body.
        meta = {"kind": "vault_file", "ref": "Inbox/note.md",
                "content_hash": hashlib.sha256(content.encode()).hexdigest()}
        self.store.plan_tool("task-verify", operation_id="op-vault", tool_name="vault_write",
                             permission="write", side_effects="write", idempotent=True,
                             verification=meta)
        self.store.update_tool("task-verify", "op-vault", "unknown")
        settled = reconcile_unknown_operations(self.store, "task-verify", vault_root=self.root)
        self.assertEqual(settled[0]["status"], "succeeded")
        self.assertEqual(self.store.pending_operations("task-verify"), [])

    def test_vault_patch_records_and_checks_final_content_hash(self):
        path = self.root / "Inbox" / "patch.md"
        path.parent.mkdir()
        path.write_text("before\n", encoding="utf-8")
        meta = build_verification_metadata(
            "vault_patch", '{"path":"Inbox/patch.md","old":"before","new":"after"}',
            vault_root=self.root,
        )
        self.assertIn("result_hash", meta)
        path.write_text("after\n", encoding="utf-8")
        evidence = verify_operation({"verification": meta}, vault_root=self.root)
        self.assertEqual(evidence["status"], "succeeded")

    def test_memory_and_bili_adapters_settle_only_real_artifacts(self):
        mem = self.root / "ark" / "memory" / "context" / "m.md"
        mem.parent.mkdir(parents=True)
        mem.write_text(
            "---\n" "id: mem-1\n" "status: revoked\n"
            "content_hash: abc123\n" "---\nold\n", encoding="utf-8")
        evidence = verify_operation(
            {"verification": {"kind": "memory_revoke", "ref": "mem-1"}},
            vault_root=self.root,
        )
        self.assertEqual(evidence["status"], "succeeded")

    def test_memory_review_verifier_matches_active_hash_and_decision(self):
        path = self.root / "ark" / "memory" / "context" / "review.md"
        path.parent.mkdir(parents=True)
        content = "reviewed fact\n"
        digest = hashlib.sha256(content.encode()).hexdigest()
        path.write_text(
            f"---\nid: mem-review\nstatus: active\ncontent_hash: {digest}\n---\n{content}",
            encoding="utf-8",
        )
        evidence = verify_operation({"verification": {
            "kind": "memory_review", "ref": "mem-review",
            "content_hash": digest, "decision": "confirm",
        }}, vault_root=self.root)
        self.assertEqual(evidence["status"], "succeeded")
        inbox = self.root / "Inbox" / "video.md"
        inbox.parent.mkdir(exist_ok=True)
        inbox.write_text("source: https://www.bilibili.com/video/BV1TEST\n", encoding="utf-8")
        evidence = verify_operation(
            {"verification": {"kind": "bili_artifact", "ref": "BV1TEST"}},
            vault_root=self.root,
        )
        self.assertEqual(evidence["status"], "succeeded")

    def test_unsupported_or_mismatched_operation_stays_unknown(self):
        self.assertIsNone(verify_operation(
            {"verification": {"kind": "article_artifact", "source_ref": "hash"}},
            vault_root=self.root,
        ))

    def test_inbox_collection_adapter_uses_operation_ledger(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inbox_collector"))
        from queue_store import InboxQueueStore

        db = self.root / "inbox" / "queue.db"
        store = InboxQueueStore(db)
        before = store.snapshot_hash()
        store.begin_collection("op-inbox", before_hash=before)
        store.complete_collection("op-inbox", after_hash=store.snapshot_hash(), new_tasks=0)
        evidence = verify_operation(
            {"verification": {"kind": "inbox_queue", "operation_id": "op-inbox"}},
            vault_root=self.root, project_root=self.root,
        )
        self.assertEqual(evidence["status"], "succeeded")

        self.store.plan_tool(
            "task-verify", operation_id="op-inbox", tool_name="inbox_collect",
            permission="read", side_effects="write", idempotent=True,
            verification={"kind": "inbox_queue", "operation_id": "op-inbox"},
        )
        self.store.update_tool("task-verify", "op-inbox", "unknown")
        settled = reconcile_unknown_operations(
            self.store, "task-verify", vault_root=self.root, project_root=self.root
        )
        self.assertEqual(settled[0]["status"], "succeeded")
        self.assertIsNone(verify_operation(
            {"verification": {"kind": "vault_file", "ref": "Inbox/missing.md",
                               "content_hash": "deadbeef"}},
            vault_root=self.root,
        ))


if __name__ == "__main__":
    unittest.main()
