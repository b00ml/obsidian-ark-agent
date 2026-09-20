import io
import argparse
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentlab.rag.index_store import RagIndexStore
import agentlab.rag_reindex as rag_reindex
from agentlab.rag_reindex import main


class _TestEmbedder:
    model = "fake-v1"

    def __init__(self, fail=False):
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("provider unavailable")
        return [[float(len(text) or 1)] for text in texts]


class TestP3Queue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = RagIndexStore(self.root / "index.sqlite", None, vault_root=self.root,
                                   batch_size=2, max_attempts=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_plan_is_side_effect_free(self):
        (self.root / "a.md").write_text("猫。", encoding="utf-8")
        plan = self.store.plan_changes(self.root)
        self.assertEqual(plan["upserts"], 1)
        self.assertFalse((self.root / "index.sqlite").exists())

    def test_enqueue_process_and_checkpoint(self):
        (self.root / "a.md").write_text("猫。", encoding="utf-8")
        (self.root / "b.md").write_text("狗。", encoding="utf-8")
        first = self.store.enqueue_changes(self.root)
        second = self.store.enqueue_changes(self.root)
        self.assertEqual(first["enqueued"], 2)
        self.assertEqual(second["enqueued"], 0)
        self.assertEqual(second["skipped"], 2)
        result = self.store.process_queue(self.root)
        self.assertEqual(result["succeeded"], 2)
        self.assertEqual(result["batches"], 1)
        status = self.store.queue_status(self.root)
        self.assertEqual(status["ready"], 0)
        self.assertEqual(status["succeeded"], 2)
        self.assertEqual(status["checkpoints"]["completed"], 1)

    def test_hidden_relative_path_is_preserved_and_requeued_after_transient_delete(self):
        hidden = self.root / ".dashboard-backup" / "snapshot.md"
        hidden.parent.mkdir()
        hidden.write_text("隐藏目录内容。", encoding="utf-8")
        first = self.store.enqueue_changes(self.root)
        self.assertEqual(first["enqueued"], 1)
        self.store.process_queue(self.root)
        self.assertTrue(self.store.search_lexical("隐藏目录内容"))
        status = self.store.index_status(self.root)
        self.assertEqual(status["indexed_files"], 1)
        self.assertEqual(status["coverage"], 1.0)

        # A previously successful queue row must not mask an index row that
        # was removed after a transient file disappearance.
        self.store.remove_document(".dashboard-backup/snapshot.md")
        plan = self.store.enqueue_changes(self.root)
        self.assertEqual(plan["enqueued"], 1)
        self.store.process_queue(self.root)
        self.assertTrue(self.store.search_lexical("隐藏目录内容"))
        self.assertEqual(self.store.index_status(self.root)["coverage"], 1.0)

    def test_since_filters_upserts_but_keeps_deletes(self):
        old = self.root / "old.md"
        changed = self.root / "changed.md"
        old.write_text("旧。", encoding="utf-8")
        changed.write_text("初始。", encoding="utf-8")
        self.store.sync_vault(self.root)
        cutoff = max(old.stat().st_mtime, changed.stat().st_mtime)
        changed.write_text("更新。", encoding="utf-8")
        os.utime(changed, (cutoff + 10, cutoff + 10))
        old.unlink()
        plan = self.store.plan_changes(self.root, since=cutoff + 1)
        self.assertEqual([(x["path"], x["operation"]) for x in plan["changes"]],
                         [("changed.md", "upsert"), ("old.md", "delete")])

    def test_new_and_modify_priority_precedes_delete(self):
        (self.root / "gone.md").write_text("将删除。", encoding="utf-8")
        self.store.sync_vault(self.root)
        (self.root / "gone.md").unlink()
        (self.root / "new.md").write_text("新。", encoding="utf-8")
        plan = self.store.enqueue_changes(self.root)
        self.assertEqual(plan["upserts"], 1)
        self.assertEqual(plan["deletes"], 1)
        conn = sqlite3.connect(self.root / "index.sqlite")
        try:
            rows = conn.execute(
                "SELECT operation,priority FROM ingest_queue ORDER BY priority DESC,id"
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual(rows, [("upsert", 100), ("delete", 10)])
        self.store.process_queue(self.root)
        self.assertEqual(self.store.search_lexical("将删除"), [])
        self.assertTrue(self.store.search_lexical("新"))

    def test_lease_expiry_recovers_running_item(self):
        (self.root / "a.md").write_text("猫。", encoding="utf-8")
        self.store.enqueue_changes(self.root)
        first_batch, rows = self.store._claim_queue_batch(1, 300)
        self.assertTrue(first_batch)
        self.assertEqual(len(rows), 1)
        conn = sqlite3.connect(self.root / "index.sqlite")
        try:
            conn.execute("UPDATE ingest_queue SET lease_until=0 WHERE status='running'")
            conn.commit()
        finally:
            conn.close()
        second_batch, rows = self.store._claim_queue_batch(1, 300)
        self.assertTrue(second_batch)
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(first_batch, second_batch)
        self.store._finish_checkpoint(second_batch)
        status = self.store.queue_status(self.root)
        self.assertEqual(status["running"], 1)
        self.assertEqual(status["checkpoints"]["started"], 0)
        self.assertEqual(status["checkpoints"]["aborted"], 1)

    def test_interruption_leaves_aborted_checkpoint_and_retries(self):
        (self.root / "a.md").write_text("猫。", encoding="utf-8")
        self.store.enqueue_changes(self.root)
        original = self.store.upsert_document

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        self.store.upsert_document = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.store.process_queue(self.root)
        status = self.store.queue_status(self.root)
        self.assertEqual(status["checkpoints"]["aborted"], 1)
        self.assertEqual(status["running"], 1)
        self.store.upsert_document = original
        conn = sqlite3.connect(self.root / "index.sqlite")
        try:
            conn.execute("UPDATE ingest_queue SET lease_until=0 WHERE status='running'")
            conn.commit()
        finally:
            conn.close()
        result = self.store.process_queue(self.root)
        self.assertEqual(result["succeeded"], 1)

    def test_retry_failed_resets_dead_item(self):
        (self.root / "a.md").write_text("猫。", encoding="utf-8")
        self.store.enqueue_changes(self.root)
        original = self.store.upsert_document

        def fail(*args, **kwargs):
            raise RuntimeError("boom")

        self.store.upsert_document = fail
        first = self.store.process_queue(self.root)
        self.assertEqual(first["dead"], 1)
        self.store.upsert_document = original
        retry = self.store.retry_queue(self.root, limit=1)
        self.assertEqual(retry["reset"], 1)
        self.assertEqual(retry["succeeded"], 1)

    def test_cli_dry_run_does_not_create_index(self):
        (self.root / "a.md").write_text("猫。", encoding="utf-8")
        index = self.root / "cli.sqlite"
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--full", "--vault", str(self.root), "--index", str(index), "--dry-run"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["upserts"], 1)
        self.assertFalse(index.exists())

    def test_cli_status_writes_hidden_snapshot(self):
        snapshot = self.root / ".agent-brain" / "rag-status.json"
        output = io.StringIO()
        with redirect_stdout(output):
            code = main([
                "--status", "--vault", str(self.root),
                "--index", str(self.root / "index.sqlite"), "--out", str(snapshot),
            ])
        self.assertEqual(code, 0)
        self.assertTrue(snapshot.exists())
        saved = json.loads(snapshot.read_text(encoding="utf-8"))
        self.assertEqual(saved["schema"], "rag-ops-status-v1")
        self.assertEqual(saved["mode"], "status")
        printed = json.loads(output.getvalue())
        self.assertEqual(printed["snapshot_path"], str(snapshot))

    def test_cli_status_uses_existing_index_metadata(self):
        index = self.root / "v2.sqlite"
        RagIndexStore(
            index,
            None,
            vault_root=self.root,
            index_version="s1-test-v2",
            parser_version="markdown-structure-v2-min64",
            chunk_strategy_version="markdown-structure-v2-min64",
        ).sync_vault(self.root)
        output = io.StringIO()
        with redirect_stdout(output):
            code = main([
                "--status", "--vault", str(self.root), "--index", str(index),
                "--config", str(self.root / "missing-config.json"),
            ])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["index_version"], "s1-test-v2")
        self.assertEqual(result["chunk_strategy_version"], "markdown-structure-v2-min64")

    def test_cli_reconcile_skips_unchanged_files_and_detects_edits(self):
        path = self.root / "note.md"
        path.write_text("初始内容。", encoding="utf-8")
        index = self.root / "reconcile.sqlite"
        store = RagIndexStore(index, None, vault_root=self.root)
        store.sync_vault(self.root)

        output = io.StringIO()
        with redirect_stdout(output):
            code = main([
                "--reconcile", "--vault", str(self.root), "--index", str(index),
                "--config", str(self.root / "missing-config.json"), "--dry-run",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["upserts"], 0)

        path.write_text("修改后的内容。", encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            code = main([
                "--reconcile", "--vault", str(self.root), "--index", str(index),
                "--config", str(self.root / "missing-config.json"), "--dry-run",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["upserts"], 1)

    def test_reconcile_repairs_missing_vectors_when_provider_is_configured(self):
        path = self.root / "note.md"
        path.write_text("已有词法内容。", encoding="utf-8")
        index = self.root / "repair.sqlite"
        lexical = RagIndexStore(index, None, vault_root=self.root)
        lexical.sync_vault(self.root)

        vector = RagIndexStore(index, _TestEmbedder(), vault_root=self.root)
        plan = vector.plan_changes(self.root)
        self.assertEqual(plan["upserts"], 1)
        self.assertEqual(plan["changes"][0]["reason"], "missing_vector")
        vector.enqueue_changes(self.root)
        result = vector.process_queue(self.root)
        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(vector.plan_changes(self.root)["upserts"], 0)

    def test_reindex_store_respects_vector_mode_off(self):
        cfg = SimpleNamespace(
            vault_root=str(self.root),
            rag=SimpleNamespace(
                vector_enabled=True,
                vector_mode="off",
                embed_base_url="http://fake",
                embed_model="fake-v1",
                embed_timeout=1.0,
                index_version="s1-p2-v1",
                chunk_strategy="markdown-structure-v1",
            ),
        )
        args = argparse.Namespace(
            config=None, vault=None, index=str(self.root / "off.sqlite"),
            index_version=None, chunk_strategy=None,
        )
        with patch.object(rag_reindex, "load_config", return_value=cfg), \
                patch("agentlab.rag.embed.OpenAIEmbedder") as embedder_cls:
            store, _vault, _index = rag_reindex._store(args)
        self.assertIsNone(store.embedder)
        embedder_cls.assert_not_called()

    def test_queue_success_keeps_lexical_fresh_when_vector_provider_fails(self):
        path = self.root / "partial.md"
        path.write_text("最新词法内容。", encoding="utf-8")
        index = self.root / "partial.sqlite"
        store = RagIndexStore(index, _TestEmbedder(fail=True), vault_root=self.root)
        store.enqueue_changes(self.root)
        result = store.process_queue(self.root)
        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["vector_failed"], 1)
        self.assertTrue(store.search_lexical("最新词法内容"))
        self.assertEqual(store.queue_status(self.root)["ready"], 0)
        self.assertEqual(store.failure_count, 1)
        self.assertEqual(store.plan_changes(self.root)["upserts"], 0,
                         "failure backoff window should prevent reconcile hot-loop")


if __name__ == "__main__":
    unittest.main()
