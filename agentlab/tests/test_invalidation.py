import tempfile
import unittest
from pathlib import Path

from agentlab.memory.invalidation import DerivedInvalidationCoordinator
from agentlab.memory.markdown_store import MemoryMarkdownStore
from agentlab.memory.ranges import RangeArchive, RangeGateway
from agentlab.runtime.task_state import TaskStateStore


class _Index:
    def __init__(self):
        self.removed = []

    def remove_document(self, path):
        self.removed.append(path)


class InvalidationTests(unittest.TestCase):
    def test_revoke_invalidates_index_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryMarkdownStore(tmp)
            mem_id = store.commit("RAG fact", project_id="p1")
            index = _Index()
            calls = []
            coordinator = DerivedInvalidationCoordinator(
                store, rag_index=index,
                cache_invalidators={"answer_cache": lambda **kwargs: calls.append(kwargs)},
            )
            result = coordinator.revoke(mem_id)
            self.assertTrue(result.source_updated)
            self.assertEqual(result.derived["rag_index"], "invalidated")
            self.assertEqual(result.derived["answer_cache"], "invalidated")
            self.assertEqual(result.derived["context_cache"], "not_configured")
            self.assertFalse(store.query("RAG fact", project_id="p1"))
            self.assertEqual(len(index.removed), 1)
            self.assertEqual(len(calls), 1)

    def test_business_source_ref_falls_back_to_markdown_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryMarkdownStore(tmp)
            mem_id = store.commit("RAG fact", source_ref="external:ticket-42")
            expected = store._find_memory_file(mem_id).relative_to(store.vault_root).as_posix()
            index = _Index()
            result = DerivedInvalidationCoordinator(store, rag_index=index).revoke(mem_id)
            self.assertTrue(result.complete)
            self.assertEqual(index.removed[0], expected)

    def test_missing_route_is_explicit_not_claimed_as_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryMarkdownStore(tmp)
            mem_id = store.commit("fact")
            result = DerivedInvalidationCoordinator(store).revoke(mem_id)
            self.assertEqual(result.derived["rag_index"], "not_configured")
            self.assertEqual(result.derived["answer_cache"], "not_configured")
            self.assertEqual(result.derived["context_cache"], "not_configured")
            self.assertTrue(result.complete)

    def test_failed_derived_cleanup_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryMarkdownStore(tmp)
            mem_id = store.commit("fact")

            def fail(**kwargs):
                raise RuntimeError("cache down")

            result = DerivedInvalidationCoordinator(store, cache_invalidators={"cache": fail}).revoke(mem_id)
            self.assertEqual(result.derived["cache"], "failed")
            self.assertFalse(result.complete)
            self.assertTrue(result.warnings)

    def test_session_range_and_task_checkpoint_are_invalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / "vault"
            sessions = Path(tmp) / "sessions"
            store = MemoryMarkdownStore(vault)
            mem_id = store.commit("旧事实", source_session="session-1")
            archive = RangeArchive(sessions)
            gateway = RangeGateway(archive)
            task_store = TaskStateStore(Path(tmp) / "task.db")
            task_store.ensure("task-1")
            task_store.patch(
                "task-1",
                {
                    "core_intent": {"memory_id": mem_id, "source_ref": "ark/memory/old.md"},
                    "completed_steps": [{"evidence_ref": f"memory:{mem_id}"}],
                },
            )
            archive_path = archive.path("session-1")
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_text("old range\n", encoding="utf-8")
            coordinator = DerivedInvalidationCoordinator(
                store,
                range_gateway=gateway,
                task_state_invalidator=lambda **kwargs: task_store.invalidate_memory(
                    "task-1", memory_id=kwargs["memory_id"], source_ref=kwargs.get("source_ref", "")
                ),
            )

            result = coordinator.revoke(mem_id)
            self.assertEqual(result.derived["session_range"], "invalidated")
            self.assertEqual(result.derived["task_state"], "invalidated")
            self.assertFalse(archive_path.exists())
            checkpoint = task_store.get("task-1")
            self.assertNotIn(mem_id, str(checkpoint.core_intent))
            self.assertNotIn(mem_id, str(checkpoint.completed_steps))
            self.assertIn(mem_id, checkpoint.context_snapshot["invalidated_memory_refs"])


if __name__ == "__main__":
    unittest.main()
