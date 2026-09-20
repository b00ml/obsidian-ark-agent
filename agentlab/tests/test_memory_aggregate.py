import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentlab.memory.aggregate import build_memory_aggregate, write_memory_aggregate
from agentlab.memory.markdown_store import MemoryMarkdownStore


class TestMemoryAggregate(unittest.TestCase):
    def test_build_groups_active_atomic_memories_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MemoryMarkdownStore(temp)
            core_id = store.commit("用户偏好深色主题", tags=["偏好", "主题"], mem_type="core")
            store.commit("项目决定使用 Markdown 真源", tags=["架构"], mem_type="decisions")
            store.commit("处理前先核对来源", tags=["流程"], mem_type="procedures")
            candidate = store.commit(
                "未确认的候选", tags=["候选"], mem_type="sessions",
                candidate_first=True, source="assistant",
            )
            before = store._find_memory_file(core_id).read_text(encoding="utf-8")
            view = build_memory_aggregate(store)
            self.assertEqual(view["items"], 3)
            self.assertIn("profile", view["groups"])
            self.assertIn("decision", view["groups"])
            self.assertIn("procedure", view["groups"])
            self.assertNotIn(candidate, json.dumps(view, ensure_ascii=False))
            self.assertEqual(store._find_memory_file(core_id).read_text(encoding="utf-8"), before)

    def test_write_is_hidden_rebuildable_json(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MemoryMarkdownStore(temp)
            store.commit("事实", tags=["topic"], mem_type="context")
            path = write_memory_aggregate(store)
            self.assertTrue(path.as_posix().endswith(".agent-brain/memory/aggregate-v1.json"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "memory-aggregate-v1")
            self.assertFalse(any("aggregate-v1" in p.name for p in (store.memory_root).rglob("*.md")))

    def test_aggregate_uses_effective_memory_lifecycle_not_stored_status_only(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MemoryMarkdownStore(temp)
            active = store.commit("active Redis 事实", tags=["redis"], mem_type="context")
            expired = store.commit(
                "expired Redis 事实", tags=["redis"], mem_type="context",
                valid_until=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            )
            future = store.commit(
                "future Redis 事实", tags=["redis"], mem_type="context",
                valid_from=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            )
            due = store.commit(
                "due Redis 事实", tags=["redis"], mem_type="context",
                review_due_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            )
            projection = json.dumps(build_memory_aggregate(store), ensure_ascii=False)
            self.assertIn(active, projection)
            self.assertNotIn(expired, projection)
            self.assertNotIn(future, projection)
            self.assertNotIn(due, projection)


if __name__ == "__main__":
    unittest.main()
