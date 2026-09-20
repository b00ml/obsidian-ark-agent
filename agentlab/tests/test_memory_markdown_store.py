"""
Unit tests for MemoryMarkdownStore.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

from agentlab.memory.markdown_store import MemoryMarkdownStore


class TestMemoryMarkdownStore(unittest.TestCase):
    
    def setUp(self):
        """Create temporary vault for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.vault_root = Path(self.temp_dir)
        self.store = MemoryMarkdownStore(str(self.vault_root))
    
    def tearDown(self):
        """Clean up temporary vault."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_directories(self):
        """Test that init creates all memory type directories."""
        for mem_type in MemoryMarkdownStore.MEMORY_TYPES:
            dir_path = self.vault_root / "ark" / "memory" / mem_type
            self.assertTrue(dir_path.exists(), f"Directory {mem_type} not created")
    
    def test_commit_basic(self):
        """Test basic memory commit."""
        mem_id = self.store.commit(
            content="User prefers dark mode",
            tags=["ui", "preference"],
            mem_type="core",
            importance=8
        )
        
        self.assertTrue(mem_id.startswith("mem-"))
        
        # Verify file exists
        file_path = self.store._find_memory_file(mem_id)
        self.assertIsNotNone(file_path)
        self.assertTrue(file_path.exists())
    
    def test_commit_with_project_context(self):
        """Test commit with project_id creates subdirectory."""
        mem_id = self.store.commit(
            content="Project uses FastAPI framework",
            tags=["framework"],
            mem_type="context",
            project_id="my-project"
        )
        
        file_path = self.store._find_memory_file(mem_id)
        self.assertIn("my-project", str(file_path))
    
    def test_commit_session_by_month(self):
        """Test session memories are organized by YYYY-MM."""
        mem_id = self.store.commit(
            content="User discussed database design",
            mem_type="sessions"
        )
        
        file_path = self.store._find_memory_file(mem_id)
        month_str = datetime.now().strftime("%Y-%m")
        self.assertIn(month_str, str(file_path))
    
    def test_commit_invalid_type(self):
        """Test commit rejects invalid memory type."""
        with self.assertRaises(ValueError):
            self.store.commit("test", mem_type="invalid_type")
    
    def test_commit_invalid_importance(self):
        """Test commit rejects out-of-range importance."""
        with self.assertRaises(ValueError):
            self.store.commit("test", importance=11)
        
        with self.assertRaises(ValueError):
            self.store.commit("test", importance=0)
    
    def test_query_basic(self):
        """Test basic query by keyword."""
        # Commit some memories
        self.store.commit("User prefers dark mode", tags=["ui"], importance=8)
        self.store.commit("System uses PostgreSQL database", tags=["db"], importance=6)
        self.store.commit("User dislikes light theme", tags=["ui"], importance=7)
        
        # Query for "dark mode"
        results = self.store.query("dark mode", limit=5)
        self.assertGreater(len(results), 0)
        self.assertIn("dark mode", results[0]["content"].lower())
    
    def test_query_cjk_bigrams(self):
        """Test CJK bigram recall (OPT-135 fix)."""
        self.store.commit("用户喜欢深色模式", tags=["ui"])
        
        results = self.store.query("深色", limit=5)
        self.assertGreater(len(results), 0)
        self.assertIn("深色", results[0]["content"])
    
    def test_query_multiple_terms(self):
        """Test multi-term OR matching."""
        self.store.commit("FastAPI framework with async support", tags=["framework"])
        
        # Should match on "FastAPI" OR "async"
        results = self.store.query("FastAPI async", limit=5)
        self.assertGreater(len(results), 0)
    
    def test_query_project_filter(self):
        """Test query filters by project_id."""
        self.store.commit("ProjectA uses Redis", project_id="project-a")
        self.store.commit("ProjectB uses Kafka", project_id="project-b")
        
        results = self.store.query("Redis Kafka", project_id="project-a")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["project_id"], "project-a")
    

    def test_bucket_commit_parse_and_query(self):
        """OPT-225：候选层月桶——追加、解析、query 命中。"""
        self.store.commit_to_bucket("sessions", "收件箱批次处理完成", tags=["收件箱"],
                                    source_session="sess-1")
        self.store.commit_to_bucket("sessions", "第二轮评测跑完", tags=["评测"])
        bucket = self.store._bucket_path("sessions", "default", "2026-09")
        self.assertTrue(bucket.exists())
        entries = self.store._parse_bucket_file(bucket)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["source_session"], "sess-1")
        q = self.store.query("收件箱 批次处理", limit=5)
        self.assertTrue(any("收件箱批次处理完成" in m["content"] for m in q))

    def test_bucket_file_not_found_by_single_lookup(self):
        """桶内条目不支持单条 update/archive 定位（候选层语义）。"""
        mem_id = self.store.commit_to_bucket("sessions", "一次性过程记录", tags=["t"])
        self.assertIsNone(self.store._find_memory_file(mem_id))


    def test_supersede_marks_status_and_query_excludes(self):
        """A4/OPT-228：supersedes 建立即置旧条 status=superseded，query 排除；lifecycle 负责物理归档。"""
        old_id = self.store.commit(content="旧结论：用 SQLite 存记忆", tags=["memory"],
                                   mem_type="decisions")
        new_id = self.store.commit(content="新结论：Markdown 是记忆真源", tags=["memory"],
                                   mem_type="decisions", supersedes=[old_id])
        old_mem = self.store.get(old_id)
        self.assertEqual(old_mem["status"], "superseded")
        self.assertEqual(old_mem["superseded_by"], new_id)
        q = self.store.query("记忆 真源 SQLite", limit=10)
        ids = [m["id"] for m in q]
        self.assertIn(new_id, ids)
        self.assertNotIn(old_id, ids)


    def test_hypothesis_cannot_supersede(self):
        """OPT-230 B7：confidence=hypothesis 禁止 supersedes 已确立结论。"""
        old_id = self.store.commit(content="已确立结论", tags=["m"], mem_type="decisions")
        with self.assertRaises(ValueError) as ctx:
            self.store.commit(content="低置信的替代猜测", tags=["m"], mem_type="decisions",
                              confidence="hypothesis", supersedes=[old_id])
        self.assertIn("MEMORY_HYPOTHESIS_NO_SUPERSEDE", str(ctx.exception))
        # 正常置信的 supersedes 不受影响
        self.store.commit(content="正常置信的替代", tags=["m"], mem_type="decisions",
                          confidence=0.9, supersedes=[old_id])
        self.assertTrue(True)

    def test_query_excludes_archived(self):
        """Test query excludes archived memories by default."""
        mem_id = self.store.commit("Old decision about API design")
        self.store.archive(mem_id, reason="superseded")
        
        results = self.store.query("API design")
        self.assertEqual(len(results), 0)
    
    def test_update_content(self):
        """Test updating memory content."""
        mem_id = self.store.commit("User prefers dark mode")
        
        success = self.store.update(mem_id, content="User prefers dark mode with high contrast")
        self.assertTrue(success)
        
        memory = self.store.get(mem_id)
        self.assertIn("high contrast", memory["content"])
    
    def test_update_tags(self):
        """Test updating memory tags."""
        mem_id = self.store.commit("System config", tags=["system"])
        
        self.store.update(mem_id, tags=["system", "config", "production"])
        
        memory = self.store.get(mem_id)
        self.assertEqual(len(memory["tags"]), 3)
        self.assertIn("production", memory["tags"])
    
    def test_update_supersedes(self):
        """Test update marks superseded memories."""
        old_id = self.store.commit("Old API uses REST")
        new_id = self.store.commit("New API uses GraphQL")
        
        self.store.update(new_id, supersedes=[old_id])
        
        old_memory = self.store.get(old_id)
        self.assertEqual(old_memory["superseded_by"], new_id)
        
        new_memory = self.store.get(new_id)
        self.assertIn(old_id, new_memory.get("supersedes", []))
    
    def test_update_nonexistent(self):
        """Test update returns False for nonexistent memory."""
        success = self.store.update("mem-nonexistent", content="test")
        self.assertFalse(success)
    
    def test_archive_moves_to_archive_dir(self):
        """Test archive moves file to archive/YYYY/."""
        mem_id = self.store.commit("Temporary decision")
        
        success = self.store.archive(mem_id, reason="lifecycle")
        self.assertTrue(success)
        
        # Original location should not exist
        original_path = self.vault_root / "ark" / "memory" / "context" / f"{mem_id}.md"
        self.assertFalse(original_path.exists())
        
        # Archive location should exist
        year = datetime.now().year
        archive_path = self.vault_root / "ark" / "memory" / "archive" / str(year) / f"{mem_id}.md"
        self.assertTrue(archive_path.exists())
        
        # Status should be updated
        memory = self.store.get(mem_id)
        self.assertEqual(memory["status"], "archived")
        self.assertEqual(memory["archive_reason"], "lifecycle")
    
    def test_normalize_tags_string(self):
        """Test tag normalization handles string input (OPT-135 fix)."""
        tags = self.store._normalize_tags("ui,preference,dark-mode")
        self.assertEqual(len(tags), 3)
        self.assertIn("ui", tags)
        self.assertIn("preference", tags)
    
    def test_normalize_tags_limit(self):
        """Test tag normalization limits to 8 tags."""
        tags = self.store._normalize_tags([f"tag{i}" for i in range(20)])
        self.assertEqual(len(tags), 8)
    
    def test_parse_and_render_roundtrip(self):
        """Test parse and render are inverse operations."""
        mem_id = self.store.commit(
            content="Test memory content",
            tags=["test", "roundtrip"],
            importance=7
        )
        
        file_path = self.store._find_memory_file(mem_id)
        memory = self.store._parse_memory_file(file_path)
        
        # Verify key fields
        self.assertEqual(memory["id"], mem_id)
        self.assertEqual(memory["importance"], 7)
        self.assertIn("test", memory["tags"])
        self.assertEqual(memory["content"], "Test memory content")
    
    def test_time_decay_calculation(self):
        """Test time decay factor calculation."""
        # Recent access = high decay factor
        recent = datetime.now(timezone.utc).isoformat()
        decay_recent = self.store._calculate_decay(recent)
        self.assertGreater(decay_recent, 0.9)
        
        # Old access (simulated) = lower decay factor
        # Would need to mock datetime for proper test
        # Here we just verify the function doesn't crash
        old = "2025-01-01T00:00:00Z"
        decay_old = self.store._calculate_decay(old)
        self.assertGreater(decay_old, 0.0)
        self.assertLess(decay_old, 1.0)
    
    def test_access_tracking(self):
        """Test that query updates access_count and last_accessed_at."""
        mem_id = self.store.commit("Trackable memory")
        
        # Query twice
        self.store.query("Trackable")
        self.store.query("Trackable")
        
        memory = self.store.get(mem_id)
        self.assertGreaterEqual(memory["access_count"], 2)

    def test_query_can_disable_access_tracking(self):
        """Read-only callers can query without changing the memory file."""
        mem_id = self.store.commit("Read-only evaluation memory")
        file_path = self.store._find_memory_file(mem_id)
        before_text = file_path.read_text(encoding="utf-8")
        before_mtime = file_path.stat().st_mtime_ns
        before_memory = self.store._parse_memory_file(file_path)

        results = self.store.query("Read-only evaluation", track_access=False)

        self.assertEqual([item["id"] for item in results], [mem_id])
        self.assertEqual(file_path.read_text(encoding="utf-8"), before_text)
        self.assertEqual(file_path.stat().st_mtime_ns, before_mtime)
        memory = self.store._parse_memory_file(file_path)
        self.assertEqual(memory["access_count"], 0)
        self.assertEqual(memory["last_accessed_at"], before_memory["last_accessed_at"])


if __name__ == "__main__":
    unittest.main()
