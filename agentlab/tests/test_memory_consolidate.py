"""
记忆巩固单元测试（F5-011 Phase 3）

测试 MemoryConsolidator 的三个方法：
1. lifecycle: 归档低价值/过期记忆
2. defrag: 合并重复、拆分臃肿记忆
3. reflect: (stub，未实现)
"""

import asyncio
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentlab.memory.consolidate import MemoryConsolidator
from agentlab.memory.markdown_store import MemoryMarkdownStore


class TestLifecycle(unittest.TestCase):
    """测试生命周期管理（归档规则）"""
    
    def setUp(self):
        """创建临时 store"""
        self.temp_dir = tempfile.mkdtemp()
        self.vault_root = Path(self.temp_dir)
        self.store = MemoryMarkdownStore(str(self.vault_root))
        self.consolidator = MemoryConsolidator(self.store)
        
        # 初始化目录结构
        for mem_type in ["core", "context", "procedures", "decisions", "sessions", "archive"]:
            (self.vault_root / "ark" / "memory" / mem_type).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _set_last_accessed(self, mem_id: str, iso_time: str):
        """辅助函数：修改记忆的 last_accessed_at 时间戳"""
        mem_file = self.store._find_memory_file(mem_id)
        content = mem_file.read_text(encoding="utf-8")
        # 使用正则替换 last_accessed_at 行
        new_content = re.sub(
            r"last_accessed_at: '[^']*'",
            f"last_accessed_at: '{iso_time}'",
            content
        )
        mem_file.write_text(new_content, encoding="utf-8")
    
    def test_archive_superseded(self):
        """测试归档被替代的记忆"""
        # 创建两条记忆，第二条替代第一条
        mem1_id = self.store.commit(
            content="旧版本流程",
            tags=["workflow"],
            mem_type="procedures",
            importance=7
        )
        
        mem2_id = self.store.commit(
            content="新版本流程",
            tags=["workflow"],
            mem_type="procedures",
            importance=8,
            supersedes=[mem1_id]
        )
        
        # 手动设置 mem1 的 superseded_by（直接修改文件）
        mem1_file = self.store._find_memory_file(mem1_id)
        content = mem1_file.read_text(encoding="utf-8")
        # 在 status: active 后插入 superseded_by
        content = content.replace(
            "status: active\n",
            f"status: active\nsuperseded_by: {mem2_id}\n"
        )
        mem1_file.write_text(content, encoding="utf-8")
        
        # dry_run 预览
        result = asyncio.run(self.consolidator.lifecycle(dry_run=True))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["scanned"], 2)
        self.assertEqual(result["archived"], 1)
        self.assertEqual(result["reasons"]["superseded"], 1)
        self.assertTrue(result["dry_run"])
        
        # 实际归档
        result = asyncio.run(self.consolidator.lifecycle(dry_run=False))
        self.assertEqual(result["archived"], 1)
        
        # 验证 mem1 已归档
        archive_dir = self.vault_root / "ark" / "memory" / "archive"
        archived_files = list(archive_dir.rglob("*.md"))
        self.assertEqual(len(archived_files), 1)
        
        # 验证 mem2 仍在 procedures
        mem2 = self.store.get(mem2_id)
        self.assertIsNotNone(mem2)
        self.assertNotEqual(mem2.get("status"), "archived")
    
    def test_archive_low_importance_old(self):
        """测试归档低重要性+长期未访问的记忆"""
        # 创建一条 importance=3、200天未访问的记忆
        old_time = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        
        mem_id = self.store.commit(
            content="过期的临时记录",
            tags=["temp"],
            mem_type="context",
            importance=3
        )
        
        # 修改 last_accessed_at
        self._set_last_accessed(mem_id, old_time)
        
        # 归档
        result = asyncio.run(self.consolidator.lifecycle(dry_run=False))
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["archived"], 1)
        self.assertGreater(result["reasons"]["low_importance"], 0)
        
        # 验证已归档
        archive_dir = self.vault_root / "ark" / "memory" / "archive"
        self.assertTrue(any(archive_dir.rglob("*.md")))
    
    def test_archive_medium_importance_very_old(self):
        """测试归档中等重要性+超长期未访问（>365天）的记忆"""
        old_time = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        
        mem_id = self.store.commit(
            content="过期的中等重要记录",
            tags=["old"],
            mem_type="decisions",
            importance=5
        )
        
        # 修改时间戳
        self._set_last_accessed(mem_id, old_time)
        
        # 归档
        result = asyncio.run(self.consolidator.lifecycle(dry_run=False))
        self.assertGreaterEqual(result["archived"], 1)
        self.assertGreater(result["reasons"]["expired"], 0)
    
    def test_no_archive_recent_or_high_importance(self):
        """测试不归档高重要性或最近访问的记忆"""
        # 高重要性（importance=9）
        self.store.commit(
            content="核心记忆",
            tags=["core"],
            mem_type="core",
            importance=9
        )
        
        # 最近访问（importance=4，但刚创建）
        self.store.commit(
            content="最近的临时记录",
            tags=["recent"],
            mem_type="context",
            importance=4
        )
        
        # 归档
        result = asyncio.run(self.consolidator.lifecycle(dry_run=False))
        self.assertEqual(result["archived"], 0)


class TestDefrag(unittest.TestCase):
    """测试碎片整理（去重+拆分）"""
    
    def setUp(self):
        """创建临时 store"""
        self.temp_dir = tempfile.mkdtemp()
        self.vault_root = Path(self.temp_dir)
        self.store = MemoryMarkdownStore(str(self.vault_root))
        self.consolidator = MemoryConsolidator(self.store)
        
        for mem_type in ["core", "context", "procedures", "decisions", "sessions", "archive"]:
            (self.vault_root / "ark" / "memory" / mem_type).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    

    def test_split_clamps_importance_floor(self):
        """B8 回归：importance=1 的超长记忆拆分时新段不再减到 0（触发校验失败）。"""
        sep = "\n\n"
        long_body = sep.join(f"段落{i}：" + "细节" * 400 for i in range(3))
        self.store.commit(content=long_body, tags=["split"], mem_type="decisions",
                          importance=1)
        out = asyncio.run(self.consolidator.defrag(dry_run=False))
        self.assertEqual(out["split"], 1)
        # 新段落记忆存在且 importance 下限为 1（source_session=split-from-*）
        split_files = [f for f in (self.vault_root / "ark/memory/decisions").glob("*.md")
                       if "split-from-" in f.read_text(encoding="utf-8")]
        self.assertTrue(split_files, "拆分新记忆未生成")
        body = split_files[0].read_text(encoding="utf-8")
        self.assertIn("importance: 1", body)

    def test_merge_duplicates(self):
        """测试合并重复记忆（Jaccard > 0.7）"""
        # 创建两条高度相似的记忆（使用更简单的重复内容确保 Jaccard > 0.7）
        content1 = "使用 markdown store 存储 agent 记忆数据"
        content2 = "使用 markdown store 存储 agent 记忆信息"
        
        mem1_id = self.store.commit(
            content=content1,
            tags=["storage"],
            mem_type="procedures",
            importance=6
        )
        
        mem2_id = self.store.commit(
            content=content2,
            tags=["storage"],
            mem_type="procedures",
            importance=7  # 更高重要性，应保留
        )
        
        # dry_run 预览
        result = asyncio.run(self.consolidator.defrag(dry_run=True))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["scanned"], 2)
        self.assertGreaterEqual(result["merged"], 1)  # 至少合并1对
        self.assertTrue(result["dry_run"])
        
        # 实际合并
        result = asyncio.run(self.consolidator.defrag(dry_run=False))
        self.assertGreaterEqual(result["merged"], 1)
        
        # 验证保留了高 importance 的 mem2
        mem2 = self.store.get(mem2_id)
        self.assertIsNotNone(mem2)
        self.assertNotEqual(mem2.get("status"), "archived")
        
        # 验证至少有一个被归档
        archive_dir = self.vault_root / "ark" / "memory" / "archive"
        archived_files = list(archive_dir.rglob("*.md"))
        self.assertGreaterEqual(len(archived_files), 1)
    
    def test_split_long_memory(self):
        """测试拆分超长记忆（>2000字符）"""
        # 创建一条超长记忆（3段）
        long_content = "\n\n".join([
            "第一段：" + "a" * 800,
            "第二段：" + "b" * 800,
            "第三段：" + "c" * 800
        ])
        
        mem_id = self.store.commit(
            content=long_content,
            tags=["long"],
            mem_type="context",
            importance=6
        )
        
        # dry_run 预览
        result = asyncio.run(self.consolidator.defrag(dry_run=True))
        self.assertEqual(result["scanned"], 1)
        self.assertEqual(result["split"], 1)
        
        # 实际拆分
        result = asyncio.run(self.consolidator.defrag(dry_run=False))
        self.assertEqual(result["split"], 1)
        
        # 验证原记忆只保留第一段
        updated_mem = self.store.get(mem_id)
        self.assertLess(len(updated_mem["content"]), 1000)
        self.assertIn("第一段", updated_mem["content"])
        
        # 验证创建了新记忆
        all_context = list((self.vault_root / "ark" / "memory" / "context").rglob("*.md"))
        self.assertGreater(len(all_context), 1)  # 原记忆 + 拆分出的新记忆
    
    def test_no_merge_dissimilar(self):
        """测试不合并低相似度记忆"""
        self.store.commit(
            content="Python 异步编程最佳实践",
            tags=["python"],
            mem_type="procedures",
            importance=7
        )
        
        self.store.commit(
            content="TypeScript 类型系统设计",
            tags=["typescript"],
            mem_type="procedures",
            importance=7
        )
        
        # 去重
        result = asyncio.run(self.consolidator.defrag(dry_run=False))
        self.assertEqual(result["merged"], 0)
    
    def test_no_split_short_memory(self):
        """测试不拆分短记忆"""
        self.store.commit(
            content="短记录内容，不到2000字符",
            tags=["short"],
            mem_type="context",
            importance=5
        )
        
        # 拆分
        result = asyncio.run(self.consolidator.defrag(dry_run=False))
        self.assertEqual(result["split"], 0)


class TestReflect(unittest.TestCase):
    """测试反思提炼（stub）"""
    
    def setUp(self):
        """创建临时 store"""
        self.temp_dir = tempfile.mkdtemp()
        self.vault_root = Path(self.temp_dir)
        self.store = MemoryMarkdownStore(str(self.vault_root))
        self.consolidator = MemoryConsolidator(self.store)
    
    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_reflect_not_implemented(self):
        """测试 reflect 返回 not_implemented"""
        result = asyncio.run(self.consolidator.reflect("test-session-123"))
        self.assertEqual(result["status"], "not_implemented")
        self.assertEqual(result["session_id"], "test-session-123")
        self.assertEqual(result["extracted"], 0)
        self.assertEqual(result["updated"], 0)


if __name__ == "__main__":
    unittest.main()
