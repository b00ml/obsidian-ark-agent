"""
记忆巩固任务（F5-011 Phase 3）

实现三个后台巩固任务：
1. reflect: 会话末反思，episodic → semantic 提炼
2. lifecycle: 归档低价值/过期记忆，写侧遗忘
3. defrag: 合并重复、拆分臃肿记忆

设计原则：
- 幂等性：重复运行安全
- 可观测：返回详细操作日志
- 可逆性：归档而非删除，支持恢复
"""

import re
from datetime import datetime, timezone
from typing import Optional

from agentlab.memory.markdown_store import MemoryMarkdownStore


class MemoryConsolidator:
    """记忆巩固调度器"""
    
    def __init__(self, store: MemoryMarkdownStore, llm_client=None):
        """
        初始化巩固器。
        
        Args:
            store: MemoryMarkdownStore 实例
            llm_client: LLM 客户端（用于 reflect 提炼，可选）
        """
        self.store = store
        self.llm = llm_client
    
    async def reflect(self, session_id: str) -> dict:
        """
        会话末反思：episodic → semantic 提炼。
        
        读取会话历史（从 run 日志或 session memories），
        LLM 提炼关键结论，与现有记忆比对：
        - 新结论 → 创建记忆
        - 冲突 → 标记 supersedes
        - 重复 → 跳过
        
        Args:
            session_id: 会话 ID
            
        Returns:
            {
                "status": "success",
                "session_id": str,
                "extracted": int,  # 提炼出的新结论数
                "updated": int,    # 更新的旧记忆数
                "conflicts": []    # 冲突列表
            }
        """
        # TODO: Phase 3 实现
        # 1. 查询 session_id 对应的 episodic 记忆
        # 2. 调用 LLM 提炼 semantic 结论（使用 prompt: memory-reflect.st）
        # 3. 与现有记忆比对，决定 create/update/skip
        # 4. 记录操作日志
        
        return {
            "status": "not_implemented",
            "session_id": session_id,
            "extracted": 0,
            "updated": 0,
            "conflicts": []
        }
    
    async def lifecycle(self, dry_run: bool = False) -> dict:
        """
        生命周期管理：归档低价值记忆。
        
        扫描所有 active 记忆，计算衰减分数（importance * decay_factor），
        低于阈值 + 长期未访问 → archive/。
        
        归档条件：
        - importance <= 3 且 180 天未访问
        - importance <= 5 且 365 天未访问
        - superseded_by 非空（已被替代）
        
        Args:
            dry_run: 仅预览，不实际归档
            
        Returns:
            {
                "status": "success",
                "scanned": int,
                "archived": int,
                "reasons": {
                    "low_importance": int,
                    "superseded": int,
                    "expired": int
                },
                "dry_run": bool,
                "details": [{"id": str, "reason": str}]
            }
        """
        now = datetime.now(timezone.utc)
        results = {
            "status": "success",
            "scanned": 0,
            "archived": 0,
            "reasons": {
                "low_importance": 0,
                "superseded": 0,
                "expired": 0
            },
            "dry_run": dry_run,
            "details": []
        }
        
        # 扫描所有 active 记忆（排除 archive）
        for mem_type in ["core", "context", "procedures", "decisions", "sessions"]:
            type_dir = self.store.memory_root / mem_type
            if not type_dir.exists():
                continue
            
            for md_file in type_dir.rglob("*.md"):
                try:
                    # OPT-225：sessions 月桶整桶老化——3 个月前的候选桶整体归档
                    m = re.match(r"(\d{4}-\d{2})\.md$", md_file.name)
                    if m:
                        y, mo = (int(x) for x in m.group(1).split("-"))
                        age_months = (now.year - y) * 12 + (now.month - mo)
                        if age_months > 3:
                            archive_dir = self.store.memory_root / "archive" / str(now.year)
                            archive_dir.mkdir(parents=True, exist_ok=True)
                            md_file.rename(archive_dir / md_file.name)
                            results["archived"] += 1
                            results["reasons"]["expired"] += 1
                            results["details"].append(
                                {"id": md_file.name, "reason": "bucket_expired"})
                        continue

                    memory = self.store._parse_memory_file(md_file)
                    results["scanned"] += 1

                    # 跳过已归档
                    if memory.get("status") == "archived":
                        continue

                    mem_id = memory["id"]
                    importance = memory.get("importance", 5)
                    last_accessed = memory.get("last_accessed_at")
                    superseded_by = memory.get("superseded_by")
                    
                    # 条件1：已被替代
                    if superseded_by:
                        if not dry_run:
                            self.store.archive(mem_id, reason="superseded")
                        results["archived"] += 1
                        results["reasons"]["superseded"] += 1
                        results["details"].append({
                            "id": mem_id,
                            "reason": f"superseded by {superseded_by}"
                        })
                        continue
                    
                    # 条件2：低重要性 + 长期未访问
                    if last_accessed:
                        try:
                            last_dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                            days_since_access = (now - last_dt).days
                            
                            should_archive = False
                            reason = ""
                            
                            if importance <= 3 and days_since_access > 180:
                                should_archive = True
                                reason = f"low importance ({importance}) + {days_since_access}d inactive"
                                results["reasons"]["low_importance"] += 1
                            elif importance <= 5 and days_since_access > 365:
                                should_archive = True
                                reason = f"importance {importance} + {days_since_access}d inactive"
                                results["reasons"]["expired"] += 1
                            
                            if should_archive:
                                if not dry_run:
                                    self.store.archive(mem_id, reason="lifecycle")
                                results["archived"] += 1
                                results["details"].append({
                                    "id": mem_id,
                                    "reason": reason
                                })
                        except Exception:
                            pass
                
                except Exception as e:
                    # 静默跳过解析失败的文件
                    continue
        
        return results
    
    async def defrag(self, dry_run: bool = False) -> dict:
        """
        碎片整理：合并重复、拆分臃肿记忆。
        
        1. 语义相似度检测（简化：关键词重叠度）
        2. 合并重复记忆（保留高 importance，supersedes 旧记录）
        3. 拆分超长记忆（>2000 字符）
        
        Args:
            dry_run: 仅预览，不实际修改
            
        Returns:
            {
                "status": "success",
                "scanned": int,
                "merged": int,
                "split": int,
                "dry_run": bool,
                "details": []
            }
        """
        results = {
            "status": "success",
            "scanned": 0,
            "merged": 0,
            "split": 0,
            "dry_run": dry_run,
            "details": []
        }
        
        # 收集所有 active 记忆
        memories = []
        for mem_type in ["core", "context", "procedures", "decisions", "sessions"]:
            type_dir = self.store.memory_root / mem_type
            if not type_dir.exists():
                continue
            
            for md_file in type_dir.rglob("*.md"):
                try:
                    memory = self.store._parse_memory_file(md_file)
                    if memory.get("status") != "archived":
                        memories.append(memory)
                        results["scanned"] += 1
                except Exception:
                    continue
        
        # 任务1：检测并合并重复
        seen = set()
        for i, mem in enumerate(memories):
            if mem["id"] in seen:
                continue
            
            content_i = mem["content"].lower()
            tokens_i = set(re.findall(r'\w+', content_i))
            
            for j in range(i + 1, len(memories)):
                mem_j = memories[j]
                if mem_j["id"] in seen:
                    continue
                
                content_j = mem_j["content"].lower()
                tokens_j = set(re.findall(r'\w+', content_j))
                
                # 计算 Jaccard 相似度
                if not tokens_i or not tokens_j:
                    continue
                
                overlap = len(tokens_i & tokens_j)
                union = len(tokens_i | tokens_j)
                similarity = overlap / union if union > 0 else 0
                
                # 高度相似（>0.7）→ 合并
                if similarity > 0.7:
                    # 保留高 importance 的
                    keep_id = mem["id"] if mem.get("importance", 5) >= mem_j.get("importance", 5) else mem_j["id"]
                    discard_id = mem_j["id"] if keep_id == mem["id"] else mem["id"]
                    
                    if not dry_run:
                        # 标记被替代
                        self.store.update(keep_id, supersedes=[discard_id])
                        self.store.archive(discard_id, reason="merged")
                    
                    results["merged"] += 1
                    results["details"].append({
                        "action": "merge",
                        "kept": keep_id,
                        "discarded": discard_id,
                        "similarity": f"{similarity:.2f}"
                    })
                    
                    seen.add(discard_id)
                    break
        
        # 任务2：拆分超长记忆
        MAX_LENGTH = 2000
        for mem in memories:
            if mem["id"] in seen:
                continue
            
            content = mem["content"]
            if len(content) > MAX_LENGTH:
                # 简化拆分：按段落
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                
                if len(paragraphs) > 1:
                    # 保留第一段，其余作为新记忆
                    if not dry_run:
                        # 更新原记忆为第一段
                        self.store.update(mem["id"], content=paragraphs[0])
                        
                        # 创建后续段落记忆（B8：importance 下限 1，原文为 1 时不再降）
                        new_importance = max(1, mem.get("importance", 5) - 1)
                        for para in paragraphs[1:]:
                            self.store.commit(
                                content=para,
                                tags=mem.get("tags", []),
                                mem_type=mem["type"],
                                project_id=mem.get("project_id", "default"),
                                importance=new_importance,
                                source_session=f"split-from-{mem['id']}"
                            )
                    
                    results["split"] += 1
                    results["details"].append({
                        "action": "split",
                        "original_id": mem["id"],
                        "original_length": len(content),
                        "parts": len(paragraphs)
                    })
        
        return results


def create_consolidator(vault_root: str, llm_client=None) -> MemoryConsolidator:
    """
    工厂函数：创建巩固器实例。
    
    Args:
        vault_root: Vault 根目录
        llm_client: LLM 客户端（可选）
        
    Returns:
        MemoryConsolidator 实例
    """
    store = MemoryMarkdownStore(vault_root)
    return MemoryConsolidator(store, llm_client)
