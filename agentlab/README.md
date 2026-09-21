# agentlab

自研轻量可复用 Python Agent 框架：**Agent = LLM + Tools + Loop**，内核克制、能力外置。

设计文档见 `docs/`（00 方案总览 → 06 测试评估与实施计划）。

## 设计哲学

- 循环内核纯函数化：只发事件、不碰 UI，可单测、可并发。
- 能力外置:工具/记忆/技能通过注册表与钩子注入，`core` 不感知具体业务。
- usage 真实 token 回写,压缩/预算判定基于事实。
- `length` 截断整批失败、`terminate` 主动终止、并行/串行调度、steering 双队列。

## 快速开始

```bash
cd agentlab
python -m pip install -e ".[dev]"

# 离线连通性自检（tools list）
python -m agentlab.runtime.cli tools list

# 需要真实 LLM：先复制 config/config.example.json 为 config/config.json 并填 Key
# 或用环境变量 AGENT_LLM_API_KEY
python -m agentlab.runtime.cli run "echo 与 add 分别做什么"

# 交互 REPL
python -m agentlab.runtime.cli repl
```

## 测试

```bash
python -m unittest discover -s tests -v
```

## 增量索引演练

在真实 Vault 上验证新增、修改、删除的增量索引闭环时，使用受限前缀和独立派生索引：

```bash
cd agentlab
python -m agentlab.rag_index_drill --vault C:/path/to/your/obsidian-vault --run-id opt-xxx \
  --out .tmp/roadmap/YYYYMMDD/opt-xxx/real-vault-drill.json
```

该命令不调用 embedding，临时 Markdown 仅创建在 `Inbox/.agentlab-drill-<run-id>/`，无论成功
或失败都会清理它和对应 SQLite 文件；它不会修改 `raw/` 或既有笔记。

## 记忆派生失效演练

在真实 Vault 上验证记忆撤销/纠正后 Markdown、RAG、session range 和 TaskState
派生物同步失效时，使用独立演练目录和唯一 run id：

```bash
cd agentlab
python -m agentlab.memory_lifecycle_drill --vault C:/path/to/your/obsidian-vault --run-id opt-xxx \
  --out .tmp/roadmap/YYYYMMDD/opt-xxx/memory-lifecycle.json
```

命令只创建带 run id 的临时记忆和 `.agent-brain/drills/` 派生物；成功或失败都会
清理记忆文件、审计日志、索引、session range 和 TaskState，不修改既有 Vault 笔记。

## 成本与预算

单轮成本 = **工具调用次数 × 每步上下文大小**。`limits.max_steps` 只管"批数"，
一次响应可以带多个 tool_call，所以它是控制不住成本的：全量基线实测有任务在 15 步内
跑了 46 次工具、373k token（`q-memory-recall`），live 会话"整理收件箱"45 次 / 596k token。

```jsonc
// config/config.json → limits
"max_tool_calls": 40,     // 单轮工具调用次数上界；0 = 不限。超过即 guardrail 收尾并说明缺什么
"tool_call_nudge": 0.6    // 用量占比达 60% 时先注入一次"收敛"提示（改用已有结果作答）
```

两个阈值的作用不同：`tool_call_nudge` 是**软**的（提醒模型别再逐条试探，把还需要的一次发齐），
`max_tool_calls` 是**硬**的（再调就收尾，并把"因未查完而无法确认"的部分讲清楚）。
真实任务确实需要更多调用时再调高——但先看看能不能靠**一次批量取回**解决。

## 目录

```
agentlab/           源码包（可 pip install -e .）
  core/             数据模型、LLM、路由、循环、上下文、guardrails
  tools/            注册表、装饰器、connectors（业务耦合隔离）
  memory/           工作记忆、技能注入
  rag/              Agentic RAG：多路召回 + 融合 + 充分性评估
  eval/             golden 集、Ragas 类指标、LLM-as-judge
  runtime/          cli、trace、config
  prompts/          .st 模板（对齐项目规范）
config/             配置（.example 占位，Key 不入库）
tests/              单元测试
```

## 记忆存储架构

自 OPT-145 起，agentlab 采用 **Markdown 文件作为长期记忆存储**，取代早期 SQLite 方案。记忆文件存放在 Obsidian vault 的 `ark/memory/` 目录，用户可在 Obsidian 中直接编辑。

### 目录结构

```
ark/memory/
  core/         核心事实（项目配置、用户偏好、常驻知识）
  context/      上下文信息（项目状态、临时关联）
  procedures/   操作流程（标准流程、工作流步骤）
  decisions/    决策记录（架构决策、方案选型）
  sessions/     会话记录（重要对话、问题解决过程）
  archive/      已归档记忆（低价值、已被替代、过期）
```

### Frontmatter 架构

每个记忆文件遵循统一的 YAML Frontmatter 格式：

```yaml
---
id: mem-<12位十六进制>          # 唯一标识
type: core|context|procedures|decisions|sessions  # 类型
project_id: default             # 项目隔离（防止跨项目污染）
importance: 1-10                # 重要度（用于生命周期管理）
confidence: 0.0-1.0             # 可信度（来源质量）
tags: [tag1, tag2]              # 标签（支持多维检索）
status: active|archived         # 状态
source_session: <路径>          # 来源会话（可追溯性）
created_at: ISO时间戳
updated_at: ISO时间戳
last_accessed_at: ISO时间戳     # 用于时间衰减
access_count: 整数              # 访问频次（衰减因子）
superseded_by: <mem-id>         # （可选）被哪条记忆替代
---
记忆内容主体（Markdown 格式）
```

### 生命周期管理

`agentlab.memory.consolidate.MemoryConsolidator` 提供三项巩固任务。当前调用方式是异步实例方法，项目没有内置定时调度器：

- **`lifecycle()`**：自动归档低价值记忆
  - 低重要度（1-3）且 180 天未访问 → 归档
  - 中等重要度（4-6）且 365 天未访问 → 归档
  - 已被 `superseded_by` 替代 → 立即归档
- **`defrag()`**：使用正文关键词的 Jaccard 重叠度（阈值 `>0.7`）合并重复内容，并按段落拆分超过 2000 字符的记忆。
- **`reflect()`**：接口已预留，但当前返回 `status: "not_implemented"`，不会调用 LLM 或写入新记忆。

示例：

```python
from agentlab.memory.consolidate import create_consolidator

consolidator = create_consolidator("C:/path/to/your/obsidian-vault")
result = await consolidator.lifecycle(dry_run=True)
```

### RAG 集成

brain 的 `vault_search` 工具会索引 `ark/memory/` 目录，`agentlab.rag.recall` 负责将它与 `memory_query` 等召回路编排起来：

```python
# Agent 内部调用示例
result = vault_search(query="项目中使用了哪些 LLM 模型？", top_k=5)
# 返回相关记忆文件的路径和摘要
```

`MemoryMarkdownStore.query()` 当前是文件扫描 + 关键词子串匹配，使用 CJK 二元组扩展查询词，再按命中数、重要度和 30 天半衰期排序；不是 BM25 或向量检索。

### 用户交互

- **直接编辑**：用户可在 Obsidian 中打开 `ark/memory/*.md` 文件进行编辑、删除、添加标签等操作，下次 Agent 查询时自动生效。
- **归档恢复**：归档文件在 `archive/` 目录，用户可手动移回对应类型目录以恢复。
- **禁止修改**：不建议编辑 `archive/` 中的文件，不要改动 Frontmatter 架构字段（仅 `tags/importance/content` 可安全编辑）。

### 迁移工具

提供 SQLite → Markdown 一次性迁移脚本（`agentlab/scripts/migrate_memory_to_markdown.py`）：

```bash
cd agentlab
python -m agentlab.scripts.migrate_memory_to_markdown \
  --vault-root C:/path/to/your/obsidian-vault --dry-run
python -m agentlab.scripts.migrate_memory_to_markdown \
  --vault-root C:/path/to/your/obsidian-vault --verify
```

脚本读取 `<vault>/.agent-brain/memory/sessions.sqlite`，按内容和标签推断类型，写入 `ark/memory/`，并在实际迁移后创建 `sessions.sqlite.pre-f5011` 备份。迁移期间 MCP 的 `memory_commit` / `memory_query` 签名保持兼容；OPT-223 后 Markdown 是唯一运行时读写源，Markdown 不可用/写入失败时显式报错（MEMORY_MARKDOWN_* 错误族），SQLite 仅保留迁移与人工恢复入口。
