# Modules

| 路径 | 类型 | 职责 | 主要入口 |
|---|---|---|---|
| `ark/` | TypeScript | Obsidian 工作台、设置、任务、搜索、回顾、SSE UI | `ark/src/main.ts` |
| `agentlab/agentlab/core/` | Python | Agent Loop、上下文、消息、路由、预算 | `agentlab.runtime.cli` |
| `agentlab/agentlab/runtime/` | Python | CLI、HTTP/SSE、鉴权、审批、幂等、任务状态 | `agentlab.runtime.serve` |
| `agentlab/agentlab/tools/` | Python | 工具注册、权限契约、MCP connector、RAG 工具 | `agentlab.tools.registry` |
| `agentlab/agentlab/memory/` | Python | 工作记忆、Markdown 记忆、提取、生命周期 | `agentlab.memory.store` |
| `agentlab/agentlab/rag/` | Python | Markdown 切分、索引、关键词/向量召回和引用 | `agentlab.rag.recall` |
| `obsidian_agent_brain/` | Python | FastMCP Server 和 Vault/内容工具薄封装 | `mcp_server.py` |
| `bili_summarizer/` | Python | B 站字幕/转写/视觉和文章总结 | `bili_transcript.py` |
| `inbox_collector/` | Python | Agent Mail 拉取、URL 分类和 SQLite 队列 | `inbox_poll.py` |
| `pipelines/` | Python | 公共边界 facade，避免消费者绑定实现目录 | `pipelines.content`, `pipelines.intake` |
| `services/agent_hub/` | Python | Agent Hub 兼容入口，转发到 `agentlab` | `services.agent_hub` |
| `templates/` | Markdown | Vault 笔记模板 | 按内容类型选择 |
| `scripts/` | PowerShell/Python | 全量测试、环境诊断和契约检查 | `test-all.ps1`, `doctor.ps1` |

## 工具组

MCP 核心工具按以下组组织：

- `vault_*`：读取、搜索、图谱、健康检查和受控写入
- `brain_*`：brain 索引查询和重建
- `memory_*`：记忆提交、召回、纠正、撤销、恢复和复核
- `bili_*`：视频元信息、转写、截图和视觉分析
- `article_*`：文章抓取和总结
- `inbox_*`：收件箱采集和队列读取

工具的实际数量以运行时 `tools/list` 为准；不要在客户端硬编码总数。

## 兼容入口

- `services.agent_hub` -> `agentlab.runtime`
- `pipelines.content` -> `bili_summarizer`
- `pipelines.intake` -> `inbox_collector`
- `packages.contracts` -> Agent/HTTP 公共数据契约
