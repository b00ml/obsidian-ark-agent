# Architecture

## 分层

```text
Ark Obsidian Plugin (TypeScript)
        |
        | HTTP/SSE: /health, /v1/responses
        v
Agent Hub: agentlab/
        |
        | tool connector / MCP stdio
        v
Vault Gateway: obsidian_agent_brain/
        |
        +--> bili_summarizer/      视频与文章处理
        +--> inbox_collector/      邮件链接采集与队列
        +--> Obsidian Vault        Markdown、模板、记忆与产物
```

## 组件边界

### Ark 插件

`ark/` 是用户界面和交互层，负责工作台、任务、卡片、搜索、回顾、设置和 SSE 流渲染。它不直接实现 Agent Loop，也不保存 API Key 到源码；Agent 后端地址、Token 和本地路径来自插件设置。

### Agent Hub

`agentlab/` 是本地 Agent 运行时：

- `core/`：消息、上下文、循环、路由、预算和 resilience
- `tools/`：工具注册、权限契约和业务连接器
- `memory/`：工作记忆、Markdown 长期记忆、会话范围和生命周期
- `rag/`：结构化切分、关键词/向量召回、融合、引用和答案门禁
- `runtime/`：CLI、HTTP/SSE、鉴权、审批、幂等和任务状态

Agent Hub 只通过工具访问业务能力；工具失败应返回可解释的降级结果，不让 UI 耦合 Vault 文件实现。

### Vault Gateway / MCP

`obsidian_agent_brain/` 使用 FastMCP/stdio 暴露 Vault、memory、B 站、文章、收件箱和 brain 工具。`tool_registry.py` 是核心工具契约的单一真相源，工具描述包含权限、side effects 和幂等信息。

### 内容处理

`bili_summarizer/` 负责 B 站字幕、Whisper 转写、截图/视觉分析、文章总结和知识编译。Prompt 放在 `prompts/*.st`，模型配置从本地 JSON 读取。

### 接收层

`inbox_collector/` 调用 Agent Mail CLI 拉取邮件，从正文提取 URL，按域名白名单分类，并通过 SQLite 队列进行任务和 seen 标记的事务提交。

## 主要数据流

### B 站视频

```text
BV 号/URL -> B站字幕 API -> yt-dlp 字幕 -> 音频 + Whisper -> 结构化 Markdown -> Vault
```

每一层失败都保留状态和缓存；只有用户明确需要视觉分析时才进入截图/视觉模型路径。

### Agent 对话

```text
Ark -> POST /v1/responses -> Bearer 校验 -> Agent Loop
    -> tool registry / approval gate -> MCP 或本地连接器
    -> SSE tool events + text deltas -> Ark 渲染并持久化会话摘要
```

### 记忆与 RAG

长期记忆以 Vault 中的 Markdown 为主数据源；RAG 索引、session range 和 task state 都是可重建派生物。撤销或纠正记忆时，必须同步处理派生索引。

## 外部合同

- HTTP：`GET /health`、`POST /v1/responses`
- 鉴权：服务配置的 Bearer token；token 缺失时服务启动 fail-closed
- SSE：`response.output_text.delta`、工具调用开始/结束和完成事件
- MCP：stdio，工具参数由 Python 函数签名和 `tool_registry.py` 生成
- Vault：相对路径写入，`raw/` 目录只读，写操作受路径守卫、审批和结构化校验约束
