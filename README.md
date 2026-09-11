# Ark

面向 Obsidian 的 AI 知识工作台：将视频与文章加工为结构化 Markdown，连接 Agent、MCP 工具、任务协作与知识检索。

## 项目组成

- `ark/`：Obsidian 插件，提供 AI 工作台、任务管理、工作记录、知识库、捕获、搜索和内容产出。
- `agentlab/`：轻量 Python Agent 运行时，包含 Agent Loop、工具调度、会话、记忆、RAG、HTTP/SSE 服务和审批流程。
- `obsidian_agent_brain/`：MCP 能力层，负责 Vault 读写、搜索、内容处理和 B 站/文章工具。
- `bili_summarizer/`：B 站视频与文章处理模块，支持字幕获取、Whisper 转写和可选视觉分析。
- `inbox_collector/`：从 Agent Mail 收集链接并写入 SQLite 队列。

## 快速开始

要求：Python 3.10+、Node.js、Obsidian；视频转写和多模态功能还需要 ffmpeg。

### Agent Hub

```powershell
cd agentlab
python -m pip install -e ".[dev]"
Copy-Item config/config.example.json config/config.json
# 编辑 config/config.json：至少填写 vault_root；需要模型调用时填写 API Key
python -m unittest discover -s tests -v
python -m agentlab.runtime.cli tools list
```

### MCP 能力层

```powershell
cd obsidian_agent_brain
python -m pip install -r requirements.txt
Copy-Item config.example.json config.json
# 编辑 config.json：填写 vault_path 和 project_root
python -m unittest test_mcp.py -v
```

### Ark 插件

```powershell
cd ark
npm install
npm run build
$env:OBSIDIAN_VAULT_PATH = "C:\path\to\your\vault"
node scripts/deploy.mjs
```

也可以直接把 `ark/main.js`、`ark/manifest.json` 和 `ark/styles.css` 复制到：

```text
<your-vault>/.obsidian/plugins/ark/
```

### 内容处理

```powershell
python bili_summarizer/bili_transcript.py <BV号> --vault "C:\path\to\your\vault"
python bili_summarizer/article_summarizer.py <URL> --vault "C:\path\to\your\vault"
```

B 站字幕按 `B站 API → yt-dlp → Whisper` 三级策略获取。API Key、Cookie 和本地配置只应保存在被 `.gitignore` 排除的实际配置文件中。

## 测试

```powershell
python -m unittest
cd ark
npm run build
```

## 设计原则

- Markdown 是知识内容的主数据源，插件缓存只做索引。
- Agent 能力通过工具注册表和 MCP 注入，核心循环与具体业务解耦。
- 写入操作经过路径守卫、审批和结构化校验。
- Prompt 使用独立模板文件，模型配置由示例配置驱动。

## License

MIT License，详见 [LICENSE](LICENSE)。
