# Configuration

## 配置文件总览

| 文件 | 用途 | 提交策略 |
|---|---|---|
| `agentlab/config/config.example.json` | Agent Hub、LLM、记忆、RAG、serve | 只提交 example |
| `obsidian_agent_brain/config.example.json` | MCP/Vault 路径、队列和外部命令 | 只提交 example |
| `bili_summarizer/config/visual_models.json.example` | 文本/视觉模型 endpoint、模型名和参数 | 只提交 example |
| `inbox_collector/config.example.json` | 邮件采集白名单和队列路径 | 只提交 example |
| `mcp.config.example.json` | MCP 客户端 stdio 启动模板 | 只提交 example |

复制 example 后生成的实际文件名为 `config.json` 或 `visual_models.json`，这些文件禁止进入 Git。

## Agent Hub 配置

`agentlab/config/config.json` 的关键字段：

```jsonc
{
  "llm": {"base_url": "https://provider.example/v1", "api_key": "", "model": "model-name"},
  "vault_root": "C:/path/to/vault",
  "serve": {"host": "127.0.0.1", "port": 8643, "token": "change-me"},
  "rag": {"vector_enabled": false, "embed_base_url": "", "embed_api_key": ""}
}
```

建议先以 `vector_enabled: false` 或 `vector_mode: shadow` 启动，确认关键词检索正常后再开启向量 provider。

重要分组：

- `llm`：主模型 endpoint、模型名、超时和 token
- `vault_root`：Vault 根目录
- `brain.config_path`：MCP Gateway 配置路径
- `memory`：Markdown 记忆目录、置信度、候选和审计策略
- `limits`：步数、上下文预算、工具调用上限和超时
- `serve`：HTTP host、端口、Bearer token、审批等待时间
- `context`：上下文装配模式、预算和 task state 路径
- `rag`：关键词/向量/融合模式、embedding provider 和答案门禁

## MCP 配置

`obsidian_agent_brain/config.json` 至少需要 `vault_path`、`project_root`，以及队列、Cookie、视觉模型和 task state 的本地路径。`mcp.config.example.json` 只演示 stdio 命令和 `OBSIDIAN_VAULT_PATH` 环境变量。

## 内容处理配置

`bili_summarizer/config/visual_models.json` 使用 `default`、`visual`、`grid`、`screenshot` 分组：

- `default`：文章/文本总结模型
- `visual`：单张网格图视觉分析模型
- `grid`：截图网格的行列、尺寸、间隔和数量
- `screenshot`：单帧数量、宽度和质量

Cookie 文件只保存到 `bili_summarizer/bilibili_cookie.json`，不要将 Cookie 粘贴到 prompt、Issue 或日志。

## 环境变量

| 变量 | 用途 |
|---|---|
| `AGENT_LLM_API_KEY` | 覆盖 Agent Hub LLM Key |
| `AGENT_REMOTE_OPERATION_TOKEN` | 远程操作状态查询 token |
| `OBSIDIAN_VAULT_PATH` | Ark 部署目标和 MCP/Vault 默认路径 |
| `AGENTLAB_APPROVAL_MODE` | serve 启动时覆盖审批模式 |
| `AGENT_RAG_*` | Ark 启动 serve 时覆盖 RAG provider 和模式 |
| `PYTHONUTF8` / `PYTHONIOENCODING` | Windows 子进程统一 UTF-8 输出 |
| `HF_ENDPOINT` | Whisper 模型下载镜像，可选 |
| `BILI_WHISPER_MODEL` | 覆盖 Whisper 模型名，可选 |

## 配置安全规则

1. API Key、Bearer token、Cookie 只进本地配置或环境变量。
2. 配置示例使用空值、占位域名或 `sk-xxx`，不可填入真实凭据。
3. Vault、队列、日志和索引路径可不同于仓库目录；公开文档只使用占位路径。
4. 服务默认绑定 loopback；如需局域网访问，必须同时评估鉴权、CORS 和 Vault 写入风险。
