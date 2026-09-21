# Troubleshooting

## 先做三步诊断

```powershell
powershell -ExecutionPolicy Bypass -File scripts/doctor.ps1
powershell -ExecutionPolicy Bypass -File scripts/test-all.ps1 -SkipArk
Set-Location ark; npm run build
```

确认当前工作目录、Python 解释器和配置文件路径，再判断业务错误。不要先删除队列数据库或 Vault 数据。

## Agent Hub 无法启动

### `token 未配置（fail-closed）`

在 `agentlab/config/config.json` 设置非空的 `serve.token`，或检查 Ark 是否将 token 保存到了插件设置。不要为了启动而关闭鉴权。

### 端口 8643 被占用

```powershell
Set-Location agentlab
..\.venv\Scripts\python.exe -m agentlab.runtime.serve_manage status
..\.venv\Scripts\python.exe -m agentlab.runtime.serve_manage stop
..\.venv\Scripts\python.exe -m agentlab.runtime.serve_manage start
```

如果 status 显示端口有进程但 pid 无法确认，先人工核对进程，不要强制杀掉未知服务。

### Ark 显示 Agent 离线

1. 访问 `http://127.0.0.1:8643/health`。
2. 核对 Ark 的 `agentlabUrl` 是否以 `/v1` 结尾。
3. 核对 `agentlabWorkdir` 指向含 `config/config.json` 的目录。
4. 核对 `agentlabExePath` 是否使用了正确的解释器。
5. 查看 `logs/serve.log`，其中可能包含配置或 provider 错误。

## MCP 工具不可用

- 确认 MCP 客户端使用 `mcp.config.example.json` 改写后的本地配置。
- 确认 `project_root` 指向仓库根目录，`OBSIDIAN_VAULT_PATH` 指向真实 Vault。
- 直接运行 `python obsidian_agent_brain/mcp_server.py`，检查依赖导入错误。
- 脚本方式运行时需要保留 `obsidian_agent_brain` 的 flat-import 目录结构。

## B站没有字幕

按状态顺序检查：

1. `[PARSING]`：BV 号或 URL 是否正确。
2. `[SUBTITLE]`：Cookie、字幕 API 和字幕语言是否可用。
3. `[TRANSCRIBING]`：yt-dlp、ffmpeg、Whisper 模型和磁盘空间是否可用。
4. `[DONE]`：输出路径是否在 Vault 允许写入范围内。

不要绕过三级降级直接判定失败。没有登录态时先确认视频是否允许公开字幕；需要 Cookie 时重新生成本地 `bilibili_cookie.json`。

## 文章总结失败

- 先用 `article_fetch` 或直接 requests 验证网页可访问。
- 检查 `visual_models.json` 中 `default.api_base`、`model` 和 `api_key`。
- 检查 prompt 文件是否存在：`bili_summarizer/prompts/article-summary-user.st`。
- JSON 解析失败时查看 trace；处理器应返回结构化降级结果，不要把原始 Key 写入日志。

## RAG 结果为空或不稳定

- 先将 `rag.vector_enabled` 关闭或保持 `vector_mode: shadow`，验证关键词召回。
- 检查 Vault Markdown 是否在允许的前缀目录中。
- 检查 `.agent-brain/` 和派生索引是否有写权限；索引损坏时优先使用重建命令，而不是手工编辑数据库。
- 向量 provider 超时、Key 缺失或 embedding 数量不一致时，应降级为关键词结果。

## 测试导入错误

测试必须从对应目录或仓库根目录的正确入口运行：

```powershell
Set-Location agentlab
..\.venv\Scripts\python.exe -m unittest discover -s tests

Set-Location ..
$env:PYTHONPATH = (Get-Location).Path
..\.venv\Scripts\python.exe -m unittest obsidian_agent_brain.test_mcp
```

不要用 `python -m unittest discover -s agentlab/tests` 代替 agentlab 的目录入口；那会让 `agentlab` 包解析到错误层级。

## 安全事故处理

如果 API Key、Cookie 或 Bearer token 曾被写入 Git、日志或聊天记录：立即撤销并重发凭据，随后从本地配置和日志中移除；仅修改 `.gitignore` 不能让已发布的秘密失效。
