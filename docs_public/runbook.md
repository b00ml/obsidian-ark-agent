# Runbook

以下命令默认在仓库根目录执行。Windows 使用 PowerShell；Linux/macOS 将 `.venv\Scripts\python.exe` 替换为 `.venv/bin/python`。

## 安装基础环境

要求：Python 3.10+、Node.js/npm。视频音频处理需要 ffmpeg；Whisper、视觉模型和 Agent Mail 属于可选集成。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".\agentlab[dev]"
.venv\Scripts\python.exe -m pip install -r bili_summarizer\requirements.txt
.venv\Scripts\python.exe -m pip install -r obsidian_agent_brain\requirements.txt
Set-Location ark
npm install
Set-Location ..
```

## 创建本地配置

```powershell
Copy-Item agentlab/config/config.example.json agentlab/config/config.json
Copy-Item obsidian_agent_brain/config.example.json obsidian_agent_brain/config.json
Copy-Item bili_summarizer/config/visual_models.json.example bili_summarizer/config/visual_models.json
Copy-Item inbox_collector/config.example.json inbox_collector/config.json
```

编辑本地配置时至少设置 Vault 根目录、项目根目录、LLM endpoint/model/key，以及 Agent Hub 的 `serve.token`。实际配置文件已被 `.gitignore` 排除。

## 环境诊断

```powershell
powershell -ExecutionPolicy Bypass -File scripts/doctor.ps1
powershell -ExecutionPolicy Bypass -File scripts/doctor.ps1 -Strict
```

诊断脚本会检查 Python、主要依赖、配置模板、Vault、端口 `8643` 和可选命令。`ffmpeg`、`agently-cli`、`obsidian` 缺失时普通模式会标记为可选跳过。

## 启动 Agent Hub

```powershell
Set-Location agentlab
..\.venv\Scripts\python.exe -m agentlab.runtime.cli tools list
..\.venv\Scripts\python.exe -m agentlab.runtime.serve_manage start
..\.venv\Scripts\python.exe -m agentlab.runtime.serve_manage status
Invoke-WebRequest http://127.0.0.1:8643/health
```

停止服务：

```powershell
..\.venv\Scripts\python.exe -m agentlab.runtime.serve_manage stop
```

`serve_manage` 使用 pidfile、端口反查和连树终止避免重复实例；没有 token 时启动会拒绝执行。

## 启动 MCP Server

MCP Server 通常由 MCP 客户端按 stdio 拉起：

```powershell
.venv\Scripts\python.exe obsidian_agent_brain\mcp_server.py
```

不要把该进程作为 HTTP 服务暴露；客户端应使用 `mcp.config.example.json`，将 Vault 路径和 Python 可执行文件替换为本机值。

## 构建与部署 Ark

```powershell
Set-Location ark
npm run check:css
npm run test:unit
npm run build
$env:OBSIDIAN_VAULT_PATH = "C:\path\to\your\vault"
npm run deploy
```

部署脚本只复制 `main.js`、`manifest.json` 和 `styles.css`。生产环境建议显式设置 `OBSIDIAN_VAULT_PATH`。

## 直接运行内容处理器

```powershell
.venv\Scripts\python.exe bili_summarizer\bili_transcript.py BV号 --vault "C:\path\to\vault"
.venv\Scripts\python.exe bili_summarizer\article_summarizer.py "https://mp.weixin.qq.com/s/..." --vault "C:\path\to\vault"
```

无字幕视频会自动进入 Whisper；视觉模式需要额外模型配置和 ffmpeg。处理器应保留 `[PARSING]`、`[SUBTITLE]`、`[TRANSCRIBING]`、`[DONE]` 等状态输出。

## 全量验证

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test-all.ps1
git status --short
git diff --cached --name-only
```
