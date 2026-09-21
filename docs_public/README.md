# Obsidian Ark Public Documentation

这组文档面向第一次接手公开仓库的开发者，描述当前代码真实提供的能力、安装方式、配置边界、运行流程和故障排查方法。

## 文档导航

- [architecture.md](architecture.md)：系统分层、组件边界和主要数据流
- [modules.md](modules.md)：目录与模块职责、调用关系
- [runbook.md](runbook.md)：安装、启动、停止、构建、部署和验证
- [configuration.md](configuration.md)：配置文件、环境变量、路径和密钥管理
- [dependencies.md](dependencies.md)：Python、Node.js、系统命令和可选能力
- [troubleshooting.md](troubleshooting.md)：诊断步骤与常见故障
- [contracts/](contracts/)：HTTP、SSE、MCP 和数据模型的生成契约 schema

## 项目定位

Obsidian Ark 将视频、文章和收件箱链接加工为结构化 Markdown 笔记：

```text
外部输入 -> 内容处理 -> MCP/Vault Gateway -> Agent Hub -> Ark 插件 -> Obsidian Vault
```

公开仓库包含可复用的 Agent 运行时、Obsidian 插件、B 站/文章处理器、MCP 能力层和收件箱采集器。真实 Vault、API Key、Cookie、运行日志、队列数据库和本地索引不属于发布内容。

## 最小验证路径

```powershell
# 环境检查
powershell -ExecutionPolicy Bypass -File scripts/doctor.ps1

# Python 与 MCP 测试
powershell -ExecutionPolicy Bypass -File scripts/test-all.ps1 -SkipArk

# Ark 插件构建
Set-Location ark
npm install
npm run build
```

真实运行前，先阅读 [configuration.md](configuration.md) 和 [runbook.md](runbook.md)。

## 当前事实

- Agent Hub 默认监听 `127.0.0.1:8643`，提供 `/health` 和 Hermes 兼容的 `/v1/responses` SSE 接口。
- MCP Server 通过 stdio 启动，工具实现位于 `obsidian_agent_brain/`。
- B 站字幕按 `B站 API -> yt-dlp -> Whisper` 三级策略降级。
- RAG 向量检索是可选能力；没有 embedding provider 时使用关键词检索或 shadow 模式。
- `ark/main.js` 是构建产物，不入库，由 `npm run build` 生成。

## 不包含在公开仓库中的内容

- 真实 Obsidian Vault 及其笔记
- `config.json`、API Key、Cookie、Agent Mail 凭据
- `.agent-brain/`、队列数据库、服务 pid/log、评测临时产物
- 私有项目治理文档、历史分析素材和机器专属路径
