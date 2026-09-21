# Dependencies

## 必需环境

| 依赖 | 作用 | 版本/来源 |
|---|---|---|
| Python | Agent Hub、MCP、内容处理和测试 | 3.10+ |
| Node.js/npm | Ark TypeScript 构建和单测 | 当前 LTS 建议 |
| Obsidian | 插件运行和 Vault UI | 用户本地安装 |

## Python 依赖

根目录 `pyproject.toml` 覆盖基础内容处理依赖；子项目有独立依赖文件：

- `agentlab/pyproject.toml`：`aiohttp`、`pydantic`、`httpx`，可选 `sqlite-vec`
- `obsidian_agent_brain/requirements.txt`：`mcp`、`pydantic`
- `bili_summarizer/requirements.txt`：`yt-dlp`、`faster-whisper`、`requests`、`Pillow`、`imageio-ffmpeg`、`bilibili-api-python`

建议所有 Python 命令使用同一个项目 `.venv`，避免 MCP、Agent Hub 和内容处理器加载不同解释器中的包。

## 系统命令

- `ffmpeg`：音频抽取、视频转码和截图；没有系统版本时可由 `imageio-ffmpeg` 提供静态二进制
- `npm`：Ark 安装、测试和生产构建
- `obsidian`：可选，Vault CLI 集成
- `agently-cli`：可选，Agent Mail 收件箱采集；需要单独安装并完成认证

## LLM/外部服务

- OpenAI 兼容文本模型：Agent Hub 和文章总结使用
- OpenAI 兼容视觉模型：视觉网格分析使用
- Embedding provider：RAG 向量模式可选
- B站公开 API、字幕接口和 `yt-dlp`：视频元数据与字幕
- 微信公众号公开网页：文章抓取
- Agent Mail CLI：收件箱采集

这些服务由网络、账号权限、限流和模型可用性决定；离线单测不会调用真实 LLM、B站或邮箱。

## GPU / Whisper

Whisper 可以 CPU 运行，但速度较慢。Windows AMD64 环境可按 requirements 安装 CUDA 运行库；GPU 不可用时应将 Whisper 设置为 CPU 或使用较小模型。首次运行可能需要下载模型文件，模型缓存不应提交到仓库。

## 依赖故障原则

- 缺少可选依赖时优先降级到关键词、纯文本或 API 字幕路径。
- 缺少必需依赖时，在启动或 doctor 阶段报出具体包/命令名。
- 不要在代码中硬编码 API Key、机器路径或模型服务私有地址。
