# Public Sync Guide

公开仓库不是私有仓库的完整镜像，而是由私有仓库生成的公开投影。同步入口是私有仓库中的 `scripts/sync-public.ps1`。

## 推荐流程

```powershell
Set-Location <private-repo-root>

# 只查看将要同步的目录，不改文件
powershell -ExecutionPolicy Bypass -File scripts/sync-public.ps1

# 应用复制和公开化替换
powershell -ExecutionPolicy Bypass -File scripts/sync-public.ps1 -Apply

# 检查公开工作区，再运行测试
Set-Location E:\vibe_coding\obsidian_ark_public
powershell -ExecutionPolicy Bypass -File scripts/test-all.ps1 -SkipArk
Set-Location ark
npm run build
Set-Location ..

# 确认 diff 后提交并推送（此时不再重复复制）
Set-Location <private-repo-root>
powershell -ExecutionPolicy Bypass -File scripts/sync-public.ps1 -Commit -Push
```

`-Apply` 开始前默认要求公开工作区干净；如果公开目录中有人工修改，脚本会停止，避免覆盖。Apply 完成后先检查 diff，再用 `-Commit -Push` 提交当前已审核的公开投影。只有确认 Apply 可以覆盖现有改动时，才使用 `-AllowDirty`。

如果私有源目录删除了文件，普通 Apply 不会删除公开目录中的旧文件。确认公开目录没有需要保留的手工文件后，可显式使用 `-Apply -Prune`，按同步边界清理源目录已不存在的内容。

## 同步边界

会同步：`agentlab`、`ark`、`bili_summarizer`、`obsidian_agent_brain`、`inbox_collector`、公共 facade、模板、测试、诊断脚本和 `docs_public/`。

不会同步：私有 `docs/`、Vault、`.agent-brain/`、配置密钥、Cookie、队列/数据库、日志、UI harness、原型、构建产物和 `scripts/sync-public.ps1` 本身。

脚本会把个人 Vault 路径、Hermes Desktop 路径和 Agentlab 私有工作目录转换为公开占位值或自动探测逻辑。新增机器专属路径时，应同时在脚本的 `$replacements` 中增加规则。

## 提交前检查

```powershell
git -C E:\vibe_coding\obsidian_ark_public status --short
git -C E:\vibe_coding\obsidian_ark_public diff --check
git -C E:\vibe_coding\obsidian_ark_public diff --cached --name-only
```

暂存区不应出现 `config.json`、`bilibili_cookie.json`、`visual_models.json`、`.db`、`.log`、`main.js` 或 `node_modules`。

## 发布状态

`-Commit` 只创建本地公开分支提交；`-Push` 必须和 `-Commit` 一起使用，才会推送到 `public/main`。网络不稳定时脚本使用项目约定的 `127.0.0.1:7897` 代理。推送后用 `git ls-remote public refs/heads/main` 核对远端提交号。
