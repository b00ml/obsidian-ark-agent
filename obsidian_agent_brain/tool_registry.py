"""
统一工具注册表：20 个核心 brain 工具的单一真相源

结构：BrainToolSpec(NamedTuple)——保持按序解包兼容旧消费方
（mcp_server._register_brain_tools / agentlab connectors.brain_tools）。
- module_suffix: "vault" → tools_vault.py 的 vault_{tool_name}()
- permission: "read" | "write"
- timeout_seconds: None（无限制）或浮点数（秒）
- side_effects: none|cache|index|write|external（写入类必须显式声明，不得默认）
- idempotent: 同参数重复调用后系统状态是否与单次一致（写入类必须显式声明）
  判定口径：状态收敛即 True（如 vault_write 同内容重写）；产生新增条目/新渲染产物即 False
- requires_approval: 预留（当前无消费点，不接入审批链）

运行时另有显式注册工具（obsidian_* 5 个 + bili 异步任务 3 个），
tools/list 实测总数以 tools/list 为准，不在此硬编码。
"""
from __future__ import annotations

from typing import NamedTuple


class BrainToolSpec(NamedTuple):
    name: str
    module_suffix: str
    description: str
    permission: str  # "read" | "write"
    timeout: float | None
    side_effects: str = ""  # ""=未声明（write 工具不允许，validate_specs 拦截）
    idempotent: bool | None = None  # None=未声明（write 工具不允许）
    requires_approval: bool = False


BRAIN_TOOL_SPECS: list[BrainToolSpec] = [
    # ==== 组名: vault (7 工具) ====
    BrainToolSpec("vault_read", "vault", "读取 Vault 内 Markdown 笔记全文（path 为相对路径）", "read", None,
                  side_effects="none", idempotent=True),
    BrainToolSpec("vault_search", "vault", "关键词检索全库笔记（标题+正文+tag），返回路径/摘要/tags", "read", None,
                  side_effects="none", idempotent=True),
    BrainToolSpec("vault_graph", "vault", "wikilink 邻接图：从指定笔记遍历 [[wikilink]] 关联", "read", None,
                  side_effects="none", idempotent=True),
    BrainToolSpec("vault_scan", "vault", "全库统计：主题分布/tag 聚合/wikilink 密度/最近笔记", "read", None,
                  side_effects="none", idempotent=True),
    BrainToolSpec("vault_health", "vault", "结构性健康检查：死链（wikilink 指向不存在笔记）+ 孤儿笔记/无引用笔记", "read", None,
                  side_effects="none", idempotent=True),
    BrainToolSpec("vault_write", "vault", "写入/覆盖 Vault 笔记；raw/ 只读拒绝；传 expected_revision 可做 CAS 冲突检测", "write", None,
                  side_effects="write", idempotent=True),
    BrainToolSpec("vault_patch", "vault", "精确替换笔记中某段文本（old 须唯一）；传 expected_revision 可做 CAS 冲突检测", "write", None,
                  side_effects="write", idempotent=True),

    # ==== 组名: brain (3 工具) ====
    BrainToolSpec("brain_search", "brain", "检索 brain 内部数据（索引）", "read", None,
                  side_effects="cache", idempotent=True),
    BrainToolSpec("brain_scan", "brain", "brain 数据扫描/统计", "read", None,
                  side_effects="cache", idempotent=True),
    BrainToolSpec("brain_reindex", "brain", "重建 brain 索引（预留）——非纯 read：会重写索引存储", "write", None,
                  side_effects="index", idempotent=True),

    # ==== 组名: memory (2 工具) ====
    BrainToolSpec("memory_commit", "memory", "沉淀一条可复用记忆（content+tags）到长期记忆", "write", None,
                  side_effects="write", idempotent=False),
    BrainToolSpec("memory_query", "memory", "按 topic 召回记忆（tag+内容关键词）", "read", None,
                  side_effects="none", idempotent=True),
    BrainToolSpec("memory_conflicts", "memory", "列出同 scope/type/subject 的未解决记忆冲突，供人工处理", "read", None,
                  side_effects="none", idempotent=True),
    BrainToolSpec("memory_correct", "memory", "纠正一条长期记忆：创建新版本并立即隐藏旧版本", "write", None,
                  side_effects="write", idempotent=False),
    BrainToolSpec("memory_revoke", "memory", "撤销一条长期记忆（保留最小审计，不再参与默认召回）", "write", None,
                  side_effects="write", idempotent=True),
    BrainToolSpec("memory_delete", "memory", "按用户明确请求删除长期记忆（默认物理删除）", "write", None,
                  side_effects="write", idempotent=True),
    BrainToolSpec("memory_restore", "memory", "恢复未被新版本替代的归档/撤销记忆", "write", None,
                  side_effects="write", idempotent=True),
    BrainToolSpec("memory_review", "memory", "按内容哈希确认或延期待复核记忆（必须提供 reviewer/reason）", "write", None,
                  side_effects="write", idempotent=True),

    # ==== 组名: bili (4 工具) ====
    BrainToolSpec("bili_meta", "bili", "B站视频元信息", "read", 60.0,
                  side_effects="external", idempotent=True),
    BrainToolSpec("bili_transcribe", "bili", "B站视频转写（3 层降级）；返回文本不落库", "read", 1800.0,
                  side_effects="cache", idempotent=True),
    BrainToolSpec("bili_screenshot", "bili", "B站视频截图/网格图（brain 内部渲染并写 Inbox raw/screenshots）。仅在用户明确提出视觉/画面/截图分析时使用——默认工具面隐藏，生成笔记走 bili_transcribe 轻量路径", "write", 600.0,
                  side_effects="write", idempotent=False),
    BrainToolSpec("bili_visual", "bili", "B站视频视觉分析（brain 内部转写+视觉+渲染+写 Inbox，实测约 4 分钟）。仅在用户明确提出视觉/画面分析时使用——默认工具面隐藏，生成笔记走 bili_transcribe 轻量路径", "write", 1800.0,
                  side_effects="write", idempotent=False),

    # ==== 组名: article (2 工具) ====
    BrainToolSpec("article_fetch", "article", "抓取公众号文章正文（readability）", "read", 60.0,
                  side_effects="external", idempotent=True),
    BrainToolSpec("article_summarize", "article", "公众号文章总结（brain 内部渲染并写 Inbox）", "write", 300.0,
                  side_effects="write", idempotent=False),

    # ==== 组名: inbox (2 工具) ====
    BrainToolSpec("inbox_collect", "inbox", "从收件箱拉取新链接并入队——写 SQLite 队列（同 URL 去重兜底，状态收敛）", "write", 60.0,
                  side_effects="write", idempotent=True),
    BrainToolSpec("inbox_read_queue", "inbox", "读取待处理任务队列", "read", None,
                  side_effects="none", idempotent=True),
]


def validate_specs(specs: list[BrainToolSpec] | None = None) -> list[str]:
    """契约校验：返回违规清单（空=通过）。

    规则：permission=write 的工具必须显式声明 side_effects（非 ""）和 idempotent
    （非 None）；side_effects 取值必须在受控枚举内。防止新增工具静默吃默认权限。
    """
    allowed = {"none", "cache", "index", "write", "external"}
    errors: list[str] = []
    for s in (specs if specs is not None else BRAIN_TOOL_SPECS):
        if s.permission == "write":
            if not s.side_effects:
                errors.append(f"{s.name}: write 工具未声明 side_effects")
            if s.idempotent is None:
                errors.append(f"{s.name}: write 工具未声明 idempotent")
        if s.side_effects and s.side_effects not in allowed:
            errors.append(f"{s.name}: side_effects={s.side_effects} 不在 {sorted(allowed)}")
    return errors


def spec_contract_line(s: BrainToolSpec) -> str:
    """生成附到工具 description 末尾的契约摘要，供 tools/list 程序化读取。"""
    idem = "unknown" if s.idempotent is None else ("yes" if s.idempotent else "no")
    return f"[contract] permission={s.permission} side_effects={s.side_effects or 'unknown'} idempotent={idem}"
