"""记忆触发 + 去重上下文注入（对齐 open-note `memory_extractor.dart` 思路，docs/07 §4.3）。

open-note 的做法：13 个触发词 + 每 N 轮批量触发 + 基于已存记忆做去重上下文注入，
避免重复沉淀、省 token。此处落地为纯函数，不侵入 loop 核心——由外层（工具/CLI）
按需调用；周期触发通过 `period()` 在每 N 轮判定。
"""
from __future__ import annotations

# 触发词集（"该沉淀到记忆"的信号；可扩展为配置项）
TRIGGERS: tuple[str, ...] = (
    "值得记", "要记住", "记住", "沉淀", "口诀", "方法论", "经验",
    "结论是", "复盘", "重点", "关键", "我学到", "补充到记忆", "下次记得",
    "规律", "教训", "建议是", "核心观点", "checkpoint", "milestone",
)

# 捕获写入治理（OPT-090 / 学 tau Tier-1）：沉积单条记忆的身体预算 + 保留尾部
CAPTURE_ITEM_CHARS = 2000
_CAPTURE_TAIL = 240


def govern_capture(content: str, per_item_chars: int = CAPTURE_ITEM_CHARS,
                   tail: int = _CAPTURE_TAIL) -> str:
    """捕获写入治理（OPT-090 / 学 tau Tier-1）。

    沉积到长期记忆的单条正文超预算 → head+tail 截断并留 `…[truncated N chars]…`
    标记；零 LLM、纯函数。身份锚（source/tags）由调用方携带，这里只治理正文本体。
    """
    if not content:
        return content
    if per_item_chars > 0 and len(content) > per_item_chars:
        return (content[:per_item_chars - tail]
                + f"\n…[truncated {len(content) - per_item_chars} chars]…\n"
                + content[-tail:])
    return content


def should_capture(text: str) -> bool:
    """命中任一触发词 → 建议触发记忆沉淀。"""
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in TRIGGERS)


def period(round_index: int, every: int = 5) -> bool:
    """每 N 轮触发一次批量提取（round_index 从 1 起）。"""
    return round_index >= 1 and round_index % max(1, every) == 0


def dedup_context(existing: list, limit: int = 10) -> str:
    """把已存记忆前 limit 条拼成注入块，供提取前自查避免重复沉淀。

    existing: list[dict]（含 content 字段）或带 .content 的对象；
    空则返回空串，调用方可跳过注入。
    """
    items: list[str] = []
    for it in existing or []:
        content = (it.get("content") if isinstance(it, dict) else getattr(it, "content", "")) or ""
        if content.strip():
            items.append("- " + content.strip())
        if len(items) >= limit:
            break
    if not items:
        return ""
    return "[已有记忆，提取前自查避免重复沉淀]\n" + "\n".join(items)


def extract_snippets(entries: list, limit: int = 8) -> list[str]:
    """从回合消息中抽取命中触发词的可沉淀片段（轻量提取，供 loop 周期调度）。

    entries: Message 列表（含 .role/.content）；只考虑 user/assistant 的文本，
    空白片段丢弃、按原文去重保序，最多返回 limit 条。纯函数、可独立测试；
    是否需要语义级 summarizer 提取由上层决定，这里只做触发词金标准。
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in entries or []:
        role = getattr(m, "role", "")
        content = (m.content if hasattr(m, "content") else "") or ""
        if role not in ("user", "assistant") or not content.strip():
            continue
        if not should_capture(content):
            continue
        seg = " ".join(content.split())
        if seg in seen:
            continue
        seen.add(seg)
        out.append(seg)
        if len(out) >= limit:
            break
    return out