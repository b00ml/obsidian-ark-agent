"""工作记忆（对齐 docs/04 §3.1 完整压缩；03 §4.1）。

- estimate/messages：随消息真实 usage 计数。
- compact()：容器内压缩，结构化增量摘要（full/update），保留近端 retained_tail，
  副作用（读写的文件/笔记）追踪写进摘要。prompt 统一走 prompts/*.st（AGENTS.md）。
- condense(messages)：纯函数压缩任意消息列表（供 loop 超预算接入），无 summarizer
  时退回"丢弃超预算段"兜底。
"""
from __future__ import annotations

import re
from typing import Sequence

from agentlab.core.context import estimate_tokens
from agentlab.core.message import Message
from agentlab.core.llm import LLMProvider
from agentlab.prompts import load_prompt

_FOLD_START_ROLES = {"user"}  # 回合起点：切点只允许落在这些角色消息之前
_CHECKPOINT_TAG = "[工作记忆检查点]"
_PATH_RE = re.compile(r'["\'](?:path|bvid|kw)["\']\s*:\s*["\']([^"\']+)["\']')
# 非"用户真实指令"的注入消息前缀：_pin_directive 钉待办时跳过——
# steer/nudge/硬截断标记若被钉住，会把系统提示当用户待办逐字回放（L10/OPT-106）
_NON_DIRECTIVE_PREFIXES = ("[待办指令]", "[AGENT_INCOMPLETE]", "[上下文提示]",
                           "[AGENT_OUTPUT_TRUNCATED]", "[续写指令]")


def _find_cut(entries: Sequence[Message], recent: int,
              fold_start_roles: frozenset[str] | set[str] | None = None,
              floor: int = 0) -> int:
    """从末尾往回找切点，落在回合起点，不切断工具调用（04 §3.1 ②）。

    fold_start_roles 默认 {user}（回合起点=用户消息）。L9/OPT-104 放宽选项：
    单用户长工具流（锚点冻住唯一 user 边界）时传 {"user","assistant"}——
    assistant 步边界同样安全（其后随的工具结果与它同侧，永不切在 tool 中段）。
    floor（OPT-106）：只考虑 >floor 的边界——锚点之前全是冻结区，切在那里
    必然空折叠；无进展时让调用方走放宽/放行路径而非误报切点。
    """
    roles = fold_start_roles if fold_start_roles is not None else _FOLD_START_ROLES
    cutoff: int | None = None
    acc = 0
    for i in range(len(entries) - 2, floor, -1):  # 最后一条 user 输入必须保留
        m = entries[i]
        acc += estimate_tokens(m)
        if m.role in roles:
            cutoff = i
        if acc >= recent and cutoff is not None:
            return cutoff
    return floor if cutoff is None else cutoff


def _find_forward_cut(entries: Sequence[Message], start: int, budget_tokens: int,
                      roles: frozenset[str] | set[str] | None = None) -> int | None:
    """正向找切点（OPT-110 四期分片）：从 start 起累计到 budget_tokens 时，切在
    最后一个回合起点角色处，保证单次折叠区段 ≈ ≤ budget_tokens（小步多次折叠）。
    区段累计不足 budget → None（调用方沿用原切点，整段一次折叠）。"""
    roles = roles if roles is not None else _FOLD_START_ROLES
    cut: int | None = None
    acc = 0
    for i in range(start, len(entries)):
        m = entries[i]
        if acc >= budget_tokens and cut is not None:
            return cut
        acc += estimate_tokens(m)
        if m.role in roles:
            cut = i
    return None


def _pin_directive(entries: Sequence[Message], cut: int) -> str | None:
    """取"待办指令"锚点，逐字钉回保留段（OPT-087/088 / 学 tau、pi）。

    - 优先：collapsed（索引 < cut）内**最近一条用户指令**原文——最贴近当前待办。
    - 兜底：若折叠段内无 user（罕见），回退首条 user 原文（原始任务锚，
      与 tau「first user message always preserved」对齐）。
    - steer/nudge 等注入消息（_NON_DIRECTIVE_PREFIXES）不是用户待办，一律跳过。
    """
    if cut <= 0:
        return None
    for i in range(cut - 1, -1, -1):
        if entries[i].role == "user":
            c = entries[i].content or ""
            if not c.startswith(_NON_DIRECTIVE_PREFIXES):
                return c
    for m in entries:  # 折叠段无 user：钉首条 user（原始任务）
        if m.role == "user":
            c = m.content or ""
            if not c.startswith(_NON_DIRECTIVE_PREFIXES):
                return c
    return None


def _mask_observations(entries: Sequence[Message], keep_recent_tokens: int) -> list[Message]:
    """Tier-2 观察遮盖（OPT-088 / 学 tau「mechanical only」）。

    - 比摘要成本为零、零幻觉：把保留段里**较旧**的 tool 结果替换成占位符
      `[output from <tool> omitted]`，但**保留工具名/参数与 assistant 的 tool_calls**，
      模型仍能看到"做过什么"，只是省掉大段结果。
    - 只返回新列表，仅对被遮盖的 tool 消息做深拷贝（不污染原存储 = 盘全量、视图裁剪）。
    """
    if keep_recent_tokens <= 0 or not entries:
        return list(entries)
    acc = 0
    masked_start: int | None = None
    for i in range(len(entries) - 1, -1, -1):
        acc += estimate_tokens(entries[i])
        if acc >= keep_recent_tokens:
            masked_start = i
            break
    if masked_start is None:
        return list(entries)
    out: list[Message] = []
    for i, m in enumerate(entries):
        if i < masked_start and m.role == "tool" and m.name:
            out.append(m.model_copy(deep=True))
            out[-1].content = f"[output from {m.name} omitted]"
        else:
            out.append(m)
    return out


def _last_checkpoint(entries: Sequence[Message]) -> str:
    """取出最近一次检查点摘要正文（供 update 增量复用），无则空串。"""
    for m in reversed(entries):
        if m.role == "system" and m.content and m.content.startswith(_CHECKPOINT_TAG):
            return m.content[len(_CHECKPOINT_TAG):].strip()
    return ""


def _side_effects(messages: Sequence[Message]) -> list[str]:
    """副作用追踪（04 §3.1 ④）：抽被压缩段读写的 path/bvid/kw，去重保值序。"""
    seen: dict[str, None] = {}
    for m in messages:
        seg = m.content or ""
        if m.tool_calls:
            for tc in m.tool_calls:
                seg += " " + tc.function.arguments
        for mm in _PATH_RE.findall(seg):
            seen.setdefault(mm.strip(), None)
    return list(seen)


async def _summarize(provider: LLMProvider, messages: list[Message], prev: str) -> str:
    """渲染 summarize-user.st：增量（带旧检查点）或全量。"""
    seg = "\n".join(f"{m.role}: {m.content}" for m in messages)
    prev_block = f"## 旧检查点\n{prev}" if prev else "（无旧检查点，全量压缩）"
    prompt = load_prompt("summarize-user", transcript=seg, prev_block=prev_block)
    resp = await provider.chat([Message(role="user", content=prompt)], tools=None, temperature=0.2)
    return resp.content or ""


class WorkingMemory:
    def __init__(
        self,
        budget: int = 8000,
        context_window: int = 32000,
        reserve_tokens: int = 16384,
        keep_recent_tokens: int = 20000,
        summarizer: LLMProvider | None = None,
    ):
        self.budget = budget
        self.context_window = context_window
        self.reserve_tokens = reserve_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.summarizer = summarizer
        self._entries: list[Message] = []
        # 最近一次 condense 在锚点处插入的头部消息数（检查点+钉住指令）；
        # 供调用方（loop）前移压缩锚点、冻结字节稳定前缀（L9/OPT-104）。无进展=0。
        self.last_condense_head = 0
        # 最近一次折叠掉的区段原文（L11/OPT-111）：供调用方归档 {sid}.ranges.jsonl，
        # 让"窗口外内容"可召回而非不可逆丢失。无折叠=空列表。
        self.last_folded: list[Message] = []

    def add(self, msg: Message) -> None:
        self._entries.append(msg)

    def messages(self) -> list[Message]:
        return list(self._entries)

    def estimate(self) -> int:
        return sum(estimate_tokens(m) for m in self._entries)

    @property
    def over_budget(self) -> bool:
        return self.estimate() > (self.context_window - self.reserve_tokens)

    def find_cut_point(self, keep_recent_tokens: int | None = None) -> int:
        return _find_cut(self._entries, keep_recent_tokens or self.keep_recent_tokens)

    # —— 容器内压缩 ——
    async def compact(self) -> None:
        """cut_point 之前的消息 → 结构化检查点摘要；摘要 + 保留段替换被压缩段。"""
        if not self.over_budget or len(self._entries) < 4:
            return
        cut = self.find_cut_point()
        if cut <= 0:
            return
        collapsed = self._entries[:cut]
        kept = self._entries[cut:]
        self.last_folded = list(collapsed)  # L11：与 condense 同语义（compact 仅测试/CLI 用）
        summary = await self._build_checkpoint(collapsed)
        base = [Message(role="system", content=f"{_CHECKPOINT_TAG}\n{summary}")] if summary else []
        self._entries = base + _mask_observations(kept, self.keep_recent_tokens)

    async def _build_checkpoint(self, collapsed: list[Message]) -> str:
        """生成检查点摘要：update 增量（若已有旧检查点）+ 副作用。"""
        prev = _last_checkpoint(self._entries)
        body = ""
        if self.summarizer is not None:
            body = await _summarize(self.summarizer, collapsed, prev)
        effects = _side_effects(collapsed)
        if effects:
            body = f"{body}\n\n## Side Effects（曾读写的文件/笔记）\n" + "\n".join(f"- {p}" for p in effects)
        return body.strip()

    # —— 纯函数压缩（供 loop 接入，不改容器） ——
    async def condense(self, messages: list[Message],
                       keep_recent_tokens: int | None = None,
                       anchor: int = 0,
                       max_fold_tokens: int | None = None) -> list[Message]:
        """对任意消息列表做压缩：折叠区 → 检查点摘要，返回 冻结前缀+摘要+保留段。

        - anchor（L9/OPT-104 前缀缓存友好）：messages[:anchor] 为**冻结前缀**，
          逐字节原样保留、永不重折叠——已折叠过的历史检查点跨轮稳定，provider
          前缀缓存得以跨压缩命中；本次只折叠 [anchor, cut)，摘要消息插在 anchor 位。
          anchor=0 时与旧全量折叠行为逐字节兼容。
        - max_fold_tokens（OPT-110 四期分片）：折叠区超过此 token 上限时收缩切点
          （小步多次折叠，学 pi 早折勤折），避免巨型摘要调用；None=不限。
        - 有 summarizer：结构化摘要；无 summarizer：仅丢弃折叠区（兜底，冻结前缀不动）。
        - 消息不足或切点无效（cut<=anchor）时原样返回（无进展放行），last_condense_head=0。
        - 无论是否压缩，折叠区内最近一条用户指令（_pin_directive）都逐字钉回保留段，
          保证"用户待办"在有损折叠后仍存活（OPT-087 / 学 tau「展示历史≠回放上下文」）；
          只在折叠区内找，冻结区里已钉过的指令不会重复钉。
        - 折叠发生时 last_folded 带出被折叠区段原文（L11/OPT-111），供调用方归档；
          无折叠时清空。
        """
        self.last_folded = []
        if len(messages) < 4:
            self.last_condense_head = 0
            return messages
        recent = keep_recent_tokens or self.keep_recent_tokens
        cut = _find_cut(messages, recent, floor=anchor)
        if cut <= anchor:
            # 冻结锚点吃掉唯一 user 边界（单用户长工具流，CRT 典型形态）→
            # 放宽到 assistant 步边界重找（仍不切在 tool 中段，工具对保持完整）
            cut = _find_cut(messages, recent, fold_start_roles=frozenset({"user", "assistant"}),
                            floor=anchor)
        if cut <= anchor:
            self.last_condense_head = 0
            return messages
        if max_fold_tokens:
            from agentlab.core.context import estimate_tokens as _est
            fold_est = sum(_est(m) for m in messages[anchor:cut])
            if fold_est > max_fold_tokens:
                sliced = _find_forward_cut(messages, anchor, max_fold_tokens,
                                           fold_start_roles=frozenset({"user", "assistant"}))
                if sliced is not None and sliced > anchor:
                    cut = sliced
        frozen = messages[:anchor]
        collapsed = messages[anchor:cut]
        kept = messages[cut:]
        self.last_folded = list(collapsed)  # L11：折叠区段原文带出，供调用方归档
        prev = _last_checkpoint(messages)
        directive = _pin_directive(collapsed, len(collapsed))
        body = ""
        if self.summarizer is not None:
            body = await _summarize(self.summarizer, collapsed, prev)
        effects = _side_effects(collapsed)
        if effects:
            body = f"{body}\n\n## Side Effects（曾读写的文件/笔记）\n" + "\n".join(f"- {p}" for p in effects)
        head: list[Message] = []
        if directive:
            # 钉住的用户指令逐字拼在保留段前，作为兜底锚点（即使摘要可能省掉）
            head.append(Message(role="user", content=f"[待办指令] {directive}"))
        if body:
            head.insert(0, Message(role="system", content=f"{_CHECKPOINT_TAG}\n{body}"))
        if not head:
            self.last_condense_head = 0
            return frozen + _mask_observations(kept, recent)  # 无摘要能力：纯丢弃折叠区 + 遮盖较旧观察
        self.last_condense_head = len(head)
        return frozen + head + _mask_observations(kept, recent)