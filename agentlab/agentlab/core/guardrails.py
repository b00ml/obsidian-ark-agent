"""输出校验与 JSON 抽取（对齐 docs/04 §1.1、03 §2.2bis 及 test_guardrails）。

- extract_json：从模型输出中稳健抽取 JSON（剥 `${'```'}json` 围栏/剥壳/括号配对）。
- validate_output：结构性谓词（宽度/长度/空输出）经一遍轻校验，失败则阻塞最终输出。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from agentlab.core.errors import AgentError

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> Any:
    """剥围栏与噪点后解析 JSON；失败抛 AGENT_GUARDRAIL。"""
    if not isinstance(text, str):
        raise AgentError("AGENT_GUARDRAIL", "extract_json 输入非字符串")
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    # 剥壳：找 `{...}` / `[...]` 的首尾括号配对
    start = cleaned.find("{")
    if start < 0:
        start = cleaned.find("[")
    if start < 0:
        raise AgentError("AGENT_GUARDRAIL", "输出中未找到 JSON 对象/数组")
    try:
        return json.loads(cleaned[start:])
    except json.JSONDecodeError:
        # 尝试括号配对修正（兼容模型在末尾追加说明文字）
        return _balanced_load(cleaned, start)


def _balanced_load(text: str, start: int) -> Any:
    pairs = {"{": "}", "[": "]"}
    open_ch = text[start]
    close_ch = pairs[open_ch]
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    raise AgentError("AGENT_GUARDRAIL", "JSON 括号配对后仍无法解析")


def validate_output(output: str, *, max_len: int = 200_000, min_len: int = 1) -> bool:
    """轻量输出校验：空/超长判为失败。返回 True 为通过。"""
    if output is None:
        return False
    if len(output) < min_len:
        return False
    if len(output) > max_len:
        return False
    return True


def ensure_structured(output: str, schemas: list[Callable[[Any], bool]] | None = None) -> bool:
    """结构化输出二次校验：能解析成 JSON 且经由 schemas 谓词逐个通过。

    供需要"必须输出 JSON"的固定流水线步骤调用。
    """
    try:
        obj = extract_json(output)
    except AgentError:
        return False
    if schemas is None:
        return True
    return all(pred(obj) for pred in schemas)