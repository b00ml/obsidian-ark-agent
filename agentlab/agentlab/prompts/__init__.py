"""prompt 加载渲染（复用项目 prompt_loader 的 .st 惯例，见 AGENTS.md 豁免）。

只渲染 {{var}} 占位符；未提供变量的占位符保留原文。
"""
from __future__ import annotations

import re
from pathlib import Path

_DIR = Path(__file__).parent
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def load_prompt(name: str, **variables) -> str:
    """按名称加载 .st 模板并渲染。name 不含 .st。"""
    fp = _DIR / f"{name}.st"
    if not fp.exists():
        raise FileNotFoundError(f"prompt 模板不存在：{name}")
    text = fp.read_text(encoding="utf-8")

    def repl(m: re.Match) -> str:
        key = m.group(1)
        return str(variables.get(key, m.group(0)))

    return _PLACEHOLDER.sub(repl, text)