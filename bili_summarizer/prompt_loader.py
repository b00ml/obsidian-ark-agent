#!/usr/bin/env python3
"""Prompt 加载器（LLM 调用标准化基础设施）

从 prompts/ 目录加载 .st 模板并渲染占位符。

规则（参考 04-llm-call-standardization.md）:
  - load_prompt(name): 读取 prompts/{name}.st，带 lru_cache
  - render(template, params): 用 str.replace("{{k}}", v) 注入占位符
    （不用 .format()，避免与 JSON 大括号冲突；缺占位符留空串）
"""
import os
import re
from functools import lru_cache

# prompts/ 目录（相对本文件，支持从任意 cwd 调用）
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


class PromptNotFoundError(FileNotFoundError):
    """Prompt 模板不存在"""


@lru_cache(maxsize=128)
def load_prompt(name: str) -> str:
    """加载 prompts/{name}.st 模板内容（带缓存）"""
    path = os.path.join(PROMPTS_DIR, f"{name}.st")
    if not os.path.exists(path):
        raise PromptNotFoundError(
            f"Prompt 模板不存在: {path}\n请检查 prompts/ 目录。")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render(template: str, params: dict | None = None) -> str:
    """渲染模板，用 {{param}} 占位符注入参数

    使用 str.replace 而非 .format()，避免模板内 JSON 大括号冲突。
    缺失的占位符留空串（regex 兜底替换）。
    """
    if params:
        for key, value in params.items():
            template = template.replace("{{" + key + "}}", str(value))
    # 未被提供的占位符统一留空串
    return re.sub(r"\{\{\s*\w+\s*\}\}", "", template)


def clear_cache() -> None:
    """清空 prompt 缓存（测试用）"""
    load_prompt.cache_clear()
