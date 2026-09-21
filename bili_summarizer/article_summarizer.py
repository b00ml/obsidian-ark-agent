#!/usr/bin/env python3
"""微信公众号文章总结处理器（Ingest Pipeline 处理层）

功能:
  - fetch_article:  抓取公众号文章 HTML（公开网页，浏览器 UA）
  - extract_article: 提取标题/作者/正文纯文本（标准库，无第三方依赖）
  - summarize_article: LLM 总结（prompt 走 article-summary-user.st，配置走 visual_models.json）
  - build_note:     渲染 Obsidian 笔记（YAML frontmatter + 结构化正文）

用法:
  python article_summarizer.py <url> [--vault C:/path/to/your/obsidian-vault] [--output 自定义路径]

关键约束（见 DESIGN-INGEST-PIPELINE.md §3.8 / 禁止事项 2）:
  - Prompt 必须走 bili_summarizer/prompts/article-summary-user.st + prompt_loader
  - LLM 调用必须记录 trace（trace_id / cost / latency）
  - 处理器不直接写 Vault（由 Trae 通过 MCP 写入）——本脚本只产出笔记文本到 stdout 或本地文件
"""
import argparse
import json
import os
import re
import sys
import time
from html.parser import HTMLParser

import requests

from prompt_loader import load_prompt, render
from visual_analyzer import load_visual_config
from knowledge_compiler import compile_note
from provider_client import ProviderError, TextModelClient

# 配置文件默认路径（相对本文件）
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "visual_models.json"
)

# 知识编译开关（优化设计文档4.0 执行线#2）：笔记落盘后 LLM 提取概念/实体
# → 增量更新 vault 的 wiki 概念/实体页。编译为 best-effort，任何异常只打
# [COMPILE] 警告，绝不影响主笔记产出；仅在同时提供 --vault 时生效。
KNOWLEDGE_COMPILE_ENABLED = True

# 浏览器 UA（公众号文章是公开网页，普通请求即可，见难点 5.1）
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


class _TextExtractor(HTMLParser):
    """从 HTML 中提取纯文本（忽略 script/style/标签）"""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in ("p", "br", "h1", "h2", "h3", "li", "tr", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        # 折叠多余空行
        lines = [ln.strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)


def fetch_article(url: str, timeout: int = 30) -> str:
    """抓取公众号文章 HTML"""
    resp = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def extract_article(html_text: str) -> dict:
    """从 HTML 提取 {title, author, content}

    - title: <h1 class="rich_media_title"> 或 <title>
    - author: var nickname = "xxx" 或 meta
    - content: <div id="js_content"> 内正文纯文本
    """
    # 标题: rich_media_title → og:title → <title>
    title = ""
    m = re.search(r'<h1[^>]*class="rich_media_title"[^>]*>(.*?)</h1>',
                  html_text, re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    if not title:
        m = re.search(
            r'<meta[^>]+property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
            html_text, re.IGNORECASE,
        )
        if m:
            title = m.group(1).strip()
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).strip()
    title = re.sub(r"\s+", " ", title)

    # 作者（公众号名）: var nickname → og:article:author → js_name
    author = ""
    m = re.search(r'var\s+nickname\s*=\s*["\']([^"\']+)["\']', html_text)
    if m:
        author = m.group(1).strip()
    if not author:
        m = re.search(r'<span[^>]*class="rich_media_meta_nickname"[^>]*>(.*?)</span>',
                      html_text, re.IGNORECASE | re.DOTALL)
        if m:
            author = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    if not author:
        m = re.search(
            r'<meta[^>]+property=["\']og:article:author["\']'
            r'[^>]*content=["\']([^"\']+)["\']',
            html_text, re.IGNORECASE,
        )
        if m:
            author = m.group(1).strip()

    # 正文：优先 #js_content
    content = ""
    m = re.search(r'<div[^>]*id="js_content"[^>]*>(.*?)</div>\s*</div>',
                  html_text, re.IGNORECASE | re.DOTALL)
    if m:
        parser = _TextExtractor()
        parser.feed(m.group(1))
        content = parser.text()
    if not content:
        # 兜底：提取整个 body 文本
        m = re.search(r"<body[^>]*>(.*?)</body>", html_text, re.IGNORECASE | re.DOTALL)
        if m:
            parser = _TextExtractor()
            parser.feed(m.group(1))
            content = parser.text()
    # 限制长度（防超长）
    if len(content) > 20000:
        content = content[:20000] + "\n...（正文过长已截断）"
    return {"title": title, "author": author, "content": content}


def summarize_article(article: dict, url: str, config: dict) -> dict:
    """LLM 总结文章 → 结构化 JSON

    config: visual_models.json 的 default 键（文本模型）
    """
    model_cfg = config.get("default", config)
    prompt = render(
        load_prompt("article-summary-user"),
        {
            "article_title": article.get("title", "") or "未知标题",
            "article_content": article.get("content", "")[:12000],
            "article_url": url,
            "article_author": article.get("author", ""),
        },
    )

    # trace 记录（trace_id / cost / latency）
    start = time.time()

    try:
        result = TextModelClient(
            api_base=model_cfg.get("api_base", ""), api_key=model_cfg.get("api_key", ""),
            model=model_cfg.get("model", ""), timeout=120, transport=requests.post,
        ).chat(prompt, max_tokens=1500)
    except ProviderError as exc:
        raise RuntimeError(str(exc)) from exc
    content = result.content

    # trace 打印（CLI 工具：打印 [ARTICLE] 状态即可，见 llm_calls.md）
    # Use the client trace id as the source of truth; the old local id could
    # diverge from the provider trace printed by the shared adapter.
    trace_id = result.trace_id
    latency = time.time() - start
    usage = result.usage
    cost = (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)) / 1000000
    print(f"[ARTICLE] LLM 调用完成 trace_id={trace_id} latency={latency:.1f}s "
          f"tokens={usage.get('total_tokens', 0)} est_cost=${cost:.4f}")

    # 二次校验：解析 JSON（失败则用降级结构）
    text = content.strip().strip("```")
    text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("[ARTICLE] LLM 返回非 JSON，使用降级结构")
        data = {
            "title": article.get("title", ""),
            "one_sentence": "",
            "points": [],
            "key_info": [],
            "summary": text.strip()[:500],
        }
    # 字段兜底
    data.setdefault("title", article.get("title", "") or "未知标题")
    data.setdefault("one_sentence", "")
    data.setdefault("points", [])
    data.setdefault("key_info", [])
    data.setdefault("summary", "")
    if not isinstance(data.get("points"), list):
        data["points"] = []
    if not isinstance(data.get("key_info"), list):
        data["key_info"] = []
    return data


def kebab(s: str) -> str:
    """标题 → kebab-case 文件名（特殊字符替换）"""
    s = re.sub(r"[\\/:*?\"<>|\s]+", "-", s.strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:60] or "article"


def build_note(article: dict, summary: dict, url: str,
               created: str = "") -> str:
    """渲染 Obsidian 笔记（YAML frontmatter + 结构化正文）"""
    if not created:
        import datetime as _dt
        created = _dt.date.today().isoformat()

    points = "\n".join(
        f"- {p}" for p in summary.get("points", [])
    ) or "- （无）"
    key_rows = "\n".join(
        f"| {k} | {v} |" for k, v in summary.get("key_info", [])
    ) or "| （无） | |"

    return f"""---
title: "{summary.get('title', article.get('title', ''))}"
type: article-summary
source: "{url}"
author: "{article.get('author', '')}"
date: {created}
tags: [article, summary]
created: {created}
updated: {created}
---

# {summary.get('title', article.get('title', ''))}

> **链接**: {url}

## 🎯 一句话核心

{summary.get('one_sentence', '') or '（无）'}

## 💡 主要论据

{points}

## 📊 关键信息表

| 概念 | 说明 |
|------|------|
{key_rows}

## 📝 总结

{summary.get('summary', '') or '（无）'}
"""


def make_llm_call(model_cfg: dict):
    """包装既有 OpenAI 兼容客户端 → knowledge_compiler 所需的 llm_call(prompt)->str

    复用 summarize_article 同款请求构造（config/visual_models.json 的 default 模型）。
    """
    def llm_call(prompt: str) -> str:
        try:
            return TextModelClient(
                api_base=model_cfg.get("api_base", ""), api_key=model_cfg.get("api_key", ""),
                model=model_cfg.get("model", ""), timeout=60, transport=requests.post,
            ).chat(prompt, max_tokens=800).content
        except ProviderError as exc:
            raise RuntimeError(str(exc)) from exc
    return llm_call


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="公众号文章总结处理器")
    parser.add_argument("url", help="公众号文章 URL")
    parser.add_argument("--output", "-o", default=None, help="笔记输出路径（默认 stdout）")
    parser.add_argument("--model-config", default=DEFAULT_CONFIG_PATH,
                        help="模型配置文件路径 (默认: config/visual_models.json)")
    parser.add_argument("--vault", default=None,
                        help="Obsidian Vault 路径（预留；实际写入由 Trae 通过 MCP 完成）")
    args = parser.parse_args()

    try:
        print("[ARTICLE] 抓取文章...")
        html_text = fetch_article(args.url)
        article = extract_article(html_text)
        print(f"[ARTICLE] 提取到标题: {article['title']} 正文长度: {len(article['content'])}")

        config = load_visual_config(args.model_config)
        summary = summarize_article(article, args.url, config)
        note = build_note(article, summary, args.url)

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(note)
            print(f"[ARTICLE] 笔记已写入: {args.output}")

            # 知识编译（优化设计文档4.0 #2，best-effort）：仅在 --vault 时生效，
            # 任何异常只警告，不影响主笔记产出
            if KNOWLEDGE_COMPILE_ENABLED and args.vault:
                try:
                    page_title = os.path.splitext(os.path.basename(args.output))[0]
                    result = compile_note(args.vault, args.output, page_title, note,
                                          make_llm_call(config.get("default", config)))
                    print(f"[COMPILE] 知识编译完成: 新建{len(result['created'])} "
                          f"合并{len(result['updated'])} 冲突{len(result['conflicts'])} "
                          f"跳过{result['skipped']}")
                except Exception as e:
                    print(f"[COMPILE] 警告: 知识编译失败（不影响主笔记）: {e}")
        else:
            print(note)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
