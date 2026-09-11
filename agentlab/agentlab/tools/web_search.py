"""web_search 工具：DuckDuckGo 全网检索（read，不属 brain 21 工具）。

框架自带的最小联网工具：纯标准库 urllib + 正则解析，避免额外依赖。
反爬可用性低于专用 API，失败时降级为空结果，不抛错中断循环。
"""
from __future__ import annotations

import html as _html
import re
import urllib.parse
import urllib.request

from agentlab.tools.base import tool

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_TIMEOUT = 10

# DDG html 版每个结果：
#   <a rel="nofollow" class="result__a" href="https://...">标题<span class="result__icon">...</span></a>
#   <a class="result__snippet" ...>摘要<span ...>...</span></a>
_RE_A = re.compile(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_RE_SNIP = re.compile(r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', re.S)


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def _ddg_html(query: str) -> str:
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _restore_direct(url: str) -> str:
    m = re.search(r"[?&]uddg=([^&]+)", url)
    if not m:
        return url
    try:
        return urllib.parse.unquote(m.group(1))
    except Exception:
        return url


@tool(
    name="web_search",
    description="全网网页检索（DuckDuckGo）。返回标题/URL/摘要前若干条；反爬失败时返回空列表。",
    permission="read",
)
def web_search(query: str, limit: int = 5) -> list[dict]:
    """按 query 检索网页，返回 {title,url,snippet} 列表。"""
    try:
        page = _ddg_html(query)
    except Exception as e:  # 网络/反爬失败：降级为空结果，不抛错
        return [{"title": f"[web_search 失败] {type(e).__name__}", "url": "", "snippet": ""}]
    return _parse_results(page, limit)


def _parse_results(page: str, limit: int = 5) -> list[dict]:
    """解析 DuckDuckGo html 结果页，返回 {title,url,snippet} 列表（可离线单测）。"""
    titles = [m for m in _RE_A.findall(page)]        # [(href, inner_html)]
    snips_raw = _RE_SNIP.findall(page)
    snips = [_strip_tags(s) for s in snips_raw]
    if len(snips) < len(titles):
        snips = snips + [""] * (len(titles) - len(snips))

    results = []
    limit = max(1, min(limit, len(titles)))
    for (href, inner), snip in zip(titles[:limit], snips[:limit]):
        results.append({
            "title": _html.unescape(_strip_tags(inner)).strip(),
            "url": _html.unescape(_restore_direct(href)),
            "snippet": _html.unescape(snip)[:200].strip(),
        })
    return results


WEB_SEARCH_TOOLS = [web_search]