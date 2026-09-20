#!/usr/bin/env python3
"""知识编译器（优化设计文档4.0 执行线#2，Karpathy LLM Wiki 模式）

把知识加工从"单源笔记生产"升级为"持续编译"：摄入一份资料产出一篇总结之后，
再从总结中 LLM 提取概念/实体 → 命中 wiki 已有页则增量合并（矛盾不静默覆盖，
写 `## 知识冲突` 区块保留两说）→ 无则按 templates 风格新建 → 同步更新
wiki/index.md 总目录与 wiki/log.md 编译日志。

职责边界：
  - 本模块只负责 wiki 页的编译写入；"编译失败不影响主笔记产出"由调用方
    try/except 兜底（article_summarizer / bili_transcript 接线处均 best-effort）。
  - 本模块不直接发起网络请求：extract_knowledge 接收 llm_call(prompt)->str
    可调用对象（测试注入 fake；生产由调用方用既有 OpenAI 兼容客户端包装，
    见 article_summarizer.make_llm_call / bili_transcript.compile_knowledge_for_note）。

用法（编译一篇已落盘的笔记）:
    from knowledge_compiler import compile_note
    result = compile_note(vault_root, note_path, note_title, note_content, llm_call)
    # {"created": [...], "updated": [...], "conflicts": [...], "skipped": N}
"""
import json
import os
import re
from datetime import date

from prompt_loader import load_prompt, render
from vault_io import controlled_write, guard_vault_path  # P0-01 受控写入

# 提取条目上限（prompt 已同步约束，二次校验兜底）
MAX_ENTRIES = 12
# 合法 kind -> wiki 子目录
VALID_KINDS = ("concept", "entity")
KIND_DIR = {"concept": "concepts", "entity": "entities"}
# 传给 LLM 的笔记正文长度上限（控制 token，超长截断）
MAX_CONTENT_CHARS = 12000
# 页内"一句话核心"可能出现的标题关键词（依次探测，兼容人工模板的"定义"）
_CORE_HEADINGS = ("一句话核心", "定义")


# ========== 提取层 ==========

def sanitize_page_name(name: str) -> str:
    """条目名 → Windows/Obsidian 双安全的 wiki 页名（非法字符替换、首尾清理）"""
    s = re.sub(r'[\\/:*?"<>|#^\[\]]', "_", str(name or "").strip())
    s = re.sub(r"\s+", " ", s).strip(" ._")
    return s


def _parse_json_array(text: str) -> list:
    """LLM 返回文本 → JSON 数组（剥代码围栏/截取首尾中括号；解析失败返回 []）"""
    if not text:
        return []
    t = str(text).strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def extract_knowledge(llm_call, title: str, content: str) -> list[dict]:
    """LLM 提取概念/实体 → 二次校验后的条目列表

    llm_call: callable(prompt: str) -> str（由调用方注入）
    返回: [{"name", "kind", "one_line"}, ...]，最多 MAX_ENTRIES 条；
          空 name 丢弃、非法 kind 兜底 "concept"、同批重名去重、解析失败返回 []。
    """
    prompt = render(
        load_prompt("knowledge-extract-user"),
        {"note_title": title or "未命名笔记",
         "note_content": (content or "")[:MAX_CONTENT_CHARS]},
    )
    entries: list[dict] = []
    seen: set[str] = set()
    for item in _parse_json_array(llm_call(prompt)):
        if not isinstance(item, dict):
            continue
        name = sanitize_page_name(item.get("name", ""))
        if not name or name in seen:
            continue  # 空 name 丢弃 / 同批重名只取首次
        seen.add(name)
        kind = item.get("kind")
        if kind not in VALID_KINDS:
            kind = "concept"  # 非法 kind 兜底
        entries.append({
            "name": name,
            "kind": kind,
            "one_line": str(item.get("one_line", "") or "").strip(),
        })
        if len(entries) >= MAX_ENTRIES:
            break
    return entries


# ========== 页面文本工具 ==========

def _norm(s: str) -> str:
    """去空白归一（用于内容重复/包含判断）"""
    return re.sub(r"\s+", "", s or "")


def _extract_section(text: str, heading_keyword: str) -> str:
    """取 markdown 中第一个含 heading_keyword 的 `## ` 标题区块正文（至下一 `## ` 为止）"""
    m = re.search(rf"^##\s*[^\n]*{re.escape(heading_keyword)}[^\n]*\n",
                  text, re.MULTILINE)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"^##\s", rest, re.MULTILINE)
    return (rest[:nxt.start()] if nxt else rest).strip()


def _extract_core(page_text: str) -> str:
    """读原页"一句话核心"（依次探测 _CORE_HEADINGS，人工页常见"定义"也能命中）"""
    for kw in _CORE_HEADINGS:
        body = _extract_section(page_text, kw)
        if body:
            return body
    return ""


def _read_sources(page_text: str) -> list[str]:
    """best-effort 读 frontmatter 的 sources 列表里的 wikilink 名（冲突块标旧说来源用）"""
    m = re.search(r"^---\n(.*?)\n---", page_text, re.DOTALL)
    if not m:
        return []
    sm = re.search(r"^sources:\s*\n((?:[ \t]+-[^\n]*\n?)+)", m.group(1), re.MULTILINE)
    if not sm:
        return []
    return [s.strip() for s in re.findall(r"\[\[([^\]|#]+)", sm.group(1))]


def _is_conflict(old_core: str, new_core: str) -> bool:
    """矛盾判据（简单版）：两者都非空且互不包含"""
    o, n = (old_core or "").strip(), (new_core or "").strip()
    return bool(o) and bool(n) and (o not in n) and (n not in o)


def _already_noted(page_text: str, note_title: str, new_core: str) -> bool:
    """本笔记对本页的贡献是否已存在（幂等判据：同一笔记跑两遍不重复追加）。

    判据 = 页内已引用本笔记（[[note_title]] 出现在 sources/来源/`## 来自` 小节/
    冲突块中）**且** 新概括文本已在页内（新建页的"一句话核心"、`## 来自` 小节
    或 `## 知识冲突` 区块三处之一）。只看内容重合不看来源会把"首次合并"
    （新概括恰好与旧核心同文）误判为已处理，故必须同时校验双链归属。
    """
    if f"[[{note_title}]]" not in page_text:
        return False
    return _norm(new_core) in _norm(page_text)


# ========== 页面区块渲染 ==========

def _new_page(name: str, kind: str, one_line: str, note_title: str, today: str) -> str:
    """新建 wiki 页：frontmatter(type/title/description/sources/generated/status)
    + 一句话核心 + 来源双链（字段风格对齐 templates/concept.md）"""
    core = one_line or "（待补充概括）"
    return (
        f'---\ntitle: "{name}"\ntype: {kind}\n'
        f'description: "{core}"\n'
        f'sources:\n  - "[[{note_title}]]"\n'
        f"generated: {today}\nstatus: draft\n---\n\n"
        f"# {name}\n\n"
        f"## 一句话核心\n\n{core}\n\n"
        f"## 来源\n\n- [[{note_title}]]\n"
    )


def _merge_section(note_title: str, one_line: str, today: str) -> str:
    """增量合并小节：来自源笔记的概括 + 一行说明"""
    return (
        f"\n## 来自 [[{note_title}]]\n\n"
        f"- 概括：{one_line or '（无概括）'}\n\n"
        f"（知识编译自动追加于 {today}，来源：[[{note_title}]]）\n"
    )


def _conflict_block(old_core: str, new_core: str, note_title: str,
                    old_sources: list[str], today: str) -> str:
    """知识冲突区块：两说并存 + 各自来源，不静默覆盖，待人工核对"""
    src = f"（原页来源：[[{old_sources[0]}]]）" if old_sources else ""
    return (
        f"\n## 知识冲突\n\n"
        f"- 旧说{src}：{old_core}\n"
        f"- 新说（[[{note_title}]]）：{new_core}\n\n"
        f"> 两说并存待人工核对，未做静默覆盖（编译于 {today}）。\n"
    )


# ========== index / log ==========

def _pinyin_sort_key(name: str) -> str:
    """拼音/字母排序键（pypinyin 可选依赖：未安装时退回小写码点序）"""
    try:
        from pypinyin import lazy_pinyin
        return "".join(lazy_pinyin(name)).lower()
    except Exception:
        return name.lower()


def _update_index(wiki_root: str, items: list[tuple[str, str]],
                  vault_root: str | None = None) -> None:
    """更新 wiki/index.md 总目录：概念/实体分节、排序去重（全量重写保证幂等）。

    非托管分节（用户手工加的其他目录节）原样保留；条目只增不删。
    items: [(页名, kind), ...] 本次触碰到的页。
    """
    path = os.path.join(wiki_root, "index.md")
    text = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    known: dict[str, list[str]] = {"concept": [], "entity": []}
    others: list[tuple[str, str]] = []  # 非托管分节 (标题行, 正文) 原样保留
    sec_re = re.compile(r"^##[^\n]*\n", re.MULTILINE)
    matches = list(sec_re.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header, body = m.group(0), text[m.end():end]
        title = header.lstrip("#").strip()
        if title.startswith("概念"):
            known["concept"] = re.findall(r"\[\[([^\]|#]+)\]\]", body)
        elif title.startswith("实体"):
            known["entity"] = re.findall(r"\[\[([^\]|#]+)\]\]", body)
        else:
            others.append((header, body))

    for name, kind in items:
        if kind in known and name not in known[kind]:
            known[kind].append(name)

    out = ["# Wiki 总目录", "",
           "> 由知识编译自动维护（优化设计文档4.0 #2），条目按拼音/字母序排列。", ""]
    for title, kind in (("## 概念", "concept"), ("## 实体", "entity")):
        names = sorted(set(known.get(kind, [])), key=_pinyin_sort_key)
        if not names:
            continue
        out.append(title)
        out.append("")
        out.extend(f"- [[{n}]]" for n in names)
        out.append("")
    for header, body in others:
        out.append(header.rstrip("\n"))
        out.append("")
        out.append(body.rstrip("\n"))
        out.append("")

    controlled_write(path, "\n".join(out).rstrip("\n") + "\n",
                     vault_root=vault_root, overwrite=True, actor="knowledge-compile")


def _append_log(wiki_root: str, note_title: str, created_n: int, updated_n: int,
                conflict_n: int, today: str, vault_root: str | None = None) -> None:
    """wiki/log.md 末尾追加一行：日期 | 笔记标题 | 新建N/合并M/冲突K。

    幂等：完全相同的行（同日期同笔记同计数）不重复追加。
    """
    path = os.path.join(wiki_root, "log.md")
    guard_vault_path(vault_root, os.path.abspath(path))
    line = f"{today} | {note_title} | 新建{created_n}/合并{updated_n}/冲突{conflict_n}"
    old = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            old = f.read()
    last = ""
    for ln in old.rstrip("\n").splitlines()[::-1]:
        if ln.strip():
            last = ln.strip()
            break
    if last == line:
        return
    header = ""
    if not old.strip():
        header = ("# 知识编译日志\n\n"
                  "> 每行一次编译：日期 | 笔记标题 | 新建N/合并M/冲突K\n\n")
    with open(path, "a", encoding="utf-8") as f:
        if header:
            f.write(header)
        elif old and not old.endswith("\n"):
            f.write("\n")
        f.write(line + "\n")


# ========== 编译入口 ==========

def compile_note(vault_root: str, note_path: str, note_title: str, note_content: str,
                 llm_call, confirm=None, wiki_dir: str = "wiki") -> dict:
    """编译一篇笔记 → 增量更新 wiki 概念/实体页 + index/log

    Args:
        vault_root: Vault 根目录（wiki 落在 {vault_root}/{wiki_dir}/）
        note_path: 源笔记路径（note_title 缺省时用其文件名兜底）
        note_title: 源笔记标题（用作 [[双链]] 目标与提取上下文）
        note_content: 源笔记正文（送 LLM 提取）
        llm_call: callable(prompt: str) -> str
        confirm: 可选 HITL 钩子 confirm(page_name, old_core, new_core)
                 -> "merge"|"conflict"|None；有矛盾时先问，
                 返回 "merge" 走正常合并，否则（含 None）写冲突块
        wiki_dir: wiki 目录名（默认 "wiki"）

    Returns:
        {"created": [页名...], "updated": [...], "conflicts": [...], "skipped": N}

    Raises:
        文件读写异常原样上抛，由调用方决定如何处置（本函数不做兜底）。
    """
    wiki_root = os.path.join(vault_root, wiki_dir)
    today = date.today().isoformat()
    if not note_title:
        note_title = os.path.splitext(os.path.basename(note_path or ""))[0] or "未命名笔记"

    entries = extract_knowledge(llm_call, note_title, note_content or "")
    created: list[str] = []
    updated: list[str] = []
    conflicts: list[str] = []
    skipped = 0
    touched: list[tuple[str, str]] = []

    for ent in entries:
        name, kind, new_core = ent["name"], ent["kind"], ent["one_line"]
        page_dir = os.path.join(wiki_root, KIND_DIR[kind])
        page_path = os.path.join(page_dir, f"{name}.md")

        if os.path.exists(page_path):
            with open(page_path, "r", encoding="utf-8") as f:
                page_text = f.read()
            # 幂等去重：本笔记对该页的贡献已在页内（新建核心/合并小节/冲突块）→ 跳过
            if _already_noted(page_text, note_title, new_core):
                skipped += 1
                continue
            old_core = _extract_core(page_text)
            decision = None
            if _is_conflict(old_core, new_core):
                # 有矛盾：给了 confirm 钩子先问（"merge" 走合并，其余写冲突块），没给默认冲突
                decision = confirm(name, old_core, new_core) if confirm else "conflict"
            if _is_conflict(old_core, new_core) and decision != "merge":
                page_text = (page_text.rstrip("\n") + "\n" +
                             _conflict_block(old_core, new_core,
                                             note_title, _read_sources(page_text), today))
                conflicts.append(name)
            else:
                page_text = page_text.rstrip("\n") + "\n" + \
                    _merge_section(note_title, new_core, today)
                updated.append(name)
            controlled_write(page_path, page_text, vault_root=vault_root,
                             overwrite=True, actor="knowledge-compile")
        else:
            controlled_write(page_path, _new_page(name, kind, new_core, note_title, today),
                             vault_root=vault_root, overwrite=True, actor="knowledge-compile")
            created.append(name)
        touched.append((name, kind))

    if touched:
        _update_index(wiki_root, touched, vault_root=vault_root)
        _append_log(wiki_root, note_title, len(created), len(updated),
                    len(conflicts), today, vault_root=vault_root)

    return {"created": created, "updated": updated,
            "conflicts": conflicts, "skipped": skipped}
