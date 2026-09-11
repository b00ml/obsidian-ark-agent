"""Vault 工具：读 / 写 / 补丁 / 搜索 / wikilink 图 / 全库统计。

全部走 common 的目录权限约束（raw/ 只读、默认 Inbox/）。
"""
import os
from collections import Counter, defaultdict

from common import (SKIP_DIRS, assert_writable, extract_links, list_notes,
                    note_title, parse_frontmatter, read_note, resolve_note_path,
                    tag_snippet, vault_root)
from vault_gateway import VaultGateway


def vault_read(config: dict, path: str) -> str:
    """读取 Vault 内 Markdown 笔记全文。"""
    return read_note(config, path)


def vault_write(config: dict, path: str, content: str) -> dict:
    """写入/覆盖 Vault 笔记。

    path 为 Vault 相对路径；裸文件名（不含 /）默认写入 Inbox/。
    raw/ 只读拒绝；父目录自动创建。
    """
    return VaultGateway(config).write(path, content)


def vault_patch(config: dict, path: str, old: str, new: str,
                expected_revision: str | None = None) -> dict:
    """精确替换笔记中的某段文本（润色用）。

    要求 old 在文件中唯一；未命中或不唯一时报错，不做覆盖写。
    """
    return VaultGateway(config).patch(path, old, new,
                                      expected_revision=expected_revision)


def _note_tags(content: str) -> list[str]:
    fm = parse_frontmatter(content)
    tags = fm.get("tags", [])
    if isinstance(tags, str):
        return [tags]
    return [str(t) for t in tags if t] if isinstance(tags, list) else []


def vault_search(config: dict, keyword: str, limit: int = 20) -> dict:
    """关键词检索全库（标题 + 正文 + frontmatter tag）。

    返回命中笔记路径 + 摘要 + tags + 命中次数，**按相关性排序**。

    排序为什么必须改（2026-09-11 全量基线实测）：旧实现是
    `key=(0 if 标题命中 else 1, path)`——正文命中一律**按路径字母序**排，
    相关性完全不参与。实际后果：`q-memory-recall` 一条任务里 agent 打了 25 次
    不同关键词的搜索、13 次 `vault_read`（46 次工具调用 / 373k token），因为
    排序给不出"哪条更相关"，只能逐条读回来看。改为
    `(标题命中 > tag 命中 > 正文命中, 命中次数降序, 路径)`，并回传 `hits` 次数。
    """
    kw = keyword.strip().lower()
    if not kw:
        raise ValueError("keyword 不能为空")
    hits: list[dict] = []
    for rel in list_notes(config):
        try:
            content = read_note(config, rel)
        except Exception:
            continue
        fm = parse_frontmatter(content)
        title = note_title(content, rel)
        tags = _note_tags(content)
        title_hit = kw in title.lower()
        tag_hit = kw in " ".join(tags).lower()
        body = content.lower()
        count = body.count(kw)
        body_idx = body.find(kw)
        if title_hit or tag_hit or body_idx >= 0:
            hits.append({
                "path": rel,
                "title": title,
                "tags": tags,
                "type": fm.get("type", ""),
                "hits": count,
                # 0=标题命中 / 1=tag 命中 / 2=仅正文命中；排完序即删，不污染返回结构
                "rank": 0 if title_hit else (1 if tag_hit else 2),
                "snippet": ("标题命中" if (title_hit or tag_hit) and body_idx < 0
                            else tag_snippet(content, kw)),
            })
    # 相关性排序：命中位置（标题>tag>正文）→ 命中次数降序 → 路径（稳定）
    hits.sort(key=lambda h: (h["rank"], -h["hits"], h["path"]))
    for h in hits:
        h.pop("rank", None)
    return {"query": keyword, "total": len(hits), "results": hits[:limit]}


def vault_graph(config: dict, note: str, depth: int = 1) -> dict:
    """wikilink 邻接图：从指定笔记出发，遍历 [[wikilink]] 关联。

    depth=1 返回直接邻居；depth=2 额外返回邻居的邻居（含被引/反引关系）。
    """
    root = vault_root(config)
    target = note.replace("\\", "/").strip().lstrip("./")
    if "/" not in target and not target.lower().endswith(".md"):
        target = target + ".md"
    # 定位笔记：精确路径优先，否则按文件名匹配
    full = None
    if os.path.exists(os.path.join(root, target)):
        full = os.path.join(root, target)
    else:
        base = os.path.basename(target).lower()
        for rel in list_notes(config):
            if os.path.basename(rel).lower() == base:
                full = os.path.join(root, rel)
                target = rel
                break
    if not full or not os.path.exists(full):
        raise FileNotFoundError(f"笔记不存在: {note}")

    def _links_of(rel_path: str) -> list[str]:
        try:
            return extract_links(read_note(config, rel_path))
        except Exception:
            return []

    outbound = _links_of(target)
    nodes: dict[str, dict] = {target: {"title": note_title(read_note(config, target), target),
                                       "links": outbound}}
    if depth >= 2:
        for link in outbound:
            link = link.replace("\\", "/").lstrip("./")
            if not link.lower().endswith(".md"):
                link += ".md"
            if link in nodes:
                continue
            if os.path.exists(os.path.join(root, link)):
                nodes[link] = {"title": note_title(read_note(config, link), link),
                               "links": _links_of(link)}
    # 反引：库内哪些笔记引用了目标
    inbound = [rel for rel in list_notes(config)
               if rel != target and target.split("/")[-1].replace(".md", "")
               in {l.split("/")[-1] for l in _links_of(rel)}]
    return {"note": target, "outbound": outbound, "inbound": inbound[:30],
            "nodes": {k: v for k, v in list(nodes.items())[:50]}}


def vault_scan(config: dict) -> dict:
    """全库统计：主题分布 / tag 聚合 / wikilink 密度（供产出链路分析）。"""
    total = 0
    by_type: Counter = Counter()
    tags: Counter = Counter()
    linked = 0
    link_count = 0
    by_dir: Counter = Counter()
    recent: list[dict] = []

    for rel in list_notes(config):
        try:
            content = read_note(config, rel)
        except Exception:
            continue
        total += 1
        fm = parse_frontmatter(content)
        t = str(fm.get("type", "") or "note")
        by_type[t] += 1
        by_dir[rel.split("/", 1)[0] if "/" in rel else "root"] += 1
        for tag in _note_tags(content):
            tags[tag] += 1
        links = extract_links(content)
        if links:
            linked += 1
            link_count += len(links)
        mtime = os.path.getmtime(os.path.join(vault_root(config), rel))
        recent.append({"path": rel, "title": note_title(content, rel),
                       "type": t, "mtime": mtime})

    recent.sort(key=lambda r: r["mtime"], reverse=True)
    return {
        "total_notes": total,
        "by_type": dict(by_type.most_common()),
        "by_dir": dict(by_dir.most_common()),
        "top_tags": dict(tags.most_common(30)),
        "linked_notes": linked,
        "total_wikilinks": link_count,
        "link_density": round(link_count / total, 2) if total else 0,
        "recent_notes": [{**r, "mtime": _fmt_mtime(r["mtime"])}
                         for r in recent[:15]],
    }


def _resolve_link(link: str, notes: list[str]) -> str | None:
    """把 wikilink 目标解析到实际笔记（相对路径）；解析不到返回 None（死链）。

    匹配优先级：相对路径精确命中（去 .md）→ 裸文件名首命中。大小写不敏感。
    """
    t = link.strip().lstrip("./").replace("\\", "/")
    if t.lower().endswith(".md"):
        t = t[:-3]
    for rel in notes:
        rel_noext = rel[:-3] if rel.lower().endswith(".md") else rel
        if rel_noext.lower() == t.lower():
            return rel
    base = os.path.basename(t).lower()
    for rel in notes:
        rel_noext = rel[:-3] if rel.lower().endswith(".md") else rel
        if os.path.basename(rel_noext).lower() == base:
            return rel
    return None


# 健康检查清单的回传样本上限：**计数在前、清单只要样本**。
# 为什么必须截样（2026-09-11 两轮全量基线实测）：原先回传完整清单，真实 Vault 一次返回
# 135 条死链 + 48 条孤儿 + 68 条无引用笔记，单次远超 agentlab 的
# `max_tool_result_chars=12000`，被头尾截断后 `broken_links_count` / `orphan_count`
# 落进"被省略的中段"——agent 与评审都看不到，于是把答案里**正确的**数字判成编造
# （`q-orphan-notes` 连续两轮被判 3 条幻觉，逐条核对后全是假阳性）。
HEALTH_SAMPLE_LIMIT = 20


def vault_health(config: dict, sample: int = HEALTH_SAMPLE_LIMIT) -> dict:
    """结构性健康检查：死链（wikilink 指向不存在笔记）+ 孤儿笔记。

    对齐 claude-obsidian 的"8 类健康检查"中可离线判定的结构项（语义/矛盾类
    需 LLM，此处不做）；作为 P5 笔记质量工具进 brain，不侵入核心 loop。
    返回：broken_links（含来源笔记）、orphan_notes（无入向也无出向的孤立岛）、
    unlinked_notes（无任何笔记引用它，仅作参考）、health_score（出向链接中可解析占比）。

    **载荷有界**：三类清单各最多回传 `sample` 条（默认 20），被截掉的部分在
    `sampled` 里给出 `returned/total`，需要全量清单时按 `sample` 参数再取。
    计数（`*_count`）固定排在清单之前，任何下游截断都不会先丢掉它们。
    """
    notes = list_notes(config)
    outbound: dict[str, list[str]] = {n: [] for n in notes}
    inbound: dict[str, set[str]] = {n: set() for n in notes}
    broken: list[dict] = []

    for rel in notes:
        try:
            content = read_note(config, rel)
        except Exception:
            continue
        links = extract_links(content)
        outbound[rel] = links
        for l in links:
            target = _resolve_link(l, notes)
            if target is None:
                broken.append({"note": rel, "link": l})
            elif target != rel:
                inbound[target].add(rel)

    orphan_notes = sorted(
        rel for rel in notes if not outbound.get(rel) and not inbound.get(rel)
    )
    unlinked = sorted(rel for rel in notes if not inbound.get(rel))
    total_outbound = sum(len(v) for v in outbound.values())
    try:
        limit = max(0, min(int(sample), 200))
    except (TypeError, ValueError):
        limit = HEALTH_SAMPLE_LIMIT
    lengths = {"broken_links": len(broken), "orphan_notes": len(orphan_notes),
               "unlinked_notes": len(unlinked)}
    return {
        "total_notes": len(notes),
        "health_score": round(1 - len(broken) / max(1, total_outbound), 3),
        # 计数在前：下游按长度截断时先丢清单、不丢计数
        "broken_links_count": len(broken),
        "orphan_count": len(orphan_notes),
        "unlinked_count": len(unlinked),
        "broken_links": broken[:limit],
        "orphan_notes": orphan_notes[:limit],
        "unlinked_notes": unlinked[:limit],
        "sample_size": limit,
        "sampled": {k: {"returned": limit, "total": n}
                    for k, n in lengths.items() if n > limit},
    }


def _fmt_mtime(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
