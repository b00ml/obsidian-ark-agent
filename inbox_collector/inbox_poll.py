#!/usr/bin/env python3
"""Agent Mail 邮件采集脚本（Ingest Pipeline 接收层）

功能:
  - 调 agently-cli 拉取收件箱邮件
  - 提取正文中的 URL → 域名白名单分流（bili / wechat_article）
  - SQLite 队列原子写入任务与去重标记（queue.jsonl/seen.txt 仅保留导入源）

用法:
  python inbox_poll.py [--config inbox_collector/config.json] [--dry-run]

关键约束（见 DESIGN-INGEST-PIPELINE.md §3.1 / §3.0.1）:
  - 必须设置 HOME / USERPROFILE 指向项目根目录（agently-cli 凭证位置）
  - 不调用任何 LLM（禁止事项 1）
  - 状态输出 [COLLECTOR] 前缀
"""
import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from queue_store import InboxQueueStore

# 项目根目录（本文件在 inbox_collector/ 下）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config.json")

# agently-cli 凭证目录：必须重定向 HOME/USERPROFILE 到项目根目录（踩坑记录 1）
AGENTLY_ENV = {"HOME": PROJECT_ROOT, "USERPROFILE": PROJECT_ROOT}

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")

# 域名规则: (域名片段, 类型, 标准URL重建函数)
DOMAIN_RULES = [
    ("bilibili.com", "bili", None),
    ("b23.tv", "bili", None),
    ("mp.weixin.qq.com", "wechat_article", None),
    ("weixin.qq.com", "wechat_article", None),
]


def print_status(tag: str, msg: str) -> None:
    """统一 [STATE] 状态输出"""
    print(f"[{tag}] {msg}", flush=True)


# ========== 配置 ==========

def load_config(config_path: str = DEFAULT_CONFIG) -> dict:
    """加载采集配置（配置驱动，无死配置）"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"采集配置文件不存在: {config_path}\n"
            f"请复制 {os.path.join(SCRIPT_DIR, 'config.example.json')} 为 config.json 并调整参数"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ========== agently-cli 封装 ==========

def _find_agently() -> str:
    """定位 agently-cli 可执行文件（PATH 优先，npm 全局目录兜底）"""
    exe = shutil.which("agently-cli")
    if exe:
        return exe
    # Windows npm 全局目录兜底（%APPDATA%\npm\agently-cli.cmd）
    npm_dir = os.path.join(os.environ.get("APPDATA", ""), "npm")
    for name in ("agently-cli.cmd", "agently-cli"):
        cand = os.path.join(npm_dir, name)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "COLLECTOR_CLI_MISSING agently-cli 未安装或不在 PATH。"
        "请先执行: npm install -g @tencent-qqmail/agently-cli，并完成 auth login"
    )


def run_agently(args: list[str]) -> dict:
    """执行 agently-cli 命令，返回解析后的 JSON

    前置条件: HOME/USERPROFILE 已指向项目根目录（凭证位置）
    Raises:
        RuntimeError: 含错误码（COLLECTOR_CLI_MISSING / COLLECTOR_CLI_FAIL）
    """
    env = os.environ.copy()
    env.update(AGENTLY_ENV)
    try:
        cli = _find_agently()
        proc = subprocess.run(
            [cli, *args],
            capture_output=True, text=True, encoding="utf-8",
            env=env, timeout=60,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "COLLECTOR_CLI_MISSING agently-cli 未安装或不在 PATH。"
            "请先执行: npm install -g @tencent-qqmail/agently-cli，并完成 auth login"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("COLLECTOR_CLI_FAIL agently-cli 调用超时")

    if proc.returncode != 0:
        raise RuntimeError(
            f"COLLECTOR_CLI_FAIL agently-cli 退出码 {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip()[:200]}"
        )

    out = proc.stdout
    start = out.find("{")
    if start == -1:
        raise RuntimeError("COLLECTOR_CLI_FAIL agently-cli 输出非 JSON")
    try:
        return json.loads(out[start:])
    except json.JSONDecodeError:
        raise RuntimeError("COLLECTOR_CLI_FAIL agently-cli 输出 JSON 解析失败")


def list_messages(limit: int) -> list[dict]:
    """拉取收件箱邮件列表（返回原始邮件对象）"""
    result = run_agently(["message", "+list", "--limit", str(limit)])
    if not result.get("ok"):
        raise RuntimeError(
            f"COLLECTOR_CLI_FAIL 拉取邮件失败: {result.get('error', {}).get('message', 'unknown')}"
        )
    return result.get("data", {}).get("data", [])


def read_message(message_id: str) -> dict:
    """读取单封邮件全文"""
    result = run_agently(["message", "+read", "--id", message_id])
    if not result.get("ok"):
        raise RuntimeError(
            f"COLLECTOR_CLI_FAIL 读取邮件失败: {result.get('error', {}).get('message', 'unknown')}"
        )
    return result.get("data", {})


# ========== URL 提取与分流 ==========

def strip_html(body: str) -> str:
    """HTML → 纯文本（用标准库，不引入依赖）"""
    # 移除 script/style 内容
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.IGNORECASE | re.DOTALL)
    # 提取 <a href="..."> 保留链接（分享邮件的链接往往在 href 中）
    body = re.sub(r'<a[^>]+href="([^"]+)"[^>]*>', r" \1 ", body, flags=re.IGNORECASE)
    # 剥离其余标签
    body = re.sub(r"<[^>]+>", " ", body)
    return html.unescape(body)


def extract_urls(text: str) -> list[str]:
    """从文本中提取所有 http(s) URL"""
    return URL_RE.findall(text)


def extract_bvid(url: str) -> str | None:
    """从 B站 URL 中提取 BV 号"""
    m = re.search(r"(BV[0-9A-Za-z]{10})", url)
    return m.group(1) if m else None


def normalize_url(url: str, content_type: str) -> str:
    """URL 归一化 → 标准可处理 URL（去 query/fragment）

    - bili: 统一为 https://www.bilibili.com/video/{BV号}
    - wechat_article: 保留 host + path（去 query）
    """
    url = url.split("#")[0]
    if content_type == "bili":
        bvid = extract_bvid(url)
        if bvid:
            return f"https://www.bilibili.com/video/{bvid}"
    # 公众号: 去 query 保留 path
    m = re.match(r"(https?://[^/]+(/[^?]*)?)", url)
    if m:
        return m.group(1)
    return url.split("?")[0]


def fingerprint(url: str, content_type: str) -> str:
    """URL 去重指纹（同一链接重复转发丢弃）"""
    if content_type == "bili":
        bvid = extract_bvid(url)
        if bvid:
            return f"bili:{bvid}"
    # 其他: 归一化后的完整 URL
    return f"{content_type}:{normalize_url(url, content_type)}"


def classify(url: str) -> tuple[str, str] | None:
    """域名白名单分流: (content_type, normalized_url) 或 None（不在白名单）"""
    host = re.match(r"https?://([^/]+)", url)
    if not host:
        return None
    host = host.group(1).lower().lstrip("www.")
    for domain, content_type, _ in DOMAIN_RULES:
        if host == domain or host.endswith("." + domain) or host == domain.replace("www.", ""):
            return content_type, normalize_url(url, content_type)
    return None


def _host_of(url: str) -> str:
    """提取 URL 的 host（去 www. 前缀）"""
    m = re.match(r"https?://([^/]+)", url)
    if not m:
        return ""
    return m.group(1).lower().lstrip("www.")


def _in_whitelist(url: str, whitelist: list[str]) -> bool:
    """host 级白名单校验（host == d 或以 .d 结尾，避免子串误判）"""
    if not whitelist:
        return True
    host = _host_of(url)
    return any(host == d or host.endswith("." + d) for d in whitelist)


# ========== 队列 ==========

def open_queue_store(config: dict, queue_path: str, seen_path: str) -> InboxQueueStore:
    """Open the configured SQLite queue store."""
    db_path = config.get("queue_db_path", "inbox/queue.db")
    if not os.path.isabs(db_path):
        db_path = os.path.join(PROJECT_ROOT, db_path)
    return InboxQueueStore(db_path, queue_path=queue_path, seen_path=seen_path)


# ========== 主流程 ==========

def poll(config: dict, dry_run: bool = False) -> int:
    """执行一轮采集，返回新入队任务数"""
    queue_path = config.get("queue_path", "inbox/queue.jsonl")
    seen_path = config.get("seen_path", "inbox/seen.txt")
    max_fetch = int(config.get("max_fetch", 20))
    whitelist = config.get("whitelist_domains", [])

    # 路径解析（相对项目根目录）
    if not os.path.isabs(queue_path):
        queue_path = os.path.join(PROJECT_ROOT, queue_path)
    if not os.path.isabs(seen_path):
        seen_path = os.path.join(PROJECT_ROOT, seen_path)

    store = open_queue_store(config, queue_path, seen_path)
    store.migrate_legacy()
    seen = store.seen_markers()
    new_tasks = 0
    dup_count = 0

    messages = list_messages(max_fetch)
    print_status("COLLECTOR", f"拉取 {len(messages)} 封邮件")

    processed_ids: list[str] = []
    for msg in messages:
        msg_id = msg.get("message_id", "")
        if not msg_id or msg_id in seen:
            dup_count += 1
            continue

        # 读全文提取 URL
        try:
            detail = read_message(msg_id)
        except RuntimeError as e:
            print_status("COLLECTOR", f"[ERROR] {e}")
            continue

        body = detail.get("body", "") or ""
        if detail.get("body_format") == "HTML":
            text = strip_html(body)
        else:
            text = body
        subject = detail.get("subject", "") or msg.get("subject", "")

        # 从正文 + 主题提取 URL 并分流
        found = False
        for url in extract_urls(text) + extract_urls(subject):
            cls = classify(url)
            if not cls:
                continue
            # 白名单二次校验（配置可缩小范围）
            content_type, norm_url = cls
            if not _in_whitelist(norm_url, whitelist):
                continue
            fp = fingerprint(norm_url, content_type)
            if fp in seen:
                dup_count += 1
                continue

            task = {
                "id": f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{new_tasks:03d}",
                "type": content_type,
                "url": norm_url,
                "status": "pending",
                "retry_count": 0,
                "received_at": detail.get("created_at", ""),
                "email_subject": subject,
                "source": "agent_mail",
                "error": None,
            }
            print_status("COLLECTOR", f"新增任务 {content_type} {norm_url}")
            if not dry_run:
                inserted = store.enqueue(task, dedupe_key=fp, markers=[fp, msg_id])
                if inserted:
                    new_tasks += 1
                else:
                    dup_count += 1
                seen.add(fp)
                seen.add(msg_id)
            else:
                new_tasks += 1
            found = True
            break  # 一封邮件只入一条任务（取第一个白名单 URL）

        if not found and not dry_run:
            # 无匹配 URL 的邮件也标记已处理，避免重复扫描
            store.mark_seen([msg_id])
            seen.add(msg_id)

    print_status("COLLECTOR", f"新增 {new_tasks} 条任务，去重跳过 {dup_count}")
    return new_tasks


def main() -> None:
    # Windows 终端编码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Agent Mail 邮件采集脚本")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要入队的任务，不写队列/去重文件")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)

    try:
        poll(config, dry_run=args.dry_run)
    except RuntimeError as e:
        print_status("COLLECTOR", f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
