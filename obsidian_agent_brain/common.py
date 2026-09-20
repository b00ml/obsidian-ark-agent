"""公共工具：配置加载、路径解析、目录权限、wikilink/frontmatter 解析。

遵循 AGENTS.md 目录权限约束：
- raw/ 只读（vault_write 拒绝）
- Inbox/ 临时（默认产物目录）
- wiki/ 成品
- .agent-brain/ 知识库"大脑"数据目录（不进 raw/）
"""
import json
import os
import re
import sys
from contextlib import contextmanager

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(PACKAGE_DIR, "config.json")

# Vault 内跳过搜索/统计的隐藏或系统目录
SKIP_DIRS = {".obsidian", ".trash", ".agent-brain", ".dashboard-backup",
             "media-lib", ".git", "node_modules"}

# Obsidian wikilink: [[target]] / [[target#heading]] / [[target|alias]] / [[#heading]]
LINK_RE = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]")


def load_config(config_path: str | None = None) -> dict:
    """加载 server 配置（路径类，不含密钥）"""
    path = config_path or DEFAULT_CONFIG
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"配置文件不存在: {path}\n"
            f"请复制 config.example.json 为 config.json 并调整参数"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_paths(config: dict) -> None:
    """把共享处理器和 agentlab 包根加入 sys.path。"""
    for key in ("project_root",):
        p = config.get(key)
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    for sub in ("bili_summarizer", "inbox_collector", "agentlab"):
        d = os.path.join(config.get("project_root", ""), sub)
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


def vault_root(config: dict) -> str:
    """Vault 绝对路径；配置缺失时显式报错。

    为什么不让它退化成 `abspath("")`：那等于"当前工作目录当 Vault"，而 brain 的部分
    调用方会切工作目录（`tools_bili._run_in_bili` → `chdir(<project_root>/bili_summarizer)`，
    见 tools_bili.py:89-94）。实测复现：cfg 里没有 vault_path 时 `memory_query` 把
    `<cwd>/.agent-brain/memory/sessions.sqlite` 当记忆库读写，**不报错**，于是在仓库里
    留下一个 `bili_summarizer/.agent-brain/`（2026-09-11 实际发生过）。
    """
    raw = str(config.get("vault_path") or "").strip()
    if not raw:
        raise ValueError(
            "config 缺少 vault_path：brain 工具会把当前工作目录当成 Vault（并在其下"
            "生成 .agent-brain/）。请在 config.json 显式配置 vault_path。"
        )
    return os.path.abspath(raw)


def brain_dir(config: dict) -> str:
    return os.path.join(vault_root(config), config.get("brain_dir", ".agent-brain"))


def resolve_note_path(config: dict, rel_path: str) -> str:
    """把 Vault 相对路径解析为绝对路径（防目录穿越）。

    rel_path 可为 "Inbox/xx.md" / "xx"（裸文件名默认补 .md）。
    """
    root = os.path.abspath(vault_root(config))
    p = rel_path.replace("\\", "/").strip().lstrip("./")
    if not p:
        raise ValueError("路径不能为空")
    if not p.lower().endswith(".md") and not os.path.splitext(p)[1]:
        p = p + ".md"
    full = os.path.abspath(os.path.join(root, p))
    if not full.startswith(root + os.sep) and full != root:
        raise ValueError(f"路径越界 Vault 根目录: {rel_path}")
    return full


def assert_writable(config: dict, rel_path: str) -> None:
    """写入权限校验：raw/ 只读拒绝，其余目录允许（产物默认 Inbox/）"""
    p = rel_path.replace("\\", "/").strip().lstrip("./")
    first = p.split("/", 1)[0].lower()
    if first == "raw":
        raise PermissionError(f"raw/ 目录只读，禁止写入: {rel_path}")


@contextmanager
def chdir(path: str):
    """临时切换工作目录（B站/公众号处理器依赖相对临时文件路径）"""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# A2/OPT-227：记忆归档区（ark/memory/archive/）默认从通用 Vault 检索排除——
# 归档 = 不可召回历史；与 agentlab rag/vector_index._EXCLUDE_DIRS 的判断保持一致。
MEMORY_ARCHIVE_REL = "ark/memory/archive"


def list_notes(config: dict) -> list[str]:
    """遍历 Vault 内全部 .md 笔记（相对路径，跳过隐藏/系统目录与记忆归档区）"""
    root = vault_root(config)
    notes: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel_dir == MEMORY_ARCHIVE_REL or rel_dir.startswith(MEMORY_ARCHIVE_REL + "/"):
            dirnames[:] = []  # 归档区整枝剪掉，不再深入
            continue
        for fn in filenames:
            if fn.lower().endswith(".md"):
                full = os.path.join(dirpath, fn)
                notes.append(os.path.relpath(full, root).replace("\\", "/"))
    return notes


def read_note(config: dict, rel_path: str) -> str:
    full = resolve_note_path(config, rel_path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"笔记不存在: {rel_path} -> {full}")
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def parse_frontmatter(content: str) -> dict:
    """解析 YAML frontmatter（失败返回空 dict，不抛错）"""
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return {}
    try:
        import yaml
        data = yaml.safe_load(m.group(1)) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def extract_links(content: str) -> list[str]:
    """提取笔记内全部 [[wikilink]] 目标名（去空白/去别名）"""
    return [g.strip() for g in LINK_RE.findall(content) if g.strip()]


def note_title(content: str, fallback: str = "") -> str:
    """取笔记标题：frontmatter title > 首个 # 标题 > fallback"""
    fm = parse_frontmatter(content)
    t = str(fm.get("title", "") or "").strip()
    if t:
        return t
    m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return fallback


def tag_snippet(content: str, keyword: str, radius: int = 120) -> str:
    """命中关键词附近的上下文摘要（去换行压缩）"""
    idx = content.lower().find(keyword.lower())
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(content), idx + len(keyword) + radius)
    snippet = re.sub(r"\s+", " ", content[start:end]).strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""
    return f"{prefix}{snippet}{suffix}"
