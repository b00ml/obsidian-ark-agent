"""Obsidian CLI 接入工具（优化设计文档4.0 执行线 #3）。

通过官方 Obsidian CLI（1.12+；安装后在 设置→通用→命令行界面 注册）执行
只读检索与原子变更：rename/move/property 走 Obsidian 自身的链接图谱引擎，
自动更新全库 [[wikilink]]，等价于在 Obsidian 内重命名/移动文件的效果。

安全白名单：只接入「只读检索 + 原子属性 + move/rename」五类能力；
绝不接入 eval（可执行任意 JS）、dev:cdp（调试协议）、delete 等高危命令。
raw/ 只读约束与既有工具一致（rename/move/property_set 拒绝 raw/ 路径）。

优雅降级：每个工具开头先探测 CLI 可用性（shutil.which + --version，每次探测
毫秒级开销可接受）；CLI 缺失/未注册/执行失败一律用返回值表达
（status=degraded / error），绝不抛异常，不影响既有文件系统路工具。

子命令语法以官方 `obsidian --help`（https://obsidian.md/help/cli）为准，
全部集中在模块级 COMMANDS 常量表，官方语法变动时一处修正。
"""
import json
import os
import shutil
import subprocess

from common import assert_writable

# ---- 子命令常量表（官方 obsidian --help，1.12+；语法变动在此一处修正） ----
# 参数形态：key=value 键值对 + 布尔旗标（无 -- 前缀）；含空格的值作为单个 argv
# 传入即可（subprocess 列表参数天然免引号）。vault= 可作首个参数显式指定库。
COMMANDS = {
    "links": "links",                  # 列出笔记出向链接（定位 file=/path=）
    "backlinks": "backlinks",          # 列出引用笔记的来源（官方支持 format=json）
    "unresolved": "unresolved",        # 全库未解析链接（官方支持 format=json）
    "orphans": "orphans",              # 无入向链接的笔记（另有 total 旗标仅取计数）
    "deadends": "deadends",            # 无出向链接的笔记（另有 total 旗标）
    "property_read": "property:read",  # name=<属性名>（另有 file=/path= 定位）
    "property_set": "property:set",    # name=<属性名> value=<值>（可选 type=text|list|number|checkbox|date）
    "rename": "rename",                # name=<新文件名>（保持扩展名，自动更新全库引用）
    "move": "move",                    # to=<目标路径>（移动/改名，自动更新双链）
}

# 官方文档确认支持 format=json 的子命令（其余保持默认输出，解析失败回退原文）
_JSON_FORMAT_COMMANDS = {"backlinks", "unresolved"}

_PROBE_TIMEOUT_DEFAULT = 10  # obsidian_available 的 --version 探测默认超时
_RUN_TIMEOUT_DEFAULT = 15    # 业务子命令默认超时（与 config.example.json 对齐）


def _cli_bin(config: dict) -> str:
    """CLI 可执行名（config.obsidian_cli.bin，默认 obsidian）。"""
    return config.get("obsidian_cli", {}).get("bin", "obsidian")


def _cli_timeout(config: dict, default: float) -> float:
    """CLI 超时秒数（config.obsidian_cli.timeout，缺省用调用方默认值）。"""
    return config.get("obsidian_cli", {}).get("timeout", default)


def obsidian_available(config: dict) -> dict:
    """探测 Obsidian CLI 是否可用：shutil.which 找到 + `--version` 能应答。

    返回 {"available": bool, "version": str|None}；任何异常都归为不可用，
    绝不抛出（探测失败本身就是"不可用"这一事实的表达）。
    """
    bin_ = _cli_bin(config)
    try:
        resolved = shutil.which(bin_)
        if not resolved:
            return {"available": False, "version": None}
        proc = subprocess.run([resolved, "--version"], capture_output=True,
                              timeout=_cli_timeout(config, _PROBE_TIMEOUT_DEFAULT))
        if proc.returncode != 0:
            return {"available": False, "version": None}
        version = (proc.stdout or b"").decode("utf-8", errors="replace").strip() or None
        return {"available": True, "version": version}
    except Exception:  # noqa: BLE001 探测失败 = 不可用
        return {"available": False, "version": None}


def _run(config: dict, args: list[str], timeout: float | None = None) -> str:
    """执行 CLI 子命令并返回 stdout（utf-8 errors=replace 解码）。

    先经 shutil.which 解析真实路径（Windows 下 CLI 常为 .cmd shim，
    带扩展名的完整路径才能被 CreateProcess 正确拉起）。
    超时/二进制缺失/非零退出抛 RuntimeError，由上层工具转为 error 返回值。
    """
    timeout = timeout if timeout is not None else _cli_timeout(config, _RUN_TIMEOUT_DEFAULT)
    resolved = shutil.which(_cli_bin(config)) or _cli_bin(config)
    try:
        proc = subprocess.run([resolved, *args], capture_output=True, timeout=timeout)
    except FileNotFoundError as e:
        raise RuntimeError(f"obsidian CLI 不存在: {resolved}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"obsidian CLI 执行超时(>{timeout}s): {args[0] if args else ''}") from e
    err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"obsidian CLI 退出码 {proc.returncode}: {err or out}")
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def _run_safe(config: dict, args: list[str]) -> tuple[str | None, str | None]:
    """执行子命令并把 RuntimeError 转为返回值 (输出, None) / (None, 错误信息)。"""
    try:
        return _run(config, args), None
    except RuntimeError as e:
        return None, str(e)


def _parse_output(out: str):
    """尝试 JSON 解析 CLI 输出；非 JSON / 空输出原样返回文本。"""
    s = (out or "").strip()
    if not s:
        return ""
    try:
        return json.loads(s)
    except ValueError:
        return s


def _count_of(value) -> int | None:
    """列表取长度；文本/空输出无法计数时返回 None。"""
    return len(value) if isinstance(value, list) else None


def _target_args(note: str) -> list[str]:
    """把笔记引用转成 CLI 定位参数。

    官方规则：file= 按 [[wikilink]] 同款规则解析（不含目录/扩展名）；
    path= 为 Vault 根相对精确路径。含路径分隔符或 .md 后缀 → path=，否则 file=。
    """
    n = note.replace("\\", "/").strip().lstrip("./")
    if "/" in n or n.lower().endswith(".md"):
        return [f"path={n}"]
    return [f"file={n}"]


def _degraded(tool: str, **extra) -> dict:
    """CLI 不可用时的统一降级返回（绝不抛异常，文件系统路工具仍可用）。"""
    return {"status": "degraded", "tool": tool,
            "reason": "obsidian cli not found",
            "hint": "需 Obsidian 1.12+ 并在 设置→通用→命令行界面 注册 CLI，"
                    "或检查 config.obsidian_cli.bin",
            **extra}


def _guard_available(config: dict, tool: str, **extra) -> dict | None:
    """探测可用性；不可用返回 degraded dict，可用返回 None。"""
    if not obsidian_available(config).get("available"):
        return _degraded(tool, **extra)
    return None


def _fmt_value(value) -> str:
    """CLI value= 一律传字符串；bool 转 true/false 对齐 checkbox 语义。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def obsidian_links(config: dict, note: str) -> dict:
    """指定笔记的双链检索：出向 links + 入向 backlinks（Obsidian 官方图谱结果）。

    返回 {"status": "ok", "note", "outbound", "backlinks"}；单项失败不影响
    另一项（记入 errors，status=partial / error）。CLI 不可用返回 degraded。
    """
    deg = _guard_available(config, "obsidian_links", note=note)
    if deg:
        return deg
    target = _target_args(note)
    outbound, backlinks, errors = None, None, []
    out, err = _run_safe(config, [COMMANDS["links"], *target])
    if err:
        errors.append(f"links: {err}")
    else:
        outbound = _parse_output(out)
    out, err = _run_safe(config, [COMMANDS["backlinks"], *target, "format=json"])
    if err:
        errors.append(f"backlinks: {err}")
    else:
        backlinks = _parse_output(out)
    if outbound is None and backlinks is None:
        status = "error"
    elif errors:
        status = "partial"
    else:
        status = "ok"
    result = {"status": status, "note": note,
              "outbound": outbound, "backlinks": backlinks}
    if errors:
        result["errors"] = errors
    return result


def obsidian_health(config: dict) -> dict:
    """全库链接体检：unresolved（未解析链接）/ orphans（无入向）/ deadends（无出向）。

    三项 best-effort 独立执行，单项失败记入 errors 不影响其余。
    与 tools_vault.vault_health（纯文件系统解析）互补：CLI 走 Obsidian 官方
    图谱引擎，可正确处理 heading/block 链接等文件系统路无法判定的情形。
    """
    deg = _guard_available(config, "obsidian_health")
    if deg:
        return deg
    sections: dict[str, object] = {}
    errors: list[str] = []
    for key, cmd in (("unresolved", COMMANDS["unresolved"]),
                     ("orphans", COMMANDS["orphans"]),
                     ("deadends", COMMANDS["deadends"])):
        extra = ["format=json"] if cmd in _JSON_FORMAT_COMMANDS else []
        out, err = _run_safe(config, [cmd, *extra])
        if err:
            errors.append(f"{key}: {err}")
        else:
            sections[key] = _parse_output(out)
    if not sections:
        status = "error"
    elif errors:
        status = "partial"
    else:
        status = "ok"
    result = {"status": status,
              "unresolved": sections.get("unresolved"),
              "orphans": sections.get("orphans"),
              "deadends": sections.get("deadends"),
              "counts": {k: _count_of(v) for k, v in sections.items()}}
    if errors:
        result["errors"] = errors
    return result


def obsidian_rename(config: dict, old_path: str, new_name: str) -> dict:
    """重命名笔记（CLI rename 自动更新全库引用，保持双链不断）。

    new_name 为纯文件名（目录/扩展名由 CLI 规则处理，传 .md 后缀会被剥掉）。
    raw/ 只读拒绝；含路径分隔符的名字拒绝（移动目录请用 obsidian_move）。
    一律以返回值表达错误（status=error），不抛异常。
    """
    tool = "obsidian_rename"
    deg = _guard_available(config, tool, old_path=old_path, new_name=new_name)
    if deg:
        return deg
    try:
        assert_writable(config, old_path)
    except PermissionError as e:
        return {"status": "error", "tool": tool, "reason": str(e)}
    name = (new_name or "").strip()
    if name.lower().endswith(".md"):  # CLI rename 保持原扩展名，这里剥掉避免双重后缀
        name = name[:-3]
    if not name:
        return {"status": "error", "tool": tool, "reason": "new_name 不能为空"}
    if "/" in name or "\\" in name:
        return {"status": "error", "tool": tool,
                "reason": "new_name 应为纯文件名（不含路径）；移动目录请用 obsidian_move"}
    out, err = _run_safe(config, [COMMANDS["rename"], *_target_args(old_path),
                                  f"name={name}"])
    if err:
        return {"status": "error", "tool": tool, "old": old_path,
                "new_name": name, "reason": err}
    return {"status": "ok", "old": old_path, "new_name": name,
            "links_updated": True, "output": _parse_output(out)}


def obsidian_move(config: dict, path: str, folder: str) -> dict:
    """移动笔记到指定目录（CLI move 保持双链，自动更新全库引用）。

    目标路径 = folder + 原文件名；源与目标均受 raw/ 只读约束。
    """
    tool = "obsidian_move"
    deg = _guard_available(config, tool, path=path, folder=folder)
    if deg:
        return deg
    folder = (folder or "").strip().replace("\\", "/").strip("/")
    path_n = (path or "").replace("\\", "/").strip().lstrip("./")
    if not folder or not path_n or os.path.basename(path_n) in ("", ".", ".."):
        return {"status": "error", "tool": tool, "reason": "path 与 folder 均不能为空"}
    dest = f"{folder}/{os.path.basename(path_n)}"
    try:
        assert_writable(config, path_n)
        assert_writable(config, dest)
    except PermissionError as e:
        return {"status": "error", "tool": tool, "path": path, "to": dest,
                "reason": str(e)}
    out, err = _run_safe(config, [COMMANDS["move"], *_target_args(path),
                                  f"to={dest}"])
    if err:
        return {"status": "error", "tool": tool, "from": path, "to": dest,
                "reason": err}
    return {"status": "ok", "from": path, "to": dest, "links_updated": True,
            "output": _parse_output(out)}


def obsidian_property_set(config: dict, path: str, key: str, value) -> dict:
    """frontmatter 原子属性更新（CLI property:set，只动属性不动正文）。

    value 接受 str/数字/布尔（bool 转 true/false）；类型由 CLI 按库设置推断。
    raw/ 只读拒绝；错误以 status=error 返回值表达。
    """
    tool = "obsidian_property_set"
    deg = _guard_available(config, tool, path=path, key=key)
    if deg:
        return deg
    try:
        assert_writable(config, path)
    except PermissionError as e:
        return {"status": "error", "tool": tool, "reason": str(e)}
    key = (key or "").strip()
    if not key:
        return {"status": "error", "tool": tool, "reason": "key 不能为空"}
    out, err = _run_safe(config, [COMMANDS["property_set"], *_target_args(path),
                                  f"name={key}", f"value={_fmt_value(value)}"])
    if err:
        return {"status": "error", "tool": tool, "path": path, "key": key,
                "reason": err}
    return {"status": "ok", "path": path, "key": key, "value": _fmt_value(value),
            "output": _parse_output(out)}
