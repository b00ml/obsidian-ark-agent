"""鉴权垢（serve 拆职责：S6，从 serve.py 抽出）。

- check_bearer：Authorization 头是否匹配 Bearer token（白盒、单点）。
- serve_confirm：web 链路的工具确认策略（F4/fail-closed：danger 仅白名单放行、
  write 仅放行被认可的 Vault/memory 写入器；其余一律拒绝）。

write 放行依赖 brain wrapper 的只读守卫（raw/、templates/）：serve_confirm 只做
"这个写工具是否被认可"，真正的路径安全边界由大脑侧的写入守卫兜底，避免越权写库到
只读目录。
"""
from __future__ import annotations

_BEARER = "Bearer "

# CRT/web 链路放行的 write 工具：产物落 Vault（Inbox/wiki）或 memory，raw/、templates/
# 由 brain wrapper 只读守卫拒绝。其余未列出的 write 工具默认仍 fail-closed。
# 可用配置 `write_allowlist` 覆盖（为空则视为不覆盖、用本默认集）。
_DEFAULT_WRITE_ALLOW = {
    "vault_write",
    "vault_patch",
    "bili_visual",
    "bili_screenshot",
    "article_summarize",
    "memory_commit",
}


def check_bearer(authorization: str, expected: str) -> bool:
    """校验 Authorization 头：须为 `Bearer <token>` 且 token 精确匹配 expected。

    expected 为空视为未鉴权（返回 False，由上层 fail-closed 处理），不在此兜底。
    """
    if not expected or not authorization.startswith(_BEARER):
        return False
    return authorization[len(_BEARER):].strip() == expected


def serve_confirm(cfg, tool, prompt: str) -> bool:
    """serve 链路的工具确认（F4 修复：web 主链路此前绕过 HITL 直通）。

    web 场景默认使用 risk_based 策略：
    - danger：仅显式 danger_allowlist 内的工具策略放行；
    - write（被认可的 Vault/memory 写入器）：放行（路径边界由 brain wrapper 守卫）；
    - 其余（未认可 write、未知）一律拒绝——避免越权写库。
    `approval_mode=allow_all` 时，已注册的 read/write/danger 工具按策略自动放行，
    但不绕过 Bearer、工具注册和 Vault 路径守卫。
    要彻底收紧可显式配置 write_allowlist 排除某些写入器。
    """
    mode = str(getattr(cfg, "approval_mode", "risk_based") or "risk_based").strip().lower()
    if mode not in {"risk_based", "allow_all"}:
        # 配置错误 fail-closed，避免拼写错误意外变成全放行。
        return False
    if mode == "allow_all" and tool.permission in {"read", "write", "danger"}:
        return True

    allow = set(getattr(cfg, "danger_allowlist", None) or [])
    if tool.permission == "danger":
        return tool.name in allow
    if tool.permission == "write":
        wc = getattr(cfg, "write_allowlist", None)
        # None=未配置→用代码默认集；[]=显式全拒；非空列表=显式放行集
        allowed = _DEFAULT_WRITE_ALLOW if wc is None else set(wc)
        return tool.name in allowed
    return False


# 旧名兼容：serve.py / test 早期按 `_serve_confirm` 引用，保留别名避免破坏导入
_serve_confirm = serve_confirm
