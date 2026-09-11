"""会话职责（serve 拆职责：S6，从 serve.py 抽出）。

- history：把前端回传的 [{role,content}] 拆成本次历史 + 最新输入（客户端侧会话）。
- persist_delta：把本轮"新增"消息追加进 JsonlSessionStorage（服务端侧会话，跳过
  system 与已重放历史，避免重复存储）。
"""
from __future__ import annotations

from agentlab.core.message import Message


def history(user_msgs: list[dict]) -> tuple[list[Message], str]:
    """把 ark 传来的 [{role,content},...] 拆成历史 ctx 与最新用户输入。

    system 消息忽略（agentlab 用自身含工具 schema 的 system 指令）；
    仅最后一条 user 作为本次输入，其余 user/assistant 轮进入历史。
    """
    items = [m for m in (user_msgs or []) if str(m.get("role", "")) in ("user", "assistant")]
    hist: list[Message] = []
    latest = ""
    if items:
        users = [i for i, m in enumerate(items) if str(m.get("role", "")) == "user"]
        if users:
            last_ui = users[-1]
            latest = str(items[last_ui].get("content") or "")
            hist = [
                Message(role="user" if m.get("role") == "user" else "assistant",
                        content=str(m.get("content") or ""))
                for i, m in enumerate(items)
                if i != last_ui
            ]
        else:
            hist = [Message(role="assistant", content=str(m.get("content") or "")) for m in items]
    return hist, latest


def persist_delta(store, session_id: str, hist: list[Message], messages: list[Message]) -> None:
    """把（system + 已重放历史 + 本轮新增）里的"新增"部分追加进会话存储。

    messages 结构 = [system(若有), *hist, user, 各轮 assistant/tool...]，
    故新增部分 = messages[1 + len(hist):]（当 system 存在）。容忍 system 缺失。
    """
    base = 0
    if messages and messages[0].role == "system":
        base = 1 + len(hist)
    elif messages:
        base = len(hist)
    for m in messages[base:]:
        store.append(session_id, m)


# 旧名兼容：serve.py / test 早期按 `_history` / `_persist_delta` 引用，保留别名避免破坏导入
_history = history
_persist_delta = persist_delta