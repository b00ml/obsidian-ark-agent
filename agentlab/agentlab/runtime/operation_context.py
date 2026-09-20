"""Request-local operation identity for side-effect adapters.

The Runner owns the durable operation id.  Adapters that cannot accept an
extra model-visible argument (for example ``inbox_collect``) read it from
this context instead, so their external ledger can be queried after a crash.
"""
from __future__ import annotations

from contextvars import ContextVar, Token


_CURRENT_OPERATION_ID: ContextVar[str] = ContextVar(
    "agentlab_current_operation_id", default=""
)


def current_operation_id() -> str:
    return _CURRENT_OPERATION_ID.get()


def bind_operation_id(operation_id: str) -> Token[str]:
    return _CURRENT_OPERATION_ID.set(str(operation_id or "").strip()[:256])


def reset_operation_id(token: Token[str]) -> None:
    _CURRENT_OPERATION_ID.reset(token)


__all__ = ["bind_operation_id", "current_operation_id", "reset_operation_id"]
