"""Public HTTP/SSE contract facade.

The implementation owner is still ``agentlab/agentlab/runtime/serve_contract.py``.
Consumers should import contract types from this boundary so a future package
move does not require changing Ark or external integrations.
"""
from __future__ import annotations

from packages._paths import ensure_legacy_importable

ensure_legacy_importable("agentlab")

from agentlab.runtime.serve_contract import (  # noqa: E402,F401
    CONFLICT_MESSAGE,
    CONTRACT,
    REPLAY_HEADER,
    ErrorCode,
    ResponsesRequest,
    StandardResponse,
    make_json_response,
    map_exception_to_error,
    request_id_middleware,
)

__all__ = [
    "CONFLICT_MESSAGE",
    "CONTRACT",
    "REPLAY_HEADER",
    "ErrorCode",
    "ResponsesRequest",
    "StandardResponse",
    "make_json_response",
    "map_exception_to_error",
    "request_id_middleware",
]
