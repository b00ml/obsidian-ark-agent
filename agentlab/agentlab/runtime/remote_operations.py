"""Configuration-driven remote operation status queries.

The adapter is intentionally small and synchronous because recovery runs
before a request is dispatched.  It accepts only a bounded JSON status
document and never treats a missing endpoint, transport failure, or malformed
response as success.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


_TERMINAL = {"succeeded", "failed", "cancelled"}


@dataclass(frozen=True)
class RemoteOperationConfig:
    base_url: str = ""
    status_path: str = "/operations/{operation_id}"
    token_env: str = "AGENT_REMOTE_OPERATION_TOKEN"
    timeout: float = 10.0
    source: str = "remote-operation-api"

    @classmethod
    def from_value(cls, value: Any = None) -> "RemoteOperationConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls()
        try:
            timeout = max(0.5, min(float(value.get("timeout", 10.0)), 60.0))
        except (TypeError, ValueError):
            timeout = 10.0
        return cls(
            base_url=str(value.get("base_url") or "").strip().rstrip("/"),
            status_path=str(value.get("status_path") or "/operations/{operation_id}"),
            token_env=str(value.get("token_env") or "AGENT_REMOTE_OPERATION_TOKEN"),
            timeout=timeout,
            source=str(value.get("source") or "remote-operation-api")[:128],
        )


def query_remote_operation(
    operation_id: str,
    config: RemoteOperationConfig | Mapping[str, Any] | None = None,
    *,
    opener=urlopen,
) -> dict[str, str] | None:
    """Query one operation and return normalized evidence when terminal."""
    cfg = RemoteOperationConfig.from_value(config)
    op_id = str(operation_id or "").strip()
    if not cfg.base_url or not op_id:
        return None
    path = cfg.status_path.replace("{operation_id}", quote(op_id, safe=""))
    url = f"{cfg.base_url}/{path.lstrip('/')}"
    headers = {"Accept": "application/json"}
    token = os.environ.get(cfg.token_env, "") if cfg.token_env else ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers, method="GET")
    try:
        with opener(request, timeout=cfg.timeout) as response:
            if int(getattr(response, "status", 200)) != 200:
                return None
            raw = response.read(64 * 1024)
    except (OSError, HTTPError, URLError, TimeoutError):
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    status = str(payload.get("status") or payload.get("state") or "").strip().lower()
    if status not in _TERMINAL:
        return None
    evidence_ref = str(payload.get("evidence_ref") or f"{cfg.source}:{op_id}").strip()
    result_ref = str(payload.get("result_ref") or payload.get("result_id") or "").strip()
    error_code = str(payload.get("error_code") or "").strip()
    result = {
        "status": status,
        "source": cfg.source[:128],
        "evidence_ref": evidence_ref[:256],
    }
    if result_ref:
        result["result_ref"] = result_ref[:256]
    if error_code:
        result["error_code"] = error_code[:128]
    return result


__all__ = ["RemoteOperationConfig", "query_remote_operation"]
