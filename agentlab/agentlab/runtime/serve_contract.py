"""前后端契约与 SSE 翻译（serve 拆职责：S6，从 serve.py 抽出，对照 S2 §4.7）。

- CONTRACT：契约版本号，随 SSE 响应头与心跳事件携带，前端可识别不匹配（§4.5）。
- ResponsesRequest：/v1/responses 请求体 schema，非法载荷由 serve 转 400（复用 pydantic）。
- _sse / _Sink：把 Runner 事件翻译成 Hermes 兼容 SSE 字符串（输出不变，纯搬移）。
- ErrorCode / StandardResponse：统一错误码与响应格式（F5-007）。
- request_id_middleware：自动生成请求追踪 ID（F5-007）。

本模块只做"翻译与声明"，不做鉴权/会话/日志归属——那些分属 serve_auth / serve_session / serve。
"""
from __future__ import annotations

import json
import uuid
from enum import IntEnum
from typing import Any, Callable, Optional

from aiohttp import web
from pydantic import BaseModel

# 前后端 SSE/请求契约版本：前端未知事件跳过、此处对版本做校验防隐身不匹配（§4.5）
CONTRACT = "v1"

# 高度相关的幂等契约头（§4.7 幂等）：命中缓存重放时响应置 1，前端可据此识别
REPLAY_HEADER = "X-Agentlab-Replay"
CONFLICT_MESSAGE = "request in progress"


# ---- F5-007：统一错误码与 request_id ----

class ErrorCode(IntEnum):
    """错误码枚举（HTTP 状态码 + 业务语义）"""

    # 2xx 成功
    OK = 200
    CREATED = 201

    # 4xx 客户端错误
    BAD_REQUEST = 400           # 请求格式错误、参数缺失
    UNAUTHORIZED = 401          # 鉴权失败
    FORBIDDEN = 403             # 无权限
    NOT_FOUND = 404             # 资源不存在
    CONFLICT = 409              # 状态冲突（如会话已提交）
    UNPROCESSABLE = 422         # 语义错误（JSON 合法但业务不合法）
    DEPENDENCY_MISSING = 424    # 依赖服务不可用（如 MCP 工具不可达）

    # 5xx 服务端错误
    INTERNAL_ERROR = 500        # 未分类的内部错误
    SERVICE_UNAVAILABLE = 503   # 服务降级（如记忆仓不可用）


class StandardResponse:
    """标准响应格式（所有 API 返回必须符合此结构）"""

    @staticmethod
    def success(
        data: Any = None,
        request_id: Optional[str] = None,
        message: str = "ok"
    ) -> dict:
        """成功响应"""
        return {
            "ok": True,
            "code": ErrorCode.OK,
            "message": message,
            "request_id": request_id or _generate_request_id(),
            "data": data
        }

    @staticmethod
    def error(
        code: ErrorCode,
        message: str,
        request_id: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[dict] = None
    ) -> dict:
        """错误响应"""
        resp = {
            "ok": False,
            "code": int(code),
            "message": message,
            "request_id": request_id or _generate_request_id()
        }
        if error_type:
            resp["error_type"] = error_type
        if details:
            resp["details"] = details
        return resp


def _generate_request_id() -> str:
    """生成请求追踪 ID（UUID4 前 8 位）"""
    return str(uuid.uuid4())[:8]


@web.middleware
async def request_id_middleware(request: web.Request, handler):
    """request_id 中间件：生成 UUID 并注入请求/响应"""
    request_id = _generate_request_id()
    request["request_id"] = request_id  # 注入 request 供后续使用

    try:
        response = await handler(request)
        # 注入响应头（便于日志关联）
        if isinstance(response, web.Response):
            response.headers["X-Request-Id"] = request_id
        return response
    except web.HTTPException as e:
        # HTTP 异常（如 404）也需要注入 request_id
        e.headers["X-Request-Id"] = request_id
        raise


def make_json_response(
    data: dict,
    status: int,
    request_id: Optional[str] = None
) -> web.Response:
    """统一 JSON 响应构造（自动注入 request_id）"""
    if request_id and "request_id" not in data:
        data["request_id"] = request_id

    resp = web.json_response(data, status=status)
    if request_id:
        resp.headers["X-Request-Id"] = request_id
    return resp


def map_exception_to_error(
    exc: Exception,
    request_id: Optional[str] = None
) -> tuple[ErrorCode, str]:
    """异常到错误码的映射规则"""

    # ValidationError（Pydantic）
    if exc.__class__.__name__ == "ValidationError":
        return ErrorCode.BAD_REQUEST, f"validation failed: {exc}"

    # ValueError（通用参数错误）
    if isinstance(exc, ValueError):
        return ErrorCode.BAD_REQUEST, str(exc)

    # KeyError（缺少必需字段）
    if isinstance(exc, KeyError):
        return ErrorCode.BAD_REQUEST, f"missing required field: {exc}"

    # PermissionError（权限不足）
    if isinstance(exc, PermissionError):
        return ErrorCode.FORBIDDEN, str(exc)

    # FileNotFoundError（资源不存在）
    if isinstance(exc, FileNotFoundError):
        return ErrorCode.NOT_FOUND, str(exc)

    # 其他未分类异常
    return ErrorCode.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}"


# ---- 原有契约定义（保持不变）----

class ResponsesRequest(BaseModel):
    """/v1/responses 请求体 schema（复用 pydantic，零新依赖，非法载荷直接 400）。

    input 为消息数组、stream 为布尔、previous_response_id 为会话标识；
    request_id 为幂等键（§4.7）：同 id 重复提交由 serve 重放首次结果，避免重复烧 token/写库。
    project_id（P0-2/OPT-107）：Project 长期任务空间——服务端据此注入项目规则/背景
    （<vault>/ark/projects/<id>/）；非法或不存在 → 静默降级为全局行为。
    其余字段（如前端 Hermes 兼容的 model/tools）放行忽略，不做强校验以免破坏现有客户端。
    """

    input: list[dict] | None = None
    stream: bool = True
    previous_response_id: str | None = None
    request_id: str | None = None
    project_id: str | None = None
    # P2-2/OPT-121：multi-agent 并行名单（agents.external 注册名 + 可选保留名 "agentlab"）。
    # 给定即走并行+汇总路；空/缺省 = 原单 agent 语义不变。
    multi: list[str] | None = None


def _sse(obj: dict[str, Any]) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


class _Sink:
    """把 Runner 事件翻译成 Hermes 兼容 SSE 字符串。

    内聚：只做事件→SSE 的映射；日志（slog）通过构造参数注入，避免与本模块耦合 serve。
    """

    def __init__(self, emit: Callable[[str], None], slog: Callable[..., None] | None = None) -> None:
        self._emit = emit
        self._slog = slog or (lambda *a: None)

    def text(self, content: str):
        if not content:
            return
        self._emit(_sse({"type": "response.output_text.delta", "delta": content}))

    def tool_start(self, name: str, arguments: str = "{}"):
        """工具开始事件。arguments 透传模型给出的真实入参（F5-016/OPT-182）：
        此前硬编码 "{}"，前端无法显示「正在咨询 @谁：什么问题」。默认值保留，
        兼容未传参的旧调用点。"""
        self._slog("tool_start", name)
        self._emit(_sse({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": name,
                     "arguments": arguments if arguments else "{}"},
        }))

    def tool_end(self, name: str, result: str):
        # 兜底上限：实际截断发生在 serve._build_backend 的 build() 处（对齐 max_tool_result_chars）
        self._slog("tool_end", name, f"len={len(result or '')}")
        self._emit(_sse({
            "type": "response.output_item.done",
            "item": {"type": "function_call_output", "name": name,
                     "output": (result or "")[:20000]},
        }))

    def event(self, payload: dict):
        """自定义契约事件透传（P2-2 multi 分支进度等；加法演进，旧前端跳过未识别事件）。"""
        self._emit(_sse(payload))
