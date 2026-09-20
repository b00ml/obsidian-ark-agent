"""工具装饰器与权限（对齐 docs/03 §3.1、04 §2.5）。

用 inspect.signature + typing 归一化生成 OpenAI function schema；
语义字段只影响调度与安全，不改动业务逻辑。
"""
from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable, Literal

ToolPermission = Literal["read", "write", "danger"]
ExecutionMode = Literal["parallel", "sequential"]
SideEffects = Literal["none", "cache", "index", "write", "external"]


@dataclass
class Tool:
    name: str
    description: str
    permission: ToolPermission = "read"
    execution_mode: ExecutionMode = "parallel"
    can_terminate: bool = False
    prepare_arguments: Callable[[dict], dict] | None = None
    disable_model_invocation: bool = False
    execution_timeout: float | None = None  # 单工具执行超时（秒）；None=用配置全局值/不设限
    side_effects: str | None = None  # none|cache|index|write|external；None=未声明（write/danger 不允许）
    idempotent: bool | None = None  # 同参数重复调用状态是否收敛；None=未声明（write/danger 不允许）
    schema: dict = None  # OpenAI function schema（由装饰器注入）
    fn: Callable = None  # 实际执行函数，参数已校验


def validate_contract(tools: list[Tool]) -> list[str]:
    """工具契约校验：返回违规清单（空=通过）。

    write/danger 工具必须显式声明 side_effects 与 idempotent，防止静默吃默认值；
    side_effects 取值受控。调用方（连接器装配、契约测试）消费本函数结果。
    """
    allowed = {"none", "cache", "index", "write", "external"}
    errors: list[str] = []
    for t in tools:
        if t.permission in ("write", "danger"):
            if t.side_effects is None:
                errors.append(f"{t.name}: {t.permission} 工具未声明 side_effects")
            if t.idempotent is None:
                errors.append(f"{t.name}: {t.permission} 工具未声明 idempotent")
        if t.side_effects is not None and t.side_effects not in allowed:
            errors.append(f"{t.name}: side_effects={t.side_effects} 不在 {sorted(allowed)}")
    return errors


_type_map = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
    "Any": "string",
}


def _py_type_to_json(tp: Any) -> dict:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Literal and args:
        return {"type": "string", "enum": list(args)}
    if origin is typing.Union:
        # 仅处理 Optional[X] = Union[X, None]；多项 Union 退化为 string
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _py_type_to_json(non_none[0])
        return {"type": "string"}
    tname = getattr(tp, "__name__", str(tp))
    jtype = _type_map.get(tname, "string")
    return {"type": jtype}


def _schema_for(name: str, description: str, fn: Callable) -> dict:
    sig = inspect.signature(fn)
    props: dict[str, dict] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        prop = _py_type_to_json(ann)
        props[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, **({"required": required} if required else {})}},
    }


def tool(
    *,
    name: str | None = None,
    description: str,
    permission: ToolPermission = "read",
    execution_mode: ExecutionMode = "parallel",
    can_terminate: bool = False,
    prepare_arguments: Callable[[dict], dict] | None = None,
    disable_model_invocation: bool = False,
    execution_timeout: float | None = None,
    side_effects: str | None = None,
    idempotent: bool | None = None,
) -> Callable:
    """装饰器：用函数签名自动生成 schema，装配成 Tool 对象。"""

    def deco(fn: Callable) -> Tool:
        tname = name or fn.__name__
        t = Tool(
            name=tname,
            description=description,
            permission=permission,
            execution_mode=execution_mode,
            can_terminate=can_terminate,
            prepare_arguments=prepare_arguments,
            disable_model_invocation=disable_model_invocation,
            execution_timeout=execution_timeout,
            side_effects=side_effects,
            idempotent=idempotent,
            schema=_schema_for(tname, description, fn),
            fn=fn,
        )
        return t

    return deco