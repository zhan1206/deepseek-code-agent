"""
工具基类 + 结果包装器 + Schema 生成 — 纯抽象层，无注册/加载逻辑。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, get_origin, get_type_hints
from enum import IntEnum


# ── 危险等级 ────────────────────────────────────────────────────────────────

class DangerLevel(IntEnum):
    SAFE = 0       # 读文件、搜索，无需确认
    MODERATE = 1  # 写文件、网络请求，需确认
    SENSITIVE = 2  # 高风险写操作，需确认
    DANGEROUS = 3  # 删除、执行命令，双重确认


# ── 工具结果包装器 ─────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """所有工具执行结果的统一包装。"""
    success: bool
    data: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "data": self.data, "error": self.error}

    def to_str(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def ok(cls, data: Any) -> "ToolResult":
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, error=error)


# ── 工具基类 ───────────────────────────────────────────────────────────────

@dataclass
class Tool:
    """工具定义。"""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    danger_level: DangerLevel = DangerLevel.SAFE
    require_approval: bool = False
    read_only: bool = False
    examples: Optional[List[Dict]] = None

    def __post_init__(self):
        if not self.parameters:
            self.parameters = generate_schema(self.func, self.description)

    def to_openai_schema(self) -> Dict[str, Any]:
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        if self.examples:
            schema["function"]["parameters"]["examples"] = self.examples
        return schema

    async def execute(self, **kwargs) -> ToolResult:
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(str(e))


# ── Schema 自动生成 ────────────────────────────────────────────────────────

_PYTHON_TO_JSON = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object", Any: "string",
}


def _python_type_to_json(annotation) -> str:
    origin = get_origin(annotation)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    if origin is Union:
        args = [a for a in annotation.__args__ if a is not type(None)]
        if args:
            return _PYTHON_TO_JSON.get(args[0], "string")
        return "string"
    return _PYTHON_TO_JSON.get(annotation, "string")


def generate_schema(func: Callable, description: str = "") -> Dict[str, Any]:
    """根据函数签名自动生成 JSON Schema。"""
    sig = inspect.signature(func)
    properties = {}
    required = []
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        json_type = "string"
        if param_name in hints:
            json_type = _python_type_to_json(hints[param_name])
        elif param.default is not inspect.Parameter.empty:
            if isinstance(param.default, bool): json_type = "boolean"
            elif isinstance(param.default, int): json_type = "integer"
            elif isinstance(param.default, float): json_type = "number"
            elif isinstance(param.default, dict): json_type = "object"
            elif isinstance(param.default, list): json_type = "array"

        prop: Dict[str, Any] = {"type": json_type}
        _DESC_MAP = {"path": "文件/目录路径", "command": "Shell 命令", "content": "文件内容", "pattern": "搜索模式（支持正则）", "url": "URL 地址"}
        if param_name in _DESC_MAP:
            prop["description"] = _DESC_MAP[param_name]
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)
        properties[param_name] = prop

    result = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


# ── 装饰器 ─────────────────────────────────────────────────────────────────

def tool(
    name: Optional[str] = None,
    description: str = "",
    danger_level: DangerLevel = DangerLevel.SAFE,
    require_approval: Optional[bool] = None,
    read_only: bool = False,
    examples: Optional[List[Dict]] = None,
) -> Callable[[Callable], Tool]:
    """装饰器：快速注册工具。"""
    def decorator(func: Callable) -> Tool:
        _name = name or func.__name__
        _approval = require_approval if require_approval is not None else (danger_level >= DangerLevel.SENSITIVE)
        return Tool(
            name=_name, description=description or (func.__doc__ or "").strip().split("\n")[0],
            func=func, danger_level=danger_level, require_approval=_approval,
            read_only=read_only, examples=examples,
        )
    return decorator


# ── 类型别名 ────────────────────────────────────────────────────────────────

ToolCallSpec = Dict[str, Any]