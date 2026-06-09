"""
MCP 协议核心类型 — JSON-RPC 2.0 标准化定义。
与官方 MCP Spec 2024-11-05 兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ── JSON-RPC 基础 ────────────────────────────────────────────────────────

@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 请求对象。"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 响应对象。"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


# ── MCP 核心类型 ────────────────────────────────────────────────────────

@dataclass
class MCPTool:
    """MCP 工具定义。"""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolResult:
    """MCP 工具执行结果。"""
    content: List[Dict[str, Any]]  # [{"type": "text", "text": "..."}]
    is_error: bool = False


@dataclass
class MCPResource:
    """MCP 资源（暂未使用）。"""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


@dataclass
class MCPPrompt:
    """MCP 提示模板（暂未使用）。"""
    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MCPProgress:
    """进度通知。"""
    progress_token: str
    progress: float
    total: Optional[float] = None


# ── MCP 错误码 ──────────────────────────────────────────────────────────

class MCPCode:
    """MCP 错误码常量（与 JSON-RPC 基础码对齐）。"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP 扩展错误码
    TOOL_NOT_FOUND = -32001
    TOOL_EXECUTION_ERROR = -32002
    RESOURCE_NOT_FOUND = -32003
    SAMPLING_ERROR = -32004


def mcp_error(code: int, message: str, data: Any = None) -> Dict[str, Any]:
    """构建 MCP 错误对象。"""
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return err
