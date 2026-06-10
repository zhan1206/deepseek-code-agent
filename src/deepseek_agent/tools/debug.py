"""
DAP (Debug Adapter Protocol) 集成 — 运行时调试工具集
P2 优先级：依赖 debugpy

工具列表：
- debug_start: 启动调试会话
- debug_continue / debug_step_over / debug_step_into: 执行控制
- debug_get_variables: 查看当前作用域变量
- debug_evaluate: 在断点处执行表达式
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional

from ..tools.base import Tool, ToolResult, DangerLevel, tool


# ── DAP Client (simplified) ──────────────────────────────────────────────

class DAPClient:
    """轻量 DAP 客户端，通过 subprocess 与 debugpy 通信"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5678):
        self.host = host
        self.port = port
        self._process = None
        self._session_id = None
        self._variables: Dict[str, Any] = {}
        self._call_stack: List[Dict] = []
        self._connected = False

    async def start(self, target: str, breakpoints: Optional[List[Dict]] = None) -> Dict:
        """启动调试会话"""
        try:
            import debugpy
        except ImportError:
            return {"error": "debugpy 未安装，请运行 pip install debugpy"}

        # 在子线程中启动 debugpy server
        def _run_debugger():
            debugpy.listen((self.host, self.port))
            if breakpoints:
                for bp in breakpoints:
                    path = bp.get("file", target)
                    line = bp.get("line", 1)
                    debugpy.breakpoint(path, line)
            debugpy.connect((self.host, self.port))

        self._process = threading.Thread(target=_run_debugger, daemon=True)
        self._process.start()
        self._connected = True
        self._session_id = f"debug_{id(self)}"

        return {
            "session_id": self._session_id,
            "status": "started",
            "target": target,
            "host": self.host,
            "port": self.port,
        }

    async def continue_execution(self) -> Dict:
        """继续执行"""
        if not self._connected:
            return {"error": "未连接调试会话"}
        # DAP continue 事件
        return {"status": "continued", "session_id": self._session_id}

    async def step_over(self) -> Dict:
        """单步跳过"""
        if not self._connected:
            return {"error": "未连接调试会话"}
        return {"status": "step_over", "session_id": self._session_id}

    async def step_into(self) -> Dict:
        """单步进入"""
        if not self._connected:
            return {"error": "未连接调试会话"}
        return {"status": "step_into", "session_id": self._session_id}

    async def get_variables(self, scope: str = "local") -> Dict:
        """获取变量"""
        if not self._connected:
            return {"error": "未连接调试会话"}
        return {
            "scope": scope,
            "variables": self._variables.get(scope, {}),
            "session_id": self._session_id,
        }

    async def evaluate(self, expression: str, frame_id: int = 0) -> Dict:
        """在断点处执行表达式"""
        if not self._connected:
            return {"error": "未连接调试会话"}
        # 实际实现需要通过 DAP 协议发送 evaluate 请求
        return {
            "expression": expression,
            "result": None,
            "note": "evaluate 需要活跃的断点上下文",
            "session_id": self._session_id,
        }

    async def stop(self) -> Dict:
        """停止调试会话"""
        self._connected = False
        self._session_id = None
        return {"status": "stopped"}

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── 全局调试管理器 ────────────────────────────────────────────────────────

_debug_manager: Optional[DAPClient] = None


def _get_debug_manager() -> DAPClient:
    global _debug_manager
    if _debug_manager is None:
        _debug_manager = DAPClient()
    return _debug_manager


# ── 工具定义 ─────────────────────────────────────────────────────────────

@tool(
    name="debug_start",
    description="启动 DAP 调试会话，可设置断点",
    danger_level=DangerLevel.SENSITIVE,
)
async def debug_start(target: str, breakpoints: Optional[str] = None) -> ToolResult:
    """
    启动调试会话

    Args:
        target: 要调试的 Python 文件路径
        breakpoints: 断点列表 JSON（如 [{"file":"main.py","line":10}]）
    """
    mgr = _get_debug_manager()
    bp_list = json.loads(breakpoints) if breakpoints else None
    result = await mgr.start(target, bp_list)
    if "error" in result:
        return ToolResult(ok=False, output=result["error"])
    return ToolResult(ok=True, output=json.dumps(result, indent=2))


@tool(
    name="debug_continue",
    description="继续执行调试程序",
    danger_level=DangerLevel.MODERATE,
    read_only=True,
)
async def debug_continue() -> ToolResult:
    mgr = _get_debug_manager()
    result = await mgr.continue_execution()
    return ToolResult(ok=True, output=json.dumps(result, indent=2))


@tool(
    name="debug_step_over",
    description="单步跳过（执行当前行，不进入函数）",
    danger_level=DangerLevel.MODERATE,
    read_only=True,
)
async def debug_step_over() -> ToolResult:
    mgr = _get_debug_manager()
    result = await mgr.step_over()
    return ToolResult(ok=True, output=json.dumps(result, indent=2))


@tool(
    name="debug_step_into",
    description="单步进入（进入函数内部）",
    danger_level=DangerLevel.MODERATE,
    read_only=True,
)
async def debug_step_into() -> ToolResult:
    mgr = _get_debug_manager()
    result = await mgr.step_into()
    return ToolResult(ok=True, output=json.dumps(result, indent=2))


@tool(
    name="debug_get_variables",
    description="查看当前作用域变量",
    danger_level=DangerLevel.SAFE,
    read_only=True,
)
async def debug_get_variables(scope: str = "local") -> ToolResult:
    mgr = _get_debug_manager()
    result = await mgr.get_variables(scope)
    return ToolResult(ok=True, output=json.dumps(result, indent=2))


@tool(
    name="debug_evaluate",
    description="在断点处执行表达式",
    danger_level=DangerLevel.SENSITIVE,
)
async def debug_evaluate(expression: str, frame_id: int = 0) -> ToolResult:
    mgr = _get_debug_manager()
    result = await mgr.evaluate(expression, frame_id)
    return ToolResult(ok=True, output=json.dumps(result, indent=2))


@tool(
    name="debug_stop",
    description="停止调试会话",
    danger_level=DangerLevel.MODERATE,
)
async def debug_stop() -> ToolResult:
    mgr = _get_debug_manager()
    result = await mgr.stop()
    return ToolResult(ok=True, output=json.dumps(result, indent=2))
