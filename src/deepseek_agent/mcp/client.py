"""
MCP Client — 连接外部 MCP Server（如 Claude Desktop 的其他 Server）。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .protocol import JSONRPCRequest, JSONRPCResponse


@dataclass
class MCPToolCall:
    """MCP 工具调用。"""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


class MCPClient:
    """
    MCP Client — 连接 MCP Server 并调用工具。

    支持两种连接方式：
    1. stdio 模式（子进程，本模块默认）
    2. HTTP 模式（未来扩展，需 Server 支持 SSE）

    用法：
        client = MCPClient()
        await client.connect(["python", "-m", "deepseek_agent.mcp.server"])
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "./README.md"})
        await client.disconnect()
    """

    def __init__(self):
        self._request_id = 0
        self._response_futures: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._tools: List[Dict[str, Any]] = []
        self._connected = False

    # ── 连接 ──────────────────────────────────────────────

    async def connect(self, command: List[str], env: Optional[Dict[str, str]] = None) -> None:
        """
        启动 MCP Server 并握手。

        Args:
            command: Server 启动命令，如 ["python", "-m", "deepseek_agent.mcp.server"]
            env: 额外环境变量
        """
        merged_env = {**subprocess.os.environ, **(env or {})}

        # asyncio subprocess（仅启动一次，避免重复 fork 两个进程）
        self._stdout_reader = asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )

        proc = await self._stdout_reader

        self._reader_task = asyncio.create_task(self._read_loop(proc.stdout))

        # 发送 initialize
        init_result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "deepseek-agent-mcp-client",
                "version": "0.4.0",
            },
            "capabilities": {},
        })

        # 发送 initialized 通知
        await self._send_notification("initialized", {})

        # 获取工具列表
        tools_result = await self._send_request("tools/list", {})
        self._tools = tools_result.get("tools", [])
        self._connected = True

    async def _read_loop(self, stdout: asyncio.StreamReader) -> None:
        """持续读取 Server 响应。"""
        while True:
            try:
                line = await stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                await self._handle_response(text)
            except Exception as e:
                print(f"[MCP Client] 读取错误: {e}", file=sys.stderr)
                break

    async def _handle_response(self, text: str) -> None:
        """处理收到的 JSON-RPC 消息。"""
        try:
            data = json.loads(text)
            resp = JSONRPCResponse(**data)
            req_id = resp.id
            if req_id is not None and req_id in self._response_futures:
                future = self._response_futures.pop(req_id)
                if resp.error:
                    future.set_exception(Exception(resp.error.get("message", "Unknown error")))
                else:
                    future.set_result(resp.result or {})
        except Exception:
            pass

    # ── 协议操作 ──────────────────────────────────────────

    async def _send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """发送 JSON-RPC 请求并等待响应。"""
        self._request_id += 1
        req_id = self._request_id

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._response_futures[req_id] = future

        request = JSONRPCRequest(
            jsonrpc="2.0",
            method=method,
            params=params,
            id=req_id,
        )

        proc = await self._get_process()
        if proc is not None and proc.stdin:
            line = json.dumps(request.model_dump(exclude_none=True), ensure_ascii=False) + "\n"
            proc.stdin.write(line.encode("utf-8"))
            await proc.stdin.drain()

        return await future

    async def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """发送 JSON-RPC 通知（无响应）。"""
        proc = await self._get_process()
        if not proc:
            return
        request = JSONRPCRequest(jsonrpc="2.0", method=method, params=params)
        line = json.dumps(request.model_dump(exclude_none=True), ensure_ascii=False) + "\n"
        proc.stdin.write(line.encode("utf-8"))
        await proc.stdin.drain()

    async def _get_process(self) -> Optional[asyncio.Process]:
        if self._stdout_reader:
            try:
                return await self._stdout_reader
            except Exception:
                return None
        return None

    # ── 工具调用 ──────────────────────────────────────────

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具。"""
        if not self._tools:
            result = await self._send_request("tools/list", {})
            self._tools = result.get("tools", [])
        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        调用 MCP 工具。

        Returns:
            {"content": [{"type": "text", "text": "..."}]}
        """
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        return result

    # ── 断开连接 ──────────────────────────────────────────

    async def disconnect(self) -> None:
        """关闭 MCP 连接。"""
        self._connected = False
        if self._reader_task:
            self._reader_task.cancel()
        if self._stdout_reader:
            try:
                proc = await self._stdout_reader
                proc.terminate()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected
