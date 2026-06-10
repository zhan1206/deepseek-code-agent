"""
MCP Client — 连接外部 MCP Server（如 Claude Desktop 的其他 Server）。

v2.0 新增：
- SSE 传输（MCPSSEClient）
- 工具桥接（MCPToolBridge）
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

    支持三种连接方式：
    1. stdio 模式（子进程）
    2. SSE 模式（HTTP Server-Sent Events）— 使用 MCPSSEClient
    3. 工具桥接 — 使用 MCPToolBridge
    """

    def __init__(self):
        self._request_id = 0
        self._response_futures: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._tools: List[Dict[str, Any]] = []
        self._connected = False

    # ── 连接 ──────────────────────────────────────────────

    async def connect(self, command: List[str], env: Optional[Dict[str, str]] = None) -> None:
        """启动 MCP Server 并握手（stdio 模式）。"""
        merged_env = {**subprocess.os.environ, **(env or {})}

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
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "deepseek-agent-mcp-client",
                "version": "2.0.0",
            },
            "capabilities": {},
        })

        await self._send_notification("initialized", {})
        tools_result = await self._send_request("tools/list", {})
        self._tools = tools_result.get("tools", [])
        self._connected = True

    async def _read_loop(self, stdout: asyncio.StreamReader) -> None:
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
        if not self._tools:
            result = await self._send_request("tools/list", {})
            self._tools = result.get("tools", [])
        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        return result

    # ── 断开连接 ──────────────────────────────────────────

    async def disconnect(self) -> None:
        self._connected = False
        if self._reader_task:
            self._reader_task.cancel()
        if hasattr(self, '_stdout_reader') and self._stdout_reader:
            try:
                proc = await self._stdout_reader
                proc.terminate()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── SSE 传输 ──────────────────────────────────────────────────────────

class MCPSSEClient(MCPClient):
    """
    MCP SSE Client — 通过 HTTP + Server-Sent Events 连接 MCP Server。

    适用于远程 MCP 服务器（如 Puppeteer、Postgres MCP Server）。
    """

    def __init__(self):
        super().__init__()
        self._base_url = ""
        self._session: Optional[Any] = None

    async def connect_sse(self, url: str, api_key: Optional[str] = None) -> None:
        """
        通过 SSE 连接 MCP Server。

        Args:
            url: Server URL（如 http://localhost:3001/sse）
            api_key: 可选 API Key
        """
        import httpx

        self._base_url = url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._session = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers=headers,
        )

        # 初始化
        init_result = await self._send_http_request("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "deepseek-agent-sse-client",
                "version": "2.0.0",
            },
            "capabilities": {},
        })

        await self._send_http_notification("initialized", {})

        tools_result = await self._send_http_request("tools/list", {})
        self._tools = tools_result.get("tools", [])
        self._connected = True

    async def _send_http_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self._session:
            raise RuntimeError("未连接")

        self._request_id += 1
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method=method,
            params=params,
            id=self._request_id,
        )

        resp = await self._session.post(
            f"{self._base_url}/message",
            json=request.model_dump(exclude_none=True),
        )

        if resp.status_code != 200:
            raise Exception(f"MCP HTTP 请求失败: {resp.status_code}")

        data = resp.json()
        if data.get("error"):
            raise Exception(data["error"].get("message", "Unknown error"))
        return data.get("result", {})

    async def _send_http_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self._session:
            return
        request = JSONRPCRequest(jsonrpc="2.0", method=method, params=params)
        await self._session.post(
            f"{self._base_url}/message",
            json=request.model_dump(exclude_none=True),
        )

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self._send_http_request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })

    async def disconnect(self) -> None:
        self._connected = False
        if self._session:
            await self._session.aclose()


# ── 工具桥接 ──────────────────────────────────────────────────────────

class MCPToolBridge:
    """
    MCP 工具桥接 — 将 MCP Server 的工具自动注册到 ToolRegistry。

    用法：
        bridge = MCPToolBridge(registry)
        await bridge.connect_stdio(["python", "-m", "some_mcp_server"])
        # 或
        await bridge.connect_sse("http://localhost:3001/sse")
        # MCP 工具现在在 registry 中可用
    """

    def __init__(self, registry: Any):
        self.registry = registry
        self._clients: List[MCPClient] = []
        self._bridged_tools: Dict[str, MCPClient] = {}

    async def connect_stdio(self, command: List[str], env: Optional[Dict[str, str]] = None) -> int:
        client = MCPClient()
        await client.connect(command, env)
        return self._register_tools(client)

    async def connect_sse(self, url: str, api_key: Optional[str] = None) -> int:
        client = MCPSSEClient()
        await client.connect_sse(url, api_key)
        return self._register_tools(client)

    def _register_tools(self, client: MCPClient) -> int:
        from ..tools.base import Tool, ToolResult, DangerLevel

        count = 0
        for tool_info in client._tools:
            name = tool_info.get("name", "")
            if not name:
                continue

            bridged_name = f"mcp_{name}"
            description = tool_info.get("description", "")

            async def _make_func(client_ref: MCPClient, tool_name: str) -> Any:
                async def _bridged_func(**kwargs) -> str:
                    try:
                        result = await client_ref.call_tool(tool_name, kwargs)
                        content = result.get("content", [])
                        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                        return ToolResult.ok("\n".join(texts) or "(无输出)").to_str()
                    except Exception as e:
                        return ToolResult.fail(f"MCP 工具调用失败: {e}").to_str()
                _bridged_func.__name__ = bridged_name
                _bridged_func.__doc__ = description
                return _bridged_func

            # Synchronous closure creation
            def _make_sync(c: MCPClient, tn: str, desc: str):
                async def f(**kwargs) -> str:
                    try:
                        result = await c.call_tool(tn, kwargs)
                        content = result.get("content", [])
                        texts = [c2.get("text", "") for c2 in content if c2.get("type") == "text"]
                        return ToolResult.ok("\n".join(texts) or "(无输出)").to_str()
                    except Exception as e:
                        return ToolResult.fail(f"MCP 工具调用失败: {e}").to_str()
                f.__name__ = f"mcp_{tn}"
                f.__doc__ = desc
                return f

            bridged_func = _make_sync(client, name, description)

            bridged_tool = Tool(
                name=bridged_name,
                description=f"[MCP] {description}",
                func=bridged_func,
                danger_level=DangerLevel.SENSITIVE,
                require_approval=True,
            )

            try:
                self.registry.register(bridged_tool)
                self._bridged_tools[bridged_name] = client
                count += 1
            except Exception:
                pass

        self._clients.append(client)
        return count

    async def disconnect_all(self) -> None:
        for client in self._clients:
            await client.disconnect()
        self._clients.clear()
        self._bridged_tools.clear()

    def get_bridged_tools(self) -> List[str]:
        return list(self._bridged_tools.keys())
