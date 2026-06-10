"""
MCP Server — 将 DeepSeek Code Agent 暴露为 MCP 协议服务。

MCP (Model Context Protocol) 是一种标准化协议，
允许 AI 模型通过统一的接口调用外部工具和数据源。

本模块将 Agent 工具集暴露为 MCP 工具，
可被 Claude Desktop / VS Code Copilot 等 MCP Client 使用。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# MCP 协议核心类型（简化版，与官方 spec 兼容）
from .protocol import (
    MCPTool,
    JSONRPCRequest, JSONRPCResponse,
)


class MCPServer:
    """
    MCP Server — 暴露 Agent 工具集为 MCP 协议端点。

    协议实现：JSON-RPC 2.0 over stdio（与 Claude Desktop / VS Code MCP 兼容）

    使用方式：
        # 独立运行（stdio 模式）
        python -m deepseek_agent.mcp.server

        # 作为 Claude Desktop 插件
        # ~/.claude_desktop_config.json 添加：
        {
          "mcpServers": {
            "deepseek-agent": {
              "command": "python",
              "args": ["-m", "deepseek_agent.mcp.server"]
            }
          }
        }

        # VS Code Copilot MCP
        # .vscode/mcp.json 添加类似配置
    """

    def __init__(
        self,
        project_path: str = ".",
        api_key: Optional[str] = None,
    ):
        self.project_path = Path(project_path).resolve()
        self._tools: Dict[str, MCPTool] = {}
        self._agent = None
        self._initialized = False
        self._api_key = api_key

    # ── 初始化 ──────────────────────────────────────────────

    async def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 initialize 请求（MCP 握手）。
        返回 server 能力声明。
        """
        self._initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "deepseek-agent",
                "version": "0.4.0",
            },
            "capabilities": {
                "tools": {
                    "listChanged": True,
                },
                "resources": {},  # 暂不支持
                "prompts": {},    # 暂不支持
            },
        }

    # ── 工具注册 ──────────────────────────────────────────

    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        """注册一个 MCP 工具。"""
        self._tools[name] = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
        )

    def register_agent_tools(self) -> None:
        """
        将 AgentLoop 的工具集自动注册为 MCP 工具。

        需要先导入工具注册函数。
        """
        try:
            from ..core.client import DeepSeekClient
            from ..tools import ToolRegistry
            from ..tools.fs import (
                read_file, write_file, edit_file, list_directory,
                search_file, search_content, delete_file,
                run_shell, run_test,
            )
            from ..tools.git import (
                git_diff, git_log, git_status, git_checkout,
                git_commit, git_push, git_branch,
            )
            from ..tools.web import web_fetch, read_docs

            # 注册文件系统工具
            for tool_obj in [
                read_file, write_file, edit_file, list_directory,
                search_file, search_content, delete_file,
                run_shell, run_test,
                git_diff, git_log, git_status, git_checkout,
                git_commit, git_push, git_branch,
                web_fetch, read_docs,
            ]:
                self.register_tool(
                    name=tool_obj.name,
                    description=tool_obj.description,
                    input_schema=tool_obj.parameters,
                )
        except ImportError as e:
            print(f"[MCP] 工具注册失败: {e}", file=sys.stderr)

    # ── MCP 请求处理 ───────────────────────────────────────

    async def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """
        路由 JSON-RPC 请求到对应处理器。
        """
        method = request.method
        req_id = request.id

        try:
            if method == "initialize":
                result = await self.initialize(request.params or {})
                return JSONRPCResponse(id=req_id, result=result)

            elif method == "tools/list":
                tools_list = [
                    {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                    for t in self._tools.values()
                ]
                return JSONRPCResponse(id=req_id, result={"tools": tools_list})

            elif method == "tools/call":
                result = await self._call_tool(request.params or {})
                return JSONRPCResponse(id=req_id, result=result)

            elif method == "ping":
                return JSONRPCResponse(id=req_id, result={"pong": True})

            elif method in ("notifications/initialized", "notifications/cancelled"):
                # 无响应通知
                return None

            else:
                return JSONRPCResponse(
                    id=req_id,
                    error={"code": -32601, "message": f"Method not found: {method}"},
                )

        except Exception as e:
            return JSONRPCResponse(
                id=req_id,
                error={"code": -32603, "message": f"Internal error: {str(e)}"},
            )

    async def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用。"""
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        tool = self._tools[name]

        # 从已注册的 Agent 工具执行
        if self._agent and self._agent.registry.get(name):
            result = await self._agent.registry.execute(name, **arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": result.to_str(),
                    }
                ]
            }

        # 独立模式：执行模拟逻辑
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"[MCP] 工具 {name} 已注册，但 Agent 未初始化。请先启动 Agent。\n参数: {json.dumps(arguments, ensure_ascii=False)}",
                }
            ]
        }

    # ── stdio 协议循环 ─────────────────────────────────────

    async def run_stdio(self) -> None:
        """
        启动 stdio 协议循环（MCP 标准传输方式）。

        从 stdin 异步读取 JSON-RPC 请求，
        向 stdout 写入 JSON-RPC 响应。
        """
        import sys
        import asyncio

        self.register_agent_tools()
        loop = asyncio.get_event_loop()

        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                req_data = json.loads(line)
                request = JSONRPCRequest(**req_data)

                response = await self.handle_request(request)
                if response is not None:
                    print(json.dumps(response.model_dump(exclude_none=True), ensure_ascii=False), flush=True)

            except json.JSONDecodeError:
                error_resp = JSONRPCResponse(
                    id=None,
                    error={"code": -32700, "message": "Parse error"},
                )
                print(json.dumps(error_resp.model_dump(exclude_none=True), ensure_ascii=False), flush=True)
            except Exception as e:
                error_resp = JSONRPCResponse(
                    id=None,
                    error={"code": -32603, "message": f"Internal error: {str(e)}"},
                )
                print(json.dumps(error_resp.model_dump(exclude_none=True), ensure_ascii=False), flush=True)


def main():
    """MCP Server 入口点。"""
    import os
    project_path = os.environ.get("DEEPSEEK_PROJECT", ".")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    if not api_key:
        print("[MCP Server] 警告：DEEPSEEK_API_KEY 未设置", file=sys.stderr)

    server = MCPServer(project_path=project_path, api_key=api_key)
    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
