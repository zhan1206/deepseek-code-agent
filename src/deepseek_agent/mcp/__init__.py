"""MCP package — Model Context Protocol 支持。"""
from .server import MCPServer
from .client import MCPClient
from .protocol import MCPTool, MCPToolResult, MCPCode, mcp_error

__all__ = ["MCPServer", "MCPClient", "MCPTool", "MCPToolResult", "MCPCode", "mcp_error"]
