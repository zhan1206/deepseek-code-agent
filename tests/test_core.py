"""Phase 1 核心框架测试。"""
import pytest
from deepseek_agent.tools.base import tool, generate_schema, ToolResult, DangerLevel
from deepseek_agent.tools.registry import ToolRegistry


class TestToolResult:
    def test_ok(self):
        r = ToolResult.ok({"data": 42})
        assert r.success is True
        assert r.data == {"data": 42}
        assert r.error is None
        assert '"success": true' in r.to_str()

    def test_fail(self):
        r = ToolResult.fail("not found")
        assert r.success is False
        assert r.error == "not found"
        assert '"success": false' in r.to_str()


class TestSchemaGeneration:
    def test_basic_schema(self):
        async def echo(message: str, count: int = 1) -> str:
            return message * count

        schema = generate_schema(echo)
        assert schema["type"] == "object"
        assert "message" in schema["required"]
        assert "count" in schema["properties"]
        assert schema["properties"]["message"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["count"]["default"] == 1

    def test_tool_decorator(self):
        @tool(name="greet", description="打招呼")
        async def greet(name: str, prefix: str = "Hello") -> str:
            return f"{prefix}, {name}!"

        assert greet.name == "greet"
        assert greet.danger_level == DangerLevel.SAFE
        assert "name" in greet.parameters["required"]
        assert greet.require_approval is False


class TestToolRegistry:
    def test_register_and_execute(self):
        registry = ToolRegistry()

        @tool(name="add", description="加法")
        async def add(a: int, b: int) -> int:
            return a + b

        registry.register(add)
        assert "add" in registry.list_tools()

        import asyncio; result = asyncio.run(registry.execute("add", a=1, b=2))
        assert result.success is True
        assert result.data == 3

    def test_parallel_execution(self):
        import asyncio

        registry = ToolRegistry()

        @tool(name="double", description="翻倍")
        async def double(x: int) -> int:
            return x * 2

        registry.register(double)

        tool_calls = [
            {"id": "1", "name": "double", "arguments": {"x": 1}},
            {"id": "2", "name": "double", "arguments": {"x": 2}},
            {"id": "3", "name": "double", "arguments": {"x": 3}},
        ]

        results = asyncio.run(registry.execute_parallel(tool_calls))
        assert results["1"].data == 2
        assert results["2"].data == 4
        assert results["3"].data == 6


class TestSchemaGenerationEdge:
    def test_union_type(self):
        from typing import Union, Optional
        async def mixed(x: Union[str, int]) -> str:
            return str(x)
        schema = generate_schema(mixed)
        assert schema["properties"]["x"]["type"] == "string"

    def test_complex_signature(self):
        async def complex_fn(
            path: str,
            content: str = "",
            recursive: bool = False,
            max_depth: int = 3,
        ) -> bool:
            return True
        schema = generate_schema(complex_fn)
        assert schema["required"] == ["path"]
        assert schema["properties"]["recursive"]["type"] == "boolean"
        assert schema["properties"]["recursive"]["default"] is False
