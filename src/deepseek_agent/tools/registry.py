"""
工具注册表 + 懒加载 — 从 base.py 提取的注册与加载逻辑。
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import uuid
from typing import Any, Callable, Dict, List, Optional

from .base import Tool, ToolResult, DangerLevel, ToolCallSpec


# ── 可选插件映射 ──────────────────────────────────────────────────────────

OPTIONAL_PLUGINS: Dict[str, str] = {
    "lsp": "deepseek_agent.tools.lsp",
    "mutation": "deepseek_agent.tools.mutation",
    "debug": "deepseek_agent.tools.debug",
    "benchmark": "deepseek_agent.tools.benchmark",
    "arch_check": "deepseek_agent.tools.arch_check",
    "refactor": "deepseek_agent.tools.refactor",
}


class ToolRegistry:
    """
    工具注册与管理中心。

    支持：
    - 注册/注销工具
    - 懒加载可选工具（节省内存）
    - 获取 OpenAI 格式 schema
    - 顺序/并行执行工具调用
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._core_registered: bool = False
        self._optional_loaded: Dict[str, bool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def register_func(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: str = "",
        danger_level: DangerLevel = DangerLevel.SAFE,
        **kwargs,
    ) -> Tool:
        """直接注册函数，自动包装为 Tool。"""
        t = Tool(
            name=name or func.__name__,
            description=description or (func.__doc__ or "").strip().split("\n")[0],
            func=func, danger_level=danger_level,
            require_approval=danger_level >= DangerLevel.SENSITIVE, **kwargs,
        )
        self.register(t)
        return t

    def register_all(self) -> None:
        """注册核心工具 + 标记可选工具为懒加载。"""
        if self._core_registered:
            return
        self._core_registered = True

        core_modules = [
            ("deepseek_agent.tools.fs", "read_file"),
            ("deepseek_agent.tools.git", "git_diff"),
            ("deepseek_agent.tools.web", "web_fetch"),
            ("deepseek_agent.tools.security", "security_scan"),
            ("deepseek_agent.tools.knowledge", "find_symbol"),
            ("deepseek_agent.tools.testing", "generate_tests"),
        ]
        for module_path, _ in core_modules:
            try:
                module = importlib.import_module(module_path)
                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue
                    attr = getattr(module, attr_name, None)
                    if isinstance(attr, Tool):
                        self.register(attr)
                    elif callable(attr) and hasattr(attr, "_tool"):
                        tool_obj = attr._tool
                        if isinstance(tool_obj, Tool):
                            self.register(tool_obj)
            except ImportError:
                pass

        for plugin_name in OPTIONAL_PLUGINS:
            self._optional_loaded[plugin_name] = False

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[Tool]:
        # 懒加载：首次访问可选工具时触发导入
        if name not in self._tools and name in OPTIONAL_PLUGINS:
            if not self._optional_loaded.get(name, False):
                self._load_plugin(name)
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    # ── 执行 ────────────────────────────────────────────────────────────────

    async def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult.fail(f"Unknown tool: {name}")
        return await tool.execute(**kwargs)

    async def execute_one(self, tool_call: ToolCallSpec, **overrides) -> ToolResult:
        tc = copy.deepcopy(tool_call)
        args = tc.get("arguments", {}) or {}
        args.update(overrides)
        return await self.execute(tc["name"], **args)

    async def execute_parallel(self, tool_calls: List[ToolCallSpec]) -> Dict[str, ToolResult]:
        async def run_one(tc: ToolCallSpec) -> tuple[str, ToolResult]:
            tc_id = tc.get("id", str(uuid.uuid4()))
            result = await self.execute_one(tc)
            return tc_id, result

        results = await asyncio.gather(
            *[run_one(tc) for tc in tool_calls], return_exceptions=True,
        )
        output: Dict[str, ToolResult] = {}
        for item in results:
            if isinstance(item, Exception):
                output["unknown"] = ToolResult.fail(str(item))
            else:
                tc_id, result = item
                output[tc_id] = result
        return output

    # ── 懒加载 ──────────────────────────────────────────────────────────────

    def _load_plugin(self, plugin_name: str) -> None:
        """懒加载单个插件工具。"""
        if plugin_name in self._optional_loaded and self._optional_loaded[plugin_name]:
            return
        self._optional_loaded[plugin_name] = True
        module_path = OPTIONAL_PLUGINS.get(plugin_name)
        if not module_path:
            return
        try:
            module = importlib.import_module(module_path)
            for attr_name in dir(module):
                if attr_name.startswith("__"):
                    continue
                attr = getattr(module, attr_name)
                if callable(attr) and hasattr(attr, "name"):
                    tool_obj = getattr(attr, "_tool", None) or attr
                    if isinstance(tool_obj, Tool):
                        self.register(tool_obj)
        except ImportError:
            pass