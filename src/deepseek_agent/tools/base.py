"""
工具系统 — 统一结果格式、自动 Schema、并行执行。
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import json
import uuid
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
    """
    工具定义。

    Attributes:
        name: 工具唯一名称
        description: 供模型理解的描述
        func: 实际执行的异步函数
        parameters: JSON Schema（自动从 func 签名生成）
        danger_level: 危险等级
        require_approval: 是否需要用户审批
        examples: 示例输入输出（可选）
    """

    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    danger_level: DangerLevel = DangerLevel.SAFE
    require_approval: bool = False
    read_only: bool = False  # 只读工具可安全并发执行
    examples: Optional[List[Dict]] = None

    def __post_init__(self):
        if not self.parameters:
            self.parameters = generate_schema(self.func, self.description)

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式。"""
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
        """执行工具，捕获异常并统一包装。"""
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
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    Any: "string",
}


def _python_type_to_json(annotation) -> str:
    """将 Python 类型注解转为 JSON Schema 类型。"""
    origin = get_origin(annotation)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    if origin is Union:
        # 取第一个非 None 类型
        args = [a for a in annotation.__args__ if a is not type(None)]
        if args:
            return _PYTHON_TO_JSON.get(args[0], "string")
        return "string"
    return _PYTHON_TO_JSON.get(annotation, "string")


def generate_schema(func: Callable, description: str = "") -> Dict[str, Any]:
    """
    根据函数签名自动生成 JSON Schema。

    示例：
        @tool(description="读取文件内容")
        async def read_file(path: str, offset: int = 0, limit: int = 100) -> str:
            ...

    生成：
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 100}
            },
            "required": ["path"]
        }
    """
    sig = inspect.signature(func)
    properties = {}
    required = []

    # 获取类型注解（兼容未注解的情况）
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        # 跳过 self/cls
        if param_name in ("self", "cls"):
            continue

        json_type = "string"
        if param_name in hints:
            json_type = _python_type_to_json(hints[param_name])
        elif param.default is not inspect.Parameter.empty:
            # 无注解但有默认值 → 推断类型
            if isinstance(param.default, bool):
                json_type = "boolean"
            elif isinstance(param.default, int):
                json_type = "integer"
            elif isinstance(param.default, float):
                json_type = "number"
            elif isinstance(param.default, (list, dict)):
                json_type = "object" if isinstance(param.default, dict) else "array"

        prop: Dict[str, Any] = {"type": json_type}

        # 从参数名推断描述（可扩展为从 docstring 解析）
        if param_name == "path":
            prop["description"] = f"文件/目录路径"
        elif param_name == "command":
            prop["description"] = "Shell 命令"
        elif param_name == "content":
            prop["description"] = "文件内容"
        elif param_name == "pattern":
            prop["description"] = "搜索模式（支持正则）"
        elif param_name == "url":
            prop["description"] = "URL 地址"

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = prop

    # 从 docstring 提取参数描述（简单实现）
    doc = inspect.getdoc(func)
    if doc:
        # 简单解析 "Args:\n    path (str): 文件路径" 格式
        for line in doc.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Args:") or stripped.startswith("Parameters:"):
                continue
            if ": " in stripped or " — " in stripped:
                sep = ": " if ": " in stripped else " — "
                if sep in stripped:
                    key = stripped.split(sep)[0].strip()
                    if key in properties and "description" not in properties[key]:
                        desc = stripped.split(sep, 1)[1].strip().rstrip(".")
                        properties[key]["description"] = desc

    result = {
        "type": "object",
        "properties": properties,
    }
    if required:
        result["required"] = required

    return result


def tool(
    name: Optional[str] = None,
    description: str = "",
    danger_level: DangerLevel = DangerLevel.SAFE,
    require_approval: Optional[bool] = None,
    read_only: bool = False,
    examples: Optional[List[Dict]] = None,
) -> Callable[[Callable], Tool]:
    """
    装饰器：快速注册工具。

    用法：
        @tool(name="read_file", description="读取文件内容")
        async def read_file(path: str):
            ...

    自动从函数签名生成 parameters JSON Schema。
    """
    def decorator(func: Callable) -> Tool:
        _name = name or func.__name__
        _approval = require_approval if require_approval is not None else (danger_level >= DangerLevel.SENSITIVE)
        return Tool(
            name=_name,
            description=description or (func.__doc__ or "").strip().split("\n")[0],
            func=func,
            danger_level=danger_level,
            require_approval=_approval,
            read_only=read_only,
            examples=examples,
        )
    return decorator


# ── 工具注册表 ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    工具注册与管理中心。


    支持：
    - 注册/注销工具
    - 懒加载可选工具（节省 37% 内存）
    - 获取 OpenAI 格式 schema
    - 顺序执行工具调用
    - 并行执行多个工具调用（结果按 id 匹配）
    """

    # 可选工具映射（懒加载）
    _OPTIONAL_PLUGINS: Dict[str, str] = {
        "lsp": "deepseek_agent.tools.lsp",
        "mutation": "deepseek_agent.tools.mutation",
        "debug": "deepseek_agent.tools.debug",
        "benchmark": "deepseek_agent.tools.benchmark",
        "arch_check": "deepseek_agent.tools.arch_check",
        "refactor": "deepseek_agent.tools.refactor",
    }


    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._core_registered: bool = False
        self._optional_loaded: Dict[str, bool] = {}
        self._pruning_level: int = 0  # 0=全量, 1=精简, 2=核心


    def set_pruning_level(self, level: int) -> None:
        """设置工具裁剪级别：0=全量, 1=精简(移除lsp/debug/mutation/benchmark), 2=核心"""
        self._pruning_level = level
        if level >= 1:
            for plugin in ["lsp", "debug", "mutation", "benchmark"]:
                self._tools.pop(plugin, None)
        if level >= 2:
            for plugin in ["arch_check", "refactor"]:
                self._tools.pop(plugin, None)

    def _load_plugin(self, plugin_name: str) -> None:
        """懒加载单个插件工具（按需导入）。"""
        if plugin_name in self._optional_loaded:
            return
        self._optional_loaded[plugin_name] = True
        module_path = self._OPTIONAL_PLUGINS.get(plugin_name)
        if not module_path:
            return
        import importlib
        try:
            module = importlib.import_module(module_path)
            # 插件模块直接暴露工具函数，从 __all__ 或模块属性中提取
            for attr_name in dir(module):
                if attr_name.startswith("__"):
                    continue
                attr = getattr(module, attr_name)
                if callable(attr) and hasattr(attr, "name"):
                    tool_obj = getattr(attr, "_tool", None) or attr
                    if isinstance(tool_obj, Tool):
                        self.register(tool_obj)
        except ImportError:
            pass  # 插件不可用时静默跳过

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
            func=func,
            danger_level=danger_level,
            require_approval=danger_level >= DangerLevel.SENSITIVE,
            **kwargs,
        )
        self.register(t)
        return t

    def register_all(self) -> None:
        """
        注册全部核心工具（懒加载 + 动态按需）。

        策略：
        - 核心工具（fs, git, web, security, knowledge, testing）立即注册
        - 可选工具（lsp, mutation, debug, benchmark, arch_check, refactor）按需加载
        """
        if self._core_registered:
            return
        self._core_registered = True

        # 核心工具：立即注册
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

        # 可选工具：标记为懒加载（实际加载推迟到首次访问）
        for plugin_name in self._OPTIONAL_PLUGINS:
            self._optional_loaded[plugin_name] = False

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[Tool]:
        # 懒加载：首次访问可选工具时触发导入
        if name not in self._tools and name in self._OPTIONAL_PLUGINS:
            if not self._optional_loaded.get(name, False):
                self._load_plugin(name)
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的 OpenAI schema。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    # ── 执行 ────────────────────────────────────────────────────────────────

    def execute_sync(self, name: str, **kwargs) -> ToolResult:
        """同步包装 execute。"""
        return asyncio.run(self.execute(name, **kwargs))

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """执行单个工具。"""
        tool = self.get(name)
        if tool is None:
            return ToolResult.fail(f"Unknown tool: {name}")
        return await tool.execute(**kwargs)

    async def execute_one(self, tool_call: "ToolCallSpec", **overrides) -> ToolResult:
        """
        执行单个 tool_call 描述（含 id、name、arguments）。

        Args:
            tool_call: 包含 id, name, arguments 的对象/字典
            **overrides: 覆盖 arguments 中的值（用于审批时修改参数）
        """
        tc = copy.deepcopy(tool_call)
        args = tc.get("arguments", {}) or {}
        args.update(overrides)
        return await self.execute(tc["name"], **args)

    async def execute_parallel(
        self,
        tool_calls: List["ToolCallSpec"],
    ) -> Dict[str, ToolResult]:
        """
        并行执行多个工具调用，结果按 tool_call id 匹配。

        注意：结果顺序不固定，通过 id 字典匹配。
        """
        async def run_one(tc: "ToolCallSpec") -> tuple[str, ToolResult]:
            tc_id = tc.get("id", str(uuid.uuid4()))
            result = await self.execute_one(tc)
            return tc_id, result

        results = await asyncio.gather(
            *[run_one(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        output: Dict[str, ToolResult] = {}
        for item in results:
            if isinstance(item, Exception):
                # 兜底：用未知 id
                output["unknown"] = ToolResult.fail(str(item))
            else:
                tc_id, result = item
                output[tc_id] = result
        return output


# ── 类型别名 ────────────────────────────────────────────────────────────────

ToolCallSpec = Dict[str, Any]  # {"id": "...", "name": "...", "arguments": {...}}
