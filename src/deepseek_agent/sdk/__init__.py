"""
工具插件 SDK — 极简第三方工具注册。

使用方式：
  from deepseek_agent.sdk import tool

  @tool(name="query_feishu", description="查询飞书文档")
  def query_feishu(doc_url: str) -> str:
      return content

插件目录：~/.deepseek-agent/plugins/
每个插件一个 .py 文件，自动扫描加载。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# 复用核心 tool 装饰器
from ..tools.base import tool as _core_tool, Tool, ToolRegistry, DangerLevel


# ── SDK 接口 ─────────────────────────────────────────────────────────────

# 全局插件注册表
_plugin_registry: Dict[str, Tool] = {}


def tool(
    name: Optional[str] = None,
    description: str = "",
    danger_level: DangerLevel = DangerLevel.SAFE,
    require_approval: Optional[bool] = None,
    read_only: bool = False,
) -> Callable:
    """
    插件工具装饰器 — 与核心 tool 装饰器兼容。

    用法：
        @tool(name="my_tool", description="我的工具")
        def my_tool(arg1: str) -> str:
            return "result"
    """
    def decorator(func: Callable) -> Tool:
        t = _core_tool(
            name=name,
            description=description,
            danger_level=danger_level,
            require_approval=require_approval,
            read_only=read_only,
        )(func)
        _plugin_registry[t.name] = t
        return t

    return decorator


# ── 插件加载器 ───────────────────────────────────────────────────────────

class PluginLoader:
    """
    插件加载器 — 扫描并加载第三方工具。

    插件目录：~/.deepseek-agent/plugins/
    每个 .py 文件视为一个插件，使用 sdk.tool 装饰器注册工具。
    """

    def __init__(self, plugin_dir: Optional[str] = None):
        self.plugin_dir = Path(
            plugin_dir or os.path.expanduser("~/.deepseek-agent/plugins")
        )
        self.loaded: Dict[str, Dict[str, Any]] = {}  # name → {path, tools, error}

    def discover(self) -> List[str]:
        """发现所有插件文件。"""
        if not self.plugin_dir.exists():
            return []

        plugins = []
        for f in sorted(self.plugin_dir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            plugins.append(str(f))
        return plugins

    def load_plugin(self, plugin_path: str) -> Dict[str, Any]:
        """加载单个插件。"""
        path = Path(plugin_path)
        plugin_name = path.stem

        if plugin_name in self.loaded:
            return self.loaded[plugin_name]

        # 记录加载前的注册表
        before = set(_plugin_registry.keys())

        try:
            # 动态导入
            spec = importlib.util.spec_from_file_location(
                f"deepseek_agent_plugin_{plugin_name}",
                str(path),
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载插件: {plugin_name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"deepseek_agent_plugin_{plugin_name}"] = module
            spec.loader.exec_module(module)

            # 获取新增的工具
            after = set(_plugin_registry.keys())
            new_tools = after - before

            result = {
                "path": str(path),
                "tools": list(new_tools),
                "error": None,
            }

        except Exception as e:
            result = {
                "path": str(path),
                "tools": [],
                "error": str(e),
            }

        self.loaded[plugin_name] = result
        return result

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """加载所有插件。"""
        for path in self.discover():
            self.load_plugin(path)
        return self.loaded

    def register_to(self, registry: ToolRegistry) -> int:
        """将所有已加载插件工具注册到 ToolRegistry。"""
        count = 0
        for tool_name, tool_obj in _plugin_registry.items():
            try:
                registry.register(tool_obj)
                count += 1
            except Exception:
                pass
        return count

    def get_status(self) -> List[Dict[str, Any]]:
        """获取插件状态列表。"""
        status = []
        for name, info in self.loaded.items():
            status.append({
                "name": name,
                "path": info["path"],
                "tools": info["tools"],
                "error": info["error"],
                "status": "error" if info["error"] else "loaded",
            })
        return status
