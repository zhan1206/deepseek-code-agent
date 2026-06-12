"""
工具裁剪策略 — 根据上下文预算动态调整可用工具集。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Set

from .base import Tool
from .registry import ToolRegistry, OPTIONAL_PLUGINS


class PruningLevel(Enum):
    """工具裁剪级别。"""
    FULL = "full"        # 所有工具
    STANDARD = "standard"  # 移除 lsp/debug/mutation/benchmark
    CORE = "core"        # 仅 fs + git + 核心工具
    MINIMAL = "minimal"  # 仅 fs + git


# 各级别要移除的插件组
_LEVEL_REMOVALS: Dict[int, Set[str]] = {
    1: {"lsp", "debug", "mutation", "benchmark"},       # STANDARD
    2: {"lsp", "debug", "mutation", "benchmark", "arch_check", "refactor"},  # CORE
    3: set(OPTIONAL_PLUGINS.keys()),                    # MINIMAL
}


class ToolPruner:
    """
    工具裁剪器 — 根据裁剪级别或上下文比例动态调整工具集。
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._current_level: int = 0

    @property
    def current_level(self) -> int:
        return self._current_level

    def set_level(self, level: int) -> None:
        """
        设置裁剪级别。

        0=FULL, 1=STANDARD, 2=CORE, 3=MINIMAL
        """
        self._current_level = level
        to_remove = _LEVEL_REMOVALS.get(level, set())
        for plugin in to_remove:
            self.registry.unregister(plugin)

    def prune_by_context_ratio(self, ratio: float) -> List[str]:
        """
        根据上下文使用比例自动裁剪。

        Args:
            ratio: 上下文使用比例 (0.0 ~ 1.0+)

        Returns:
            被移除的工具名称列表
        """
        removed = []
        if ratio >= 1.0:
            # 严重超限 → MINIMAL
            for plugin in OPTIONAL_PLUGINS:
                if self.registry.unregister(plugin):
                    removed.append(plugin)
            self._current_level = 3
        elif ratio >= 0.8:
            # 超限 → CORE
            for plugin in _LEVEL_REMOVALS[2]:
                if self.registry.unregister(plugin):
                    removed.append(plugin)
            self._current_level = 2
        elif ratio >= 0.6:
            # 接近上限 → STANDARD
            for plugin in _LEVEL_REMOVALS[1]:
                if self.registry.unregister(plugin):
                    removed.append(plugin)
            self._current_level = 1
        return removed