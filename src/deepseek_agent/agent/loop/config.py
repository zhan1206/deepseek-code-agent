"""
Agent Loop 配置 — 循环模式、终止条件、双模型分离。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ...tools.base import DangerLevel


# ── 枚举 ─────────────────────────────────────────────────────────────────

class LoopMode(Enum):
    REACT = "react"           # 交替推理 + 行动
    PLAN_EXECUTE = "plan_execute"  # 规划 → 执行 → 反思


# ── 循环配置 ────────────────────────────────────────────────────────────

@dataclass
class LoopConfig:
    """循环配置。"""
    mode: LoopMode = LoopMode.REACT
    max_steps: int = 50
    max_total_tool_calls: int = 100
    max_consecutive_same_result: int = 3  # 连续 N 次相同结果 → 终止
    max_execution_time: float = 600.0     # 10 分钟
    max_reflections: int = 2              # 反思上限
    reflection_threshold: float = 0.7     # 质量分数阈值
    # ── 双模型分离 ─────────────────────────────────────────────
    planner_model: str = "deepseek-reasoner"  # 规划/推理专用模型
    executor_model: str = "deepseek-chat"      # 执行/对话专用模型
    max_planner_calls: int = 3                # 每会话 Planner 模型调用上限
    # ── 上下文预算 ──────────────────────────────────────────────
    budget_max_tokens: int = 30000              # 上下文预算上限
    budget_summary_trigger: float = 0.80       # 使用率触发总结
    # ── 动态工具裁剪 ──────────────────────────────────────────
    tool_pruning_threshold: float = 0.80     # 上下文超过此比例时裁剪工具集
    planning_tools: Optional[List[str]] = None  # 规划阶段工具（默认只读工具）
    verbose_tools: Optional[List[str]] = None   # 大输出工具（超限时移除）


# ── 权限回调 ────────────────────────────────────────────────────────────

class PermissionCallback:
    """权限回调接口，子类实现具体审批逻辑。"""

    async def requires_approval(self, tool_name: str, args: Dict[str, Any], danger_level: DangerLevel) -> bool:
        return danger_level >= DangerLevel.SENSITIVE

    async def request_approval(
        self,
        tool_name: str,
        args: Dict[str, Any],
        danger_level: DangerLevel,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        请求用户审批。

        Returns:
            (approved, modified_args)
            - approved: 是否批准
            - modified_args: 批准时可选修改后的参数
        """
        return False, None


class CLIApprovalCallback(PermissionCallback):
    """CLI 交互审批回调。"""

    async def request_approval(
        self,
        tool_name: str,
        args: Dict[str, Any],
        danger_level: DangerLevel,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        level_str = {DangerLevel.SENSITIVE: "⚠️ 敏感", DangerLevel.DANGEROUS: "🚨 危险"}.get(
            danger_level, "🔒"
        )
        print(f"\n{level_str} 操作请求确认：")
        print(f"  工具: {tool_name}")
        # 脱敏显示参数
        safe_args = {k: ("***" if "key" in k.lower() or "token" in k.lower() else v)
                     for k, v in args.items()}
        print(f"  参数: {json.dumps(safe_args, ensure_ascii=False, indent=2)}")
        print(f"  是否批准? (y/n/edit) ", end="", flush=True)
        try:
            choice = input().strip().lower()
            if choice == "y" or choice == "yes":
                return True, None
            elif choice.startswith("edit"):
                return False, None
            return False, None
        except (EOFError, KeyboardInterrupt):
            return False, None