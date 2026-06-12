"""
工具执行调度 — 并发只读工具 + 串行写工具 + 审批集成。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ...core.client import ToolCall
from ...tools.registry import ToolRegistry
from ...tools.base import ToolResult, DangerLevel
from .config import PermissionCallback


class ToolExecutor:
    """
    工具执行调度器。

    v2.0: 只读工具并发执行，写工具串行执行。
    支持权限审批集成。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission: PermissionCallback,
        task_tracker: Optional[Any] = None,  # TaskTracker, 避免循环导入
        loop_mode: Optional[Any] = None,     # LoopMode
    ):
        self.registry = registry
        self.permission = permission
        self.task_tracker = task_tracker
        self.loop_mode = loop_mode

    async def execute(
        self,
        tool_calls: List[ToolCall],
        total_tool_calls_counter: Dict[str, int],
    ) -> Dict[str, ToolResult]:
        """
        执行工具调用列表。

        Args:
            tool_calls: 待执行的工具调用列表
            total_tool_calls_counter: 共享的计数器 {"count": N}，用于更新总调用次数

        Returns:
            {tool_call_id: ToolResult}
        """
        if not tool_calls:
            return {}

        results: Dict[str, ToolResult] = {}

        # ── 分组：只读 vs 写 ──────────────────────────────────────────
        readonly_calls: List[tuple[ToolCall, Any]] = []
        write_calls: List[ToolCall] = []

        for tc in tool_calls:
            total_tool_calls_counter["count"] = total_tool_calls_counter.get("count", 0) + 1
            tool_obj = self.registry.get(tc.name)
            if tool_obj is None:
                results[tc.id] = ToolResult.fail(f"未知工具: {tc.name}")
                continue
            if tool_obj.read_only:
                readonly_calls.append((tc, tool_obj))
            else:
                write_calls.append(tc)

        # ── 并发执行只读工具 ──────────────────────────────────────────
        if readonly_calls:
            async def _run_readonly(tc: ToolCall, tool_obj: Any) -> tuple[str, ToolResult]:
                needs = await self.permission.requires_approval(
                    tc.name, tc.arguments, tool_obj.danger_level
                )
                if needs:
                    approved, modified = await self.permission.request_approval(
                        tc.name, tc.arguments, tool_obj.danger_level
                    )
                    if not approved:
                        return tc.id, ToolResult.fail("用户拒绝执行")
                    if modified:
                        tc = ToolCall(id=tc.id, name=tc.name, arguments=modified)
                result = await self.registry.execute(tc.name, **tc.arguments)
                return tc.id, result

            readonly_results = await asyncio.gather(
                *[_run_readonly(tc, tobj) for tc, tobj in readonly_calls],
                return_exceptions=True,
            )
            for item in readonly_results:
                if isinstance(item, Exception):
                    results["unknown"] = ToolResult.fail(str(item))
                else:
                    tc_id, result = item
                    results[tc_id] = result

        # ── 串行执行写工具（带审批）──────────────────────────────────────
        for tc in write_calls:
            tool_obj = self.registry.get(tc.name)
            if tool_obj is None:
                results[tc.id] = ToolResult.fail(f"未知工具: {tc.name}")
                continue

            needs_approval = await self.permission.requires_approval(
                tc.name, tc.arguments, tool_obj.danger_level
            )
            if needs_approval:
                approved, modified = await self.permission.request_approval(
                    tc.name, tc.arguments, tool_obj.danger_level
                )
                if not approved:
                    results[tc.id] = ToolResult.fail("用户拒绝执行")
                    continue
                if modified:
                    tc = ToolCall(id=tc.id, name=tc.name, arguments=modified)

            result = await self.registry.execute(tc.name, **tc.arguments)
            results[tc.id] = result

            # Plan-Execute 模式：更新任务状态
            if self.loop_mode and self.task_tracker:
                from .config import LoopMode
                if self.loop_mode == LoopMode.PLAN_EXECUTE:
                    pending = self.task_tracker.get_next_pending()
                    if pending:
                        self.task_tracker.mark_started(pending.id)
                        if result.success:
                            self.task_tracker.mark_completed(pending.id, str(result.data)[:500])
                        else:
                            self.task_tracker.mark_failed(pending.id, str(result.error))

        return results