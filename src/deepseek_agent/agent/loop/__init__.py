"""
Agent Loop 包 — 配置、规划、执行、反思、流式处理。

公共 API 从此模块导出，保持向后兼容。
"""

# 子模块导出
from .config import LoopConfig, LoopMode, PermissionCallback, CLIApprovalCallback
from .planner import TaskTracker, TaskItem, TaskStatus
from .executor import ToolExecutor
from .error_recovery import TerminationChecker, ReflectionEngine
from .stream_handler import StreamHandler

# ── AgentLoop 协调器 ────────────────────────────────────────────────────

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from ...core.client import DeepSeekClient, Response
from ...tools.registry import ToolRegistry
from ...memory.manager import MemoryManager
from ..context_budget import ContextBudget, BudgetConfig


class AgentLoop:
    """
    Agent 主循环，支持 ReAct 和 Plan-Execute 两种模式。

    逻辑委托给子模块：
    - TerminationChecker: 终止条件
    - ReflectionEngine: 反思评估
    - ToolExecutor: 工具执行调度
    - StreamHandler: 消息构建 + 上下文预算 + 工具裁剪
    - TaskTracker: 计划追踪
    """

    def __init__(
        self,
        client: DeepSeekClient,
        registry: ToolRegistry,
        memory: MemoryManager,
        config: Optional[LoopConfig] = None,
        permission_callback: Optional[PermissionCallback] = None,
    ):
        self.client = client
        self.registry = registry
        self.memory = memory
        self.config = config or LoopConfig()
        self.permission = permission_callback or PermissionCallback()
        self.task_tracker = TaskTracker()

        # 初始化子模块
        budget_config = BudgetConfig(
            max_tokens=self.config.budget_max_tokens,
            summary_trigger=self.config.budget_summary_trigger,
        )
        self.context_budget = ContextBudget(budget_config)
        self.termination = TerminationChecker(self.config)
        self.reflection = ReflectionEngine(self.config)
        self.executor = ToolExecutor(
            registry=self.registry,
            permission=self.permission,
            task_tracker=self.task_tracker,
            loop_mode=self.config.mode,
        )
        self.stream = StreamHandler(
            registry=self.registry,
            budget_config=budget_config,
            tool_pruning_threshold=self.config.tool_pruning_threshold,
            planning_tools=self.config.planning_tools,
            verbose_tools=self.config.verbose_tools,
        )
        self._planner_calls = 0

    # ── 公开 API ──────────────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        system_extra: str = "",
    ) -> AsyncGenerator[Response, None]:
        """运行 Agent。"""
        self._reset()
        self.memory.add_user_message(task)

        # 构建 system prompt
        base_system = self.stream.build_system_prompt(
            project_root=getattr(self.memory, 'project_root', None),
            existing_system_prompt=self.memory.short_term.system_prompt,
            system_extra=system_extra,
        )

        # Plan-Execute 模式：先生成计划
        if self.config.mode == LoopMode.PLAN_EXECUTE:
            if self._planner_calls >= self.config.max_planner_calls:
                yield Response(content=f"⚠️ Planner 调用已达上限（{self.config.max_planner_calls}），切换为 ReAct 模式")
                self.config.mode = LoopMode.REACT
            else:
                await self._generate_plan()
                self._planner_calls += 1
                yield Response(content=f"📋 **计划已生成**\n{self.task_tracker.get_plan_summary()}")

        # 主循环
        while not self.termination.should_terminate(self.task_tracker):
            if self.termination.step_count >= self.config.max_steps:
                yield Response(content=f"⚠️ 达到最大步数（{self.config.max_steps}），强制总结")
                yield await self._force_summarize()
                break

            # 构建消息
            messages = self.stream.build_messages(base_system, self.memory)

            # 上下文预算检查
            was_trimmed, messages, _ = self.stream.check_and_trim_budget(messages)

            tools = self.stream.select_tools(messages, self.config.budget_max_tokens)

            try:
                active_model = self.config.planner_model
                async for resp in self.client._stream_chat(messages, tools, "auto", model=active_model):
                    if resp.thinking:
                        yield resp

                    if resp.tool_calls:
                        # 工具执行
                        results = await self.executor.execute(
                            resp.tool_calls,
                            {"count": self.termination.total_tool_calls},
                        )
                        # 注入结果
                        for tc_id, result in results.items():
                            self.memory.add_tool_result(tc_id, result.to_str())
                            self.termination.track_result(result.to_str())

                        if self.termination.should_terminate(self.task_tracker):
                            break

                    elif resp.content:
                        self.memory.add_assistant_message(resp.content)
                        yield resp
                        return

            except Exception as e:
                yield Response(content=f"❌ 执行出错：{str(e)}")
                break

            self.termination.increment_step()

            # 反思检查（Plan-Execute 模式）
            if self.config.mode == LoopMode.PLAN_EXECUTE:
                needs_more = await self.reflection.reflect(self.client, self.task_tracker, self.memory)
                if not needs_more and self.reflection.max_reflections_reached:
                    yield Response(content="反思达到上限，终止")
                    return

            # 自动摘要检查
            await self.memory.summarize_if_needed(self.client)

        # 兜底
        yield await self._force_summarize()

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _reset(self):
        self.termination.reset()
        self.reflection.reset()
        self._planner_calls = 0

    async def _generate_plan(self) -> None:
        """生成初始计划。"""
        messages = [
            {"role": "system", "content": (
                "你是一个任务规划助手。请根据用户需求制定一个可执行的步骤计划。\n"
                "输出格式（必须是有效 JSON）：\n"
                '{"plan": ["步骤1", "步骤2", "步骤3"]}\n\n'
                "要求：\n- 步骤尽量细分到可独立执行\n- 每步不超过 50 字\n- 最多 10 个步骤"
            )},
            {"role": "user", "content": self.memory.short_term.get_raw_messages()[0].content if self.memory.short_term.get_raw_messages() else ""},
        ]

        resp = await self.client.chat(messages, max_tokens=1024)
        try:
            body = json.loads(resp.content or "{}")
            steps = body.get("plan", [])
            if isinstance(steps, list) and steps:
                self.task_tracker.init_plan(steps)
        except (json.JSONDecodeError, AttributeError):
            self.task_tracker.init_plan([resp.content or "执行任务"])

    async def _force_summarize(self) -> Response:
        """强制总结当前进展。"""
        messages = [
            {"role": "system", "content": (
                "你是一个总结助手。请总结当前 Agent 的执行进展，"
                "说明已完成的工作、遇到的问题、以及下一步建议。"
            )},
            *self.memory.get_messages(),
        ]
        resp = await self.client.chat(
            messages,
            max_tokens=2048,
            model=self.client.active_model,
        )
        return resp


__all__ = [
    "AgentLoop", "LoopConfig", "LoopMode", "PermissionCallback", "CLIApprovalCallback",
    "TaskTracker", "TaskItem", "TaskStatus",
    "ToolExecutor", "TerminationChecker", "ReflectionEngine", "StreamHandler",
]