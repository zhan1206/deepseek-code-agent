"""
Agent 主循环 — ReAct + Plan-Execute 双模式、多维终止条件、任务追踪器。
"""

from __future__ import annotations

import asyncio
import time
import uuid
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Generator, AsyncGenerator

from ..core.client import DeepSeekClient, Response, ToolCall
from ..tools.base import ToolRegistry, ToolResult, DangerLevel
from ..memory.manager import MemoryManager, ShortTermMemory
from .context_budget import ContextBudget, BudgetConfig, ContextPriority, ContextEntry


# ── 枚举 ─────────────────────────────────────────────────────────────────

class LoopMode(Enum):
    REACT = "react"           # 交替推理 + 行动
    PLAN_EXECUTE = "plan_execute"  # 规划 → 执行 → 反思


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ── 任务追踪器 ────────────────────────────────────────────────────────────

@dataclass
class TaskItem:
    """单个任务步骤。"""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    reflections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
        }


class TaskTracker:
    """
    任务追踪器，支持 Plan-Execute 模式。

    特性：
    - 保留计划版本历史（可回溯）
    - 支持步骤粒度的状态更新
    - 可中途修订计划
    """

    def __init__(self):
        self._plan_revisions: List[List[TaskItem]] = []
        self._current_revision: int = -1
        self._current_step: int = 0

    @property
    def plan(self) -> List[TaskItem]:
        if 0 <= self._current_revision < len(self._plan_revisions):
            return self._plan_revisions[self._current_revision]
        return []

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def is_active(self) -> bool:
        """还有未完成的任务。"""
        return any(t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS) for t in self.plan)

    def init_plan(self, steps: List[str]) -> None:
        """从字符串列表初始化计划。"""
        tasks = [
            TaskItem(id=str(uuid.uuid4())[:8], description=s)
            for s in steps
        ]
        self._plan_revisions.append(tasks)
        self._current_revision = len(self._plan_revisions) - 1
        self._current_step = 0

    def mark_started(self, step_id: str) -> None:
        for t in self.plan:
            if t.id == step_id:
                t.status = TaskStatus.IN_PROGRESS
                return

    def mark_completed(self, step_id: str, result: str) -> None:
        for t in self.plan:
            if t.id == step_id:
                t.status = TaskStatus.COMPLETED
                t.result = result
                self._current_step += 1
                return

    def mark_failed(self, step_id: str, error: str) -> None:
        for t in self.plan:
            if t.id == step_id:
                t.status = TaskStatus.FAILED
                t.result = error
                return

    def get_next_pending(self) -> Optional[TaskItem]:
        for t in self.plan:
            if t.status == TaskStatus.PENDING:
                return t
        return None

    def revise_plan(self, new_steps: List[str]) -> None:
        """
        修订计划，保留历史版本。

        策略：保留已完成步骤，替换未完成步骤。
        """
        completed = {t.id: t for t in self.plan if t.status == TaskStatus.COMPLETED}
        new_tasks: List[TaskItem] = []

        for desc in new_steps:
            # 检查是否与已完成步骤重复
            matched = next((t for t in completed.values() if t.description == desc), None)
            if matched:
                new_tasks.append(matched)
            else:
                new_tasks.append(TaskItem(
                    id=str(uuid.uuid4())[:8],
                    description=desc,
                ))

        self._plan_revisions.append(new_tasks)
        self._current_revision += 1

        # 重置当前步骤索引
        for i, t in enumerate(new_tasks):
            if t.status == TaskStatus.PENDING:
                self._current_step = i
                break

    def get_plan_summary(self) -> str:
        """生成可读的计划状态字符串。"""
        if not self.plan:
            return "(无计划)"
        lines = []
        for i, t in enumerate(self.plan):
            icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.SKIPPED: "⏭️",
            }.get(t.status, "?")
            lines.append(f"{icon} {i+1}. {t.description}")
            if t.result and len(t.result) < 200:
                lines.append(f"   └─ {t.result[:200]}")
        return "\n".join(lines)

    def get_history(self) -> List[str]:
        """获取计划版本历史摘要。"""
        return [
            f"v{i+1}: {[t.description[:40] for t in plan]}"
            for i, plan in enumerate(self._plan_revisions)
        ]


# ── Agent Loop ────────────────────────────────────────────────────────────

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
    # ── 上下文预算 ──────────────────────────────────────────────
    budget_max_tokens: int = 30000              # 上下文预算上限
    budget_summary_trigger: float = 0.80       # 使用率触发总结


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
                # 支持参数编辑
                return False, None
            return False, None
        except (EOFError, KeyboardInterrupt):
            return False, None


class AgentLoop:
    """
    Agent 主循环，支持 ReAct 和 Plan-Execute 两种模式。

    终止条件（多维度）：
    1. 达到最大步数
    2. 累计 tool_calls 超过上限
    3. 连续 N 次相同结果
    4. 执行超时
    5. 模型输出最终回复（非 tool_calls）
    6. 计划全部完成（Plan-Execute 模式）
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

        # 上下文预算
        budget_config = BudgetConfig(
            max_tokens=self.config.budget_max_tokens,
            summary_trigger=self.config.budget_summary_trigger,
        )
        self.context_budget = ContextBudget(budget_config)

        # 运行时状态
        self._step_count = 0
        self._total_tool_calls = 0
        self._consecutive_same = 0
        self._last_tool_result: Optional[str] = None
        self._start_time: Optional[float] = None
        self._reflection_count = 0

    # ── 公开 API ──────────────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        system_extra: str = "",
    ) -> AsyncGenerator[Response, None]:
        """
        运行 Agent。

        Args:
            task: 用户任务描述
            system_extra: 追加到 system prompt 的额外内容
        """
        self._reset()
        self.memory.add_user_message(task)

        # 构建 system prompt
        base_system = self._build_system_prompt()
        if system_extra:
            base_system += "\n\n" + system_extra
        if self.memory.short_term.system_prompt:
            base_system = self.memory.short_term.system_prompt + "\n" + base_system

        # Plan-Execute 模式：先生成计划
        if self.config.mode == LoopMode.PLAN_EXECUTE:
            await self._generate_plan()
            yield Response(content=f"📋 **计划已生成**\n{self.task_tracker.get_plan_summary()}")

        # 主循环
        while not self._should_terminate():
            if self._step_count >= self.config.max_steps:
                yield Response(content=f"⚠️ 达到最大步数（{self.config.max_steps}），强制总结")
                yield await self._force_summarize()
                break

            # 构建消息
            messages = self._build_messages(base_system)

            # ── 上下文预算检查 ──────────────────────────────────────────
            budget = ContextBudget.from_messages(
                messages,
                BudgetConfig(
                    max_tokens=self.config.budget_max_tokens,
                    summary_trigger=self.config.budget_summary_trigger,
                ),
            )
            was_trimmed, removed = budget.check_budget()
            if was_trimmed:
                messages = budget.to_messages()

            tools = self.registry.get_schemas()

            try:
                # ── 双模型：规划/推理用 reasoner，执行用 chat ──────────────
                active_model = self.config.planner_model
                async for resp in self.client._stream_chat(messages, tools, "auto", model=active_model):
                    # 阶段性 yield
                    if resp.thinking:
                        yield resp  # 思考过程

                    if resp.tool_calls:
                        # 工具执行阶段切换为 executor_model
                        tool_model = self.config.executor_model
                        results = await self._execute_tool_calls(resp.tool_calls)
                        # 注入结果
                        for tc_id, result in results.items():
                            self.memory.add_tool_result(tc_id, result.to_str())
                            self._track_result(result.to_str())

                        # 检查终止
                        if self._should_terminate():
                            break

                    elif resp.content:
                        # 最终回复
                        self.memory.add_assistant_message(resp.content)
                        yield resp
                        return

            except Exception as e:
                yield Response(content=f"❌ 执行出错：{str(e)}")
                break

            self._step_count += 1

            # 反思检查（Plan-Execute 模式）
            if self.config.mode == LoopMode.PLAN_EXECUTE:
                needs_more = await self._reflect()
                if not needs_more and self._reflection_count >= self.config.max_reflections:
                    yield Response(content="反思达到上限，终止")
                    return

            # 自动摘要检查
            await self.memory.summarize_if_needed(self.client)

        # 兜底
        yield await self._force_summarize()

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _reset(self):
        self._step_count = 0
        self._total_tool_calls = 0
        self._consecutive_same = 0
        self._last_tool_result = None
        self._start_time = time.time()
        self._reflection_count = 0

    def _should_terminate(self) -> bool:
        if self._step_count >= self.config.max_steps:
            return True
        if self._total_tool_calls >= self.config.max_total_tool_calls:
            return True
        if self._consecutive_same >= self.config.max_consecutive_same_result:
            return True
        if time.time() - (self._start_time or 0) > self.config.max_execution_time:
            return True
        if self.config.mode == LoopMode.PLAN_EXECUTE and not self.task_tracker.is_active:
            return True
        return False

    def _track_result(self, result: str):
        if result == self._last_tool_result:
            self._consecutive_same += 1
        else:
            self._consecutive_same = 0
            self._last_tool_result = result

    def _build_system_prompt(self) -> str:
        tools_list = "\n".join(
            f"- **{t.name}**: {t.description}"
            for t in self.registry._tools.values()
        )
        return (
            "你是一个严谨的代码助手。\n"
            f"可用工具：\n{tools_list}\n\n"
            "规则：\n"
            "1. 优先使用工具读取文件、分析代码，再做修改\n"
            "2. 修改前先确认旧字符串在文件中唯一\n"
            "3. 执行危险操作前告知用户潜在风险\n"
            "4. 如果执行失败，给出清晰的错误信息和修复建议\n"
        )

    def _build_messages(self, base_system: str) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": base_system}]
        # 注入项目上下文（如果可用）
        project_ctx = self.memory.short_term.system_prompt or ""
        msgs[0]["content"] = base_system
        for m in self.memory.get_messages():
            msgs.append(m)
        return msgs

    async def _generate_plan(self) -> None:
        """生成初始计划。"""
        messages = [
            {"role": "system", "content": (
                "你是一个任务规划助手。请根据用户需求制定一个可执行的步骤计划。\n"
                "输出格式（必须是有效 JSON）：\n"
                '{"plan": ["步骤1", "步骤2", "步骤3"]}\n\n'
                "要求：\n"
                "- 步骤尽量细分到可独立执行\n"
                "- 每步不超过 50 字\n"
                "- 最多 10 个步骤"
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
            # 降级：单步计划
            self.task_tracker.init_plan([resp.content or "执行任务"])

    async def _execute_tool_calls(
        self, tool_calls: List[ToolCall]
    ) -> Dict[str, ToolResult]:
        """
        执行工具调用列表，支持并行 + 审批。

        v2.0: 只读工具并发执行，写工具串行执行。
        """
        if not tool_calls:
            return {}

        results: Dict[str, ToolResult] = {}

        # ── 分组：只读 vs 写 ──────────────────────────────────────────
        readonly_calls: List[tuple[ToolCall, Any]] = []  # (tc, tool_obj)
        write_calls: List[ToolCall] = []

        for tc in tool_calls:
            self._total_tool_calls += 1
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
                    self.memory.add_tool_result(tc.id, results[tc.id].to_str())
                    continue
                if modified:
                    tc = ToolCall(id=tc.id, name=tc.name, arguments=modified)

            result = await self.registry.execute(tc.name, **tc.arguments)
            results[tc.id] = result

            # Plan-Execute 模式：更新任务状态
            if self.config.mode == LoopMode.PLAN_EXECUTE:
                pending = self.task_tracker.get_next_pending()
                if pending:
                    self.task_tracker.mark_started(pending.id)
                    if result.success:
                        self.task_tracker.mark_completed(pending.id, str(result.data)[:500])
                    else:
                        self.task_tracker.mark_failed(pending.id, str(result.error))

        return results

    async def _force_summarize(self) -> Response:
        """强制总结当前进展。"""
        messages = [
            {"role": "system", "content": (
                "你是一个总结助手。请总结当前 Agent 的执行进展，"
                "说明已完成的工作、遇到的问题、以及下一步建议。"
            )},
            *self.memory.get_messages(),
        ]
        resp = await self.client.chat(messages, max_tokens=2048)
        return resp

    async def _reflect(self) -> bool:
        """
        反思当前执行质量。返回是否需要继续反思。
        """
        if self._reflection_count >= self.config.max_reflections:
            return False

        messages = [
            {"role": "system", "content": (
                "你是一个反思助手。请评估最近几步的执行质量，"
                "指出问题并给出修正建议。如果质量已经足够好，回答 'OK'。"
                "否则，给出反思内容和修改计划（JSON 格式）。"
                "\n格式：{\"quality\": 0.8, \"reflection\": \"...\", \"suggestions\": [\"...\"]}"
            )},
            {"role": "assistant", "content": self.task_tracker.get_plan_summary()},
            *self.memory.get_messages()[-6:],
        ]

        resp = await self.client.chat(messages, max_tokens=2048, temperature=0.3)
        try:
            body = json.loads(resp.content or "{}")
            quality = body.get("quality", 1.0)
            if quality >= self.config.reflection_threshold:
                return False
            # 注入反思内容
            reflection = body.get("reflection", "")
            self.memory.short_term.add_reflection_message(f"反思：{reflection}")
            self._reflection_count += 1
            return True
        except (json.JSONDecodeError, AttributeError):
            return False