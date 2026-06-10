"""
ParallelExecutor — 并行任务执行器。
支持子任务分解、并行执行、结果合并、冲突检测。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from .loop import AgentLoop, LoopConfig, LoopMode
from ..core.client import DeepSeekClient
from ..tools.base import ToolRegistry, ToolResult
from ..memory.manager import MemoryManager
from ..knowledge.parser import CodeParser, SymbolTable
from ..knowledge.graph import RelationGraph


# ── 数据模型 ────────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    """子任务。"""
    id: str
    description: str
    prompt: str
    depends_on: List[str] = field(default_factory=list)
    writes_files: List[str] = field(default_factory=list)  # 检测冲突
    status: str = "pending"   # pending | running | completed | failed | conflict
    result: Optional[str] = None
    duration: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "duration": self.duration,
            "error": self.error,
            "writes_files": self.writes_files,
        }


@dataclass
class TaskGroup:
    """可并行执行的任务组。"""
    tasks: List[SubTask]
    can_run_parallel: bool = True

    def __len__(self) -> int:
        return len(self.tasks)


@dataclass
class ParallelResult:
    """并行执行结果。"""
    total_tasks: int
    completed: int
    failed: int
    conflicts: int
    groups: List[TaskGroup]
    duration: float
    total_duration: float


# ── TaskDecomposer ─────────────────────────────────────────────────────────

class TaskDecomposer:
    """
    任务分解器 — 基于知识图谱分析子任务依赖。

    策略：
    1. 用 LLM 生成子任务列表
    2. 分析子任务之间的文件依赖关系
    3. 识别无依赖的子任务组（可并行）
    4. 生成 DAG 执行顺序
    """

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = project_root or "."
        self.parser = CodeParser()
        self.graph: Optional[RelationGraph] = None
        try:
            self.graph = RelationGraph()
            self.graph.build_from_dir(self.project_root)
        except Exception:
            pass

    async def decompose(
        self,
        task: str,
        max_subtasks: int = 8,
    ) -> List[SubTask]:
        """
        分解任务为子任务。

        Returns:
            子任务列表（包含依赖关系）
        """
        client = DeepSeekClient()

        prompt = f"""请将以下复杂任务分解为 {max_subtasks} 个或更少的独立子任务。

要求：
- 每个子任务可独立执行
- 子任务之间尽量减少依赖
- 每个子任务用 JSON 格式描述：
{{"description": "简短描述", "writes_files": ["file1.py"], "depends_on": []}}
- writes_files: 该子任务会修改的文件列表
- depends_on: 该子任务依赖的其他子任务描述（可选）

任务：
{task}
"""

        messages = [
            {"role": "system", "content": "你是一个任务分解专家。输出必须是有效的 JSON 数组。"},
            {"role": "user", "content": prompt},
        ]

        resp = await client.chat(messages, max_tokens=2048)
        content = resp.content or "[]"

        # 解析 JSON
        import json
        import re
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            return [SubTask(id=str(uuid.uuid4())[:8], description=task, prompt=task)]

        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return [SubTask(id=str(uuid.uuid4())[:8], description=task, prompt=task)]

        subtasks = []
        for item in items[:max_subtasks]:
            subtasks.append(SubTask(
                id=str(uuid.uuid4())[:8],
                description=item.get("description", ""),
                prompt=item.get("description", ""),
                writes_files=item.get("writes_files", []),
                depends_on=item.get("depends_on", []),
            ))

        return subtasks

    def detect_conflicts(self, subtasks: List[SubTask]) -> List[SubTask]:
        """
        检测文件写入冲突。

        如果两个子任务修改同一文件，标记后者为 conflict。
        """
        written: Dict[str, str] = {}  # file → task_id
        conflicts: List[SubTask] = []

        for task in subtasks:
            task_has_conflict = False
            for f in task.writes_files:
                if f in written:
                    task_has_conflict = True
                    task.status = "conflict"
                    task.error = f"文件冲突：{f} 已被 {written[f]} 修改"
                    conflicts.append(task)
                    break
            if not task_has_conflict:
                for f in task.writes_files:
                    written[f] = task.id

        return conflicts

    def group_independent(self, subtasks: List[SubTask]) -> List[TaskGroup]:
        """
        将无依赖子任务分组，同组可并行。
        """
        groups: List[TaskGroup] = []
        remaining = [t for t in subtasks if t.status != "conflict"]

        while remaining:
            # 找所有入度为 0 的任务（无依赖）
            independent = []
            for task in remaining:
                deps = set(task.depends_on)
                can_run = all(
                    dep not in [t.id for t in remaining if t.id != task.id]
                    for dep in deps
                ) or not deps
                if can_run:
                    independent.append(task)

            if not independent:
                # 循环依赖保护：取第一个
                independent = [remaining[0]]

            groups.append(TaskGroup(tasks=independent))
            for t in independent:
                remaining.remove(t)

        return groups


# ── ParallelExecutor ────────────────────────────────────────────────────────

class ParallelExecutor:
    """
    并行任务执行器。

    用法：
        executor = ParallelExecutor(max_parallel=4)
        result = await executor.execute(subtasks)
    """

    def __init__(
        self,
        max_parallel: int = 4,
        project_root: Optional[str] = None,
    ):
        self.max_parallel = max_parallel
        self.project_root = project_root or "."
        self.decomposer = TaskDecomposer(project_root)

        # 冲突检测
        self.detect_conflicts = self.decomposer.detect_conflicts
        self.group_independent = self.decomposer.group_independent

    async def execute(
        self,
        subtasks: List[SubTask],
        run_fn: Optional[Callable[[SubTask], Awaitable[str]]] = None,
    ) -> ParallelResult:
        """
        执行并行任务。

        Args:
            subtasks: 子任务列表
            run_fn: 执行函数，接收 SubTask，返回结果字符串。
                   如不提供，使用默认的 LLM 执行（单任务 AgentLoop）。

        Returns:
            ParallelResult 执行结果
        """
        start_time = time.time()
        total_duration = 0.0

        # 检测冲突
        conflicts = self.detect_conflicts(subtasks)

        # 分组
        groups = self.group_independent(subtasks)
        completed = 0
        failed = 0

        for group in groups:
            group_start = time.time()

            # 并发执行同组任务
            sem = asyncio.Semaphore(self.max_parallel)

            async def run_with_sem(task: SubTask) -> SubTask:
                async with sem:
                    task_start = time.time()
                    task.status = "running"

                    try:
                        if run_fn:
                            task.result = await run_fn(task)
                        else:
                            task.result = await self._default_run(task)
                        task.status = "completed"
                        task.duration = time.time() - task_start
                    except Exception as e:
                        task.status = "failed"
                        task.error = str(e)
                        task.duration = time.time() - task_start

                    return task

            results = await asyncio.gather(
                *[run_with_sem(t) for t in group.tasks],
                return_exceptions=True,
            )

            group_duration = time.time() - group_start
            total_duration += group_duration

            for r in results:
                if isinstance(r, SubTask):
                    if r.status == "completed":
                        completed += 1
                    elif r.status == "failed":
                        failed += 1

        return ParallelResult(
            total_tasks=len(subtasks),
            completed=completed,
            failed=failed,
            conflicts=len(conflicts),
            groups=groups,
            duration=time.time() - start_time,
            total_duration=total_duration,
        )

    async def _default_run(self, task: SubTask) -> str:
        """默认执行方式：调用 LLM 完成任务。"""
        client = DeepSeekClient()
        messages = [
            {"role": "system", "content": (
                "你是一个代码助手。请完成以下任务。"
                "优先使用工具读取文件、分析代码，再做修改。"
            )},
            {"role": "user", "content": task.prompt},
        ]
        resp = await client.chat(messages, max_tokens=4096)
        return resp.content or "(无输出)"

    async def decompose_and_execute(
        self,
        task: str,
        max_subtasks: int = 8,
        run_fn: Optional[Callable[[SubTask], Awaitable[str]]] = None,
    ) -> tuple[List[SubTask], ParallelResult]:
        """
        一步完成分解 + 执行。
        """
        subtasks = await self.decomposer.decompose(task, max_subtasks=max_subtasks)
        result = await self.execute(subtasks, run_fn)
        return subtasks, result

    def merge_results(
        self,
        subtasks: List[SubTask],
        format: str = "summary",
    ) -> str:
        """
        合并子任务结果。
        """
        if format == "summary":
            lines = ["# 并行执行结果汇总", ""]
            for task in subtasks:
                icon = {"completed": "✅", "failed": "❌", "conflict": "⚠️"}.get(task.status, "⏳")
                lines.append(f"{icon} [{task.id}] {task.description}")
                if task.result:
                    lines.append(f"   结果: {task.result[:200]}")
                if task.error:
                    lines.append(f"   错误: {task.error}")
            return "\n".join(lines)
        elif format == "json":
            import json
            return json.dumps([t.to_dict() for t in subtasks], ensure_ascii=False, indent=2)
        else:
            return "\n".join(f"[{t.id}] {t.status}: {t.result or t.error}" for t in subtasks)
