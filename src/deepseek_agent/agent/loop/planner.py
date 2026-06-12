"""
任务追踪器 — Plan-Execute 模式的计划管理与状态追踪。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .config import LoopMode


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


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
        """修订计划，保留历史版本。"""
        completed = {t.id: t for t in self.plan if t.status == TaskStatus.COMPLETED}
        new_tasks: List[TaskItem] = []

        for desc in new_steps:
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