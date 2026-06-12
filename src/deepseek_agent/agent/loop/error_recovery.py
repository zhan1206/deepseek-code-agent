"""
终止条件检查 + 反思引擎 — 多维终止条件与 Plan-Execute 反思评估。
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Any

from .config import LoopConfig, LoopMode


class TerminationChecker:
    """
    多维终止条件检查器。

    终止维度：
    1. 达到最大步数
    2. 累计 tool_calls 超过上限
    3. 连续 N 次相同结果
    4. 执行超时
    5. 计划全部完成（Plan-Execute 模式）
    """

    def __init__(self, config: LoopConfig):
        self.config = config
        self._step_count: int = 0
        self._total_tool_calls: int = 0
        self._consecutive_same: int = 0
        self._last_tool_result: Optional[str] = None
        self._start_time: Optional[float] = None

    def reset(self):
        self._step_count = 0
        self._total_tool_calls = 0
        self._consecutive_same = 0
        self._last_tool_result = None
        self._start_time = time.time()

    def increment_step(self):
        self._step_count += 1

    def increment_tool_calls(self, n: int = 1):
        self._total_tool_calls += n

    def track_result(self, result: str):
        if result == self._last_tool_result:
            self._consecutive_same += 1
        else:
            self._consecutive_same = 0
            self._last_tool_result = result

    def should_terminate(self, task_tracker: Optional[Any] = None) -> bool:
        """综合检查是否应该终止循环。"""
        if self._step_count >= self.config.max_steps:
            return True
        if self._total_tool_calls >= self.config.max_total_tool_calls:
            return True
        if self._consecutive_same >= self.config.max_consecutive_same_result:
            return True
        if time.time() - (self._start_time or 0) > self.config.max_execution_time:
            return True
        if self.config.mode == LoopMode.PLAN_EXECUTE and task_tracker and not task_tracker.is_active:
            return True
        return False

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def total_tool_calls(self) -> int:
        return self._total_tool_calls


class ReflectionEngine:
    """
    反思引擎 — Plan-Execute 模式的质量评估。

    评估执行质量并决定是否需要修正计划。
    """

    def __init__(self, config: LoopConfig):
        self.config = config
        self._reflection_count: int = 0

    def reset(self):
        self._reflection_count = 0

    @property
    def reflection_count(self) -> int:
        return self._reflection_count

    @property
    def max_reflections_reached(self) -> bool:
        return self._reflection_count >= self.config.max_reflections

    async def reflect(self, client: Any, task_tracker: Any, memory: Any) -> bool:
        """
        反思当前执行质量。返回是否需要继续反思。

        Args:
            client: DeepSeekClient 实例
            task_tracker: TaskTracker 实例
            memory: MemoryManager 实例

        Returns:
            True: 需要继续反思/修正
            False: 质量足够好或达到反思上限
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
            {"role": "assistant", "content": task_tracker.get_plan_summary()},
            *memory.get_messages()[-6:],
        ]

        resp = await client.chat(messages, max_tokens=2048, temperature=0.3)
        try:
            body = json.loads(resp.content or "{}")
            quality = body.get("quality", 1.0)
            if quality >= self.config.reflection_threshold:
                return False
            reflection = body.get("reflection", "")
            memory.short_term.add_reflection_message(f"反思：{reflection}")
            self._reflection_count += 1
            return True
        except (json.JSONDecodeError, AttributeError):
            return False