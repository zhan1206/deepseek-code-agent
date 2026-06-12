"""
TestDrivenLoop — TDD 红绿重构循环。
状态机：RED → RUN_RED → GREEN → RUN_GREEN → REFACTOR → RUN_REFACTOR
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from .loop import AgentLoop, LoopConfig, LoopMode, TaskTracker, TaskStatus
from ..core.client import DeepSeekClient, Response
from ..tools.registry import ToolRegistry
from ..tools.base import ToolResult
from ..memory.manager import MemoryManager
from ..memory.checkpoint import CheckpointManager
from ..tools.mutation import MutateCode


# ── 状态机 ──────────────────────────────────────────────────────────────────

class TDDState(Enum):
    RED         = "red"          # 写失败的测试
    RUN_RED     = "run_red"      # 运行测试，确认失败
    GREEN       = "green"        # 写通过测试的最少代码
    RUN_GREEN   = "run_green"    # 运行测试，确认通过
    REFACTOR    = "refactor"     # 重构改进
    RUN_REFACTOR= "run_refactor" # 验证重构不破坏测试
    DONE        = "done"


TDD_TRANSITIONS = {
    TDDState.RED: TDDState.RUN_RED,
    TDDState.RUN_RED: TDDState.GREEN,
    TDDState.GREEN: TDDState.RUN_GREEN,
    TDDState.RUN_GREEN: TDDState.REFACTOR,
    TDDState.REFACTOR: TDDState.RUN_REFACTOR,
    TDDState.RUN_REFACTOR: TDDState.RED,
}

TDD_SUMMARIES = {
    TDDState.RED:         "🔴 写一个失败的测试（描述期望行为）",
    TDDState.RUN_RED:     "🔴 运行测试，确认失败",
    TDDState.GREEN:       "🟢 写最少代码使测试通过",
    TDDState.RUN_GREEN:   "🟢 运行测试，确认通过",
    TDDState.REFACTOR:    "♻️  重构代码（清理、改进结构）",
    TDDState.RUN_REFACTOR:"♻️  运行测试，确认重构不破坏功能",
    TDDState.DONE:        "✅ TDD 循环完成",
}


# ── TDD Loop ────────────────────────────────────────────────────────────────

@dataclass
class TDDConfig:
    """TDD 循环配置。"""
    max_cycles: int = 20           # 最大循环次数
    auto_run: bool = True          # 自动运行测试
    mutation_testing: bool = False # REFACTOR 阶段使用变异测试
    checkpoint_enabled: bool = True # 中断恢复


@dataclass
class TDDCycleRecord:
    """单次循环记录。"""
    cycle: int
    state: TDDState
    test_result: Optional[Dict[str, Any]] = None
    code_result: Optional[Dict[str, Any]] = None
    refactor_result: Optional[Dict[str, Any]] = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


class TestDrivenLoop:
    """
    TDD 循环 — 红绿重构。

    用法：
        loop = TestDrivenLoop(client, registry, memory)
        async for event in loop.run(task="为 Calculator 实现加减乘除"):
            print(event)
    """

    def __init__(
        self,
        client: DeepSeekClient,
        registry: ToolRegistry,
        memory: MemoryManager,
        config: Optional[TDDConfig] = None,
    ):
        self.client = client
        self.registry = registry
        self.memory = memory
        self.config = config or TDDConfig()
        self.checkpoint = CheckpointManager() if self.config.checkpoint_enabled else None
        self.mutator = MutateCode()

        # 运行时状态
        self._state = TDDState.RED
        self._cycle = 0
        self._history: List[TDDCycleRecord] = []
        self._test_file: Optional[str] = None
        self._impl_file: Optional[str] = None

    # ── 公开 API ────────────────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        test_file: Optional[str] = None,
        impl_file: Optional[str] = None,
    ) -> AsyncGenerator[Response, None]:
        """
        运行 TDD 循环。

        Args:
            task: 任务描述
            test_file: 测试文件路径
            impl_file: 实现文件路径
        """
        self._test_file = test_file
        self._impl_file = impl_file
        self._state = TDDState.RED
        self._cycle = 0

        # 尝试从 checkpoint 恢复
        if self.checkpoint:
            saved = self.checkpoint.load("tdd_state")
            if saved:
                self._state = TDDState(saved.get("state", "red"))
                self._cycle = saved.get("cycle", 0)

        self.memory.add_user_message(f"[TDD 模式] {task}")

        while self._cycle < self.config.max_cycles:
            if self._state == TDDState.DONE:
                break

            yield Response(content=f"\n{'='*40}\n"
                                    f"🔄 TDD 循环 #{self._cycle + 1} | "
                                    f"状态: {TDD_SUMMARIES.get(self._state, '')}")

            # 执行当前状态
            record = TDDCycleRecord(cycle=self._cycle + 1, state=self._state)
            start = time.time()

            try:
                record = await self._execute_state(record)
            except Exception as e:
                yield Response(content=f"❌ 状态执行出错: {e}")
                break

            record.duration = time.time() - start
            self._history.append(record)

            # checkpoint
            if self.checkpoint:
                self.checkpoint.save("tdd_state", {
                    "state": self._state.value,
                    "cycle": self._cycle,
                })

            # 推进状态机
            self._advance_state()

            # 检查是否完成（连续 GREEN + REFACTOR 各通过）
            if self._state == TDDState.DONE:
                break

            self._cycle += 1

        # 总结
        yield await self._summarize()

    # ── 状态执行 ────────────────────────────────────────────────────────────

    async def _execute_state(self, record: TDDCycleRecord) -> TDDCycleRecord:
        """执行当前状态的操作。"""
        if self._state == TDDState.RED:
            record = await self._do_red(record)
        elif self._state == TDDState.RUN_RED:
            record = await self._do_run_red(record)
        elif self._state == TDDState.GREEN:
            record = await self._do_green(record)
        elif self._state == TDDState.RUN_GREEN:
            record = await self._do_run_green(record)
        elif self._state == TDDState.REFACTOR:
            record = await self._do_refactor(record)
        elif self._state == TDDState.RUN_REFACTOR:
            record = await self._do_run_refactor(record)
        return record

    async def _do_red(self, record: TDDCycleRecord) -> TDDCycleRecord:
        """RED：LLM 写失败的测试。"""
        # 使用 AgentLoop 的 LLM 调用能力生成测试
        messages = [
            {"role": "system", "content": (
                "你是一个 TDD 测试工程师。请为以下任务编写 pytest 测试用例。"
                "测试应该描述期望的行为，当前尚未实现，所以测试应该失败。"
                "\n只输出测试代码（Python 文件内容），不需要解释。"
            )},
            {"role": "user", "content": self.memory.short_term.get_raw_messages()[0].content if self.memory.short_term.get_raw_messages() else "实现功能"},
        ]
        resp = await self.client.chat(messages, max_tokens=4096)
        code = resp.content or ""

        # 写入测试文件
        if self._test_file:
            from pathlib import Path
            Path(self._test_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self._test_file).write_text(code, encoding="utf-8")

        record.test_result = {"generated": True, "lines": len(code.splitlines())}
        return record

    async def _do_run_red(self, record: TDDCycleRecord) -> TDDCycleRecord:
        """RUN_RED：运行测试，确认失败。"""
        if self._test_file and self.config.auto_run:
            from ..tools.testing import run_test_suite
            result = run_test_suite(self._test_file, verbose=False)
            record.test_result = result.to_dict() if hasattr(result, "to_dict") else str(result)

            # 确认失败
            data = record.test_result
            failed = data.get("failed", 0) + data.get("errors", 0)
            if failed > 0:
                record.test_result["status"] = "red_ok"
            else:
                record.test_result["status"] = "unexpected_pass"
        return record

    async def _do_green(self, record: TDDCycleRecord) -> TDDCycleRecord:
        """GREEN：写最少代码使测试通过。"""
        messages = [
            {"role": "system", "content": (
                "你是一个 TDD 实现工程师。请编写最少量代码使测试通过。"
                "不要过度设计，只实现测试期望的行为。"
                "\n只输出代码，不需要解释。"
            )},
            {"role": "user", "content": f"测试文件: {self._test_file}\n\n"
                                        "请读取测试文件并实现相应功能。"},
        ]
        resp = await self.client.chat(messages, max_tokens=4096)
        code = resp.content or ""

        if self._impl_file:
            from pathlib import Path
            Path(self._impl_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self._impl_file).write_text(code, encoding="utf-8")

        record.code_result = {"implemented": True, "lines": len(code.splitlines())}
        return record

    async def _do_run_green(self, record: TDDCycleRecord) -> TDDCycleRecord:
        """RUN_GREEN：运行测试，确认通过。"""
        if self._test_file and self.config.auto_run:
            from ..tools.testing import run_test_suite
            result = run_test_suite(self._test_file, verbose=False)
            record.code_result = result.to_dict() if hasattr(result, "to_dict") else {}

            data = record.code_result
            if data.get("failed", 0) == 0 and data.get("errors", 0) == 0:
                record.code_result["status"] = "green_ok"
                # 两次 GREEN 循环后进入 DONE
            else:
                record.code_result["status"] = "still_red"
        return record

    async def _do_refactor(self, record: TDDCycleRecord) -> TDDCycleRecord:
        """REFACTOR：重构改进。"""
        messages = [
            {"role": "system", "content": (
                "你是一个重构专家。请改进代码质量（可读性、简洁性、设计模式）。"
                "测试必须继续通过。只输出重构后的代码。"
            )},
            {"role": "user", "content": f"实现文件: {self._impl_file}\n\n请读取文件并进行重构改进。"},
        ]
        resp = await self.client.chat(messages, max_tokens=4096)
        code = resp.content or ""

        if self._impl_file:
            from pathlib import Path
            Path(self._impl_file).write_text(code, encoding="utf-8")

        record.refactor_result = {"refactored": True, "lines": len(code.splitlines())}
        return record

    async def _do_run_refactor(self, record: TDDCycleRecord) -> TDDCycleRecord:
        """RUN_REFACTOR：验证重构不破坏测试。"""
        if self._test_file and self.config.auto_run:
            from ..tools.testing import run_test_suite
            result = run_test_suite(self._test_file, verbose=False)
            record.refactor_result = result.to_dict() if hasattr(result, "to_dict") else {}

            data = record.refactor_result
            if data.get("failed", 0) == 0 and data.get("errors", 0) == 0:
                record.refactor_result["status"] = "refactor_ok"
            else:
                record.refactor_result["status"] = "refactor_broke_tests"
        return record

    # ── 辅助 ────────────────────────────────────────────────────────────────

    def _advance_state(self) -> None:
        """推进状态机。"""
        prev = self._state

        if self._state == TDDState.RED:
            self._state = TDDState.RUN_RED
        elif self._state == TDDState.RUN_RED:
            self._state = TDDState.GREEN
        elif self._state == TDDState.GREEN:
            self._state = TDDState.RUN_GREEN
        elif self._state == TDDState.RUN_GREEN:
            # 检查是否全部通过 → 进入 DONE 或继续 REFACTOR
            green_ok = any(
                r.code_result and r.code_result.get("status") == "green_ok"
                for r in self._history[-2:]
            )
            if green_ok and self._cycle >= 2:
                self._state = TDDState.DONE
            else:
                self._state = TDDState.REFACTOR
        elif self._state == TDDState.REFACTOR:
            self._state = TDDState.RUN_REFACTOR
        elif self._state == TDDState.RUN_REFACTOR:
            self._state = TDDState.RED  # 下一轮继续 RED
            self._cycle += 1
        elif self._state == TDDState.DONE:
            pass

    async def _summarize(self) -> Response:
        """生成 TDD 循环总结。"""
        lines = ["\n" + "=" * 40, "📊 TDD 循环总结"]
        lines.append(f"循环次数: {len(self._history)}")
        lines.append(f"最终状态: {self._state.value}")

        # 统计
        passed = sum(1 for r in self._history if r.code_result and r.code_result.get("status") == "green_ok")
        refactored = sum(1 for r in self._history if r.refactor_result and r.refactor_result.get("status") == "refactor_ok")
        total_duration = sum(r.duration for r in self._history)

        lines.append(f"绿色阶段通过: {passed}")
        lines.append(f"重构验证通过: {refactored}")
        lines.append(f"总耗时: {total_duration:.1f}s")

        return Response(content="\n".join(lines))
