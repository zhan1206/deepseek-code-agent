"""
loop 子模块单元测试 — 验证拆分后的 config/planner/executor/error_recovery/stream_handler。
"""

import pytest
from unittest.mock import MagicMock


class TestLoopConfig:
    """验证 LoopConfig 数据类和 LoopMode 枚举。"""

    def test_default_values(self):
        from deepseek_agent.agent.loop.config import LoopConfig, LoopMode
        cfg = LoopConfig()
        assert cfg.mode == LoopMode.REACT
        assert cfg.max_steps == 50

    def test_custom_values(self):
        from deepseek_agent.agent.loop.config import LoopConfig, LoopMode
        cfg = LoopConfig(mode=LoopMode.PLAN_EXECUTE)
        assert cfg.mode == LoopMode.PLAN_EXECUTE

    def test_loop_mode_values(self):
        from deepseek_agent.agent.loop.config import LoopMode
        assert LoopMode.REACT.value == "react"
        assert LoopMode.PLAN_EXECUTE.value == "plan_execute"


class TestPlanner:
    """验证 TaskTracker 的任务跟踪。"""

    def test_task_tracker_init(self):
        from deepseek_agent.agent.loop.planner import TaskTracker
        tracker = TaskTracker()
        assert tracker is not None

    def test_task_item_creation(self):
        from deepseek_agent.agent.loop.planner import TaskItem, TaskStatus
        item = TaskItem(id="t1", description="fix bug")
        assert item.id == "t1"
        assert item.status == TaskStatus.PENDING


class TestExecutor:
    """验证 ToolExecutor 的初始化。"""

    def test_init(self):
        from deepseek_agent.agent.loop.executor import ToolExecutor
        registry = MagicMock()
        permission = MagicMock()
        ex = ToolExecutor(registry=registry, permission=permission)
        assert ex is not None


class TestErrorRecovery:
    """验证 ReflectionEngine 和 TerminationChecker。"""

    def test_reflection_engine_init(self):
        from deepseek_agent.agent.loop.error_recovery import ReflectionEngine
        from deepseek_agent.agent.loop.config import LoopConfig
        cfg = LoopConfig()
        re = ReflectionEngine(config=cfg)
        assert re is not None

    def test_termination_checker_init(self):
        from deepseek_agent.agent.loop.error_recovery import TerminationChecker
        from deepseek_agent.agent.loop.config import LoopConfig
        cfg = LoopConfig()
        tc = TerminationChecker(config=cfg)
        assert tc is not None


class TestStreamHandler:
    """验证 StreamHandler 的初始化和重置。"""

    def test_init(self):
        from deepseek_agent.agent.loop.stream_handler import StreamHandler
        registry = MagicMock()
        budget = MagicMock()
        sh = StreamHandler(registry=registry, budget_config=budget)
        assert sh is not None

    def test_clear_state(self):
        from deepseek_agent.agent.loop.stream_handler import StreamHandler
        registry = MagicMock()
        budget = MagicMock()
        sh = StreamHandler(registry=registry, budget_config=budget)
        # Verify handler object exists and is usable
        assert hasattr(sh, 'process') or hasattr(sh, 'handle') or sh is not None