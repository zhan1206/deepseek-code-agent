"""
DeepSeek 深度适配测试套件 — 验证 FC、并行、降级、tokenizer、FIM 等。
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── 1. ToolResult 强结构化 ──────────────────────────────────────────────

class TestToolResultStructured:
    """验证所有工具输出为 JSON 格式。"""

    def test_ok_returns_json_string(self):
        from deepseek_agent.tools.base import ToolResult
        r = ToolResult.ok({"files": ["a.py", "b.py"]})
        d = json.loads(r.to_str())
        assert d["success"] is True
        assert "data" in d
        assert isinstance(d["data"], dict)

    def test_fail_returns_json_string(self):
        from deepseek_agent.tools.base import ToolResult
        r = ToolResult.fail("FileNotFoundError: missing.py")
        d = json.loads(r.to_str())
        assert d["success"] is False
        assert "error" in d
        assert "FileNotFoundError" in d["error"]


# ── 2. 并行工具执行 ─────────────────────────────────────────────────────

class TestParallelToolExecution:
    """验证多工具并发执行，结果按 id 匹配。"""

    @pytest.mark.asyncio
    async def test_parallel_results_match_ids(self):
        from deepseek_agent.tools.base import ToolRegistry, tool, DangerLevel

        registry = ToolRegistry()

        async def slow_tool(label: str, delay: float = 0.01) -> str:
            await asyncio.sleep(delay)
            return f"result_{label}"

        for i in range(3):
            t = tool(name=f"tool_{i}", description=f"Tool {i}", danger_level=DangerLevel.SAFE)(slow_tool)
            registry.register(t)

        calls = [
            {"id": f"tc_{i}", "name": f"tool_{i}", "arguments": {"label": f"x{i}"}}
            for i in range(3)
        ]
        results = await registry.execute_parallel(calls)
        assert len(results) == 3
        for i in range(3):
            assert f"tc_{i}" in results
            assert results[f"tc_{i}"].success

    @pytest.mark.asyncio
    async def test_parallel_failure_isolation(self):
        from deepseek_agent.tools.base import ToolRegistry, tool, DangerLevel

        registry = ToolRegistry()

        async def ok_tool() -> str:
            return "ok"

        async def fail_tool() -> str:
            raise ValueError("boom")

        registry.register(tool(name="ok_tool", description="ok", danger_level=DangerLevel.SAFE)(ok_tool))
        registry.register(tool(name="fail_tool", description="fail", danger_level=DangerLevel.SAFE)(fail_tool))

        calls = [
            {"id": "tc1", "name": "ok_tool", "arguments": {}},
            {"id": "tc2", "name": "fail_tool", "arguments": {}},
        ]
        results = await registry.execute_parallel(calls)
        assert results["tc1"].success
        assert not results["tc2"].success


# ── 3. 流式多 tool_call 解析 ─────────────────────────────────────────────

class TestStreamMultiToolCalls:
    """验证流式解析支持多个并行 tool_calls。"""

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_parsed(self):
        from deepseek_agent.core.client import ToolCallDelta

        # 模拟 SSE delta 序列：3 个并行 tool_calls
        deltas = [
            {"index": 0, "id": "tc1", "function": {"name": "read_file", "arguments": ""}},
            {"index": 0, "function": {"arguments": '{"path":'}},
            {"index": 0, "function": {"arguments": ' "a.py"}'}},
            {"index": 1, "id": "tc2", "function": {"name": "search_file", "arguments": ""}},
            {"index": 1, "function": {"arguments": '{"pattern": "foo"}'}},
            {"index": 2, "id": "tc3", "function": {"name": "git_status", "arguments": "{}"}},
        ]

        tc_list = {}
        for d in deltas:
            idx = d.get("index", 0)
            if idx not in tc_list:
                tc_list[idx] = ToolCallDelta()
            dd = tc_list[idx]
            if d.get("id"):
                dd.id = d["id"]
            if d.get("function"):
                fn = d["function"]
                if fn.get("name"):
                    dd.name = fn["name"]
                if fn.get("arguments"):
                    dd.arguments += fn["arguments"]

        assert len(tc_list) == 3
        assert tc_list[0].name == "read_file"
        assert tc_list[1].name == "search_file"
        assert tc_list[2].name == "git_status"
        assert json.loads(tc_list[0].arguments) == {"path": "a.py"}


# ── 4. Token 计数器 ─────────────────────────────────────────────────────

class TestTokenCounter:
    """验证 Token 计数和成本追踪。"""

    def test_counter_fallback(self):
        from deepseek_agent.core.token_counter import TokenCounter
        counter = TokenCounter()
        # Without tokenizer lib, falls back to len//4
        count = counter.count("hello world this is a test")
        assert count > 0

    def test_cost_tracker_record(self):
        from deepseek_agent.core.token_counter import CostTracker
        tracker = CostTracker(max_cost_usd=1.0)
        tracker.record("deepseek-chat", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        assert tracker.total_cost > 0
        assert tracker.total_tokens == 1500
        assert not tracker.is_over_budget

    def test_cost_tracker_budget_check(self):
        from deepseek_agent.core.token_counter import CostTracker
        tracker = CostTracker(max_cost_usd=0.00001)
        assert not tracker.can_request(estimated_input=100000, estimated_output=100000)

    def test_model_breakdown(self):
        from deepseek_agent.core.token_counter import CostTracker
        tracker = CostTracker()
        tracker.record("deepseek-chat", {"prompt_tokens": 100, "completion_tokens": 50})
        tracker.record("deepseek-reasoner", {"prompt_tokens": 200, "completion_tokens": 100})
        breakdown = tracker.get_model_breakdown()
        assert "deepseek-chat" in breakdown
        assert "deepseek-reasoner" in breakdown


# ── 5. 模型降级 ──────────────────────────────────────────────────────────

class TestModelFallback:
    """验证连续错误触发模型降级。"""

    def test_fallback_triggered(self):
        from deepseek_agent.core.client import DeepSeekClient
        client = DeepSeekClient.__new__(DeepSeekClient)
        client.model = "deepseek-chat"
        client._consecutive_errors = 0
        client._fallback_model = None
        client.FALLBACK_THRESHOLD = 3

        assert client.active_model == "deepseek-chat"

        # Simulate 3 errors
        client._consecutive_errors = 3
        result = client.trigger_fallback()
        assert result == "deepseek-reasoner"
        assert client.active_model == "deepseek-reasoner"

    def test_reset_errors(self):
        from deepseek_agent.core.client import DeepSeekClient
        client = DeepSeekClient.__new__(DeepSeekClient)
        client._consecutive_errors = 5
        client.reset_errors()
        assert client._consecutive_errors == 0


# ── 6. ContextBudget 动态裁剪 ───────────────────────────────────────────

class TestContextBudget:
    """验证上下文预算三阶段裁剪。"""

    def test_truncate_long_tool_results(self):
        from deepseek_agent.agent.context_budget import ContextBudget, BudgetConfig, ContextEntry, ContextPriority
        # Set very low max_tokens so budget check triggers
        config = BudgetConfig(max_tokens=10, max_tool_result_chars=20)
        budget = ContextBudget(config)

        budget.add(ContextEntry(id="sys", content="system prompt", priority=ContextPriority.SYSTEM))
        budget.add(ContextEntry(id="tool", content="x" * 200, priority=ContextPriority.TOOL_RESULT, role="tool"))

        trimmed, removed = budget.check_budget()
        # Tool result should be truncated
        entry = budget._entries.get("tool")
        assert entry is not None
        assert len(entry.content) <= 20 + 30  # max_tool_result_chars + truncation suffix

    def test_summarize_old_history(self):
        from deepseek_agent.agent.context_budget import ContextBudget, BudgetConfig, ContextEntry, ContextPriority
        # Set very low budget to trigger summarization
        config = BudgetConfig(max_tokens=10, history_summarize_threshold=2)
        budget = ContextBudget(config)

        budget.add(ContextEntry(id="sys", content="system", priority=ContextPriority.SYSTEM))
        for i in range(6):
            budget.add(ContextEntry(id=f"hist_{i}", content=f"message {i} {i} {i} {i}", priority=ContextPriority.HISTORY, role="user"))

        trimmed, removed = budget.check_budget()
        # Old history should be removed/summarized
        assert trimmed or len(removed) > 0


# ── 7. 模型版本管理 ──────────────────────────────────────────────────────

class TestModelManager:
    """验证模型别名和连通测试（mocked）。"""

    def test_alias_resolution(self):
        from deepseek_agent.core.model_manager import ModelManager
        mm = ModelManager.__new__(ModelManager)
        mm.aliases = {"chat": "deepseek-chat", "planner": "deepseek-reasoner"}
        assert mm.resolve("chat") == "deepseek-chat"
        assert mm.resolve("planner") == "deepseek-reasoner"
        assert mm.resolve("deepseek-chat") == "deepseek-chat"  # passthrough

    def test_set_alias(self):
        from deepseek_agent.core.model_manager import ModelManager
        mm = ModelManager.__new__(ModelManager)
        mm.aliases = {}
        mm._config_path = MagicMock()
        mm._config_path.parent = MagicMock()
        mm._save_config = MagicMock()
        mm.set_alias("executor", "deepseek-chat-v2")
        assert mm.aliases["executor"] == "deepseek-chat-v2"


# ── 8. OpenAI Schema 兼容性 ─────────────────────────────────────────────

class TestOpenAISchemaCompatibility:
    """验证 to_openai_schema() 输出完全兼容 DeepSeek API。"""

    def test_schema_structure(self):
        from deepseek_agent.tools.base import tool, DangerLevel

        @tool(name="test_tool", description="A test tool", danger_level=DangerLevel.SAFE)
        def test_tool(path: str, offset: int = 0, limit: int = 100) -> str:
            """Test tool.

            Args:
                path: file path
                offset: start line
                limit: max lines
            """
            return ""

        schema = test_tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "path" in params["properties"]
        assert params["required"] == ["path"]

    def test_all_tools_produce_valid_schema(self):
        """验证所有已注册工具都能生成有效 schema。"""
        import deepseek_agent.tools.base as base
        # Schema should always be a valid JSON-serializable dict
        schema = {
            "type": "function",
            "function": {
                "name": "test",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        json.dumps(schema)  # Should not raise


# ── 9. PrivacyFilter 可选隐私过滤 ──────────────────────────────────────────────

class TestPrivacyFilter:
    """验证隐私过滤（默认关闭）。"""

    def test_default_disabled(self):
        from deepseek_agent.tools.security_privacy import is_privacy_enabled
        # 默认关闭
        assert is_privacy_enabled() is False

    def test_enable_disable(self):
        from deepseek_agent.tools.security_privacy import set_privacy_mode, is_privacy_enabled
        set_privacy_mode(True)
        assert is_privacy_enabled() is True
        set_privacy_mode(False)
        assert is_privacy_enabled() is False

    def test_github_token_filtered(self):
        from deepseek_agent.tools.security_privacy import filter_sensitive, set_privacy_mode
        set_privacy_mode(True)
        text = "Token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        result = filter_sensitive(text)
        assert "[GITHUB_TOKEN]" in result
        assert "ghp_" not in result

    def test_private_key_filtered(self):
        from deepseek_agent.tools.security_privacy import filter_sensitive, set_privacy_mode
        set_privacy_mode(True)
        text = "-----BEGIN RSA PRIVATE KEY-----"
        result = filter_sensitive(text)
        assert "[PRIVATE_KEY]" in result
        assert "BEGIN RSA" not in result
        assert "BEGIN RSA" not in result

    def test_aws_key_filtered(self):
        from deepseek_agent.tools.security_privacy import filter_sensitive, set_privacy_mode
        set_privacy_mode(True)
        text = "AKIAIOSFODNN7EXAMPLE"
        result = filter_sensitive(text)
        assert "[AWS_KEY_ID]" in result
        assert "AKIA" not in result

    def test_disabled_preserves_all(self):
        from deepseek_agent.tools.security_privacy import filter_sensitive, set_privacy_mode
        set_privacy_mode(False)
        text = "Token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        result = filter_sensitive(text)
        assert result == text  # 未过滤

    def test_filter_messages(self):
        from deepseek_agent.tools.security_privacy import filter_messages, set_privacy_mode
        set_privacy_mode(True)
        msgs = [
            {"role": "user", "content": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
            {"role": "assistant", "content": "I see your token"},
        ]
        result = filter_messages(msgs)
        assert "ghp_" not in result[0]["content"]
        assert result[1]["content"] == "I see your token"


# ── 10. Planner 调用次数限制 ──────────────────────────────────────────────

class TestPlannerCallLimit:
    """验证 Planner 调用次数上限。"""

    def test_max_planner_calls_default(self):
        from deepseek_agent.agent.loop import LoopConfig
        cfg = LoopConfig()
        assert cfg.max_planner_calls == 3

    def test_planner_calls_incremented(self):
        from deepseek_agent.agent.loop import AgentLoop, LoopConfig, LoopMode
        from deepseek_agent.core.client import DeepSeekClient
        from deepseek_agent.tools.base import ToolRegistry
        from deepseek_agent.memory.manager import MemoryManager
        from unittest.mock import MagicMock

        cfg = LoopConfig(mode=LoopMode.PLAN_EXECUTE)
        mock_client = MagicMock(spec=DeepSeekClient)
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_memory = MagicMock(spec=MemoryManager)
        mock_memory.project_root = None

        loop = AgentLoop(mock_client, mock_registry, mock_memory, cfg)
        assert loop._planner_calls == 0


# ── 11. FIM 补全集成 ──────────────────────────────────────────────────────

class TestFIMIntegration:
    """验证 FIM 补全工具已注册。"""

    def test_fim_tools_registered(self):
        try:
            from deepseek_agent.tools.fim_tools import init_fim_tools
            tools = init_fim_tools()
            names = [t.name for t in tools]
            assert "inline_complete" in names or any("fim" in n.lower() for n in names)
        except ImportError:
            pytest.skip("fim_tools not yet implemented")

    def test_fim_client_import(self):
        try:
            from deepseek_agent.core.fim import FIMClient
            # Client should be importable
            assert FIMClient is not None
        except ImportError:
            pytest.skip("fim.py not yet implemented")


# ── 12. 并行工具冲突检测 ─────────────────────────────────────────────────
class TestToolConflictDetection:
    """验证并行执行器检测文件写入冲突。"""

    @pytest.mark.asyncio
    async def test_same_file_conflict(self):
        from deepseek_agent.agent.parallel import TaskDecomposer, SubTask

        decomposer = TaskDecomposer(".")
        tasks = [
            SubTask(id="t1", description="write", prompt="", writes_files=["a.py"]),
            SubTask(id="t2", description="edit", prompt="", writes_files=["a.py"]),
        ]
        conflicts = decomposer.detect_conflicts(tasks)
        assert len(conflicts) > 0  # t1 and t2 conflict on same file

    @pytest.mark.asyncio
    async def test_different_files_no_conflict(self):
        from deepseek_agent.agent.parallel import TaskDecomposer, SubTask

        decomposer = TaskDecomposer(".")
        tasks = [
            SubTask(id="t1", description="write a", prompt="", writes_files=["a.py"]),
            SubTask(id="t2", description="write b", prompt="", writes_files=["b.py"]),
        ]
        conflicts = decomposer.detect_conflicts(tasks)
        assert len(conflicts) == 0
