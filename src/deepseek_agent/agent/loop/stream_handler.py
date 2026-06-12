"""
流式输出处理 — system prompt 构建 + 上下文预算集成。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...tools.registry import ToolRegistry
from ..context_budget import ContextBudget, BudgetConfig, ContextPriority


class StreamHandler:
    """
    流式输出与消息构建处理器。

    职责：
    - 构建 system prompt
    - 上下文预算检查与裁剪
    - 动态工具集选择
    """

    def __init__(
        self,
        registry: ToolRegistry,
        budget_config: BudgetConfig,
        tool_pruning_threshold: float = 0.80,
        planning_tools: Optional[List[str]] = None,
        verbose_tools: Optional[List[str]] = None,
    ):
        self.registry = registry
        self.budget_config = budget_config
        self.tool_pruning_threshold = tool_pruning_threshold
        self.planning_tools = planning_tools
        self.verbose_tools = verbose_tools

    def build_system_prompt(
        self,
        project_root: Optional[str] = None,
        existing_system_prompt: Optional[str] = None,
        system_extra: str = "",
    ) -> str:
        """构建完整的 system prompt。"""
        tools_list = "\n".join(
            f"- **{t.name}**: {t.description}"
            for t in self.registry._tools.values()
        )
        base = (
            "你是一个严谨的代码助手。\n"
            f"可用工具：\n{tools_list}\n\n"
            "规则：\n"
            "1. 优先使用工具读取文件、分析代码，再做修改\n"
            "2. 修改前先确认旧字符串在文件中唯一\n"
            "3. 执行危险操作前告知用户潜在风险\n"
            "4. 如果执行失败，给出清晰的错误信息和修复建议\n"
        )

        # 自动注入项目骨架
        try:
            from ...knowledge.ingest import ProjectIngester
            if project_root:
                ingester = ProjectIngester(project_root)
                snapshot = ingester.ingest(compression_rate=0.3, max_files=30)
                snapshot_text = snapshot[:8000]
                base += f"\n\n## 项目上下文\n{snapshot_text}"
        except Exception:
            pass

        if system_extra:
            base += "\n\n" + system_extra

        if existing_system_prompt:
            base = existing_system_prompt + "\n" + base

        return base

    def build_messages(
        self,
        base_system: str,
        memory: Any,
    ) -> List[Dict[str, str]]:
        """构建发送给模型的完整消息列表。"""
        msgs = [{"role": "system", "content": base_system}]
        for m in memory.get_messages():
            msgs.append(m)
        return msgs

    def check_and_trim_budget(
        self,
        messages: List[Dict[str, str]],
    ) -> tuple[bool, List[Dict[str, str]], List[str]]:
        """
        检查上下文预算并在超限时裁剪。

        Returns:
            (was_trimmed, trimmed_messages, removed_descriptions)
        """
        budget = ContextBudget.from_messages(messages, self.budget_config)
        was_trimmed, removed = budget.check_budget()
        if was_trimmed:
            return True, budget.to_messages(), [str(r) for r in removed]
        return False, messages, []

    def select_tools(
        self,
        messages: List[Dict[str, str]],
        budget_max_tokens: int,
    ) -> List[Dict[str, Any]]:
        """动态工具集裁剪：根据上下文预算选择工具。"""
        all_schemas = self.registry.get_schemas()

        # 估算当前上下文 token 数
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4  # 粗略估算
        ratio = estimated_tokens / budget_max_tokens if budget_max_tokens > 0 else 0

        if ratio <= self.tool_pruning_threshold:
            return all_schemas

        # 超限时移除大输出工具
        verbose = set(self.verbose_tools or ['read_file', 'search_content', 'read_docs'])
        pruned = [s for s in all_schemas if s['function']['name'] not in verbose]

        if ratio <= 1.0:
            return pruned

        # 严重超限时只保留核心工具
        core = set(self.planning_tools or [
            'list_directory', 'search_file', 'find_symbol', 'git_status'
        ])
        return [s for s in all_schemas if s['function']['name'] in core]