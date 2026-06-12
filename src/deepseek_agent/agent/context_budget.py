"""
动态上下文预算 — 智能裁剪策略，避免简单截断丢失关键信息。

优先级分层：
  P0 不可裁剪：当前任务指令、系统 prompt
  P1 可总结：工具返回结果 → 保留摘要 + 完整结果存 memory
  P2 可丢弃：历史对话轮次 → 超过 N 轮自动总结
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from ..core.client import estimate_tokens


# ── 优先级 ────────────────────────────────────────────────────────────────

class ContextPriority(IntEnum):
    SYSTEM = 0      # 不可裁剪
    TASK = 1         # 当前任务指令
    TOOL_RESULT = 2  # 可总结
    HISTORY = 3      # 可丢弃


# ── 上下文条目 ────────────────────────────────────────────────────────────

@dataclass
class ContextEntry:
    """单条上下文条目。"""
    id: str
    content: str
    priority: ContextPriority
    token_count: int = 0
    summary: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    role: str = ""  # system/user/assistant/tool

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


# ── 预算配置 ──────────────────────────────────────────────────────────────

@dataclass
class BudgetConfig:
    """上下文预算配置。"""
    max_tokens: int = 30000              # 总 token 上限
    system_reserve: float = 0.15         # 系统指令预留比例
    tool_result_max_ratio: float = 0.40  # 工具结果最大占比
    history_max_ratio: float = 0.30      # 历史消息最大占比
    summary_trigger: float = 0.80        # 使用率触发总结
    max_tool_result_chars: int = 2000    # 单条工具结果最大字符数
    history_summarize_threshold: int = 10  # 历史轮次超过此值触发总结


# ── 预算管理器 ────────────────────────────────────────────────────────────

class ContextBudget:
    """
    动态上下文预算管理器。

    工作流：
    1. 每轮循环开始时调用 check_budget()
    2. 超过阈值时按优先级裁剪
    3. 裁剪方式：总结 > 截断 > 丢弃
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self._entries: Dict[str, ContextEntry] = {}
        self._order: List[str] = []  # 插入顺序
        self._total_tokens: int = 0
        self._tokenizer_model: str = "deepseek-chat"  # 用于 estimate_tokens

    # ── 增删 ──────────────────────────────────────────────────────────

    def add(self, entry: ContextEntry) -> None:
        self._entries[entry.id] = entry
        self._order.append(entry.id)
        self._total_tokens += entry.token_count

    def remove(self, entry_id: str) -> None:
        if entry_id in self._entries:
            self._total_tokens -= self._entries[entry_id].token_count
            del self._entries[entry_id]
            self._order.remove(entry_id)

    # ── 查询 ──────────────────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def usage_ratio(self) -> float:
        return self._total_tokens / self.config.max_tokens if self.config.max_tokens > 0 else 0.0

    def get_budget_status(self) -> Dict[str, Any]:
        """返回预算使用状态。"""
        by_priority: Dict[ContextPriority, int] = {}
        for e in self._entries.values():
            by_priority[e.priority] = by_priority.get(e.priority, 0) + e.token_count

        return {
            "total_tokens": self._total_tokens,
            "max_tokens": self.config.max_tokens,
            "usage_ratio": round(self.usage_ratio, 2),
            "by_priority": {p.name: t for p, t in by_priority.items()},
            "entry_count": len(self._entries),
        }

    # ── 核心裁剪逻辑 ──────────────────────────────────────────────────

    def check_budget(self) -> Tuple[bool, List[str]]:
        """
        检查预算是否超限，如需要则自动裁剪。

        Returns:
            (was_trimmed, list of removed entry ids)
        """
        if self.usage_ratio <= self.config.summary_trigger:
            return False, []

        removed: List[str] = []

        # 阶段 1：截断过长的工具结果
        removed.extend(self._truncate_tool_results())

        # 阶段 2：总结历史消息
        removed.extend(self._summarize_history())

        # 阶段 3：丢弃最低优先级的历史（如仍超限）
        if self.usage_ratio > 1.0:
            removed.extend(self._discard_oldest_history())

        return bool(removed), removed

    def _truncate_tool_results(self) -> List[str]:
        """截断过长的工具结果，保留摘要。"""
        removed = []
        max_chars = self.config.max_tool_result_chars

        for entry_id in list(self._order):
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            if entry.priority != ContextPriority.TOOL_RESULT:
                continue
            if len(entry.content) <= max_chars:
                continue

            # 截断并生成摘要
            truncated = entry.content[:max_chars]
            entry.summary = truncated + f"\n... [截断，原始 {len(entry.content)} 字符]"
            old_tokens = entry.token_count
            entry.content = entry.summary
            entry.token_count = estimate_tokens(entry.content)
            self._total_tokens -= (old_tokens - entry.token_count)

        return removed

    def _summarize_history(self) -> List[str]:
        """总结过老的历史消息。"""
        removed = []

        # 按优先级和年龄排序，优先移除最老的历史
        history_entries = [
            e for e in self._entries.values()
            if e.priority == ContextPriority.HISTORY
        ]

        if len(history_entries) <= self.config.history_summarize_threshold:
            return removed

        # 保留最近的 N 条，总结剩余
        history_entries.sort(key=lambda e: e.created_at)
        keep_count = self.config.history_summarize_threshold
        to_summarize = history_entries[:-keep_count] if keep_count < len(history_entries) else []

        if not to_summarize:
            return removed

        # 生成总结文本
        summary_parts = []
        for e in to_summarize:
            summary_parts.append(f"[{e.role}] {e.content[:200]}")
            self.remove(e.id)
            removed.append(e.id)

        # 插入总结条目
        summary_text = "📋 历史消息总结：\n" + "\n".join(summary_parts)
        summary_entry = ContextEntry(
            id=f"summary_{int(time.time())}",
            content=summary_text,
            priority=ContextPriority.TASK,  # 总结提升到 TASK 优先级
            role="system",
        )
        self.add(summary_entry)

        return removed

    def _discard_oldest_history(self) -> List[str]:
        """紧急丢弃最老的历史消息。"""
        removed = []

        history_entries = [
            e for e in self._entries.values()
            if e.priority == ContextPriority.HISTORY
        ]
        history_entries.sort(key=lambda e: e.created_at)

        while self.usage_ratio > 1.0 and history_entries:
            oldest = history_entries.pop(0)
            self.remove(oldest.id)
            removed.append(oldest.id)

        return removed

    # ── 从消息列表构建 ────────────────────────────────────────────────

    @classmethod
    def from_messages(
        cls,
        messages: List[Dict[str, str]],
        config: Optional[BudgetConfig] = None,
    ) -> "ContextBudget":
        """从 OpenAI 格式消息列表构建预算管理器。"""
        budget = cls(config)

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                priority = ContextPriority.SYSTEM
            elif role == "tool":
                priority = ContextPriority.TOOL_RESULT
            elif role == "user":
                priority = ContextPriority.TASK
            else:
                priority = ContextPriority.HISTORY

            entry = ContextEntry(
                id=f"msg_{i}",
                content=content,
                priority=priority,
                role=role,
            )
            budget.add(entry)

        return budget

    def to_messages(self) -> List[Dict[str, str]]:
        """将当前条目导出为 OpenAI 格式消息列表。"""
        messages = []
        for entry_id in self._order:
            entry = self._entries.get(entry_id)
            if entry:
                messages.append({
                    "role": entry.role or "user",
                    "content": entry.content,
                })
        return messages

    # ── 语义重要性评分（v1.5.0 新增）────────────────────────────────

    def score_importance(self, entry: ContextEntry) -> float:
        """
        计算单条上下文的语义重要性分数（0.0 ~ 1.0）。

        评分策略：
        - system prompt: 1.0（不可删除）
        - 当前用户任务: 0.9（核心上下文）
        - 最新工具结果: 0.7（执行反馈）
        - 早期工具结果: 0.4（历史参考）
        - 历史对话: 0.3（可压缩）
        - 包含关键词（error/fix/bug）: +0.1 加权
        """
        base_scores = {
            ContextPriority.SYSTEM: 1.0,
            ContextPriority.TASK: 0.9,
            ContextPriority.TOOL_RESULT: 0.7,
            ContextPriority.HISTORY: 0.3,
        }
        score = base_scores.get(entry.priority, 0.5)

        # 关键词加权
        keywords = ["error", "bug", "fix", "crash", "fail", "exception", "not found"]
        content_lower = entry.content.lower()
        for kw in keywords:
            if kw in content_lower:
                score = min(1.0, score + 0.1)
                break

        # 越新的消息越重要（时间衰减）
        age_hours = (time.time() - entry.created_at) / 3600
        if age_hours > 24:
            score *= 0.8  # 超过 24 小时降低权重

        return score

    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        语义压缩：根据重要性分数智能裁剪，保留关键信息。

        策略：
        1. 计算每条消息的重要性分数
        2. 按分数降序排列，优先保留高重要性消息
        3. 超出预算时，低重要性消息被总结或丢弃
        4. 恢复原始消息顺序
        """
        total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        if total_tokens <= self.config.max_tokens:
            return messages

        # 评分并排序
        scored = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            priority_map = {
                "system": ContextPriority.SYSTEM,
                "tool": ContextPriority.TOOL_RESULT,
                "user": ContextPriority.TASK,
                "assistant": ContextPriority.HISTORY,
            }
            priority = priority_map.get(role, ContextPriority.HISTORY)
            entry = ContextEntry(
                id=f"msg_{i}",
                content=content,
                priority=priority,
                role=role,
            )
            score = self.score_importance(entry)
            scored.append((i, msg, score, entry))

        scored.sort(key=lambda x: x[2], reverse=True)

        # 按分数从高到低选择，保留在预算内
        result: List[Dict[str, str]] = []
        current_tokens = 0
        for i, msg, score, entry in scored:
            tokens = estimate_tokens(msg.get("content", ""))
            if current_tokens + tokens <= self.config.max_tokens:
                result.append(msg)
                current_tokens += tokens
            elif score > 0.8 and entry.priority == ContextPriority.TOOL_RESULT:
                # 高重要性工具结果：强制总结
                summarized = self._summarize_single(entry)
                summarized_tokens = estimate_tokens(summarized)
                if current_tokens + summarized_tokens <= self.config.max_tokens:
                    result.append({"role": msg["role"], "content": summarized})
                    current_tokens += summarized_tokens

        # 恢复原始顺序
        result.sort(key=lambda m: next(
            (i for i, orig in enumerate(messages) if orig.get("content") == m.get("content")),
            0
        ))
        return result

    def _summarize_single(self, entry: ContextEntry) -> str:
        """对单条工具结果生成简短摘要。"""
        MAX = 300
        content = entry.content
        if len(content) <= MAX:
            return content
        return content[:MAX] + f"\n... [已截断，原 {len(content)} 字符]"
