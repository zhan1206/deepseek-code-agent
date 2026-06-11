"""
精确 Token 计数与成本追踪

替换粗略的 len(text)//4 估算，使用 DeepSeek 官方 tokenizer。
备用方案：API usage 字段实时累计。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Token 计数器 ─────────────────────────────────────────────────────────

class TokenCounter:
    """
    DeepSeek Token 计数器。

    优先级：
    1. tokenizers 库加载 DeepSeek 官方 tokenizer.json
    2. 退化为 API usage 字段
    3. 最终退化为 len(text)//4
    """

    _instance: Optional["TokenCounter"] = None
    _encoder = None

    def __init__(self):
        self._encoder = None
        self._fallback = True
        self._try_load_tokenizer()

    def _try_load_tokenizer(self):
        """尝试加载 DeepSeek tokenizer。"""
        try:
            from tokenizers import Tokenizer
            # 尝试多个本地路径
            candidates = [
                os.path.expanduser("~/.deepseek-agent/tokenizer.json"),
            ]
            for path in candidates:
                if os.path.isfile(path):
                    self._encoder = Tokenizer.from_file(path)
                    self._fallback = False
                    return
            # 不尝试在线加载（避免网络阻塞）
        except ImportError:
            pass

    def count(self, text: str) -> int:
        """统计文本 token 数。"""
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        # 退化为 API 方式或粗略估算
        return len(text) // 4 + text.count("\n")

    @property
    def is_precise(self) -> bool:
        return self._encoder is not None

    @classmethod
    def get(cls) -> "TokenCounter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ── DeepSeek 定价 (2026-06) ─────────────────────────────────────────────

DEEPSEEK_PRICING = {
    "deepseek-chat": {"input": 0.14, "output": 0.28, "cache_input": 0.014},     # per 1M tokens
    "deepseek-reasoner": {"input": 0.55, "output": 2.19, "cache_input": 0.055},
}


# ── 成本追踪器 ─────────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cache_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """
    API 成本追踪与预算管理。

    功能：
    - 每次 API 调用后记录 usage 和费用
    - 支持设置会话/每日预算上限
    - 预算超支时触发告警或降级
    """

    def __init__(self, max_cost_usd: float = 5.0):
        self.max_cost_usd = max_cost_usd
        self._records: List[UsageRecord] = []
        self._total_cost: float = 0.0
        self._total_prompt: int = 0
        self._total_completion: int = 0

    def record(self, model: str, usage: Dict[str, int]) -> UsageRecord:
        """记录一次 API 调用的 token 使用量。"""
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        cache = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) if isinstance(usage.get("prompt_tokens_details"), dict) else 0

        pricing = DEEPSEEK_PRICING.get(model, {"input": 0.14, "output": 0.28, "cache_input": 0.014})
        cost = (
            (prompt - cache) * pricing["input"] / 1_000_000
            + cache * pricing["cache_input"] / 1_000_000
            + completion * pricing["output"] / 1_000_000
        )

        record = UsageRecord(
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_tokens=cache,
            cost_usd=round(cost, 6),
        )
        self._records.append(record)
        self._total_cost += cost
        self._total_prompt += prompt
        self._total_completion += completion
        return record

    def can_request(self, estimated_input: int = 0, estimated_output: int = 0) -> bool:
        """检查预算是否允许新请求。"""
        est_cost = (
            estimated_input * 0.14 / 1_000_000
            + estimated_output * 0.28 / 1_000_000
        )
        return (self._total_cost + est_cost) <= self.max_cost_usd

    @property
    def total_cost(self) -> float:
        return round(self._total_cost, 6)

    @property
    def total_tokens(self) -> int:
        return self._total_prompt + self._total_completion

    @property
    def is_over_budget(self) -> bool:
        return self._total_cost >= self.max_cost_usd

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": self.total_cost,
            "max_cost_usd": self.max_cost_usd,
            "budget_remaining_usd": round(self.max_cost_usd - self._total_cost, 6),
            "total_prompt_tokens": self._total_prompt,
            "total_completion_tokens": self._total_completion,
            "total_tokens": self.total_tokens,
            "call_count": len(self._records),
            "is_over_budget": self.is_over_budget,
        }

    def get_model_breakdown(self) -> Dict[str, Dict]:
        by_model: Dict[str, Dict] = {}
        for r in self._records:
            if r.model not in by_model:
                by_model[r.model] = {"calls": 0, "cost": 0.0, "tokens": 0}
            by_model[r.model]["calls"] += 1
            by_model[r.model]["cost"] += r.cost_usd
            by_model[r.model]["tokens"] += r.prompt_tokens + r.completion_tokens
        return by_model
