"""
模型版本管理 — 别名、连通测试、版本感知。

支持通过别名平滑切换模型版本，一键测试模型可用性。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


# ── 默认模型别名配置 ─────────────────────────────────────────────────────

DEFAULT_MODEL_ALIASES = {
    "chat": "deepseek-chat",
    "reasoner": "deepseek-reasoner",
    "executor": "deepseek-chat",
    "planner": "deepseek-reasoner",
    "coder": "deepseek-coder",
}


@dataclass
class ModelInfo:
    """模型信息。"""
    alias: str
    actual_name: str
    supports_fc: bool = True          # 是否支持 Function Calling
    supports_reasoning: bool = False  # 是否有思考过程
    supports_fim: bool = True         # 是否支持 FIM 补全
    description: str = ""
    deprecated: bool = False
    recommended_replacement: Optional[str] = None


@dataclass
class ConnectivityTest:
    """模型连通测试结果。"""
    alias: str
    model: str
    success: bool
    latency_ms: float = 0.0
    error: Optional[str] = None
    test_time: float = field(default_factory=time.time)


class ModelManager:
    """
    模型版本管理器。

    功能：
    - 别名映射（平滑切换模型版本）
    - 一键连通测试（延迟 + 可用性）
    - 版本感知（检查推荐版本）
    - 配置持久化
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.aliases: Dict[str, str] = dict(DEFAULT_MODEL_ALIASES)
        self._test_results: Dict[str, ConnectivityTest] = {}
        self._config_path = Path(os.path.expanduser("~/.deepseek-agent/model_config.json"))
        self._load_config()

    # ── 别名管理 ──────────────────────────────────────────────────────

    def resolve(self, alias_or_name: str) -> str:
        """将别名或模型名解析为实际模型名。"""
        return self.aliases.get(alias_or_name, alias_or_name)

    def set_alias(self, alias: str, model: str) -> None:
        """设置别名映射。"""
        self.aliases[alias] = model
        self._save_config()

    def list_models(self) -> List[ModelInfo]:
        """列出所有已配置的模型。"""
        seen = set()
        models = []
        for alias, actual in self.aliases.items():
            if actual not in seen:
                seen.add(actual)
                models.append(ModelInfo(
                    alias=alias,
                    actual_name=actual,
                    supports_fc=(actual != "deepseek-reasoner"),
                    supports_reasoning=(actual == "deepseek-reasoner"),
                    supports_fim=("coder" in actual.lower()),
                ))
        return models

    # ── 连通测试 ────────────────────────────────────────────────────────

    async def test_connectivity(
        self, alias: str, timeout: float = 10.0
    ) -> ConnectivityTest:
        """测试指定模型的连通性。"""
        model = self.resolve(alias)
        start = time.time()

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5,
                    },
                )

                latency = (time.time() - start) * 1000

                if resp.status_code == 200:
                    result = ConnectivityTest(
                        alias=alias, model=model,
                        success=True, latency_ms=round(latency, 1),
                    )
                else:
                    body = resp.json()
                    msg = body.get("error", {}).get("message", str(body))
                    result = ConnectivityTest(
                        alias=alias, model=model,
                        success=False, latency_ms=round(latency, 1),
                        error=msg,
                    )
        except Exception as e:
            latency = (time.time() - start) * 1000
            result = ConnectivityTest(
                alias=alias, model=model,
                success=False, latency_ms=round(latency, 1),
                error=str(e),
            )

        self._test_results[alias] = result
        return result

    async def test_all(self) -> Dict[str, ConnectivityTest]:
        """测试所有已配置模型的连通性。"""
        results = {}
        tasks = []
        for alias in self.aliases:
            tasks.append(self.test_connectivity(alias))

        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, ConnectivityTest):
                results[result.alias] = result

        return results

    # ── 配置持久化 ──────────────────────────────────────────────────────

    def _load_config(self):
        """从磁盘加载配置。"""
        try:
            if self._config_path.exists():
                data = json.loads(self._config_path.read_text(encoding="utf-8"))
                if "aliases" in data:
                    self.aliases.update(data["aliases"])
        except Exception:
            pass

    def _save_config(self):
        """保存配置到磁盘。"""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps({"aliases": self.aliases}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """获取模型管理器状态。"""
        return {
            "aliases": dict(self.aliases),
            "test_results": {
                alias: {
                    "success": r.success,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for alias, r in self._test_results.items()
            },
            "config_path": str(self._config_path),
        }
