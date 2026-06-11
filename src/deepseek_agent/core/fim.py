"""
FIM (Fill-in-the-Middle) 补全 — DeepSeek Coder 特有能力。

使用 /v1/completions 端点，支持 prefix/suffix 补全模式。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ..tools.base import ToolResult, ToolRegistry, DangerLevel, tool


@dataclass
class FIMResult:
    """FIM 补全结果。"""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class FIMClient:
    """
    DeepSeek FIM 补全客户端。

    使用 /v1/completions（非 Chat 端点），传入 prefix/suffix。
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    async def complete(
        self,
        prefix: str,
        suffix: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> FIMResult:
        """
        FIM 补全。

        Args:
            prefix: 光标前的代码
            suffix: 光标后的代码
            max_tokens: 最大生成长度
            temperature: 温度（代码补全建议为 0）
            stop: 停止符号列表
        """
        payload = {
            "model": model or self.model,
            "prompt": prefix,
            "suffix": suffix,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        try:
            resp = await self._client.post("/v1/completions", json=payload)
            if resp.status_code != 200:
                raise Exception(f"FIM API error: {resp.status_code} {resp.text}")

            body = resp.json()
            choice = body["choices"][0]
            usage = body.get("usage", {})

            return FIMResult(
                text=choice.get("text", ""),
                model=model or self.model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        except httpx.TimeoutException:
            raise Exception("FIM request timed out")

    async def close(self):
        await self._client.aclose()


# ── FIM 工具（注册到 Agent） ───────────────────────────────────────────────

_fim_client: Optional[FIMClient] = None


def _get_fim_client(client: DeepSeekClient) -> FIMClient:
    """从 DeepSeekClient 创建 FIM 客户端实例。"""
    global _fim_client
    if _fim_client is None:
        _fim_client = FIMClient(
            api_key=client.api_key,
            base_url=client.base_url,
            model=client.model,
        )
    return _fim_client


async def fim_complete(
    prefix: str,
    suffix: str = "",
    max_tokens: int = 256,
    file_path: Optional[str] = None,
) -> ToolResult:
    """
    FIM (Fill-in-the-Middle) 代码补全。

    输入光标前后的代码片段，返回中间补全结果。
    适合用于局部逻辑补全、函数体生成等场景。

    Args:
        prefix: 光标前的代码内容
        suffix: 光标后的代码内容
        max_tokens: 最大补全长度（默认 256）
        file_path: 可选，文件路径（用于日志）
    """
    try:
        from .client import DeepSeekClient
        # 用全局 client 初始化
        if _fim_client is None:
            raise Exception("FIM client not initialized. Call init_fim_tools() first.")
        result = await _fim_client.complete(
            prefix=prefix,
            suffix=suffix,
            max_tokens=max_tokens,
        )
        return ToolResult.ok({
            "completion": result.text,
            "tokens": result.prompt_tokens + result.completion_tokens,
        })
    except Exception as e:
        return ToolResult.fail(str(e))


def init_fim_tools(client: DeepSeekClient, registry: "ToolRegistry"):
    """初始化 FIM 工具并注册到 ToolRegistry。"""
    from .base import tool, DangerLevel

    _get_fim_client(client)

    # 内联补全工具
    inline_complete = tool(
        name="inline_complete",
        description=(
            "Fill-in-the-Middle 代码补全。输入光标前后的代码片段，返回中间补全结果。"
            "适合局部逻辑补全、函数体生成等场景。"
        ),
        danger_level=DangerLevel.SAFE,
        read_only=True,
    )(fim_complete)

    registry.register(inline_complete)
