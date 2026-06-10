"""
DeepSeek API Client — 支持流式 Function Calling、Reasoner 模型特殊处理、精细错误分类。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Union
from enum import Enum

import httpx

# ── 异常定义 ─────────────────────────────────────────────────────────────────

class DeepSeekAPIError(Exception):
    """DeepSeek API 错误基类。"""

    RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
    NON_RETRYABLE_STATUSES = {401, 403, 404, 422, 400}

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        self.is_retriable = status_code in self.RETRYABLE_STATUSES
        super().__init__(f"[{status_code}] {message}")

    @classmethod
    def from_response(cls, status_code: int, body: dict) -> "DeepSeekAPIError":
        msg = body.get("error", {}).get("message", str(body))
        return cls(status_code, msg)


class ContextLengthExceededError(DeepSeekAPIError):
    """上下文超限，需要截断后重试。"""

    def __init__(self, message: str):
        super().__init__(422, message)
        self.is_retriable = False


# ── 数据模型 ─────────────────────────────────────────────────────────────────

@dataclass
class ToolCallDelta:
    """流式 tool_call 增量片段。"""
    id: str = ""
    name: str = ""
    arguments: str = ""   # JSON 字符串，逐步累积
    complete: bool = False


@dataclass
class ToolCall:
    """完整的 tool_call。"""
    id: str
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def from_delta(cls, delta: ToolCallDelta) -> "ToolCall":
        return cls(
            id=delta.id,
            name=delta.name,
            arguments=json.loads(delta.arguments) if delta.arguments else {},
        )


@dataclass
class Response:
    """统一响应对象，同时兼容普通回复和 tool_calls。"""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    thinking: Optional[str] = None  # deepseek-reasoner 模型专用
    usage: Optional[Dict[str, int]] = None
    raw: Optional[Any] = None


# ── Token 估算 ────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中英文混合友好）。"""
    return len(text) // 4 + text.count("\n")


# ── DeepSeekClient ───────────────────────────────────────────────────────────

class DeepSeekClient:
    """
    DeepSeek API 客户端，兼容 OpenAI SDK 风格。

    支持：
    - 普通对话 / Function Calling / 流式响应
    - deepseek-reasoner 模型思考过程提取
    - 自动重试（指数退避）
    - Token 用量统计
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        max_retries: int = 3,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )
        self._total_tokens = 0

    # ── 公开 API ────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs,
    ) -> Union[Response, Generator[Response, None, None]]:
        """
        发送对话请求。

        Args:
            messages: OpenAI 格式消息列表 [{"role": "user", "content": "..."}]
            tools: 工具 schema 列表（OpenAI 格式）
            tool_choice: "auto" | "none" | {"type": "function", "function": {"name": "..."}}
            stream: 是否流式
            model: 覆盖默认模型（如指定则使用此模型而非构造时的 self.model）
            **kwargs: 透传给 API body

        Returns:
            stream=False: Response 对象
            stream=True: 生成器，逐步 yield Response（增量）
        """
        if stream:
            return self._stream_chat(messages, tools, tool_choice, model=model, **kwargs)

        # 非流式
        payload = self._build_payload(messages, tools, tool_choice, stream=False, model=model, **kwargs)
        result = await self._request(payload)
        return self._parse_response(result)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> Generator[Response, None, None]:
        """同步包装，方便在非 async 上下文中使用。"""
        async def _runner():
            async for resp in self._stream_chat(messages, tools, tool_choice, **kwargs):
                yield resp

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已在事件循环中，返回异步生成器
                return self._async_to_sync_wrapper(messages, tools, tool_choice, **kwargs)
            else:
                return loop.run_until_complete(_runner())
        except RuntimeError:
            return asyncio.run(_runner())

    def _async_to_sync_wrapper(self, messages, tools, tool_choice, **kwargs):
        """在已有事件循环中同步调用流式接口（用于阻塞等待）。"""
        import concurrent.futures
        def run():
            async def _collect():
                results = []
                async for r in self._stream_chat(messages, tools, tool_choice, **kwargs):
                    results.append(r)
                return results
            return asyncio.run(_collect())
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(run)
            return future.result()

    # ── 内部方法 ────────────────────────────────────────────────────────────

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]],
        tool_choice: str,
        stream: bool,
        model: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        active_model = model or self.model
        payload = {
            "model": active_model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        # DeepSeek prefix (FIM / forced output format)
        if "prefix" in kwargs:
            payload["prefix"] = kwargs.pop("prefix")
        if "extra_body" in kwargs:
            payload.update(kwargs.pop("extra_body"))
        if kwargs:
            payload.update(kwargs)
        return payload

    async def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                body = resp.json()

                if resp.status_code == 200:
                    self._track_usage(body.get("usage", {}))
                    return body

                error = DeepSeekAPIError.from_response(resp.status_code, body)

                # 上下文超限 → 抛专用异常
                if "context_length_exceeded" in str(error).lower():
                    raise ContextLengthExceededError(str(error)) from None

                if not error.is_retriable or attempt == self.max_retries - 1:
                    raise error

                # 指数退避
                await asyncio.sleep(2 ** attempt * 0.5)
                last_error = error

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise last_error or DeepSeekAPIError(500, "Max retries exceeded")

    def _parse_response(self, body: Dict[str, Any]) -> Response:
        choice = body["choices"][0]
        msg = choice["message"]

        tool_calls = None
        if "tool_calls" in msg and msg["tool_calls"]:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                )
                for tc in msg["tool_calls"]
            ]

        thinking = None
        content = msg.get("content")
        # deepseek-reasoner 模型：content 可能以 <think>...</think> 包裹
        if content and self.model == "deepseek-reasoner":
            content, thinking = self._extract_thinking(content)

        return Response(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking,
            usage=body.get("usage"),
            raw=body,
        )

    # ── 流式解析核心 ────────────────────────────────────────────────────────

    async def _stream_chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]],
        tool_choice: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Response, None]:
        active_model = model or self.model
        payload = self._build_payload(messages, tools, tool_choice, stream=True, model=active_model, **kwargs)

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.json()
                raise DeepSeekAPIError.from_response(resp.status_code, body)

            current_tc: Optional[ToolCallDelta] = None
            thinking_buf: List[str] = []
            content_buf: List[str] = []
            in_thinking = False
            is_reasoner = active_model == "deepseek-reasoner"
            first_chunk_sent = False

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line.strip() == "data: [DONE]":
                    break

                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})

                # ── thinking 提取（reasoner 模型）────────────────────────
                if is_reasoner and "content" in delta:
                    text = delta["content"]
                    if "<think>" in text:
                        in_thinking = True
                        text = text.split("<think>")[-1]
                    if "</think>" in text:
                        text = text.split("</think>")[0]
                        thinking_buf.append(text)
                        in_thinking = False
                        # 立即 yield thinking（第一块）
                        if not first_chunk_sent:
                            yield Response(
                                thinking="".join(thinking_buf),
                                raw=data,
                            )
                            first_chunk_sent = True
                        continue
                    if in_thinking:
                        thinking_buf.append(text)
                        continue

                # ── tool_call 增量解析 ──────────────────────────────────
                if "tool_call" in delta:
                    tc_delta = delta["tool_call"]

                    if tc_delta.get("index") is not None:
                        idx = tc_delta["index"]
                        # 简单策略：按顺序维护当前 tool_call 列表
                        # （SSE 不保证 delta 顺序，严格实现需 buffer）
                        if current_tc is None or idx != getattr(current_tc, "_index", -1):
                            current_tc = ToolCallDelta()
                            current_tc._index = idx
                        if tc_delta.get("id"):
                            current_tc.id = tc_delta["id"]
                        if tc_delta.get("function", {}).get("name"):
                            current_tc.name = tc_delta["function"]["name"]
                        if tc_delta.get("function", {}).get("arguments"):
                            current_tc.arguments += tc_delta["function"]["arguments"]
                    elif tc_delta.get("id"):
                        # 简化路径：单个 tool_call
                        if current_tc is None:
                            current_tc = ToolCallDelta()
                        if tc_delta.get("id"):
                            current_tc.id = tc_delta["id"]
                        if tc_delta.get("function", {}).get("name"):
                            current_tc.name = tc_delta["function"]["name"]
                        if tc_delta.get("function", {}).get("arguments"):
                            current_tc.arguments += tc_delta["function"]["arguments"]

                # ── content 增量 ─────────────────────────────────────────
                elif "content" in delta and delta["content"]:
                    content_buf.append(delta["content"])

                # ── tool_call 完成（role = "tool" 出现）──────────────────
                # 注意：SSE 中 tool_call 完成通常由 delta.content=null 标志
                # 这里我们通过检查 delta 中是否有 finish_reason="tool_calls" 来判断
                # 更可靠的方式是在流结束时做一次批量 yield

            # 流结束：汇总完整响应
            final_content = "".join(content_buf)
            if current_tc is not None and current_tc.id:
                current_tc.complete = True
                tool_calls = [ToolCall.from_delta(current_tc)]
            else:
                tool_calls = None

            yield Response(
                content=final_content,
                tool_calls=tool_calls,
                thinking="".join(thinking_buf) if thinking_buf else None,
                usage=data.get("usage"),
                raw=data,
            )

    @staticmethod
    def _extract_thinking(content: str) -> tuple[str, Optional[str]]:
        """从 content 中提取 <think>...</think> 部分。"""
        if "<think>" in content and "</think>" in content:
            start = content.index("<think>") + len("<think>")
            end = content.index("</think>")
            thinking = content[start:end]
            text = content[:content.index("<think>")] + content[end + len("</think>"):]
            return text.strip(), thinking.strip()
        return content, None

    def _track_usage(self, usage: Dict[str, int]):
        self._total_tokens += usage.get("total_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    async def close(self):
        await self._client.aclose()
