"""
FastAPI 服务 — HTTP API + WebSocket 流式端点。
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Agent 核心
from ..core.client import DeepSeekClient
from ..tools import (
    ToolRegistry,
    read_file, write_file, edit_file, list_directory,
    search_file, search_content, delete_file,
    run_shell, run_test,
)
from ..tools.git import (
    git_diff, git_log, git_status, git_checkout,
    git_commit, git_push, git_branch,
)
from ..tools.web import web_fetch, read_docs
from ..memory.manager import MemoryManager
from ..agent.loop import AgentLoop, LoopConfig, LoopMode, CLIApprovalCallback
from ..core.sandbox import SandboxConfig, create_sandbox


# ── 请求/响应模型 ───────────────────────────────────────────────────────

class RunRequest(BaseModel):
    task: str
    project: Optional[str] = "."
    model: str = "deepseek-chat"
    mode: str = "react"
    stream: bool = True


class RunResponse(BaseModel):
    session_id: str
    status: str


@dataclass
class Session:
    id: str
    client: DeepSeekClient
    registry: ToolRegistry
    memory: MemoryManager
    agent: AgentLoop
    sandbox: Any = None  # DockerSandbox 或 LocalSandbox
    config: Dict[str, Any] = field(default_factory=dict)


# ── 全局状态 ─────────────────────────────────────────────────────────────

sessions: Dict[str, Session] = {}


def create_session(session_id: str, project: str, model: str, mode: str) -> Session:
    """创建新会话。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置")

    client = DeepSeekClient(api_key=api_key, model=model)

    registry = ToolRegistry()
    # 注册文件系统工具
    for fn in [read_file, write_file, edit_file, list_directory,
               search_file, search_content, delete_file, run_shell, run_test]:
        registry.register_func(fn)
    # 注册 Git 工具
    for fn in [git_diff, git_log, git_status, git_checkout,
               git_commit, git_push, git_branch]:
        registry.register_func(fn)
    # 注册 Web 工具
    registry.register_func(web_fetch)
    registry.register_func(read_docs)

    memory = MemoryManager(
        project_path=project,
        long_term_dir=os.path.expanduser("~/.deepseek_agent_memory"),
    )

    loop_config = LoopConfig(
        mode=LoopMode.PLAN_EXECUTE if mode == "plan" else LoopMode.REACT,
        max_steps=50,
        max_execution_time=600,
    )

    agent = AgentLoop(
        client=client,
        registry=registry,
        memory=memory,
        config=loop_config,
        permission_callback=CLIApprovalCallback(),
    )

    return Session(
        id=session_id,
        client=client,
        registry=registry,
        memory=memory,
        agent=agent,
    )


# ── FastAPI 应用 ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # 清理所有会话
    for s in sessions.values():
        await s.client.close()


app = FastAPI(
    title="DeepSeek Code Agent API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST API ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}


@app.post("/session", response_model=RunResponse)
async def create_session_endpoint(project: str = ".", model: str = "deepseek-chat", mode: str = "react"):
    """创建新会话。"""
    session_id = str(uuid.uuid4())[:8]
    try:
        session = create_session(session_id, project, model, mode)
        sessions[session_id] = session
        return RunResponse(session_id=session_id, status="created")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """查看会话状态。"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    s = sessions[session_id]
    return {
        "id": s.id,
        "tools": s.registry.list_tools(),
        "memory_tokens": s.memory.short_term.total_tokens(),
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话。"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    s = sessions.pop(session_id)
    await s.client.close()
    return {"status": "deleted"}


@app.post("/session/{session_id}/run")
async def run_task(session_id: str, request: RunRequest):
    """运行单次任务（非流式）。"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    s = sessions[session_id]
    results = []

    async for resp in s.agent.run(request.task):
        if resp.content:
            results.append(resp.content)

    return {"session_id": session_id, "responses": results}


# ── WebSocket 流式端点 ─────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_agent(websocket: WebSocket, session_id: str):
    """
    WebSocket 双向通信，支持：
    - 发送任务
    - 接收流式响应（思考/工具调用/结果/最终回复）
    - 发送审批决策（approve/reject）
    """
    if session_id not in sessions:
        await websocket.close(code=4004, reason="Session not found")
        return

    s = sessions[session_id]
    await websocket.accept()

    pending_approvals: Dict[str, Any] = {}

    # 替换审批回调为 WebSocket 版本
    class WSApprovalCallback:
        async def requires_approval(self, tool_name, args, danger_level):
            return True  # WebSocket 模式下所有敏感操作都请求确认

        async def request_approval(self, tool_name, args, danger_level):
            # 通过 WebSocket 发送审批请求
            approval_id = str(uuid.uuid4())[:8]
            pending_approvals[approval_id] = {
                "tool_name": tool_name,
                "args": args,
                "danger_level": danger_level.value if hasattr(danger_level, "value") else int(danger_level),
            }
            await websocket.send_json({
                "type": "approval_request",
                "approval_id": approval_id,
                "tool": tool_name,
                "args": args,
                "danger_level": pending_approvals[approval_id]["danger_level"],
            })

            # 等待审批结果（超时 5 分钟）
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=300)
                approval_type = msg.get("type")
                if approval_type == "approval_response":
                    rid = msg.get("approval_id")
                    if rid == approval_id:
                        approved = msg.get("approved", False)
                        modified = msg.get("modified_args")
                        del pending_approvals[rid]
                        return approved, modified
            except asyncio.TimeoutError:
                del pending_approvals[approval_id]
                return False, None
            return False, None

    s.agent.permission = WSApprovalCallback()

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "run":
                task = msg.get("task", "")
                await websocket.send_json({"type": "status", "message": "开始执行..."})

                async for resp in s.agent.run(task):
                    event: Dict[str, Any] = {"type": "chunk"}

                    if resp.thinking:
                        event["thinking"] = resp.thinking
                    if resp.tool_calls:
                        event["tool_calls"] = [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in resp.tool_calls
                        ]
                    if resp.content:
                        event["content"] = resp.content

                    if len(event) > 1:  # 至少有一点内容
                        await websocket.send_json(event)

                await websocket.send_json({"type": "done"})

            elif msg_type == "approval_response":
                # 已在 WSApprovalCallback 中处理
                pass

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


# ── 启动入口 ─────────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
