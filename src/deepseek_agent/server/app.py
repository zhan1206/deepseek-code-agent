"""
FastAPI 服务 — HTTP API + WebSocket 流式端点 + Diff 审批管理。

REST 端点:
  GET  /health
  POST /session          — 创建会话
  GET  /session/{id}    — 会话状态
  DELETE /session/{id}  — 删除会话
  POST /session/{id}/run  — 运行任务（非流式）
  POST /diff/preview     — 生成编辑预览（不落盘）
  POST /diff/apply      — 应用已审批的编辑
  POST /diff/reject     — 拒绝编辑

WebSocket /ws/{session_id}:
  接收: {type:"run", task:"..."} | {type:"approval_response", approval_id:"...", approved:bool, modified_args:{}}
  发送: {type:"chunk", content:"...", tool_calls:[...]} | {type:"approval_request", ...} | {type:"done"} | {type:"error"}
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
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


class DiffPreviewRequest(BaseModel):
    session_id: str
    tool: str
    args: Dict[str, Any]


class DiffPreviewResponse(BaseModel):
    preview_id: str
    file: str
    original: str
    modified: str
    hunks: List[Dict[str, Any]]


class DiffApplyRequest(BaseModel):
    preview_id: str
    session_id: str
    modified_hunks: Optional[List[Dict[str, Any]]] = None  # 用户修改后的 hunks


# ── Diff 管理 ───────────────────────────────────────────────────────────

@dataclass
class PendingDiff:
    """单个待审批的编辑。"""
    preview_id: str
    session_id: str
    tool: str          # edit_file | write_file | delete_file
    args: Dict[str, Any]
    file: str
    original: str      # 编辑前内容
    modified: str       # 编辑后内容（或空串表示删除）
    hunks: List[Dict[str, Any]] = field(default_factory=list)
    approved: bool = False
    modified_hunks: Optional[List[Dict[str, Any]]] = None


class DiffManager:
    """全局 Diff 预览管理器。"""

    def __init__(self):
        self._store: Dict[str, PendingDiff] = {}

    def create_preview(
        self,
        session_id: str,
        tool: str,
        args: Dict[str, Any],
        original: str,
        modified: str,
    ) -> PendingDiff:
        """生成预览并存储。"""
        preview_id = str(uuid.uuid4())[:8]
        hunks = self._build_hunks(original, modified)

        diff = PendingDiff(
            preview_id=preview_id,
            session_id=session_id,
            tool=tool,
            args=args,
            file=args.get("path", args.get("file", "")),
            original=original,
            modified=modified,
            hunks=hunks,
        )
        self._store[preview_id] = diff
        return diff

    @staticmethod
    def _build_hunks(original: str, modified: str) -> List[Dict[str, Any]]:
        """用 difflib 生成 unified diff hunks。"""
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)

        unified = list(difflib.unified_diff(
            orig_lines, mod_lines,
            fromfile="original", tofile="modified",
            lineterm="",
        ))

        hunks = []
        i = 0
        while i < len(unified):
            line = unified[i]
            if line.startswith("@@"):
                # 解析 @@ -a,b +c,d @@
                parts = line.split(" ")
                meta = parts[1]
                m = meta.split(",")
                old_start = int(m[0][1:])
                old_count = int(m[0][1:]) if len(m) == 1 else int(m[1])
                new_start = int(parts[2][1:])
                new_count = int(parts[2][1:]) if len(parts[2].split(",")) == 1 else int(parts[2].split(",")[1])

                hunk_lines = [line]
                i += 1
                while i < len(unified) and not unified[i].startswith("@@"):
                    hunk_lines.append(unified[i])
                    i += 1

                hunks.append({
                    "old_start": old_start,
                    "old_count": old_count,
                    "new_start": new_start,
                    "new_count": new_count,
                    "lines": hunk_lines,
                })
            else:
                i += 1
        return hunks

    def apply(self, preview_id: str, modified_hunks: Optional[List[Dict[str, Any]]] = None) -> PendingDiff:
        """标记为已批准，可附带用户修改后的 hunks。"""
        diff = self._store.get(preview_id)
        if not diff:
            raise ValueError(f"Preview {preview_id} not found")
        diff.approved = True
        diff.modified_hunks = modified_hunks
        return diff

    def reject(self, preview_id: str) -> None:
        """标记为已拒绝。"""
        self._store.pop(preview_id, None)

    def get(self, preview_id: str) -> Optional[PendingDiff]:
        return self._store.get(preview_id)

    def cleanup_session(self, session_id: str) -> None:
        """清理会话相关的所有预览。"""
        to_remove = [k for k, v in self._store.items() if v.session_id == session_id]
        for k in to_remove:
            self._store.pop(k, None)


# ── 会话 ────────────────────────────────────────────────────────────────

@dataclass
class Session:
    id: str
    client: DeepSeekClient
    registry: ToolRegistry
    memory: MemoryManager
    agent: AgentLoop
    sandbox: Any = None  # DockerSandbox 或 LocalSandbox
    config: Dict[str, Any] = field(default_factory=dict)


sessions: Dict[str, Session] = {}
diff_manager = DiffManager()


def create_session(session_id: str, project: str, model: str, mode: str) -> Session:
    """创建新会话。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置")

    client = DeepSeekClient(api_key=api_key, model=model)

    registry = ToolRegistry()
    for fn in [read_file, write_file, edit_file, list_directory,
               search_file, search_content, delete_file, run_shell, run_test]:
        registry.register(fn)
    for fn in [git_diff, git_log, git_status, git_checkout,
               git_commit, git_push, git_branch]:
        registry.register(fn)
    registry.register(web_fetch)
    registry.register(read_docs)

    # Template tools (v2.0 §5.4)
    from ..templates.tools import register_template_tools
    register_template_tools(registry)

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
    for s in sessions.values():
        await s.client.close()
        diff_manager.cleanup_session(s.id)


app = FastAPI(
    title="DeepSeek Code Agent API",
    version="0.3.0",
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
async def create_session_endpoint(
    project: str = ".", model: str = "deepseek-chat", mode: str = "react"
):
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
    diff_manager.cleanup_session(session_id)
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


# ── Diff 审批 API ──────────────────────────────────────────────────────

@app.post("/diff/preview", response_model=DiffPreviewResponse)
async def diff_preview(req: DiffPreviewRequest):
    """
    生成文件编辑预览（不落盘）。
    前端在收到 tool_call 事件后调用此端点获取 diff，然后向用户展示。
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    s = sessions[req.session_id]
    tool = req.tool
    args = req.args

    try:
        original = ""
        if tool == "write_file":
            path = args.get("path", args.get("file", ""))
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    original = f.read()
            modified = args.get("content", "")
        elif tool == "edit_file":
            path = args.get("path", args.get("file", ""))
            if os.path.exists(path):
                with open(path, "r", encoding="utf" if tool != "delete_file" else "utf-8") as f:
                    original = f.read()
            # 模拟 apply_edit 后的内容（不做实际修改）
            modified = original
            if "old_text" in args and "new_text" in args:
                modified = original.replace(args["old_text"], args["new_text"], 1)
            elif "content" in args:
                modified = args["content"]
        elif tool == "delete_file":
            path = args.get("path", args.get("file", ""))
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    original = f.read()
            modified = ""
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported tool: {tool}")

        diff = diff_manager.create_preview(
            session_id=req.session_id,
            tool=tool,
            args=args,
            original=original,
            modified=modified,
        )

        return DiffPreviewResponse(
            preview_id=diff.preview_id,
            file=diff.file,
            original=diff.original,
            modified=diff.modified,
            hunks=diff.hunks,
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {args.get('path')}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diff/apply")
async def diff_apply(req: DiffApplyRequest):
    """
    应用已审批的编辑。
    前端发送用户决策后调用，后端执行真实写入。
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    diff = diff_manager.get(req.preview_id)
    if not diff:
        raise HTTPException(status_code=404, detail=f"Preview {req.preview_id} not found or already processed")

    # 执行真实工具调用
    s = sessions[req.session_id]
    tool_fn = s.registry.get_tool(diff.tool)
    if not tool_fn:
        raise HTTPException(status_code=500, detail=f"Tool not found: {diff.tool}")

    try:
        result = await tool_fn(**diff.args)
        diff_manager._store.pop(req.preview_id, None)  # 清理
        return {"status": "applied", "preview_id": req.preview_id, "result": result.to_dict()}
    except Exception as e:
        return {"status": "error", "preview_id": req.preview_id, "error": str(e)}


@app.post("/diff/reject")
async def diff_reject(preview_id: str, session_id: str):
    """拒绝编辑预览。"""
    diff_manager.reject(preview_id)
    return {"status": "rejected", "preview_id": preview_id}


# ── WebSocket 流式端点 ─────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_agent(websocket: WebSocket, session_id: str):
    """
    WebSocket 双向通信:
    - 发送 {type:"run", task:"..."} → 启动任务
    - 接收流式响应 (chunk / approval_request / done / error)
    - 发送 {type:"approval_response", approval_id:"...", approved:bool}
    """
    if session_id not in sessions:
        await websocket.close(code=4004, reason="Session not found")
        return

    s = sessions[session_id]
    await websocket.accept()

    pending_approvals: Dict[str, Any] = {}

    class WSApprovalCallback:
        """WebSocket 驱动的审批回调。"""

        async def requires_approval(self, tool_name, args, danger_level):
            return True

        async def request_approval(self, tool_name, args, danger_level):
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

            # 自动拒绝（前端会主动发 approval_response，这里设 30s 超时）
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=300)
                if msg.get("type") == "approval_response":
                    rid = msg.get("approval_id")
                    if rid == approval_id:
                        approved = msg.get("approved", False)
                        modified = msg.get("modified_args")
                        pending_approvals.pop(rid, None)
                        return approved, modified
            except asyncio.TimeoutError:
                pending_approvals.pop(approval_id, None)
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

                    if len(event) > 1:
                        await websocket.send_json(event)

                await websocket.send_json({"type": "done"})

            elif msg_type == "approval_response":
                # 审批响应（已在 WSApprovalCallback 中处理）
                pass

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


@app.get("/a2a/card")
async def a2a_card():
    """返回 Agent 身份卡（A2A 发现端点）"""
    try:
        from ..a2a import get_a2a_server
        return get_a2a_server().get_card()
    except Exception as e:
        return {"error": str(e)}


@app.post("/a2a/request")
async def a2a_request(request: Request):
    """处理来自其他 Agent 的 A2A 请求"""
    try:
        data = await request.json()
        from ..a2a import get_a2a_server
        server = get_a2a_server()
        return await server.handle_request(data)
    except Exception as e:
        return {"error": str(e)}


# ── P2 端点：语音转写 & 截图分析 ────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(request: Request):
    """语音转写（需 Whisper API 或本地 whisper）"""
    form = await request.form()
    audio = form.get("audio")
    if not audio:
        return {"error": "No audio file provided"}
    # 骨架：实际集成 Whisper API
    return {
        "text": "",
        "note": "Whisper integration pending; requires OPENAI_API_KEY or local whisper.cpp",
    }


@app.post("/api/analyze-image")
async def analyze_image(request: Request):
    """截图/图片分析（需 DeepSeek VL 或类似视觉模型）"""
    form = await request.form()
    image = form.get("image")
    prompt = (await request.form()).get("prompt", "Describe this image in detail. If it contains code, extract it.")
    if not image:
        return {"error": "No image file provided"}
    # 骨架：实际集成 DeepSeek VL
    return {
        "description": "",
        "note": "Vision model integration pending; requires DeepSeek VL or GPT-4V API key",
    }


@app.post("/feedback")
async def submit_feedback(request: Request):
    """提交错误反馈报告"""
    try:
        data = await request.json()
        from ..core.feedback import get_feedback_collector
        collector = get_feedback_collector()
        report = collector.create_report(
            user_description=data.get("description", ""),
            context=data.get("context"),
        )
        return {"status": "saved", "report": report}
    except Exception as e:
        return {"error": str(e)}


# ── 启动入口 ─────────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
