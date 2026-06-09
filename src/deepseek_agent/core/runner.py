"""
沙箱 Runner — 将 DockerSandbox 集成到 AgentLoop 的代码执行流程中。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..tools.base import ToolResult
from .sandbox import SandboxConfig, DockerSandbox, LocalSandbox, create_sandbox


@dataclass
class ExecutionContext:
    """单次代码执行的上下文。"""
    id: str
    sandbox: Any  # DockerSandbox | LocalSandbox
    workspace: str
    command: str
    language: str = "python"
    timeout: int = 60
    created_at: float = field(default_factory=time.time)
    container_id: Optional[str] = None


class SandboxRunner:
    """
    沙箱 Runner：将 Agent 的代码执行请求路由到隔离环境。

    集成流程：
    1. Agent 检测到 run_code 类型的任务
    2. SandboxRunner.start() 启动沙箱（容器或子进程）
    3. Agent 将代码写入临时文件
    4. SandboxRunner.exec() 执行代码
    5. SandboxRunner.stop() 清理

    与 AgentLoop 的集成方式：
    - 方案 A：AgentLoop 持有 SandboxRunner，代码工具内部调用
    - 方案 B：注入 registry，代码执行工具自动路由到沙箱
    """

    def __init__(
        self,
        project_path: str,
        config: Optional[SandboxConfig] = None,
        auto_start: bool = False,
    ):
        self.project_path = Path(project_path).resolve()
        self.config = config or SandboxConfig()
        self._docker_sandbox: Optional[DockerSandbox] = None
        self._local_sandbox: Optional[LocalSandbox] = None
        self._active: bool = False
        self._executions: Dict[str, ExecutionContext] = {}

        if auto_start:
            asyncio.run(self.start())

    # ── 生命周期 ──────────────────────────────────────────────

    async def start(self) -> str:
        """
        启动沙箱（优先 Docker，自动降级）。

        Returns:
            容器 ID 或 "local"
        """
        self._docker_sandbox, self._local_sandbox = create_sandbox(
            str(self.project_path), self.config
        )

        if self._docker_sandbox:
            try:
                container_id = await self._docker_sandbox.start()
                self._active = True
                return container_id
            except Exception as e:
                print(f"[SandboxRunner] Docker 启动失败，降级到本地模式: {e}")
                self._docker_sandbox = None

        if self._local_sandbox:
            await self._local_sandbox.start()
            self._active = True
            return "local"

        raise RuntimeError("无法启动任何沙箱（Docker 和 Local 都失败）")

    async def stop(self) -> None:
        """停止所有沙箱，清理容器。"""
        if self._docker_sandbox:
            await self._docker_sandbox.stop()
        if self._local_sandbox:
            await self._local_sandbox.stop()
        self._active = False
        self._executions.clear()

    @property
    def is_active(self) -> bool:
        return self._active

    # ── 代码执行 ──────────────────────────────────────────────

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
        filename: str = "",
    ) -> ToolResult:
        """
        执行代码（自动路由到当前活跃沙箱）。

        Args:
            code: 代码内容
            language: 语言（python / node / bash）
            timeout: 超时秒数
            filename: 保存的文件名
        """
        if not self._active:
            return ToolResult.fail("沙箱未启动，请先调用 runner.start()")

        # 选择语言对应的命令
        cmd_map = {
            "python": f"python3 {filename or 'temp_script.py'}",
            "node": f"node {filename or 'temp_script.js'}",
            "bash": f"bash {filename or 'temp_script.sh'}",
        }
        command = cmd_map.get(language.lower(), f"python3 {filename or 'temp_script.py'}")

        # 写入临时文件（通过 Agent 的 write_file 工具）
        exec_id = str(uuid.uuid4())[:8]
        ctx = ExecutionContext(
            id=exec_id,
            sandbox=self._docker_sandbox or self._local_sandbox,
            workspace=self.config.work_dir,
            command=command,
            language=language,
            timeout=timeout or self.config.max_runtime,
        )
        self._executions[exec_id] = ctx

        # 通过沙箱执行
        result = await ctx.sandbox.exec(command, timeout=ctx.timeout)
        return result

    async def write_and_run(
        self,
        code: str,
        language: str = "python",
        filename: str = "",
        timeout: Optional[int] = None,
    ) -> ToolResult:
        """
        写入文件并立即执行（便捷方法）。

        等价于：
            agent.run_tool("write_file", path=filename, content=code)
            agent.run_tool("sandbox_exec", command=cmd)
        """
        ext_map = {"python": ".py", "node": ".js", "bash": ".sh", "javascript": ".js"}
        ext = ext_map.get(language.lower(), ".py")
        fname = filename or f"temp_{uuid.uuid4().hex[:8]}{ext}"
        rel_path = f"/tmp/{fname}" if self.config.provider == "local" else fname

        # 写入文件（通过本地 subprocess 因为此时沙箱还未创建）
        try:
            local_path = self.project_path / fname
            local_path.write_text(code, encoding="utf-8")
            exec_result = await self.execute_code(
                code=code,
                language=language,
                timeout=timeout,
                filename=fname,
            )
            # 清理临时文件
            try:
                local_path.unlink(missing_ok=True)
            except Exception:
                pass
            return exec_result
        except Exception as e:
            return ToolResult.fail(f"写入/执行失败: {str(e)}")

    # ── 工具化 ──────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回沙箱相关工具的 Schema（用于注册到 AgentLoop）。"""
        return [
            {
                "name": "sandbox_start",
                "description": "启动代码执行沙箱（Docker 容器或本地子进程）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "provider": {
                            "type": "string",
                            "enum": ["auto", "docker", "local"],
                            "default": "auto",
                            "description": "沙箱类型"
                        },
                        "network": {
                            "type": "boolean",
                            "default": False,
                            "description": "是否允许网络访问"
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "sandbox_exec",
                "description": "在沙箱中执行代码。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "要执行的代码"
                        },
                        "language": {
                            "type": "string",
                            "enum": ["python", "node", "bash"],
                            "default": "python",
                            "description": "编程语言"
                        },
                        "filename": {
                            "type": "string",
                            "default": "",
                            "description": "文件名（可选，用于多文件场景）"
                        },
                        "timeout": {
                            "type": "integer",
                            "default": 60,
                            "description": "超时秒数"
                        },
                    },
                    "required": ["code"],
                },
            },
            {
                "name": "sandbox_stop",
                "description": "停止沙箱，清理容器。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]
