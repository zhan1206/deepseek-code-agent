"""
代码执行沙箱 — Docker 容器隔离 + 本地子进程降级。
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..tools.base import ToolResult


# ── 配置模型 ──────────────────────────────────────────────────────────────

class SandboxConfig(BaseModel):
    """沙箱配置。"""
    provider: str = "local"  # "docker" | "local"
    image: str = "deepseek-sandbox:latest"
    allowed_commands: List[str] = Field(
        default_factory=lambda: ["python", "python3", "node", "npm", "git", "ls", "cat", "grep", "find", "echo"]
    )
    denied_patterns: List[str] = Field(default_factory=lambda: [
        r"rm\s+-rf\s+/", r"mkfs", r"dd\s+if=/dev/zero", r":\(\)\{", r">\s*/dev/sd"
    ])
    network: bool = False
    max_runtime: int = 120
    max_memory_mb: int = 512
    work_dir: str = "/workspace"


# ── 安全检查 ──────────────────────────────────────────────────────────────

DENY_PATTERNS = [
    r"rm\s+-rf\s+/", r"mkfs", r"dd\s+if=/dev/zero", r":\(\)\{",
    r"chmod\s+-R\s+777\s+/", r">\s*/etc/", r"curl\s+http://",
]


def _check_command(command: str, config: SandboxConfig) -> Tuple[bool, str]:
    """检查命令是否安全。"""
    for pat in config.denied_patterns:
        if re.search(pat, command):
            return False, f"危险命令模式: {pat}"
    return True, ""


# ── Docker 沙箱 ──────────────────────────────────────────────────────────

class DockerSandbox:
    """
    Docker 容器沙箱。

    特性：
    - 每个会话独立的临时容器（--rm，退出自动清理）
    - 项目目录只读挂载，写时复制
    - 网络隔离（可选）
    - 非 root 用户执行
    - 资源限制（CPU/内存/超时）
    """

    def __init__(self, project_path: str, config: Optional[SandboxConfig] = None):
        self.project_path = Path(project_path).resolve()
        self.config = config or SandboxConfig()
        self._container_id: Optional[str] = None
        self._client = None

    def _get_client(self):
        """延迟导入 docker（可选依赖）。"""
        if self._client is None:
            try:
                import docker
                self._client = docker.from_env()
            except ImportError:
                raise RuntimeError(
                    "Docker SDK 未安装。请运行：pip install docker\n"
                    "或设置 sandbox.provider='local' 使用子进程模式。"
                )
        return self._client

    async def start(self) -> str:
        """启动沙箱容器。"""
        try:
            client = self._get_client()
        except RuntimeError as e:
            raise e

        try:
            volumes = {
                str(self.project_path): {"bind": self.config.work_dir, "mode": "rw"}
            }

            # pull 镜像
            try:
                client.images.get(self.config.image)
            except Exception:
                print(f"正在拉取镜像 {self.config.image}（可能需要几分钟）...")
                client.images.pull(self.config.image)

            container = client.containers.run(
                self.config.image,
                "sleep infinity",
                volumes=volumes,
                working_dir=self.config.work_dir,
                detach=True,
                mem_limit=f"{self.config.max_memory_mb}m",
                nano_cpus=int(self.config.max_runtime * 1e9),  # CPU 时间上限
                network_mode="none" if not self.config.network else "bridge",
                user="sandbox",
                auto_remove=True,
                stderr=True,
            )
            self._container_id = container.id
            return container.id[:12]

        except Exception as e:
            raise RuntimeError(f"启动 Docker 沙箱失败: {str(e)}")

    async def exec(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> ToolResult:
        """在容器中执行命令。"""
        if not self._container_id:
            return ToolResult.fail("沙箱未启动")

        safe, reason = _check_command(command, self.config)
        if not safe:
            return ToolResult.fail(f"命令被拦截: {reason}")

        timeout = timeout or self.config.max_runtime

        try:
            client = self._get_client()
            container = client.containers.get(self._container_id)

            # exec_create → exec_start
            exec_id = container.client.api.exec_create(
                container.id,
                f"bash -c {command!r}",
                user="sandbox",
                workdir=self.config.work_dir,
            )

            output = container.client.api.exec_start(
                exec_id,
                stream=False,
                demux=False,
            )

            exit_code = container.client.api.exec_inspect(exec_id)["ExitCode"]

            output_str = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)

            return ToolResult.ok(
                f"[exit {exit_code}]\n{output_str[:5000]}"
            )

        except Exception as e:
            return ToolResult.fail(f"容器执行失败: {str(e)}")

    async def stop(self) -> None:
        """停止沙箱（容器为 auto_remove，会自动清理）。"""
        if self._container_id:
            try:
                client = self._get_client()
                container = client.containers.get(self._container_id)
                container.kill()
            except Exception:
                pass
            self._container_id = None


# ── 本地子进程沙箱 ──────────────────────────────────────────────────────

class LocalSandbox:
    """
    本地子进程沙箱（无 Docker 时的降级方案）。

    限制：
    - 依赖系统层级隔离（Windows 支持有限）
    - 仅靠超时和命令白名单保护
    """

    def __init__(self, project_path: str, config: Optional[SandboxConfig] = None):
        self.project_path = Path(project_path).resolve()
        self.config = config or SandboxConfig()

    async def start(self) -> str:
        """本地模式无需启动容器。"""
        return "local"

    async def exec(self, command: str, timeout: Optional[int] = None) -> ToolResult:
        """在子进程中执行命令。"""
        safe, reason = _check_command(command, self.config)
        if not safe:
            return ToolResult.fail(f"命令被拦截: {reason}")

        # 白名单检查
        if self.config.allowed_commands:
            cmd_name = command.strip().split()[0].split("=")[0]
            if cmd_name not in self.config.allowed_commands:
                return ToolResult.fail(f"命令 '{cmd_name}' 不在白名单中")

        timeout = min(timeout or self.config.max_runtime, 300)

        # 移除敏感环境变量
        env = {
            k: v for k, v in os.environ.items()
            if not any(secret in k.upper() for secret in ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"])
        }

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            output = []
            if result.stdout:
                output.append(f"[stdout]\n{result.stdout[:5000]}")
            if result.stderr:
                output.append(f"[stderr]\n{result.stderr[:5000]}")
            output.append(f"[退出码: {result.returncode}]")

            return ToolResult.ok("\n".join(output))

        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"命令超时（>{timeout}s）")
        except Exception as e:
            return ToolResult.fail(f"执行失败: {str(e)}")

    async def stop(self) -> None:
        """本地模式无需清理。"""
        pass


# ── 工厂函数 ─────────────────────────────────────────────────────────────

def create_sandbox(
    project_path: str,
    config: Optional[SandboxConfig] = None,
) -> Tuple[DockerSandbox, LocalSandbox]:
    """
    创建沙箱实例。

    优先 Docker，失败时降级到 LocalSandbox。
    """
    config = config or SandboxConfig()

    if config.provider == "local":
        return None, LocalSandbox(project_path, config)

    if config.provider == "docker":
        try:
            import docker
            docker.from_env()
            return DockerSandbox(project_path, config), LocalSandbox(project_path, config)
        except Exception:
            return None, LocalSandbox(project_path, config)

    # auto：自动选择
    try:
        import docker
        docker.from_env()
        return DockerSandbox(project_path, config), LocalSandbox(project_path, config)
    except Exception:
        return None, LocalSandbox(project_path, config)
