"""
A2A (Agent-to-Agent) 协议 — P2 优先级
Agent 发现 → 能力协商 → 任务委派 → 结果回收

核心接口：
- A2AServer: 暴露自身能力，接收其他 Agent 的请求
- A2AClient: 发现远端 Agent，协商能力，委派任务
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCapability:
    """Agent 能力描述"""
    name: str
    description: str
    tools: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCard:
    """Agent 身份卡（用于发现）"""
    agent_id: str
    name: str
    version: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    endpoint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "capabilities": [
                {"name": c.name, "description": c.description, "tools": c.tools, "models": c.models}
                for c in self.capabilities
            ],
            "endpoint": self.endpoint,
            "metadata": self.metadata,
        }


@dataclass
class A2ATask:
    """跨 Agent 任务"""
    task_id: str = ""
    source_agent: str = ""
    target_agent: str = ""
    task_type: str = ""  # delegate, query, stream, callback
    payload: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{int(time.time()*1000):016x}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()


class A2AServer:
    """A2A 服务端 — 暴露自身能力给其他 Agent"""

    def __init__(self, agent_card: AgentCard):
        self.card = agent_card
        self._handlers: Dict[str, callable] = {}
        self._task_history: List[A2ATask] = []
        self._running = False

    def register_handler(self, task_type: str, handler: callable) -> None:
        """注册任务类型处理器"""
        self._handlers[task_type] = handler

    async def handle_request(self, request: Dict) -> Dict:
        """处理来自其他 Agent 的请求"""
        task = A2ATask(
            source_agent=request.get("source_agent", "unknown"),
            target_agent=request.get("target_agent", self.card.agent_id),
            task_type=request.get("task_type", "delegate"),
            payload=request.get("payload", {}),
        )

        handler = self._handlers.get(task.task_type)
        if not handler:
            task.status = TaskStatus.FAILED
            task.error = f"Unknown task type: {task.task_type}"
            return task.__dict__

        try:
            task.status = TaskStatus.RUNNING
            result = handler(task.payload)
            if asyncio.iscoroutine(result):
                result = await result
            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)

        self._task_history.append(task)
        return task.__dict__

    def get_card(self) -> Dict:
        """返回 Agent 身份卡"""
        return self.card.to_dict()

    def get_status(self) -> Dict:
        """返回服务状态"""
        return {
            "agent_id": self.card.agent_id,
            "running": self._running,
            "handlers": list(self._handlers.keys()),
            "tasks_total": len(self._task_history),
            "tasks_completed": sum(1 for t in self._task_history if t.status == TaskStatus.COMPLETED),
        }


class A2AClient:
    """A2A 客户端 — 发现和连接其他 Agent"""

    def __init__(self, self_id: str, self_name: str):
        self.self_id = self_id
        self.self_name = self_name
        self._known_agents: Dict[str, AgentCard] = {}
        self._config_path = os.path.expanduser("~/.deepseek-agent/agents.json")

    def load_config(self) -> None:
        """从配置文件加载已知 Agent 列表"""
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                agents = json.load(f)
            for agent_data in agents:
                card = AgentCard(
                    agent_id=agent_data["agent_id"],
                    name=agent_data["name"],
                    version=agent_data.get("version", "0.0.0"),
                    endpoint=agent_data.get("endpoint", ""),
                )
                for cap in agent_data.get("capabilities", []):
                    card.capabilities.append(AgentCapability(
                        name=cap["name"],
                        description=cap.get("description", ""),
                        tools=cap.get("tools", []),
                    ))
                self._known_agents[card.agent_id] = card
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save_config(self) -> None:
        """保存已知 Agent 列表"""
        try:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            data = [card.to_dict() for card in self._known_agents.values()]
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def discover(self, endpoint: str) -> Optional[AgentCard]:
        """发现远端 Agent（通过 HTTP GET /a2a/card）"""
        # 实际实现需要 httpx 或 aiohttp
        # 这里返回骨架
        return None

    def register_agent(self, card: AgentCard) -> None:
        """注册一个已知 Agent"""
        self._known_agents[card.agent_id] = card
        self.save_config()

    def list_agents(self) -> List[Dict]:
        """列出所有已知 Agent"""
        return [card.to_dict() for card in self._known_agents.values()]

    def find_agent_by_capability(self, capability_name: str) -> Optional[AgentCard]:
        """按能力查找 Agent"""
        for card in self._known_agents.values():
            for cap in card.capabilities:
                if cap.name == capability_name:
                    return card
        return None

    async def delegate(self, target_id: str, task_type: str, payload: Dict, timeout: float = 30.0) -> Dict:
        """向目标 Agent 委派任务"""
        card = self._known_agents.get(target_id)
        if not card or not card.endpoint:
            return {
                "task_id": "",
                "status": TaskStatus.FAILED.value,
                "error": f"Agent {target_id} not found or has no endpoint",
            }

        task = A2ATask(
            source_agent=self.self_id,
            target_agent=target_id,
            task_type=task_type,
            payload=payload,
        )

        # 实际实现：HTTP POST 到 target endpoint
        # return await httpx.AsyncClient().post(f"{card.endpoint}/a2a/request", json=request, timeout=timeout)

        return {
            "task_id": task.task_id,
            "status": TaskStatus.PENDING.value,
            "note": "A2A HTTP transport not yet wired; use with FastAPI integration",
            "target_endpoint": card.endpoint,
        }


# ── 全局实例 ─────────────────────────────────────────────────────────────

_a2a_server: Optional[A2AServer] = None
_a2a_client: Optional[A2AClient] = None


def get_a2a_server() -> A2AServer:
    global _a2a_server
    if _a2a_server is None:
        card = AgentCard(
            agent_id="deepseek-code-agent",
            name="DeepSeek Code Agent",
            version="2.0.0",
            endpoint="http://localhost:8000",
            capabilities=[
                AgentCapability(name="code_edit", description="代码编辑与重构", tools=["edit_file", "write_file", "auto_refactor"]),
                AgentCapability(name="code_analysis", description="代码分析与检测", tools=["arch_check", "security_scan", "find_symbol"]),
                AgentCapability(name="testing", description="测试生成与运行", tools=["generate_tests", "run_test_suite", "get_coverage"]),
                AgentCapability(name="git", description="Git 操作", tools=["git_diff", "git_commit", "git_push"]),
            ],
        )
        _a2a_server = A2AServer(card)
    return _a2a_server


def get_a2a_client() -> A2AClient:
    global _a2a_client
    if _a2a_client is None:
        _a2a_client = A2AClient(self_id="deepseek-code-agent", self_name="DeepSeek Code Agent")
        _a2a_client.load_config()
    return _a2a_client
