"""
OpenTelemetry 集成 — 可观测性
P2 优先级

功能：
- 每次 Agent 循环生成一个 Trace
- 每个工具调用生成一个 Span
- 本地导出（~/.deepseek-agent/traces/）
- 可选 OTLP 导出
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── 轻量级 Span/Trace (不强制依赖 opentelemetry) ────────────────────────

class Span:
    """单个操作的时间跨度"""

    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        self.name = name
        self.trace_id = trace_id
        self.span_id = f"{id(self):016x}"
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.status = "OK"
        self.events: List[Dict] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def finish(self, status: str = "OK") -> None:
        self.end_time = time.time()
        self.status = status

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class Trace:
    """一个完整的 Agent 循环跟踪"""

    def __init__(self, name: str, trace_id: Optional[str] = None):
        self.name = name
        self.trace_id = trace_id or f"trace_{int(time.time()*1000):016x}"
        self.spans: List[Span] = []
        self.root_span: Optional[Span] = None
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def create_span(self, name: str, parent: Optional[Span] = None) -> Span:
        span = Span(name=name, trace_id=self.trace_id, parent_id=parent.span_id if parent else None)
        self.spans.append(span)
        return span

    def finish(self) -> None:
        self.end_time = time.time()
        if self.root_span:
            self.root_span.finish()

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
        }


# ── Tracer (全局管理) ────────────────────────────────────────────────────

class Tracer:
    """全局追踪器"""

    def __init__(self, export_dir: Optional[str] = None):
        self.export_dir = export_dir or os.path.expanduser("~/.deepseek-agent/traces")
        self._active_trace: Optional[Trace] = None
        self._trace_history: List[Trace] = []
        self._enabled = True
        self._token_counter: Dict[str, int] = {}  # model -> tokens

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def start_trace(self, name: str) -> Trace:
        """开始一个新的 Trace"""
        if not self._enabled:
            trace = Trace(name=name)
            trace.root_span = trace.create_span(name)
            return trace

        trace = Trace(name=name)
        trace.root_span = trace.create_span(name)
        self._active_trace = trace
        return trace

    def end_trace(self, trace: Trace) -> None:
        """结束并导出 Trace"""
        trace.finish()
        if self._enabled:
            self._trace_history.append(trace)
            self._export_trace(trace)

    def record_tool_call(self, tool_name: str, args: Dict, result: Any, duration_ms: float, parent: Optional[Span] = None) -> None:
        """记录工具调用"""
        if not self._enabled or not self._active_trace:
            return

        span = self._active_trace.create_span(f"tool.{tool_name}", parent=parent or self._active_trace.root_span)
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.args", json.dumps(args, default=str)[:500])
        span.set_attribute("tool.duration_ms", duration_ms)
        span.set_attribute("tool.result_type", type(result).__name__)
        if isinstance(result, dict):
            span.set_attribute("tool.ok", result.get("ok", True))
        span.finish()

    def record_token_usage(self, model: str, tokens: int) -> None:
        """记录 token 使用量"""
        self._token_counter[model] = self._token_counter.get(model, 0) + tokens

    def get_stats(self) -> Dict:
        """获取追踪统计"""
        return {
            "total_traces": len(self._trace_history),
            "total_spans": sum(len(t.spans) for t in self._trace_history),
            "token_usage": dict(self._token_counter),
            "avg_trace_duration_ms": (
                sum(t.duration_ms for t in self._trace_history) / len(self._trace_history)
                if self._trace_history else 0
            ),
            "export_dir": self.export_dir,
        }

    def _export_trace(self, trace: Trace) -> None:
        """导出 trace 到本地文件"""
        try:
            Path(self.export_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.fromtimestamp(trace.start_time, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{trace.trace_id[:8]}.json"
            filepath = os.path.join(self.export_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            # 静默失败，不影响主流程
            pass

    def load_traces(self, limit: int = 20) -> List[Dict]:
        """加载历史 trace"""
        traces = []
        try:
            trace_dir = Path(self.export_dir)
            if not trace_dir.exists():
                return []
            files = sorted(trace_dir.glob("*.json"), reverse=True)[:limit]
            for f in files:
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    traces.append(data)
                except:
                    continue
        except:
            pass
        return traces


# ── 全局实例 ─────────────────────────────────────────────────────────────

_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def init_tracer(export_dir: Optional[str] = None, enabled: bool = True) -> Tracer:
    """初始化全局追踪器"""
    global _global_tracer
    _global_tracer = Tracer(export_dir=export_dir)
    if not enabled:
        _global_tracer.disable()
    return _global_tracer
