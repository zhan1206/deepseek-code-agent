"""
错误报告与反馈模块
P2 优先级

功能：
- Agent 连续 3 次工具调用失败时，自动生成错误摘要
- 收集：日志 + 上下文摘要 + Agent 思考链 + 用户描述
- 导出到本地文件（不上传第三方）
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FeedbackCollector:
    """错误报告收集器"""

    def __init__(self, feedback_dir: Optional[str] = None):
        self.feedback_dir = feedback_dir or os.path.expanduser("~/.deepseek-agent/feedback")
        self._consecutive_failures: int = 0
        self._failure_log: List[Dict] = []
        self._max_consecutive = 3

    def record_tool_result(self, tool_name: str, ok: bool, error: Optional[str] = None) -> None:
        """记录工具调用结果"""
        if ok:
            self._consecutive_failures = 0
            return

        self._consecutive_failures += 1
        self._failure_log.append({
            "tool": tool_name,
            "error": error,
            "timestamp": time.time(),
            "consecutive_count": self._consecutive_failures,
        })

        if self._consecutive_failures >= self._max_consecutive:
            self._auto_generate_report()

    def _auto_generate_report(self) -> Dict:
        """自动生成错误摘要"""
        report = {
            "type": "auto_error_summary",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "consecutive_failures": self._consecutive_failures,
            "failures": self._failure_log[-self._max_consecutive:],
            "summary": f"Agent 连续 {self._consecutive_failures} 次工具调用失败",
        }
        self._save_report(report)
        self._consecutive_failures = 0
        return report

    def create_report(
        self,
        user_description: str,
        context: Optional[Dict] = None,
        agent_thinking: Optional[str] = None,
        log_snippet: Optional[str] = None,
    ) -> Dict:
        """手动创建反馈报告"""
        report = {
            "type": "user_feedback",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "user_description": user_description,
            "context_summary": context,
            "agent_thinking": agent_thinking,
            "log_snippet": log_snippet,
            "recent_failures": self._failure_log[-5:],
        }
        self._save_report(report)
        return report

    def _save_report(self, report: Dict) -> None:
        """保存报告到本地"""
        try:
            Path(self.feedback_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"feedback_{ts}.json"
            filepath = os.path.join(self.feedback_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def list_reports(self, limit: int = 10) -> List[Dict]:
        """列出历史报告"""
        reports = []
        try:
            report_dir = Path(self.feedback_dir)
            if not report_dir.exists():
                return []
            files = sorted(report_dir.glob("feedback_*.json"), reverse=True)[:limit]
            for f in files:
                try:
                    reports.append(json.loads(f.read_text(encoding="utf-8")))
                except:
                    continue
        except:
            pass
        return reports


# ── 全局实例 ─────────────────────────────────────────────────────────────

_global_feedback: Optional[FeedbackCollector] = None


def get_feedback_collector() -> FeedbackCollector:
    global _global_feedback
    if _global_feedback is None:
        _global_feedback = FeedbackCollector()
    return _global_feedback
