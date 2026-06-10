"""
安全扫描工具 — 写入前自动检查代码安全漏洞。

集成：
- bandit（Python 安全扫描）
- 内置 JS/TS 规则（eval、innerHTML、SQL 拼接等）
- npm audit / pip audit（依赖检查）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import tool, ToolResult, DangerLevel

from dataclasses import dataclass, field


# ── 内置 JS/TS 安全规则 ──────────────────────────────────────────────────

JS_SECURITY_RULES: List[Dict[str, Any]] = [
    {
        "id": "JS001",
        "pattern": r"eval\s*\(",
        "severity": "HIGH",
        "message": "使用 eval() 可能导致代码注入",
        "suggestion": "使用 JSON.parse() 或 Function 构造器替代",
    },
    {
        "id": "JS002",
        "pattern": r"innerHTML\s*=",
        "severity": "HIGH",
        "message": "使用 innerHTML 可能导致 XSS",
        "suggestion": "使用 textContent 或 DOMPurify.sanitize()",
    },
    {
        "id": "JS003",
        "pattern": r"document\.write\s*\(",
        "severity": "MEDIUM",
        "message": "使用 document.write 可能导致 XSS",
        "suggestion": "使用 DOM API 操作",
    },
    {
        "id": "JS004",
        "pattern": r"\.exec\s*\(\s*[\"'].*(?:SELECT|INSERT|UPDATE|DELETE).*[\"']\s*\)",
        "severity": "HIGH",
        "message": "SQL 拼接可能导致 SQL 注入",
        "suggestion": "使用参数化查询",
    },
    {
        "id": "JS005",
        "pattern": r"new\s+Function\s*\(",
        "severity": "MEDIUM",
        "message": "动态函数构造可能导致代码注入",
        "suggestion": "避免动态构造函数",
    },
    {
        "id": "JS006",
        "pattern": r"localStorage\.setItem|sessionStorage\.setItem",
        "severity": "LOW",
        "message": "本地存储可能泄露敏感数据",
        "suggestion": "不要存储密码、token 等敏感信息",
    },
]

# Python 内置规则（bandit 不覆盖的简单检查）
PY_SECURITY_RULES: List[Dict[str, Any]] = [
    {
        "id": "PY001",
        "pattern": r"subprocess\.call\s*\([^)]*shell\s*=\s*True",
        "severity": "MEDIUM",
        "message": "shell=True 可能导致命令注入",
        "suggestion": "使用列表形式传参，避免 shell=True",
    },
    {
        "id": "PY002",
        "pattern": r"pickle\.loads?\s*\(",
        "severity": "HIGH",
        "message": "pickle 反序列化不安全数据可能导致任意代码执行",
        "suggestion": "使用 json 或 yaml.safe_load 替代",
    },
    {
        "id": "PY003",
        "pattern": r"yaml\.load\s*\([^)]*,\s*Loader\s*=\s*None",
        "severity": "HIGH",
        "message": "yaml.load 不指定 Loader 不安全",
        "suggestion": "使用 yaml.safe_load 或 yaml.load(data, Loader=yaml.SafeLoader)",
    },
]


# ── 扫描引擎 ─────────────────────────────────────────────────────────────

@dataclass
class SecurityFinding:
    """单个安全发现。"""
    rule_id: str
    severity: str  # HIGH / MEDIUM / LOW
    message: str
    file_path: str
    line_number: int = 0
    line_content: str = ""
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content[:100],
            "suggestion": self.suggestion,
        }


class SecurityScanner:
    """
    安全扫描器，集成多引擎。

    使用方式：
    1. scan_code(content, file_path) — 扫描代码内容
    2. scan_file(file_path) — 扫描文件
    3. scan_dependencies(project_dir) — 扫描依赖
    """

    def __init__(self, min_severity: str = "MEDIUM"):
        self.min_severity = min_severity
        self._severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        self._min_level = self._severity_order.get(min_severity, 2)

    def scan_code(self, content: str, file_path: str = "<memory>") -> List[SecurityFinding]:
        """扫描代码内容，返回发现列表。"""
        findings: List[SecurityFinding] = []

        # 选择规则集
        ext = Path(file_path).suffix.lower()
        if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs"):
            rules = JS_SECURITY_RULES
        elif ext in (".py", ".pyw"):
            rules = PY_SECURITY_RULES
        else:
            rules = JS_SECURITY_RULES + PY_SECURITY_RULES

        lines = content.splitlines()

        for rule in rules:
            pattern = rule["pattern"]
            severity = rule["severity"]

            # 过滤低严重度
            if self._severity_order.get(severity, 0) < self._min_level:
                continue

            for lineno, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(SecurityFinding(
                        rule_id=rule["id"],
                        severity=severity,
                        message=rule["message"],
                        file_path=file_path,
                        line_number=lineno,
                        line_content=line.strip(),
                        suggestion=rule["suggestion"],
                    ))

        return findings

    def scan_file(self, file_path: str) -> List[SecurityFinding]:
        """扫描文件。"""
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return [SecurityFinding(
                rule_id="SCAN_ERROR",
                severity="LOW",
                message=f"无法读取文件: {e}",
                file_path=file_path,
            )]

        findings = self.scan_code(content, file_path)

        # Python 文件额外运行 bandit
        if file_path.endswith(".py"):
            findings.extend(self._run_bandit(file_path))

        return findings

    def scan_content_before_write(
        self,
        content: str,
        file_path: str,
        block_on_high: bool = True,
    ) -> Tuple[bool, List[SecurityFinding]]:
        """
        写入前安全检查钩子。

        Returns:
            (is_safe, findings)
        """
        findings = self.scan_code(content, file_path)

        # Python 内容额外用 bandit 扫描（临时文件）
        if file_path.endswith(".py"):
            findings.extend(self._run_bandit_on_content(content))

        high_findings = [f for f in findings if f.severity == "HIGH"]

        if block_on_high and high_findings:
            return False, findings

        return True, findings

    def scan_dependencies(self, project_dir: str) -> List[SecurityFinding]:
        """扫描项目依赖安全性。"""
        findings: List[SecurityFinding] = []

        project = Path(project_dir)

        # pip audit（Python）
        if (project / "requirements.txt").exists() or (project / "pyproject.toml").exists():
            findings.extend(self._run_pip_audit(project_dir))

        # npm audit（Node.js）
        if (project / "package.json").exists():
            findings.extend(self._run_npm_audit(project_dir))

        return findings

    def _run_bandit(self, file_path: str) -> List[SecurityFinding]:
        """运行 bandit 扫描 Python 文件。"""
        try:
            result = subprocess.run(
                ["bandit", "-f", "json", "-q", file_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode in (0, 1):
                data = json.loads(result.stdout or "{}")
                findings = []
                for issue in data.get("results", []):
                    findings.append(SecurityFinding(
                        rule_id=issue.get("test_id", "BANDIT"),
                        severity=issue.get("issue_confidence", "MEDIUM").upper(),
                        message=issue.get("issue_text", ""),
                        file_path=file_path,
                        line_number=issue.get("line_number", 0),
                        line_content=issue.get("line_range", [""])[0] if issue.get("line_range") else "",
                        suggestion=issue.get("more_info", ""),
                    ))
                return findings
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        return []

    def _run_bandit_on_content(self, content: str) -> List[SecurityFinding]:
        """对内存中的 Python 代码运行 bandit。"""
        try:
            fd, tmp = tempfile.mkstemp(suffix=".py")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                return self._run_bandit(tmp)
            finally:
                os.unlink(tmp)
        except Exception:
            return []

    def _run_pip_audit(self, project_dir: str) -> List[SecurityFinding]:
        """运行 pip-audit。"""
        try:
            result = subprocess.run(
                ["pip-audit", "--format", "json", "-q"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.stdout:
                data = json.loads(result.stdout)
                findings = []
                for dep in data.get("dependencies", []):
                    for vuln in dep.get("vulns", []):
                        findings.append(SecurityFinding(
                            rule_id="PIP-AUDIT",
                            severity=vuln.get("severity", "MEDIUM").upper(),
                            message=f"{dep['name']}@{dep.get('version', '?')}: {vuln.get('description', '')}",
                            file_path=project_dir,
                            suggestion=f"升级 {dep['name']} 到修复版本",
                        ))
                return findings
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        return []

    def _run_npm_audit(self, project_dir: str) -> List[SecurityFinding]:
        """运行 npm audit。"""
        try:
            result = subprocess.run(
                ["npm", "audit", "--json"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            data = json.loads(result.stdout or "{}")
            findings = []
            for name, meta in data.get("vulnerabilities", {}).items():
                findings.append(SecurityFinding(
                    rule_id="NPM-AUDIT",
                    severity=meta.get("severity", "MEDIUM").upper(),
                    message=f"{name}: {meta.get('title', meta.get('via', ''))}",
                    file_path=project_dir,
                    suggestion=f"运行 npm audit fix 修复 {name}",
                ))
            return findings
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        return []


# ── 工具注册 ─────────────────────────────────────────────────────────────

@tool(
    name="security_scan",
    description="扫描代码或文件的安全漏洞（Python bandit + JS 内置规则 + 依赖审计）。",
    danger_level=DangerLevel.SAFE,
    read_only=True,
)
async def security_scan(
    path: str = ".",
    scan_type: str = "code",
    min_severity: str = "MEDIUM",
) -> str:
    """
    安全扫描。

    Args:
        path: 文件或目录路径
        scan_type: 扫描类型 - code（代码扫描）/ deps（依赖审计）/ all
        min_severity: 最低报告严重度 - HIGH / MEDIUM / LOW
    """
    scanner = SecurityScanner(min_severity=min_severity)
    all_findings: List[SecurityFinding] = []

    p = Path(path)

    if scan_type in ("code", "all"):
        if p.is_file():
            all_findings.extend(scanner.scan_file(str(p)))
        elif p.is_dir():
            for ext in ("*.py", "*.js", "*.jsx", "*.ts", "*.tsx"):
                for f in p.rglob(ext):
                    # 跳过 node_modules 和虚拟环境
                    if any(skip in str(f) for skip in ["node_modules", ".venv", "__pycache__", ".git"]):
                        continue
                    all_findings.extend(scanner.scan_file(str(f)))

    if scan_type in ("deps", "all") and p.is_dir():
        all_findings.extend(scanner.scan_dependencies(str(p)))

    if not all_findings:
        return ToolResult.ok("✅ 未发现安全漏洞").to_str()

    # 按严重度排序
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_findings.sort(key=lambda f: severity_order.get(f.severity, 3))

    # 格式化输出
    high = sum(1 for f in all_findings if f.severity == "HIGH")
    medium = sum(1 for f in all_findings if f.severity == "MEDIUM")
    low = sum(1 for f in all_findings if f.severity == "LOW")

    lines = [
        f"🔒 安全扫描结果：{len(all_findings)} 个发现",
        f"   🔴 HIGH: {high}  🟡 MEDIUM: {medium}  🟢 LOW: {low}",
        "",
    ]

    for f in all_findings[:30]:  # 最多显示 30 条
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(f.severity, "⚪")
        lines.append(f"{icon} [{f.rule_id}] {f.message}")
        lines.append(f"   📄 {f.file_path}:{f.line_number}")
        if f.suggestion:
            lines.append(f"   💡 {f.suggestion}")
        lines.append("")

    if len(all_findings) > 30:
        lines.append(f"... 以及 {len(all_findings) - 30} 个其他发现")

    return ToolResult.ok("\n".join(lines)).to_str()
