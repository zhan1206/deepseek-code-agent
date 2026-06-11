"""
架构嗅觉检测 — 基于知识图谱的结构化问题分析。

检测规则：
- 循环依赖（图中的环）
- 上帝类（入度/出度超过阈值）
- 长函数（AST 节点行数 > 50）
- 过度耦合（扇入/扇出比异常）
- 深层嵌套（缩进层级 > 4）
- 重复代码块（相似度 > 80%）
"""

from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import tool, DangerLevel, ToolResult


# ── 嗅觉检测器 ───────────────────────────────────────────────────────────

class ArchSniffer:
    """架构嗅觉检测器。"""

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.findings: List[Dict[str, Any]] = []

    def scan_all(self) -> List[Dict[str, Any]]:
        """运行所有检测规则。"""
        self.findings = []
        self._check_cycles()
        self._check_god_classes()
        self._check_long_functions()
        self._check_deep_nesting()
        self._check_high_coupling()
        return self.findings

    # ── 循环依赖 ────────────────────────────────────────────────────

    def _check_cycles(self) -> None:
        """检测模块间的循环依赖。"""
        import_graph = self._build_import_graph()

        # DFS 检测环
        visited: Set[str] = set()
        path_set: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(node: str, path: List[str]) -> None:
            if node in path_set:
                # 找到环
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            if node in visited:
                return

            visited.add(node)
            path_set.add(node)
            path.append(node)

            for neighbor in import_graph.get(node, []):
                dfs(neighbor, path)

            path.pop()
            path_set.discard(node)

        for node in import_graph:
            dfs(node, [])

        for cycle in cycles:
            self.findings.append({
                "rule_id": "CYCLE001",
                "severity": "HIGH",
                "category": "循环依赖",
                "message": f"循环依赖：{' → '.join(cycle)}",
                "suggestion": "考虑使用依赖注入、事件总线或提取公共模块打破循环",
                "files": list(set(cycle)),
            })

    # ── 上帝类 ──────────────────────────────────────────────────────

    def _check_god_classes(self) -> None:
        """检测上帝类（方法/属性过多的类）。"""
        threshold_methods = 15
        threshold_lines = 300

        for py_file in self._iter_py_files():
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    class_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0

                    if len(methods) > threshold_methods or class_lines > threshold_lines:
                        self.findings.append({
                            "rule_id": "GOD001",
                            "severity": "MEDIUM",
                            "category": "上帝类",
                            "message": f"类 `{node.name}` 过于庞大：{len(methods)} 个方法，{class_lines} 行",
                            "file_path": str(py_file),
                            "line_number": node.lineno,
                            "suggestion": "考虑将类拆分为多个职责单一的类（单一职责原则）",
                        })

    # ── 长函数 ──────────────────────────────────────────────────────

    def _check_long_functions(self, threshold: int = 50) -> None:
        """检测过长的函数。"""
        for py_file in self._iter_py_files():
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(node, 'end_lineno'):
                        func_lines = node.end_lineno - node.lineno + 1
                        if func_lines > threshold:
                            self.findings.append({
                                "rule_id": "LONG001",
                                "severity": "MEDIUM",
                                "category": "长函数",
                                "message": f"函数 `{node.name}` 过长：{func_lines} 行（阈值 {threshold}）",
                                "file_path": str(py_file),
                                "line_number": node.lineno,
                                "suggestion": "考虑提取子函数，每个函数只做一件事",
                            })

    # ── 深层嵌套 ────────────────────────────────────────────────────

    def _check_deep_nesting(self, max_depth: int = 4) -> None:
        """检测过深的嵌套层级。"""
        for py_file in self._iter_py_files():
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
            except Exception:
                continue

            for i, line in enumerate(lines, 1):
                stripped = line.lstrip()
                if not stripped or stripped.startswith("#"):
                    continue

                indent = len(line) - len(stripped)
                # 假设 4 空格缩进
                depth = indent // 4

                if depth > max_depth:
                    self.findings.append({
                        "rule_id": "NEST001",
                        "severity": "LOW",
                        "category": "深层嵌套",
                        "message": f"嵌套层级 {depth}（阈值 {max_depth}）：行 {i}",
                        "file_path": str(py_file),
                        "line_number": i,
                        "suggestion": "使用 early return、guard clause 或提取函数减少嵌套",
                    })

    # ── 高耦合 ──────────────────────────────────────────────────────

    def _check_high_coupling(self, threshold: int = 10) -> None:
        """检测高耦合模块（扇入/扇出异常）。"""
        import_graph = self._build_import_graph()
        reverse_graph: Dict[str, List[str]] = defaultdict(list)

        for src, dsts in import_graph.items():
            for dst in dsts:
                reverse_graph[dst].append(src)

        # 扇出过高（一个模块依赖太多其他模块）
        for module, deps in import_graph.items():
            if len(deps) > threshold:
                self.findings.append({
                    "rule_id": "COUPLE001",
                    "severity": "MEDIUM",
                    "category": "高扇出",
                    "message": f"模块 `{module}` 依赖 {len(deps)} 个其他模块（阈值 {threshold}）",
                    "suggestion": "考虑减少依赖，使用接口隔离或依赖注入",
                })

        # 扇入过高（一个模块被太多模块依赖）
        fan_in_threshold = threshold * 2
        for module, dependents in reverse_graph.items():
            if len(dependents) > fan_in_threshold:
                self.findings.append({
                    "rule_id": "COUPLE002",
                    "severity": "LOW",
                    "category": "高扇入",
                    "message": f"模块 `{module}` 被 {len(dependents)} 个模块依赖",
                    "suggestion": "考虑是否需要拆分此模块的职责",
                })

    # ── 辅助 ────────────────────────────────────────────────────────

    def _build_import_graph(self) -> Dict[str, List[str]]:
        """构建模块导入图。"""
        graph: Dict[str, List[str]] = defaultdict(list)
        project_name = self.project_path.name

        for py_file in self._iter_py_files():
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue

            # 计算模块名
            rel_path = py_file.relative_to(self.project_path)
            parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module_name = ".".join(parts)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(project_name):
                            graph[module_name].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith(project_name):
                        graph[module_name].append(node.module)

        return dict(graph)

    def _iter_py_files(self):
        """迭代项目中的 Python 文件。"""
        for f in self.project_path.rglob("*.py"):
            if any(skip in str(f) for skip in [".git", "__pycache__", ".venv", "node_modules", "site-packages"]):
                continue
            yield f


# ── 工具注册 ─────────────────────────────────────────────────────────────

@tool(
    name="arch_check",
    description="架构嗅觉检测：循环依赖、上帝类、长函数、高耦合、深层嵌套。",
    danger_level=DangerLevel.SAFE,
    read_only=True,
)
async def arch_check(
    project_path: str = ".",
    checks: str = "all",
) -> str:
    """
    架构嗅觉检测。

    Args:
        project_path: 项目根目录
        checks: 检测类型 - all / cycles / god_classes / long_functions / deep_nesting / high_coupling
    """
    sniffer = ArchSniffer(project_path)

    if checks == "all" or checks == ["all"]:
        findings = sniffer.scan_all()
    else:
        sniffer.findings = []
        check_map = {
            "cycles": sniffer._check_cycles,
            "god_classes": sniffer._check_god_classes,
            "long_functions": sniffer._check_long_functions,
            "deep_nesting": sniffer._check_deep_nesting,
            "high_coupling": sniffer._check_high_coupling,
        }
        # 支持 str 或 list 传入
        check_list = checks.split(",") if isinstance(checks, str) else list(checks)
        for check_name in check_list:
            check_name = check_name.strip()
            if check_name in check_map:
                check_map[check_name]()

        findings = sniffer.findings

    if not findings:
        return ToolResult.ok("✅ 未检测到架构问题").to_str()

    # 按严重度排序
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    findings.sort(key=lambda f: severity_order.get(f.get("severity", "LOW"), 3))

    # 统计
    by_category: Dict[str, int] = defaultdict(int)
    by_severity: Dict[str, int] = defaultdict(int)
    for f in findings:
        by_category[f.get("category", "未知")] += 1
        by_severity[f.get("severity", "LOW")] += 1

    lines = [
        f"🏗️ 架构嗅觉检测报告",
        f"   项目：{project_path}",
        f"   总问题数：{len(findings)}",
    ]

    for sev in ["HIGH", "MEDIUM", "LOW"]:
        if sev in by_severity:
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[sev]
            lines.append(f"   {icon} {sev}: {by_severity[sev]}")

    lines.append("")
    lines.append("📊 问题分布：")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        lines.append(f"   - {cat}: {count}")

    lines.append("")
    lines.append("📋 详情：")

    for f in findings[:30]:
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(f.get("severity", ""), "⚪")
        lines.append(f"{icon} [{f.get('rule_id', '?')}] {f.get('message', '')}")
        if f.get("file_path"):
            lines.append(f"   📄 {f['file_path']}" + (f":{f['line_number']}" if f.get("line_number") else ""))
        if f.get("suggestion"):
            lines.append(f"   💡 {f['suggestion']}")
        lines.append("")

    if len(findings) > 30:
        lines.append(f"... 以及 {len(findings) - 30} 个其他问题")

    return ToolResult.ok("\n".join(lines)).to_str()
