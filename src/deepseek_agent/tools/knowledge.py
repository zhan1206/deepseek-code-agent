"""
知识图谱工具集 — 提供符号搜索、调用者分析、导入关系、影响分析。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import tool, ToolResult, DangerLevel
from ..knowledge.parser import CodeParser, SymbolTable, Symbol
from ..knowledge.graph import RelationGraph

# ── 全局解析器实例（进程内共享缓存）───────────────────────────────────────

_parser: Optional[CodeParser] = None
_graph: Optional[RelationGraph] = None


def _get_parser() -> CodeParser:
    global _parser
    if _parser is None:
        _parser = CodeParser()
    return _parser


def _get_graph(root: str) -> RelationGraph:
    global _graph
    if _graph is None:
        try:
            _graph = RelationGraph()
            _graph.build_from_dir(root)
        except ImportError:
            # networkx 未安装时返回空图
            class _EmptyGraph:
                def find_callers(self, s): return []
                def find_dependents(self, s): return []
                def analyze_impact(self, s, depth=3): return []
            _graph = _EmptyGraph()  # type: ignore
    return _graph


# ── 工具 ────────────────────────────────────────────────────────────────────

@tool(
    name="find_symbol",
    description=(
        "在项目代码库中搜索符号（函数、类、方法、变量）。"
        "支持按名称模糊匹配、按类型和文件路径过滤。"
    ),
    danger_level=DangerLevel.SAFE,
)
def find_symbol(
    name: str,
    kind: Optional[str] = None,
    path: Optional[str] = None,
) -> ToolResult:
    """
    在项目中搜索代码符号。

    Args:
        name: 符号名称（支持子串匹配）
        kind: 符号类型过滤 (function|class|method|variable|import)
        path: 限定文件路径
    """
    parser = _get_parser()
    try:
        table = parser.parse_dir(os.getcwd())
    except Exception:
        return ToolResult.fail(f"无法解析当前目录: {os.getcwd()}")

    results = table.query(name, kind=kind, path=path)
    if not results:
        return ToolResult.ok({
            "query": name, "kind": kind, "path": path,
            "total": 0, "results": [],
        })

    output = [
        {
            "name": s.name,
            "kind": s.kind,
            "file": s.file_path,
            "lines": f"{s.line_start}-{s.line_end}",
            "modifiers": s.modifiers,
            "signature": s.signature[:100],
            "docstring": s.docstring[:100] if s.docstring else "",
        }
        for s in results[:20]
    ]
    return ToolResult.ok({
        "query": name, "kind": kind,
        "total": len(results),
        "results": output,
    })


@tool(
    name="get_callers",
    description="查找调用了指定符号的所有文件和函数。",
    danger_level=DangerLevel.SAFE,
)
def get_callers(symbol: str) -> ToolResult:
    """
    获取调用了指定符号的所有调用者。

    Args:
        symbol: 符号名称
    """
    try:
        graph = _get_graph(os.getcwd())
        callers = graph.find_callers(symbol)
        return ToolResult.ok({
            "symbol": symbol,
            "callers": callers[:50],
            "count": len(callers),
        })
    except Exception as e:
        return ToolResult.fail(str(e))


@tool(
    name="get_imports",
    description="查看模块的导入关系（被导入的模块列表）。",
    danger_level=DangerLevel.SAFE,
)
def get_imports(module_path: str) -> ToolResult:
    """
    获取模块的导入关系。

    Args:
        module_path: Python 模块文件路径（相对或绝对）
    """
    parser = _get_parser()
    try:
        abs_path = str(Path(module_path).resolve())
        table = parser.parse_file(abs_path)
        imports = sorted(table.imports)
        return ToolResult.ok({
            "module": module_path,
            "imports": imports,
            "count": len(imports),
        })
    except Exception as e:
        return ToolResult.fail(f"解析失败: {e}")


@tool(
    name="analyze_impact",
    description="分析修改某符号会影响哪些其他代码（递归深度可配置）。",
    danger_level=DangerLevel.SAFE,
)
def analyze_impact(
    target_symbol: str,
    change_type: Optional[str] = None,
    depth: int = 3,
) -> ToolResult:
    """
    影响分析工具。

    Args:
        target_symbol: 目标符号名称
        change_type: 变更类型描述（可选，用于 LLM 推理）
        depth: 递归分析深度（默认 3）
    """
    try:
        graph = _get_graph(os.getcwd())
        impacted = graph.analyze_impact(target_symbol, depth=depth)

        output = []
        for s in impacted[:30]:
            output.append({
                "name": s.name,
                "kind": s.kind,
                "file": s.file_path,
                "lines": f"{s.line_start}-{s.line_end}",
                "docstring": s.docstring[:80] if s.docstring else "",
            })

        return ToolResult.ok({
            "target": target_symbol,
            "change_type": change_type,
            "depth": depth,
            "impacted": output,
            "total": len(impacted),
        })
    except Exception as e:
        return ToolResult.fail(f"影响分析失败: {e}")


@tool(
    name="ingest_project",
    description="全量摄入项目代码，生成压缩骨架快照（可注入为系统消息上下文）。",
    danger_level=DangerLevel.SAFE,
)
def ingest_project(
    path: Optional[str] = None,
    exclude: Optional[List[str]] = None,
    compression_rate: float = 0.6,
) -> ToolResult:
    """
    项目全量摄入工具。

    Args:
        path: 项目根路径（默认当前目录）
        exclude: 额外排除的目录名列表
        compression_rate: 压缩率 0.0-1.0，越高压缩越多
    """
    try:
        from ..knowledge.ingest import ProjectIngester
        root = path or os.getcwd()
        exclude = exclude or []
        ingester = ProjectIngester(root, exclude_dirs=exclude)
        snapshot = ingester.ingest(compression_rate=compression_rate)
        return ToolResult.ok({
            "root": root,
            "compression_rate": compression_rate,
            "snapshot": snapshot[:8000],
        })
    except Exception as e:
        return ToolResult.fail(f"摄入失败: {e}")
