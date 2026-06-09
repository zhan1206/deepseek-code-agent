"""LSP 工具集 — 5 个代码理解工具，基于正则的轻量实现（降级模式）。"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import tool, ToolResult

# ── 辅助 ──────────────────────────────────────────────────────────────

def _extract_symbols(content: str) -> List[Dict[str, Any]]:
    """从文件内容中提取函数/类/变量定义（正则降级）。"""
    results = []
    lines = content.split("\n")
    patterns = [
        (r"^\s*def\s+(\w+)", "function"),
        (r"^\s*class\s+(\w+)", "class"),
        (r"^\s*(\w+)\s*=\s*(?:async\s+)?def\s+", "function"),
        (r"^\s*(\w+)\s*=\s*(?:class|literal|const)\s+", "constant"),
    ]
    for lineno, line in enumerate(lines, 1):
        for pat, kind in patterns:
            m = re.match(pat, line.strip())
            if m:
                name = m.group(1)
                if not name.startswith("_") or kind == "class":
                    results.append({"name": name, "kind": kind, "lineno": lineno, "line": line.strip()[:100]})
                break
    return results

def _find_references(content: str, symbol: str) -> List[str]:
    """查找符号的所有引用（简单文本搜索）。"""
    lines = content.split("\n")
    hits = []
    for lineno, line in enumerate(lines, 1):
        if symbol in line and not re.match(rf"^\s*(def|class|async\s+def)\s+{re.escape(symbol)}\b", line.strip()):
            hits.append(f"  行 {lineno}: {line.strip()[:120]}")
    return hits

# ── 工具 ──────────────────────────────────────────────────────────────

@tool(name="get_symbols", description="获取文件中定义的符号（函数、类、变量）。", danger_level=0)
async def get_symbols(path: str) -> str:
    """返回文件中所有顶层符号定义。"""
    try:
        content = open(path, encoding="utf-8", errors="ignore").read()
        symbols = _extract_symbols(content)
        if not symbols:
            return ToolResult.ok(f"文件 {path} 中未找到符号定义").to_str()
        lines = [f"  [{s['kind']:8}] {s['name']} (行 {s['lineno']})" for s in symbols]
        return ToolResult.ok(f"文件 {path} 的符号定义 ({len(symbols)} 个):\n" + "\n".join(lines)).to_str()
    except Exception as e:
        return ToolResult.fail(f"获取符号失败: {e}").to_str()

@tool(name="find_references", description="查找符号的所有引用位置。", danger_level=0)
async def find_references(symbol: str, path: str) -> str:
    """返回符号在文件中的所有引用。"""
    try:
        content = open(path, encoding="utf-8", errors="ignore").read()
        refs = _find_references(content, symbol)
        if not refs:
            return ToolResult.ok(f"未在 {path} 中找到 '{symbol}' 的引用").to_str()
        return ToolResult.ok(f"'{symbol}' 在 {path} 中的引用 ({len(refs)} 处):\n" + "\n".join(refs)).to_str()
    except Exception as e:
        return ToolResult.fail(f"查找引用失败: {e}").to_str()

@tool(name="go_to_definition", description="跳转到符号定义处。", danger_level=0)
async def go_to_definition(symbol: str, path: str) -> str:
    """返回符号定义的行号和内容。"""
    try:
        content = open(path, encoding="utf-8", errors="ignore").read()
        lines = content.split("\n")
        for lineno, line in enumerate(lines, 1):
            if re.match(rf"^\s*(def|class|async\s+def)\s+{re.escape(symbol)}\b", line.strip()):
                return ToolResult.ok(f"'{symbol}' 定义于 {path}:{lineno}\n{line}").to_str()
        return ToolResult.fail(f"未在 {path} 中找到 '{symbol}' 的定义").to_str()
    except Exception as e:
        return ToolResult.fail(f"跳转定义失败: {e}").to_str()

@tool(name="get_hover_info", description="获取悬停信息（类型、文档字符串）。", danger_level=0)
async def get_hover_info(path: str, line: int, col: int = 0) -> str:
    """返回指定行附近的类型信息和文档。"""
    try:
        content = open(path, encoding="utf-8", errors="ignore").read()
        lines = content.split("\n")
        if not (0 <= line - 1 < len(lines)):
            return ToolResult.fail(f"行号 {line} 超出范围").to_str()
        target = lines[line - 1]
        # 尝试找同名的 def/class
        for search_line in [line - 1, line, line - 2]:
            if 0 <= search_line < len(lines):
                l = lines[search_line]
                if re.match(r"\s*(def|class)\s+", l):
                    doc_lines = []
                    for dl in range(search_line + 1, min(search_line + 5, len(lines))):
                        if lines[dl].strip().startswith('"""') or lines[dl].strip().startswith("'''"):
                            break
                        doc_lines.append(lines[dl].strip())
                    doc = "\n".join(doc_lines) if doc_lines else "(无文档)"
                    return ToolResult.ok(f"类型: {l.strip()[:80]}\n文档: {doc}").to_str()
        return ToolResult.ok(f"行 {line}: {target[:120]}").to_str()
    except Exception as e:
        return ToolResult.fail(f"获取悬停信息失败: {e}").to_str()

@tool(name="get_diagnostics", description="获取当前文件的 lint 错误和警告。", danger_level=0)
async def get_diagnostics(path: str) -> str:
    """运行 ruff/flake8/pylint 获取诊断信息（降级为语法检查）。"""
    import subprocess, tempfile
    try:
        # 尝试 ruff
        result = subprocess.run(
            ["python", "-m", "py_compile", path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return ToolResult.ok(f"✓ {path}: 语法检查通过（无错误）").to_str()
        return ToolResult.fail(f"✗ {path}: 语法错误\n{result.stderr}").to_str()
    except FileNotFoundError:
        # python 不在 PATH，尝试直接 py_compile
        try:
            import py_compile
            py_compile.compile(path, doraise=True)
            return ToolResult.ok(f"✓ {path}: 语法检查通过").to_str()
        except py_compile.PyCompileError as e:
            return ToolResult.fail(f"✗ {path}: 语法错误\n{e}").to_str()
        except Exception as e:
            return ToolResult.fail(f"无法检查语法: {e}").to_str()
    except Exception as e:
        return ToolResult.fail(f"诊断失败: {e}").to_str()
