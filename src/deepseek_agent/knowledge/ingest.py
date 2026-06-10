"""
ingest — 项目全量摄入，生成压缩骨架快照。
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .parser import CodeParser, SymbolTable, Symbol


# ── 重要度排序 ──────────────────────────────────────────────────────────────

def _file_importance(file_path: str, root: Path) -> float:
    """
    为文件分配重要度分数（越高越重要，优先摄入）。

    策略：
    - 入口文件（__main__.py, __init__.py）优先
    - 与根目录越近越优先
    - 排除测试和资源文件
    """
    rel = Path(file_path).relative_to(root)
    parts = rel.parts

    # 排除
    exclude = {"tests", "test_", "_test.", ".git", "__pycache__", ".venv", "venv", "resources", "node_modules"}
    if any(p in exclude or p.startswith(".") for p in parts):
        return -1.0

    depth = len(parts)
    score = 10.0 - depth * 0.5  # 越浅越重要

    # 特殊文件
    if parts[-1] in ("__init__.py", "__main__.py"):
        score += 3.0
    elif parts[-1] == "setup.py":
        score += 2.0
    elif parts[-1] in ("pyproject.toml", "setup.cfg", "config.py", "settings.py"):
        score += 1.5

    return score


def _compress_function(node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str) -> str:
    """
    用 AST 压缩函数体为骨架。
    保留签名、docstring、import/export，函数体替换为 `# ... (N lines)`。
    """
    lines: List[str] = []

    # 装饰器
    for dec in node.decorator_list:
        lines.append(f"@{ast.unparse(dec)}")

    # 签名
    lines.append(ast.unparse(ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=node.returns,
        type_params=node.type_params,
    )))

    # docstring
    doc = ast.get_docstring(node)
    if doc:
        indent = "    "
        lines.append(f'{indent}"""{doc[:200]}"""')

    # 体行数
    body_lines = (node.end_lineno or node.lineno) - node.lineno - 1
    if body_lines > 0:
        lines.append(f"    # ... ({body_lines} lines)")

    return "\n".join(lines)


def _compress_class(node: ast.ClassDef, file_path: str) -> str:
    """压缩类定义，保留签名和 docstring。"""
    lines: List[str] = []

    for dec in node.decorator_list:
        lines.append(f"@{ast.unparse(dec)}")

    # 基类
    if node.bases:
        bases = [ast.unparse(b) for b in node.bases]
        lines.append(f"class {node.name}({', '.join(bases)}):")
    else:
        lines.append(f"class {node.name}:")

    doc = ast.get_docstring(node)
    if doc:
        lines.append(f'    """{doc[:200]}"""')

    # 成员方法（简化）
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(f"    def {item.name}(...): ...")

    return "\n".join(lines)


def _compress_file_content(content: str, file_path: str) -> str:
    """
    将文件内容压缩为骨架形式。
    """
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return content  # 解析失败返回原始内容

    parts: List[str] = []
    imports: List[str] = []
    docstring = ast.get_docstring(tree)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_") or node.name in ("__init__", "__call__", "__enter__", "__exit__"):
                parts.append(_compress_function(node, file_path))
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                parts.append(_compress_class(node, file_path))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    parts.append(f"{t.id} = ...")

    result_parts = []
    if imports:
        result_parts.append("# ── imports ──")
        result_parts.extend(imports[:30])  # 最多 30 个 import
    if docstring:
        result_parts.append(f'"""\n{docstring[:300]}\n"""')
    if parts:
        result_parts.append("# ── definitions ──")
        result_parts.extend(parts)

    return "\n".join(result_parts) if result_parts else content[:500]


# ── 主类 ────────────────────────────────────────────────────────────────────

class ProjectIngester:
    """
    项目全量摄入器。

    用法：
        ingester = ProjectIngester("/path/to/project")
        snapshot = ingester.ingest()
        # 返回压缩骨架快照，可注入为系统消息
    """

    def __init__(self, root_path: str, exclude_dirs: Optional[List[str]] = None):
        self.root = Path(root_path).resolve()
        self.exclude_dirs = exclude_dirs or [
            "__pycache__", ".git", ".venv", "venv", "tests", "resources",
            "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ]

    def ingest(
        self,
        compression_rate: float = 0.5,
        max_files: int = 100,
    ) -> str:
        """
        执行全量摄入。

        Args:
            compression_rate: 压缩率（0.0-1.0），越高压缩越多
            max_files: 最多处理文件数

        Returns:
            压缩后的项目骨架快照字符串
        """
        # 收集文件
        py_files: List[Path] = []
        for py_file in self.root.rglob("*.py"):
            rel = py_file.relative_to(self.root)
            if any(part in self.exclude_dirs or part.startswith(".") for part in rel.parts):
                continue
            py_files.append(py_file)

        # 按重要度排序
        scored = [(f, _file_importance(str(f), self.root)) for f in py_files]
        scored = [(f, s) for f, s in scored if s >= 0]
        scored.sort(key=lambda x: -x[1])

        # 应用压缩率
        take = max(5, int(len(scored) * compression_rate))
        take = min(take, max_files)
        scored = scored[:take]

        # 生成快照
        sections: List[str] = [f"# Project Snapshot: {self.root.name}", f"# Path: {self.root}", f"# Files: {len(scored)}"]

        parser = CodeParser()

        for file_path, score in scored:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            rel = file_path.relative_to(self.root)
            sections.append(f"\n# ═══ {rel} ═══")
            sections.append(_compress_file_content(content, str(file_path)))

        return "\n".join(sections)

    def ingest_structured(self) -> Dict[str, Any]:
        """
        生成结构化的摄入结果（JSON 格式）。
        """
        py_files: List[Path] = []
        for py_file in self.root.rglob("*.py"):
            rel = py_file.relative_to(self.root)
            if any(part in self.exclude_dirs or part.startswith(".") for part in rel.parts):
                continue
            py_files.append(py_file)

        scored = [(f, _file_importance(str(f), self.root)) for f in py_files]
        scored = [(f, s) for f, s in scored if s >= 0]
        scored.sort(key=lambda x: -x[1])

        parser = CodeParser()
        files_data: List[Dict[str, Any]] = []

        for file_path, _ in scored[:50]:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            table = parser.parse_file(str(file_path))
            classes = table.get_classes()
            funcs = table.get_functions()

            files_data.append({
                "path": str(file_path.relative_to(self.root)),
                "classes": [{"name": c.name, "signature": c.signature, "docstring": c.docstring[:100]} for c in classes],
                "functions": [{"name": f.name, "signature": f.signature, "docstring": f.docstring[:100]} for f in funcs],
                "compressed": _compress_file_content(content, str(file_path)),
            })

        return {
            "root": str(self.root),
            "file_count": len(files_data),
            "files": files_data,
        }