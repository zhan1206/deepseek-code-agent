"""
CodeParser — Python AST 解析器，生成符号表。
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ── 数据模型 ────────────────────────────────────────────────────────────────

@dataclass
class Symbol:
    """
    单个代码符号。

    Attributes:
        name: 符号名称
        kind: 类型（function/class/method/variable/import/constant）
        file_path: 文件路径（绝对路径）
        line_start: 起始行号（1-indexed）
        line_end: 结束行号（1-indexed）
        modifiers: 修饰符列表（async, public, export, property, static 等）
        docstring: 文档字符串
        signature: 函数/方法签名（原始字符串）
        parent: 父符号（所属类/模块名）
    """
    name: str
    kind: str  # function | class | method | variable | import | constant
    file_path: str
    line_start: int = 1
    line_end: int = 1
    modifiers: List[str] = field(default_factory=list)
    docstring: str = ""
    signature: str = ""
    parent: str = ""


# ── AST 访问器 ──────────────────────────────────────────────────────────────

class _SymbolVisitor(ast.NodeVisitor):
    """遍历 AST 收集符号信息。"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.symbols: List[Symbol] = []
        self._class_stack: List[str] = []
        self._imports: Set[str] = set()

    def visit_Module(self, node: ast.Module):
        # 模块级 docstring
        self._add_docstring(node, "")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.symbols.append(Symbol(
                name=alias.asname or alias.name,
                kind="import",
                file_path=self.file_path,
                line_start=self._line(node),
                line_end=self._line_end(node),
                modifiers=["public"],
                signature=alias.name,
            ))
            self._imports.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            full_name = alias.asname or alias.name
            self.symbols.append(Symbol(
                name=full_name,
                kind="import",
                file_path=self.file_path,
                line_start=self._line(node),
                line_end=self._line_end(node),
                modifiers=["public"],
                signature=f"from {module} import {alias.name}",
                parent=module,
            ))
            self._imports.add(f"{module}.{alias.name}" if module else alias.name)

    def visit_ClassDef(self, node: ast.ClassDef):
        # 基类列表
        bases = [b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", str(b))
                 for b in node.bases]

        # 装饰器
        modifiers = self._extract_modifiers(node.decorator_list)
        if not any(m in modifiers for m in ("public", "private", "protected")):
            modifiers.append("public")

        docstring = ast.get_docstring(node) or ""
        self.symbols.append(Symbol(
            name=node.name,
            kind="class",
            file_path=self.file_path,
            line_start=self._line(node),
            line_end=self._line_end(node),
            modifiers=modifiers,
            docstring=docstring,
            signature=self._format_class_signature(node),
        ))

        # 递归访问类成员
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._add_function(node, kind="function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._add_function(node, kind="function", extra_modifier="async")

    def visit_Lambda(self, node):
        # lambda 不单独作为符号，但记录位置
        pass

    def visit_Assign(self, node: ast.Assign):
        # 模块/类级赋值作为变量符号
        if not self._class_stack:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.symbols.append(Symbol(
                        name=target.id,
                        kind="variable",
                        file_path=self.file_path,
                        line_start=self._line(node),
                        line_end=self._line_end(node),
                        modifiers=["public"],
                    ))

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # 带类型注解的赋值
        if isinstance(node.target, ast.Name):
            kind = "constant" if node.value is None else "variable"
            self.symbols.append(Symbol(
                name=node.target.id,
                kind=kind,
                file_path=self.file_path,
                line_start=self._line(node),
                line_end=self._line_end(node),
                modifiers=["public"],
            ))

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _add_function(self, node, kind: str, extra_modifier: str = ""):
        parent = self._class_stack[-1] if self._class_stack else ""
        func_kind = "method" if parent else kind

        modifiers = self._extract_modifiers(node.decorator_list)
        if extra_modifier:
            modifiers.append(extra_modifier)

        # 判断 public/private
        if node.name.startswith("_") and not node.name.startswith("__"):
            modifiers.append("private")
        elif not node.name.startswith("_"):
            if "public" not in modifiers and "private" not in modifiers:
                modifiers.append("public")

        # property 装饰器
        if any(d == "property" or (isinstance(d, ast.Name) and d.id == "property")
               for d in node.decorator_list):
            modifiers.append("property")

        docstring = ast.get_docstring(node) or ""
        self.symbols.append(Symbol(
            name=node.name,
            kind=func_kind,
            file_path=self.file_path,
            line_start=self._line(node),
            line_end=self._line_end(node),
            modifiers=modifiers,
            docstring=docstring,
            signature=self._format_function_signature(node),
            parent=parent,
        ))

    @staticmethod
    def _extract_modifiers(decorators: list) -> List[str]:
        mods = []
        for d in decorators:
            if isinstance(d, ast.Name):
                mods.append(d.id)
            elif isinstance(d, ast.Attribute):
                mods.append(d.attr)
        return mods

    @staticmethod
    def _format_function_signature(node) -> str:
        args = node.args
        parts = [a.arg for a in args.args]
        for i, default in enumerate(args.defaults):
            # 倒序对应
            idx = len(parts) - len(args.defaults) + i
            if idx >= 0:
                parts[idx] += "=..."
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")
        return f"def {node.name}({', '.join(parts)})"

    @staticmethod
    def _format_class_signature(node) -> str:
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(f"{b.value.id}.{b.attr}")
            else:
                bases.append(getattr(b, "id", "?"))
        if bases:
            return f"class {node.name}({', '.join(bases)})"
        return f"class {node.name}"


    def _line(self, node) -> int:
        return getattr(node, 'lineno', 1)

    def _line_end(self, node) -> int:
        return getattr(node, 'end_lineno', self._line(node))

    def _add_docstring(self, node, parent: str):
        doc = ast.get_docstring(node)
        if doc:
            # Module/FunctionDef 等顶层节点的 lineno 存在性不同
            line_start = getattr(node, "lineno", 1)
            line_end = getattr(node, "end_lineno", line_start)
            self.symbols.append(Symbol(
                name="__doc__",
                kind="docstring",
                file_path=self.file_path,
                line_start=line_start,
                line_end=line_end,
                docstring=doc,
                parent=parent,
            ))


# ── CodeParser ──────────────────────────────────────────────────────────────

class CodeParser:
    """
    Python 代码 AST 解析器。

    用法：
        parser = CodeParser()
        table = parser.parse_file("/path/to/module.py")
        symbols = table.query("MyClass", kind="class")
        table.get_exported_symbols("/path/to/module.py")
    """

    def __init__(self):
        self._cache: Dict[str, SymbolTable] = {}

    def parse_file(self, file_path: str) -> SymbolTable:
        """
        解析单个 Python 文件，返回符号表。
        """
        path = Path(file_path).resolve()
        key = str(path)

        if key in self._cache:
            return self._cache[key]

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return SymbolTable(file_path=str(path), error=str(e))

        try:
            tree = ast.parse(content, filename=str(path))
        except SyntaxError as e:
            return SymbolTable(file_path=str(path), error=str(e))

        visitor = _SymbolVisitor(str(path))
        visitor.visit(tree)

        table = SymbolTable(
            file_path=str(path),
            symbols=visitor.symbols,
            imports=visitor._imports,
        )
        self._cache[key] = table
        return table

    def parse_dir(self, root_path: str, exclude_dirs: Optional[List[str]] = None) -> SymbolTable:
        """
        递归解析目录下所有 .py 文件，合并符号表。
        """
        root = Path(root_path).resolve()
        exclude_dirs = exclude_dirs or ["__pycache__", ".git", ".venv", "venv", "tests", "resources"]

        all_symbols: List[Symbol] = []
        all_imports: Set[str] = set()
        file_paths: List[str] = []

        for py_file in root.rglob("*.py"):
            rel = py_file.relative_to(root)
            if any(part in exclude_dirs for part in rel.parts):
                continue
            table = self.parse_file(str(py_file))
            if table.error:
                continue
            all_symbols.extend(table.symbols)
            all_imports.update(table.imports)
            file_paths.append(str(py_file))

        merged = SymbolTable(
            file_path=str(root),
            symbols=all_symbols,
            imports=all_imports,
        )
        merged._file_paths = file_paths
        return merged

    def invalidate(self, file_path: str) -> None:
        """清除单个文件的缓存（用于增量更新）。"""
        key = str(Path(file_path).resolve())
        self._cache.pop(key, None)


# ── SymbolTable ─────────────────────────────────────────────────────────────

class SymbolTable:
    """
    符号表 — 查询和管理代码符号。

    Attributes:
        file_path: 所属文件/目录路径
        symbols: 所有符号列表
        imports: 模块级 import 列表
        error: 解析错误信息（如有）
    """

    def __init__(
        self,
        file_path: str,
        symbols: Optional[List[Symbol]] = None,
        imports: Optional[Set[str]] = None,
        error: Optional[str] = None,
    ):
        self.file_path = file_path
        self.symbols = symbols or []
        self.imports = imports or set()
        self.error = error
        self._file_paths: List[str] = []

    # ── 查询 ────────────────────────────────────────────────────────────────

    def query(
        self,
        name: str,
        kind: Optional[str] = None,
        path: Optional[str] = None,
    ) -> List[Symbol]:
        """
        按名称/类型/路径查询符号。

        Args:
            name: 符号名称（支持子串匹配）
            kind: 符号类型过滤（function/class/method/variable/import）
            path: 文件路径过滤
        """
        results = []
        for s in self.symbols:
            if name in s.name:
                if kind and s.kind != kind:
                    continue
                if path and s.file_path != path:
                    continue
                results.append(s)
        return results

    def get_exported_symbols(self, module: str) -> List[Symbol]:
        """
        获取模块的公开导出符号（public 修饰符，不含下划线开头）。
        """
        results = []
        for s in self.symbols:
            if s.file_path != module:
                continue
            if "public" in s.modifiers and not s.name.startswith("_"):
                results.append(s)
            elif s.kind == "class" and not s.name.startswith("_"):
                results.append(s)
        return results

    def get_by_kind(self, kind: str) -> List[Symbol]:
        return [s for s in self.symbols if s.kind == kind]

    def get_functions(self) -> List[Symbol]:
        return self.get_by_kind("function")

    def get_classes(self) -> List[Symbol]:
        return self.get_by_kind("class")

    def get_methods(self) -> List[Symbol]:
        return self.get_by_kind("method")

    def get_file_symbols(self, file_path: str) -> List[Symbol]:
        return [s for s in self.symbols if s.file_path == file_path]

    def summary(self) -> str:
        """生成可读摘要。"""
        if self.error:
            return f"解析错误: {self.error}"

        lines = []
        files = sorted(set(s.file_path for s in self.symbols))
        for f in files:
            syms = self.get_file_symbols(f)
            classes = [s for s in syms if s.kind == "class"]
            funcs = [s for s in syms if s.kind in ("function", "method")]
            imports = [s for s in syms if s.kind == "import"]
            if classes:
                lines.append(f"📦 {f}")
                for c in classes:
                    lines.append(f"  class {c.name}{c.signature.split('(')[1] if '(' in c.signature else ''}")
                    # 方法
                    for m in syms:
                        if m.kind == "method" and m.parent == c.name:
                            lines.append(f"    def {m.name}(...)")
            if funcs and not classes:
                lines.append(f"📄 {f}")
                for fn in funcs:
                    lines.append(f"  {fn.modifiers} {fn.name}{fn.signature}")
        return "\n".join(lines) if lines else "(无符号)"