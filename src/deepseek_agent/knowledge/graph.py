"""
RelationGraph — 基于 NetworkX 的代码关系图谱。
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from .parser import CodeParser, SymbolTable, Symbol


# ── 依赖解析 ────────────────────────────────────────────────────────────────

def _resolve_import(module_path: str, import_name: str) -> Optional[str]:
    """
    将 import 名称解析为实际文件路径。

    简单实现：基于相对路径解析。
    """
    base_dir = os.path.dirname(module_path)
    parts = import_name.split(".")

    # 尝试相对于当前文件目录
    for i in range(len(parts), 0, -1):
        rel = os.path.join(base_dir, *parts[:i]) + ".py"
        if os.path.exists(rel):
            return rel
        # __init__.py
        init = os.path.join(base_dir, *parts[:i], "__init__.py")
        if os.path.exists(init):
            return init

    # 尝试 sys.path（包安装路径）
    import sys
    for sp in sys.path:
        for i in range(len(parts), 0, -1):
            candidate = os.path.join(sp, *parts[:i], "__init__.py")
            if os.path.exists(candidate):
                return candidate
            candidate2 = os.path.join(sp, *parts[:i]) + ".py"
            if os.path.exists(candidate2):
                return candidate2

    return None


class _CallGraphVisitor(ast.NodeVisitor):
    """遍历 AST 提取函数调用和类继承信息。"""

    def __init__(self, file_path: str, known_symbols: Dict[str, Symbol]):
        self.file_path = file_path
        self.known_symbols = known_symbols
        self.calls: Set[str] = set()        # 函数调用
        self.inherits: Set[str] = set()     # 继承的类
        self.decorator_calls: Dict[str, Set[str]] = defaultdict(set)

    def visit_Call(self, node: ast.Call):
        # func = 函数名或方法调用
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # a.b() → 记录 b
            self.calls.add(node.func.attr)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.inherits.add(base.id)
            elif isinstance(base, ast.Attribute):
                self.inherits.add(base.attr)
        self.generic_visit(node)


# ── RelationGraph ───────────────────────────────────────────────────────────

class RelationGraph:
    """
    代码关系图谱。

    支持：
    - 模块依赖图（import → 被导入模块）
    - 函数调用图（caller → callee）
    - 类继承链（子类 → 基类）
    - 影响分析（修改某符号会影响哪些其他符号）

    需要 networkx>=3.0。
    """

    def __init__(self, parser: Optional[CodeParser] = None):
        if not HAS_NETWORKX:
            raise ImportError(
                "RelationGraph 需要 networkx，请安装：pip install networkx>=3.0"
            )

        self.parser = parser or CodeParser()
        self._call_graph = nx.DiGraph()  # caller → callee
        self._dep_graph = nx.DiGraph()   # module → dependency
        self._inheritance: Dict[str, List[str]] = defaultdict(list)  # class → bases
        self._reverse_inherit: Dict[str, List[str]] = defaultdict(list)  # base → derived
        self._symbol_to_files: Dict[str, List[str]] = defaultdict(list)

        # 增量更新追踪
        self._file_mtimes: Dict[str, float] = {}

    # ── 初始化 ──────────────────────────────────────────────────────────────

    def build_from_dir(self, root_path: str, exclude_dirs: Optional[List[str]] = None) -> None:
        """
        从目录构建完整图谱。
        """
        exclude_dirs = exclude_dirs or ["__pycache__", ".git", ".venv", "tests", "resources"]
        table = self.parser.parse_dir(root_path, exclude_dirs)

        file_paths = getattr(table, "_file_paths", [])
        if not file_paths:
            file_paths = sorted(set(s.file_path for s in table.symbols))

        # 符号索引
        all_symbols: Dict[str, Symbol] = {}
        for s in table.symbols:
            key = s.name
            all_symbols[key] = s
            self._symbol_to_files[key].append(s.file_path)

        for fp in file_paths:
            self._analyze_file(fp, all_symbols)
            self._analyze_imports(fp, table)

        self._build_inheritance_reverse()

    def _analyze_file(self, file_path: str, all_symbols: Dict[str, Symbol]) -> None:
        """分析单个文件的调用关系。"""
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content, filename=file_path)
        except Exception:
            return

        visitor = _CallGraphVisitor(file_path, all_symbols)
        visitor.visit(tree)

        # 加入调用图（当前文件 → 调用的函数）
        file_key = file_path
        for callee in visitor.calls:
            self._call_graph.add_edge(file_key, callee, label="calls")

        # 继承关系
        for base in visitor.inherits:
            self._call_graph.add_edge(file_key, base, label="inherits")

        self._file_mtimes[file_path] = os.path.getmtime(file_path)

    def _analyze_imports(self, file_path: str, table: SymbolTable) -> None:
        """分析模块依赖关系。"""
        for s in table.symbols:
            if s.file_path != file_path or s.kind != "import":
                continue
            target = _resolve_import(file_path, s.signature)
            if target:
                self._dep_graph.add_edge(file_path, target, label="imports")

    def _build_inheritance_reverse(self) -> None:
        """构建反向继承链（基类 → 派生类）。"""
        for file_path, data in self._call_graph.nodes(data=True):
            # 查找类继承
            for s in self.parser.parse_file(file_path).symbols:
                if s.kind == "class":
                    bases = []
                    if "(" in s.signature:
                        base_str = s.signature.split("(", 1)[1].rstrip(")")
                        for base in base_str.split(","):
                            base = base.strip()
                            if base:
                                bases.append(base)
                    self._inheritance[s.name] = bases
                    for b in bases:
                        self._reverse_inherit[b].append(s.name)

    # ── 增量更新 ────────────────────────────────────────────────────────────

    def update_file(self, file_path: str, new_content: Optional[str] = None) -> None:
        """
        增量更新：只重解析受影响文件。
        """
        # 清除该文件的旧数据
        if file_path in self._file_mtimes:
            del self._file_mtimes[file_path]
        if file_path in self._call_graph:
            self._call_graph.remove_node(file_path)
        if file_path in self._dep_graph:
            self._dep_graph.remove_node(file_path)

        # 重新解析
        if new_content is not None:
            # 更新缓存
            self.parser.invalidate(file_path)

        # 重新分析
        all_symbols = {}
        for names in self._symbol_to_files.values():
            for fp in names:
                for s in self.parser.parse_file(fp).symbols:
                    all_symbols[s.name] = s

        self._analyze_file(file_path, all_symbols)

        # 标记依赖此文件的文件也需要重分析
        dependents = self.find_dependents(file_path)
        for dep in dependents:
            self._analyze_file(dep, all_symbols)

    # ── 查询 API ───────────────────────────────────────────────────────────

    def find_callers(self, symbol: str) -> List[str]:
        """
        查找调用了指定符号的所有文件/函数。
        """
        callers = []
        for src, dst, data in self._call_graph.in_edges(data=True):
            if dst == symbol or symbol in dst:
                callers.append(src)
        return callers

    def find_dependents(self, module: str) -> List[str]:
        """
        查找依赖指定模块的所有文件。
        """
        try:
            return list(nx.descendants(self._dep_graph, module))
        except nx.NetworkXError:
            return []

    def find_class_hierarchy(self, base_class: str) -> List[str]:
        """
        查找类的完整继承链（基类 → 派生类）。
        """
        result = [base_class]
        queue = list(self._reverse_inherit.get(base_class, []))
        seen = set(result)

        while queue:
            cls = queue.pop(0)
            if cls in seen:
                continue
            result.append(cls)
            seen.add(cls)
            queue.extend(self._reverse_inherit.get(cls, []))

        return result

    def analyze_impact(
        self,
        target_symbol: str,
        depth: int = 3,
    ) -> List[Symbol]:
        """
        影响分析：修改某符号会影响哪些符号。

        Args:
            target_symbol: 目标符号名
            depth: 递归深度

        Returns:
            受影响的符号列表（按文件分组）
        """
        impacted: List[Symbol] = []
        visited: Set[str] = set()

        def _recurse(symbol: str, d: int):
            if d <= 0 or symbol in visited:
                return
            visited.add(symbol)

            # 调用该符号的地方
            callers = self.find_callers(symbol)
            for caller in callers:
                for s in self.parser.parse_file(caller).symbols:
                    if s.name not in visited:
                        impacted.append(s)
                        if d > 1:
                            _recurse(s.name, d - 1)

            # 继承该类的子类
            for derived in self._reverse_inherit.get(symbol, []):
                if derived not in visited:
                    for fp in self._symbol_to_files.get(derived, []):
                        for s in self.parser.parse_file(fp).symbols:
                            if s.name == derived and s not in impacted:
                                impacted.append(s)

        _recurse(target_symbol, depth)
        return impacted

    def get_dependency_tree(self, module: str) -> Dict[str, Any]:
        """
        获取模块的依赖树。
        """
        try:
            descendants = nx.descendants(self._dep_graph, module)
            tree: Dict[str, Any] = {"module": module, "depends_on": []}
            for dep in descendants:
                sub = self.get_dependency_tree(dep)
                tree["depends_on"].append(sub)
            return tree
        except nx.NetworkXError:
            return {"module": module, "depends_on": []}

    def stats(self) -> Dict[str, int]:
        """返回图谱统计信息。"""
        return {
            "nodes_call": self._call_graph.number_of_nodes(),
            "edges_call": self._call_graph.number_of_edges(),
            "nodes_dep": self._dep_graph.number_of_nodes(),
            "edges_dep": self._dep_graph.number_of_edges(),
            "classes_tracked": len(self._inheritance),
        }