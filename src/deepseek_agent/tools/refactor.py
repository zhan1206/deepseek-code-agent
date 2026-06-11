"""
语义重构工具 — 基于知识图谱的智能代码重构。

支持重构类型：
- 重命名（函数/类/变量/方法）
- 提取函数
- 内联函数
- 移动模块（含 import 更新）
"""

from __future__ import annotations

import ast
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import tool, DangerLevel, ToolResult

try:
    from ..knowledge import RelationGraph
    _HAS_RELATION_GRAPH = True
except ImportError:
    _HAS_RELATION_GRAPH = False


# ── 重构引擎 ─────────────────────────────────────────────────────────────

class RefactorEngine:
    """基于 AST + 关系图谱的语义重构引擎。"""

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self._rg = RelationGraph() if _HAS_RELATION_GRAPH else None

    # ── 重命名 ──────────────────────────────────────────────────────

    def rename_symbol(
        self,
        old_name: str,
        new_name: str,
        file_path: Optional[str] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        语义重命名：在所有引用点替换符号名。

        策略：
        1. 在知识图谱中查找符号定义位置
        2. 查找所有引用点（调用、导入、注释）
        3. 生成批量替换计划
        4. 验证替换不会破坏语法
        """
        changes: List[Dict[str, Any]] = []

        # 查找定义
        definitions = self._find_definitions(old_name, file_path)

        if not definitions:
            # Fallback：纯文本搜索
            definitions = self._text_search_definitions(old_name)

        # 查找引用
        references = self._find_references(old_name, definitions)

        # 生成替换计划
        all_sites = definitions + references

        for site in all_sites:
            fp = site["file_path"]
            try:
                content = Path(fp).read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()

                if site.get("line_number", 0) > len(lines):
                    continue

                line = lines[site["line_number"] - 1]

                # 精确替换：确保替换的是完整符号名而非子串
                new_line = self._replace_symbol_in_line(line, old_name, new_name)

                if new_line != line:
                    changes.append({
                        "file_path": fp,
                        "line_number": site["line_number"],
                        "old_line": line.rstrip(),
                        "new_line": new_line.rstrip(),
                        "change_type": site.get("type", "reference"),
                    })
            except Exception:
                continue

        # 验证替换不会破坏语法
        syntax_errors = self._validate_changes(changes) if not dry_run else []

        # 执行替换
        if not dry_run and not syntax_errors:
            self._apply_changes(changes)

        return {
            "old_name": old_name,
            "new_name": new_name,
            "total_changes": len(changes),
            "changes": changes[:50],  # 最多返回 50 条
            "syntax_errors": syntax_errors,
            "dry_run": dry_run,
        }

    # ── 提取函数 ────────────────────────────────────────────────────

    def extract_function(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        new_func_name: str,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        提取代码片段为独立函数。

        策略：
        1. 分析选中代码的输入变量（从上层作用域读取的）
        2. 分析输出变量（选中代码修改的变量）
        3. 生成函数定义 + 调用替换
        """
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
        except Exception as e:
            return {"error": str(e)}

        if start_line < 1 or end_line > len(lines):
            return {"error": f"行范围无效: {start_line}-{end_line}, 文件共 {len(lines)} 行"}

        # 提取选中代码
        selected_lines = lines[start_line - 1:end_line]
        selected_code = "\n".join(selected_lines)

        # 简单分析：查找可能的外部变量（未在选中代码中定义的标识符）
        # 这是一个近似分析，精确分析需要 AST
        defined_vars = set(re.findall(r"(?:^|\s)(\w+)\s*=", selected_code))
        used_vars = set(re.findall(r"\b([a-zA-Z_]\w*)\b", selected_code))
        # 排除 Python 关键字
        import keyword
        external_vars = used_vars - defined_vars - set(keyword.kwlist) - {"self", "cls", "True", "False", "None"}

        # 生成参数列表
        params = list(external_vars)

        # 生成缩进
        indent = len(selected_lines[0]) - len(selected_lines[0].lstrip())
        indent_str = " " * indent

        # 生成函数定义
        func_def = textwrap.dedent(
            f"""def {new_func_name}({', '.join(params)}):
            {textwrap.indent(selected_code.strip(), '    ')}
        """
        ).strip()

        # 生成调用
        call_line = f"{indent_str}{new_func_name}({', '.join(params)})"

        # 修改后的文件内容
        new_lines = lines[:start_line - 1] + [call_line] + lines[end_line:]
        new_content = "\n".join(new_lines)

        # 在文件末尾或类方法后插入函数定义
        # 简化：在文件顶部（import 之后）插入
        import_end = self._find_import_end(new_lines)
        new_lines.insert(import_end, "")
        new_lines.insert(import_end + 1, func_def)
        new_content_with_func = "\n".join(new_lines)

        changes = [
            {
                "file_path": file_path,
                "change_type": "extract_function",
                "new_func_name": new_func_name,
                "params": params,
                "original_range": [start_line, end_line],
                "replaced_with_call": call_line,
                "func_definition": func_def,
            }
        ]

        if not dry_run:
            Path(file_path).write_text(new_content_with_func, encoding="utf-8")

        return {
            "file_path": file_path,
            "new_func_name": new_func_name,
            "params": params,
            "extracted_lines": end_line - start_line + 1,
            "changes": changes,
            "dry_run": dry_run,
        }

    # ── 辅助方法 ────────────────────────────────────────────────────

    def _find_definitions(self, name: str, file_path: Optional[str] = None) -> List[Dict]:
        """在 AST 中查找符号定义，优先使用 RelationGraph。"""
        results = []

        # 尝试使用 RelationGraph 的 build_from_dir + stats
        if self._rg is not None:
            try:
                self._rg.build_from_dir(str(self.project_path))
                # 使用 find_callers/find_dependents 查找关联符号
                callers = self._rg.find_callers(name)
                for c in callers[:20]:
                    results.append({
                        "file_path": c if isinstance(c, str) else str(c),
                        "line_number": 0,
                        "type": "definition",
                        "kind": "symbol",
                    })
            except Exception:
                pass

        # Fallback: AST 搜索
        if not results:
            results = self._ast_search_definitions(name, file_path)

        return results

    def _ast_search_definitions(self, name: str, file_path: Optional[str] = None) -> List[Dict]:
        """AST 搜索符号定义。"""
        results = []
        search_dirs = [Path(file_path).parent] if file_path else [self.project_path]

        for search_dir in search_dirs:
            for py_file in search_dir.rglob("*.py"):
                try:
                    tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                            results.append({
                                "file_path": str(py_file),
                                "line_number": node.lineno,
                                "type": "definition",
                                "kind": "function",
                            })
                        elif isinstance(node, ast.ClassDef) and node.name == name:
                            results.append({
                                "file_path": str(py_file),
                                "line_number": node.lineno,
                                "type": "definition",
                                "kind": "class",
                            })
                except Exception:
                    continue
        return results

    def _text_search_definitions(self, name: str) -> List[Dict]:
        """Fallback: 纯文本搜索定义。"""
        results = []
        patterns = [
            rf"^(def|class|async\s+def)\s+{re.escape(name)}\b",
            rf"^{re.escape(name)}\s*=",
        ]

        for py_file in self.project_path.rglob("*.py"):
            if any(skip in str(py_file) for skip in [".git", "__pycache__", ".venv", "node_modules"]):
                continue
            try:
                for i, line in enumerate(py_file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    for pat in patterns:
                        if re.search(pat, line):
                            results.append({
                                "file_path": str(py_file),
                                "line_number": i,
                                "type": "definition",
                                "kind": "function" if "def " in line else "variable",
                            })
                            break
            except Exception:
                continue
        return results

    def _find_references(self, name: str, definitions: List[Dict]) -> List[Dict]:
        """查找所有引用点。"""
        results = []
        seen = set()

        # 从定义文件中查找引用
        def_files = {d["file_path"] for d in definitions if d.get("file_path")}

        # 全局搜索引用
        search_scope = list(self.project_path.rglob("*.py"))
        for py_file in search_scope:
            if any(skip in str(py_file) for skip in [".git", "__pycache__", ".venv", "node_modules"]):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    # 跳过定义行
                    if str(py_file) in def_files and any(
                        d["line_number"] == i for d in definitions if d["file_path"] == str(py_file)
                    ):
                        continue

                    # 检查是否引用了目标符号
                    if re.search(rf"\b{re.escape(name)}\b", line):
                        key = (str(py_file), i)
                        if key not in seen:
                            seen.add(key)
                            # 判断引用类型
                            if "import" in line:
                                ref_type = "import"
                            elif "def " in line or "class " in line:
                                ref_type = "definition_reference"
                            elif "#" in line.split(name)[0]:
                                ref_type = "comment"
                            else:
                                ref_type = "usage"
                            results.append({
                                "file_path": str(py_file),
                                "line_number": i,
                                "type": ref_type,
                                "line_content": line.strip()[:100],
                            })
            except Exception:
                continue

        return results

    def _replace_symbol_in_line(self, line: str, old_name: str, new_name: str) -> str:
        """精确替换符号名，避免替换子串。"""
        # 使用 word boundary 替换
        return re.sub(rf"\b{re.escape(old_name)}\b", new_name, line)

    def _validate_changes(self, changes: List[Dict]) -> List[str]:
        """验证变更不会破坏 Python 语法。"""
        errors = []

        # 按文件分组
        by_file: Dict[str, List[Dict]] = {}
        for c in changes:
            by_file.setdefault(c["file_path"], []).append(c)

        for fp, file_changes in by_file.items():
            try:
                content = Path(fp).read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()

                # 应用替换
                for c in file_changes:
                    idx = c["line_number"] - 1
                    if 0 <= idx < len(lines):
                        lines[idx] = c["new_line"]

                # 验证语法
                new_content = "\n".join(lines)
                try:
                    ast.parse(new_content)
                except SyntaxError as e:
                    errors.append(f"{fp}: 语法错误 - {e}")
            except Exception as e:
                errors.append(f"{fp}: 无法验证 - {e}")

        return errors

    def _apply_changes(self, changes: List[Dict]) -> None:
        """执行文件修改。"""
        by_file: Dict[str, List[Dict]] = {}
        for c in changes:
            by_file.setdefault(c["file_path"], []).append(c)

        for fp, file_changes in by_file.items():
            try:
                content = Path(fp).read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()

                for c in file_changes:
                    idx = c["line_number"] - 1
                    if 0 <= idx < len(lines):
                        lines[idx] = c["new_line"]

                Path(fp).write_text("\n".join(lines), encoding="utf-8")
            except Exception:
                continue

    def _find_import_end(self, lines: List[str]) -> int:
        """找到 import 区域的结束行。"""
        last_import = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = i + 1
            elif stripped.startswith("#") or stripped == "":
                continue
            elif last_import > 0:
                break
        return last_import


# ── 工具注册 ─────────────────────────────────────────────────────────────

@tool(
    name="auto_refactor",
    description="语义级代码重构（重命名/提取函数/内联），基于知识图谱定位所有引用点。",
    danger_level=DangerLevel.SENSITIVE,
)
async def auto_refactor(
    action: str = "rename",
    old_name: str = "",
    new_name: str = "",
    file_path: str = "",
    start_line: int = 0,
    end_line: int = 0,
    new_func_name: str = "",
    project_path: str = ".",
    dry_run: bool = True,
) -> str:
    """
    语义重构工具。

    Args:
        action: 重构类型 - rename / extract_function
        old_name: 旧符号名（rename 时必填）
        new_name: 新符号名（rename 时必填）
        file_path: 文件路径（extract_function 时必填）
        start_line: 起始行号（extract_function 时必填）
        end_line: 结束行号（extract_function 时必填）
        new_func_name: 新函数名（extract_function 时必填）
        project_path: 项目根目录
        dry_run: 是否仅预览（默认 True，安全起见）
    """
    engine = RefactorEngine(project_path)

    if action == "rename":
        if not old_name or not new_name:
            return ToolResult.fail("rename 需要 old_name 和 new_name 参数").to_str()

        result = engine.rename_symbol(old_name, new_name, file_path, dry_run)

        lines = [
            f"🔧 重构预览：重命名 `{old_name}` → `{new_name}`",
            f"   影响文件：{len(set(c['file_path'] for c in result['changes']))} 个",
            f"   总变更点：{result['total_changes']}",
        ]

        if result.get("syntax_errors"):
            lines.append(f"   ⚠️ 语法错误：{len(result['syntax_errors'])} 个")
            for err in result["syntax_errors"][:3]:
                lines.append(f"     - {err}")

        if result["changes"]:
            lines.append("")
            lines.append("📋 变更详情：")
            for c in result["changes"][:20]:
                icon = {"definition": "📝", "usage": "🔗", "import": "📦"}.get(c.get("change_type", ""), "·")
                lines.append(f"  {icon} {c['file_path']}:{c['line_number']}")
                if c.get("old_line"):
                    lines.append(f"     - {c['old_line'][:80]}")
                    lines.append(f"     + {c['new_line'][:80]}")

        if dry_run:
            lines.append("")
            lines.append("💡 这是预览模式，设置 dry_run=false 执行实际修改")

        return ToolResult.ok("\n".join(lines)).to_str()

    elif action == "extract_function":
        if not file_path or not new_func_name:
            return ToolResult.fail("extract_function 需要 file_path 和 new_func_name 参数").to_str()
        if start_line <= 0 or end_line <= 0:
            return ToolResult.fail("extract_function 需要 start_line 和 end_line 参数").to_str()

        result = engine.extract_function(file_path, start_line, end_line, new_func_name, dry_run)

        if "error" in result:
            return ToolResult.fail(result["error"]).to_str()

        lines = [
            f"🔧 提取函数预览：`{new_func_name}`",
            f"   源文件：{file_path}",
            f"   提取范围：行 {start_line}-{end_line}（{result['extracted_lines']} 行）",
            f"   参数：{', '.join(result['params']) or '无'}",
        ]

        for c in result.get("changes", []):
            lines.append(f"   函数定义：{c['new_func_name']}({', '.join(c['params'])})")
            lines.append(f"   调用替换：{c['replaced_with_call']}")

        if dry_run:
            lines.append("")
            lines.append("💡 这是预览模式，设置 dry_run=false 执行实际修改")

        return ToolResult.ok("\n".join(lines)).to_str()

    else:
        return ToolResult.fail(f"不支持的重构类型: {action}，可选: rename, extract_function").to_str()
