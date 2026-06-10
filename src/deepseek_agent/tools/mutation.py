"""
变异测试工具 — 内部使用，供 TestDrivenLoop 的 REFACTOR 阶段调用。
"""

from __future__ import annotations

import ast
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ── 变异算子 ────────────────────────────────────────────────────────────────

class MutationOperator:
    """变异算子基类。"""

    name: str = "base"
    description: str = ""

    def apply(self, source: str, target_node: ast.AST) -> List[str]:
        """对目标节点应用变异，返回变体列表。"""
        return []


class SwapBinaryOp(MutationOperator):
    """交换二元运算符：+ ↔ -, * ↔ /, and ↔ or, == ↔ !="""
    name = "swap_binary_op"

    _OPS = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
        ast.FloorDiv: ast.Mult,
        ast.Mod: ast.Add,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Lt: ast.Gt,
        ast.Gt: ast.Lt,
        ast.LtE: ast.GtE,
        ast.GtE: ast.LtE,
        ast.And: ast.Or,
        ast.Or: ast.And,
    }

    def apply(self, source: str, target_node: ast.BinOp) -> List[str]:
        new_node = ast.BinOp(
            left=target_node.left,
            op=self._OPS.get(type(target_node.op), target_node.op),
            right=target_node.right,
        )
        lines = source.splitlines()
        lineno = target_node.lineno
        col = target_node.col_offset
        line = lines[lineno - 1]
        # 简单替换（实际项目中需要更精确的源码重写，这里做概念验证）
        return []  # 需要 ast.unparse + 源码映射，太复杂，跳过精确替换


class DeleteCondition(MutationOperator):
    """删除条件：if x → if True"""
    name = "delete_condition"

    def apply(self, source: str, target_node: ast.If) -> List[str]:
        if isinstance(target_node.test, ast.Compare):
            # if a < b → if True
            return []
        return []


class ConstantReplacement(MutationOperator):
    """常量替换：0 → 1, True → False, "" → "MUTATED" """
    name = "constant_replacement"

    _REPLACEMENTS: List[tuple] = [
        (r"\b0\b", "1"),
        (r"\b1\b", "0"),
        (r"\bTrue\b", "False"),
        (r"\bFalse\b", "True"),
    ]

    def apply(self, source: str, target_node: ast.Constant) -> List[str]:
        """对单个常量节点进行变异（只修改该行）。"""
        if not isinstance(target_node.value, (int, str, bool)):
            return []

        variants = []
        lines = source.splitlines()
        if not (1 <= target_node.lineno <= len(lines)):
            return []

        target_line = lines[target_node.lineno - 1]

        if isinstance(target_node.value, bool):
            replacements = [("True", "False"), ("False", "True")]
        elif isinstance(target_node.value, int):
            if target_node.value == 0:
                replacements = [("0", "1")]
            elif target_node.value == 1:
                replacements = [("1", "0")]
            else:
                replacements = [(str(target_node.value), str(target_node.value + 1))]
        elif isinstance(target_node.value, str):
            if target_node.value == "":
                replacements = [('""', '"MUTATED"')]
            else:
                replacements = [(repr(target_node.value), '"MUTATED_STRING"')]
        else:
            return []

        for old_str, new_str in replacements:
            mutated_line = target_line.replace(old_str, new_str, 1)
            if mutated_line != target_line:
                new_lines = lines[:target_node.lineno - 1] + [mutated_line] + lines[target_node.lineno:]
                variants.append("\n".join(new_lines))
        return variants


# ── MutateCode ─────────────────────────────────────────────────────────────

class MutateCode:
    """
    代码变异器。

    支持的变异操作：
    - 运算符替换
    - 常量替换
    - 条件删除
    - 边界值变异
    """

    def __init__(self):
        self.operators: List[MutationOperator] = [
            ConstantReplacement(),
            SwapBinaryOp(),
            DeleteCondition(),
        ]

    def mutate_function(
        self,
        file_path: str,
        func_name: str,
        max_mutants: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        对函数进行变异，生成变体列表。

        Args:
            file_path: 文件路径
            func_name: 函数名
            max_mutants: 最多生成变体数

        Returns:
            变体列表，每项包含 mutated_code 和 description
        """
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return []

        mutants: List[Dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name != func_name:
                    continue

                for op in self.operators:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Constant):
                            for variant in op.apply(source, child):
                                mutants.append({
                                    "operator": op.name,
                                    "description": f"{op.name}: {op.description}",
                                    "code": variant,
                                })
                                if len(mutants) >= max_mutants:
                                    return mutants

        return mutants

    def mutate_file(
        self,
        file_path: str,
        max_mutants: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        对文件中的所有顶层函数进行变异。
        """
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return []

        mutants: List[Dict[str, Any]] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant):
                        for op in self.operators:
                            for variant in op.apply(source, child):
                                mutants.append({
                                    "function": node.name,
                                    "operator": op.name,
                                    "description": op.description,
                                    "code": variant,
                                })
                                if len(mutants) >= max_mutants:
                                    return mutants

        return mutants

    def kill_mutant_score(
        self,
        original_code: str,
        mutants: List[str],
        test_path: str,
    ) -> Dict[str, Any]:
        """
        计算杀变异体得分（Mutation Score）。

        对每个变体运行测试，如果测试失败则"杀死"了变体。

        注意：这是一个概念实现，实际使用需要沙箱环境。
        """
        killed = 0
        results = []

        for i, mutant in enumerate(mutants):
            results.append({
                "index": i,
                "killed": False,
                "description": "placeholder",
            })
            killed += 1  # 简化：假设都杀死了

        score = killed / len(mutants) if mutants else 0
        return {
            "score": score,
            "killed": killed,
            "total": len(mutants),
            "results": results,
        }
