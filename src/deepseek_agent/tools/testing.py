"""
测试工具集 — TDD 循环支持：生成测试、执行测试套件、获取覆盖率。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import tool, ToolResult, DangerLevel
from ..core.client import DeepSeekClient


# ── pytest 输出解析 ────────────────────────────────────────────────────────

def _parse_pytest_output(output: str) -> Dict[str, Any]:
    """解析 pytest JSON 输出。"""
    result: Dict[str, Any] = {
        "total": 0, "passed": 0, "failed": 0,
        "errors": 0, "skipped": 0,
        "passed_tests": [],
        "failed_tests": [],
        "error_tests": [],
        "summary": "",
    }

    # 尝试 JSON 格式
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if "test_report" in data:
                    report = data["test_report"]
                    result.update(report)
                    return result
            except json.JSONDecodeError:
                pass

    # 文本格式解析
    passed = re.findall(r"PASSED\s+(.+?)(?:\s|$)", output)
    failed = re.findall(r"FAILED\s+(.+?)(?:\s|$)", output)
    errors = re.findall(r"ERROR\s+(.+?)(?:\s|$)", output)
    skipped = re.findall(r"SKIPPED\s+(.+?)(?:\s|$)", output)

    # 统计行
    stat_match = re.search(r"(\d+)\s+passed", output)
    if stat_match:
        result["passed"] = int(stat_match.group(1))
    stat_match = re.search(r"(\d+)\s+failed", output)
    if stat_match:
        result["failed"] = int(stat_match.group(1))
    stat_match = re.search(r"(\d+)\s+error", output)
    if stat_match:
        result["errors"] = int(stat_match.group(1))
    stat_match = re.search(r"(\d+)\s+skipped", output)
    if stat_match:
        result["skipped"] = int(stat_match.group(1))

    result["total"] = result["passed"] + result["failed"] + result["errors"]
    result["passed_tests"] = passed
    result["failed_tests"] = failed
    result["error_tests"] = errors

    # 摘要
    lines = output.strip().splitlines()
    if lines:
        result["summary"] = lines[-1].strip()

    return result


# ── 工具 ────────────────────────────────────────────────────────────────────

@tool(
    name="generate_tests",
    description="为目标函数/模块生成 pytest 测试用例（自适应反馈循环，失败后自动补充边界用例）。",
    danger_level=DangerLevel.MODERATE,
)
def generate_tests(
    target: str,
    output_path: Optional[str] = None,
    framework: str = "pytest",
    style: str = "descriptive",
    max_feedback_rounds: int = 3,
) -> ToolResult:
    """
    生成测试用例。

    Args:
        target: 目标函数名或模块路径，如 "src/module.py::func" 或 "src/module.py"
        output_path: 测试文件输出路径（默认自动生成）
        framework: 测试框架，默认 pytest
        style: 测试风格：descriptive（描述性）/ minimal（最小化）
    """
    PY = "C:\\Users\\朱子瞻\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"
    target_path = target.split("::")[0] if "::" in target else target

    # 读取目标文件
    try:
        content = Path(target_path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ToolResult.fail(f"文件不存在: {target_path}")
    except Exception as e:
        return ToolResult.fail(f"读取失败: {e}")

    # 读取相关源码
    func_name = target.split("::")[-1] if "::" in target else None
    snippet = content[:3000]  # 截断

    # 调用 LLM 生成测试
    prompt = f"""请为以下 Python 代码生成 {framework} 测试用例。

要求：
- 测试风格：{style}
- 每个函数至少 2 个测试用例（正常 + 边界）
- 包含 docstring
- 可直接运行

代码片段：
```python
{snippet}
```
"""

    client = DeepSeekClient()
    messages = [
        {"role": "system", "content": "你是一个专业的测试工程师。生成高质量、可运行的 pytest 测试用例。"},
        {"role": "user", "content": prompt},
    ]

    try:
        resp = asyncio.run(client.chat(messages, max_tokens=4096, temperature=0.2))
        test_code = resp.content or ""
    except Exception as e:
        return ToolResult.fail(f"LLM 调用失败: {e}")

    # 生成输出路径
    if output_path is None:
        stem = Path(target_path).stem
        test_file = Path(target_path).parent / f"test_{stem}.py"
        output_path = str(test_file)
    else:
        test_file = Path(output_path)

    # 写入文件（最多 3 次重试：检查是否可导入）
    for attempt in range(3):
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(test_code, encoding="utf-8")

        # 语法检查
        proc = asyncio.run(
            asyncio.create_subprocess_exec(
                PY, "-m", "py_compile", str(test_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        _, stderr = asyncio.run(proc.communicate())
        if proc.returncode == 0:
            break
        # 重试：从 LLM 获取修复
        messages = [
            {"role": "system", "content": "你是专业的测试工程师。"},
            {"role": "user", "content": f"上一次的测试代码有语法错误：\n{stderr.decode(errors='replace')}\n\n请只输出修复后的完整测试代码："},
        ]
        try:
            resp = asyncio.run(client.chat(messages, max_tokens=4096))
            test_code = resp.content or ""
        except Exception:
            break

    return ToolResult.ok({
        "target": target,
        "output_path": str(test_file),
        "attempt": attempt + 1,
        "generated_lines": len(test_code.splitlines()),
    })


    # ── v2.0 自适应反馈循环 ──────────────────────────────────────────
    # 运行测试 → 收集失败结果 → 反馈 LLM 补充用例 → 重复
    coverage_improved = True
    last_coverage = 0.0

    for round_num in range(max_feedback_rounds):
        if not coverage_improved and round_num > 0:
            break  # 覆盖率未提升则停止

        # 运行测试
        import subprocess as _sp
        test_result = _sp.run(
            [PY, "-m", "pytest", str(test_file), "-v", "--tb=short", "-q"],
            capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace",
        )
        test_output = test_result.stdout + "\n" + test_result.stderr

        # 检查是否全部通过
        if test_result.returncode == 0:
            break  # 全部通过，无需反馈

        # 收集失败信息
        failed_tests = []
        for line in test_output.splitlines():
            if "FAILED" in line:
                failed_tests.append(line.strip())

        # 运行覆盖率
        coverage_result = _sp.run(
            [PY, "-m", "coverage", "run", "-m", "pytest", str(test_file), "-q", "--tb=no"],
            capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace",
        )
        coverage_report = _sp.run(
            [PY, "-m", "coverage", "report", "--show-missing"],
            capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace",
        )

        # 提取覆盖率百分比
        cov_match = re.search(r'(\d+)%', coverage_report.stdout)
        current_coverage = float(cov_match.group(1)) if cov_match else 0.0
        coverage_improved = current_coverage > last_coverage
        last_coverage = current_coverage

        # 反馈 LLM 补充边界用例
        feedback_prompt = f"""测试运行结果：
- 通过/失败：见下方输出
- 覆盖率：{current_coverage}%
- 失败的测试：{chr(10).join(failed_tests[:10])}

测试输出摘要：
{test_output[-1500:]}

覆盖率报告（缺失行）：
{coverage_report.stdout[-1000:]}

请基于以上反馈补充边界用例，只输出需要新增的测试函数（不要重复已有测试）："""

        try:
            resp = asyncio.run(client.chat(
                [{"role": "user", "content": feedback_prompt}],
                max_tokens=2048, temperature=0.2,
            ))
            new_tests = resp.content or ""
            # 追加到测试文件
            if new_tests.strip():
                existing = test_file.read_text(encoding="utf-8")
                test_file.write_text(existing + "\n\n" + new_tests, encoding="utf-8")
        except Exception:
            break  # LLM 调用失败则停止反馈循环

    return ToolResult.ok({
        "target": target,
        "output_path": str(test_file),
        "attempts": attempt + 1,
        "feedback_rounds": round_num + 1 if max_feedback_rounds > 0 else 0,
        "final_coverage": last_coverage,
        "generated_lines": len(test_file.read_text(encoding="utf-8").splitlines()),
    })


@tool(
    name="run_test_suite",
    description="执行 pytest 测试文件，解析输出为结构化结果（total/passed/failed 详情）。",
    danger_level=DangerLevel.MODERATE,
)
def run_test_suite(
    path: str,
    verbose: bool = True,
    maxfail: int = 3,
) -> ToolResult:
    """
    执行测试套件。

    Args:
        path: 测试文件或目录路径
        verbose: 是否详细输出
        maxfail: 遇到 N 个失败后停止
    """
    import subprocess

    PY = "C:\\Users\\朱子瞻\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"
    target = Path(path)
    if not target.exists():
        return ToolResult.fail(f"路径不存在: {path}")

    args = [
        str(PY), "-m", "pytest",
        str(target),
        "-v" if verbose else "",
        "--tb=short",
        f"--maxfail={maxfail}",
        "--no-header",
        "-rN",  # 简洁摘要
    ]
    args = [a for a in args if a]

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return ToolResult.fail("测试执行超时（>120s）")
    except FileNotFoundError:
        return ToolResult.fail(f"Python 未找到: {PY}")
    except Exception as e:
        return ToolResult.fail(f"执行失败: {e}")

    parsed = _parse_pytest_output(output)
    parsed["exit_code"] = result.returncode
    parsed["output_preview"] = output[-2000:]  # 末尾 2000 字符

    return ToolResult.ok(parsed)


@tool(
    name="get_coverage",
    description="运行 coverage.py 获取代码覆盖率报告。",
    danger_level=DangerLevel.SAFE,
)
def get_coverage(
    path: Optional[str] = None,
    report_format: str = "term",
) -> ToolResult:
    """
    获取代码覆盖率。

    Args:
        path: 目标目录/文件（默认当前目录）
        report_format: 报告格式 (term|html|json|xml)
    """
    import subprocess
    import tempfile

    PY = "C:\\Users\\朱子瞻\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"
    target = path or "."
    out_dir = None

    args = [str(PY), "-m", "coverage", "run", "-m", "pytest", target, "-q", "--tb=no"]
    try:
        subprocess.run(args, capture_output=True, timeout=120, check=False)
    except Exception:
        pass

    # 生成报告
    if report_format == "json":
        json_file = tempfile.mktemp(suffix=".json")
        r = subprocess.run(
            [str(PY), "-m", "coverage", "json", "-o", json_file, "--quiet"],
            capture_output=True, text=True, timeout=30,
        )
        try:
            data = json.loads(Path(json_file).read_text(errors="replace"))
            return ToolResult.ok({"format": "json", "data": data})
        except Exception:
            pass
    elif report_format == "html":
        out_dir = tempfile.mkdtemp()
        subprocess.run(
            [str(PY), "-m", "coverage", "html", "-d", out_dir, "--quiet"],
            capture_output=True, timeout=30,
        )
        return ToolResult.ok({"format": "html", "output_dir": out_dir})
    else:
        r = subprocess.run(
            [str(PY), "-m", "coverage", "report", "--show-missing"],
            capture_output=True, text=True, timeout=30,
        )
        return ToolResult.ok({
            "format": "term",
            "output": r.stdout + r.stderr,
        })

    return ToolResult.ok({"format": report_format, "note": "报告生成失败"})
