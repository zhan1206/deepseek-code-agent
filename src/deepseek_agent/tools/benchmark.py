"""
性能基准守护工具 — 基于 pytest-benchmark 的自动回归检测。

用法：
  1. 生成基线：benchmark init --project ./myproject
  2. 修改代码后运行：benchmark run --project ./myproject
  3. 对比结果：benchmark compare --project ./myproject
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import tool, DangerLevel, ToolResult


# ── 基准管理 ─────────────────────────────────────────────────────────────

BENCHMARK_DIR = ".deepseek-benchmarks"

class BenchmarkRunner:
    """基准测试运行器。"""

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.bench_dir = self.project_path / BENCHMARK_DIR

    def init_baseline(self) -> Dict[str, Any]:
        """初始化基准线。"""
        self.bench_dir.mkdir(exist_ok=True)
        result = self._run_pytest_benchmark()
        if result:
            baseline_path = self.bench_dir / "baseline.json"
            baseline_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return {
                "status": "initialized",
                "baseline_path": str(baseline_path),
                "benchmarks": len(result.get("benchmarks", [])),
            }
        return {"status": "no_benchmarks_found"}

    def run_comparison(self, regression_threshold: float = 5.0) -> Dict[str, Any]:
        """运行对比检测。"""
        baseline_path = self.bench_dir / "baseline.json"
        if not baseline_path.exists():
            return {"status": "no_baseline", "message": "请先运行 init 建立基线"}

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        current = self._run_pytest_benchmark()

        if not current:
            return {"status": "no_benchmarks_found"}

        # 对比
        regressions = []
        improvements = []
        baseline_map = {b["name"]: b for b in baseline.get("benchmarks", [])}

        for bench in current.get("benchmarks", []):
            name = bench["name"]
            if name not in baseline_map:
                continue

            base_mean = baseline_map[name].get("mean", 0)
            curr_mean = bench.get("mean", 0)

            if base_mean <= 0:
                continue

            change_pct = ((curr_mean - base_mean) / base_mean) * 100

            comparison = {
                "name": name,
                "baseline_mean": round(base_mean, 6),
                "current_mean": round(curr_mean, 6),
                "change_pct": round(change_pct, 2),
            }

            if change_pct > regression_threshold:
                regressions.append(comparison)
            elif change_pct < -regression_threshold:
                improvements.append(comparison)

        # 保存当前结果
        current_path = self.bench_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        current_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "status": "compared",
            "total_benchmarks": len(current.get("benchmarks", [])),
            "regressions": regressions,
            "improvements": improvements,
            "regression_threshold": regression_threshold,
        }

    def _run_pytest_benchmark(self) -> Optional[Dict]:
        """运行 pytest-benchmark 并收集结果。"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                output_path = f.name

            result = subprocess.run(
                ["python", "-m", "pytest",
                 str(self.project_path),
                 "--benchmark-only",
                 "--benchmark-json", output_path,
                 "-q"],
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
                errors="replace",
            )

            if os.path.exists(output_path):
                data = json.loads(Path(output_path).read_text(encoding="utf-8"))
                os.unlink(output_path)
                return data

        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

        return None


# ── 简易自包含基准 ──────────────────────────────────────────────────────

def quick_benchmark(func_code: str, iterations: int = 100) -> Dict[str, Any]:
    """对一段 Python 代码进行简易基准测试。"""
    setup = f"""
import time
{func_code}
start = time.perf_counter()
for _ in range({iterations}):
    _bench_func()
elapsed = time.perf_counter() - start
mean = elapsed / {iterations}
print(json.dumps({{"mean": mean, "iterations": {iterations}}}))
"""
    # 替换函数名
    if "def " in func_code:
        # 提取函数名
        import re
        match = re.search(r"def\s+(\w+)", func_code)
        if match:
            func_name = match.group(1)
            setup = setup.replace("_bench_func()", f"{func_name}()")

    try:
        result = subprocess.run(
            ["python", "-c", f"import json\n{setup}"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        # 最后一行是 JSON 结果
        for line in reversed(result.stdout.strip().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return {"mean": 0, "iterations": iterations, "error": "benchmark failed"}


# ── 工具注册 ─────────────────────────────────────────────────────────────

@tool(
    name="benchmark",
    description="性能基准测试与回归检测（基于 pytest-benchmark）。",
    danger_level=DangerLevel.SAFE,
    read_only=True,
)
async def benchmark(
    action: str = "compare",
    project_path: str = ".",
    regression_threshold: float = 5.0,
) -> str:
    """
    性能基准测试。

    Args:
        action: 操作类型 - init（建立基线）/ compare（对比检测）/ status
        project_path: 项目根目录
        regression_threshold: 回归阈值百分比（默认 5%）
    """
    runner = BenchmarkRunner(project_path)

    if action == "init":
        result = runner.init_baseline()
        if result["status"] == "initialized":
            return ToolResult.ok(
                f"📊 基线已建立\n"
                f"   基线文件：{result['baseline_path']}\n"
                f"   基准数量：{result['benchmarks']}"
            ).to_str()
        return ToolResult.ok("⚠️ 未找到基准测试（需在项目中创建 pytest-benchmark 用例）").to_str()

    elif action == "compare":
        result = runner.run_comparison(regression_threshold)

        if result["status"] == "no_baseline":
            return ToolResult.fail("未找到基线，请先运行 init").to_str()

        lines = [
            f"📊 性能对比报告",
            f"   基准数量：{result.get('total_benchmarks', 0)}",
            f"   回归阈值：{regression_threshold}%",
        ]

        if result.get("regressions"):
            lines.append(f"\n🔴 性能回归（{len(result['regressions'])} 个）：")
            for r in result["regressions"]:
                lines.append(
                    f"   {r['name']}: {r['baseline_mean']:.6f}s → {r['current_mean']:.6f}s "
                    f"(+{r['change_pct']:.1f}%)"
                )

        if result.get("improvements"):
            lines.append(f"\n🟢 性能提升（{len(result['improvements'])} 个）：")
            for r in result["improvements"]:
                lines.append(
                    f"   {r['name']}: {r['baseline_mean']:.6f}s → {r['current_mean']:.6f}s "
                    f"({r['change_pct']:.1f}%)"
                )

        if not result.get("regressions") and not result.get("improvements"):
            lines.append("\n✅ 无显著性能变化")

        return ToolResult.ok("\n".join(lines)).to_str()

    elif action == "status":
        bench_dir = runner.bench_dir
        if not bench_dir.exists():
            return ToolResult.ok("📊 尚未初始化基准测试").to_str()

        runs = sorted(bench_dir.glob("*.json"))
        lines = [
            f"📊 基准测试状态",
            f"   目录：{bench_dir}",
            f"   历史运行：{len(runs)} 次",
        ]
        if (bench_dir / "baseline.json").exists():
            lines.append("   ✅ 基线已建立")
        else:
            lines.append("   ⚠️ 基线未建立")

        return ToolResult.ok("\n".join(lines)).to_str()

    else:
        return ToolResult.fail(f"不支持的操作: {action}，可选: init, compare, status").to_str()
