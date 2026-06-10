"""
文件系统工具集 — read/write/edit/search/list。
包含对标 Claude Code 的 edit_file 精确替换引擎。
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import tool, DangerLevel, ToolResult
from .security import SecurityScanner


# ── 路径安全检查 ─────────────────────────────────────────────────────────

def _safe_path(path: str, cwd: Optional[str] = None) -> Path:
    """解析路径并确保不超出工作目录。"""
    p = Path(path).resolve()
    if cwd:
        root = Path(cwd).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            raise ValueError(f"路径 '{path}' 超出工作目录 '{cwd}'")
    return p


def _ensure_path(path: str) -> Path:
    """确保父目录存在。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── read_file ─────────────────────────────────────────────────────────────

@tool(
    name="read_file",
    description="读取文件全部或部分内容，支持行号和范围。",
    danger_level=DangerLevel.SAFE,
    read_only=True,
)
async def read_file(path: str, offset: int = 0, limit: int = 100) -> str:
    """
    读取文件内容。

    Args:
        path: 文件路径（绝对或相对）
        offset: 起始行号（0-based）
        limit: 最大读取行数

    Returns:
        带行号格式的文件内容。
    """
    try:
        p = _safe_path(path)
        lines = p.read_text(encoding="utf-8").splitlines()

        # 支持负索引（从末尾）
        if offset < 0:
            offset = max(0, len(lines) + offset)

        segment = lines[offset : offset + limit]
        numbered = "\n".join(f"{i + offset + 1:>4}: {line}" for i, line in enumerate(segment))

        suffix_info = ""
        if offset + limit < len(lines):
            suffix_info = f"\n... ({len(lines) - offset - limit} 行未显示)"
        if offset > 0:
            suffix_info = f"... (从第 {offset + 1} 行开始)\n" + suffix_info

        return numbered + suffix_info + f"\n[共 {len(lines)} 行]"

    except FileNotFoundError:
        return ToolResult.fail(f"文件未找到: {path}").to_str()
    except UnicodeDecodeError:
        return ToolResult.fail(f"无法解码文件（非 UTF-8）: {path}").to_str()
    except Exception as e:
        return ToolResult.fail(f"读取失败: {str(e)}").to_str()


# ── write_file ────────────────────────────────────────────────────────────

@tool(
    name="write_file",
    description="创建或覆盖文件（危险操作）。",
    danger_level=DangerLevel.DANGEROUS,
)
async def write_file(path: str, content: str) -> str:
    """
    写入文件，原子性操作（写临时文件 → 重命名）。

    v2.0: 写入前自动执行安全扫描，HIGH 级别漏洞阻止写入。

    Args:
        path: 文件路径
        content: 文件内容
    """
    try:
        # ── 安全扫描钩子 ──────────────────────────────────────────
        scanner = SecurityScanner(min_severity="MEDIUM")
        is_safe, findings = scanner.scan_content_before_write(content, path, block_on_high=True)
        if not is_safe:
            high_findings = [f for f in findings if f.severity == "HIGH"]
            issues = "\n".join(
                f"  🔴 [{f.rule_id}] {f.message} (行 {f.line_number})"
                for f in high_findings
            )
            return ToolResult.fail(
                f"写入被安全扫描阻止：发现 {len(high_findings)} 个高危漏洞\n{issues}\n"
                f"如需强制写入，请使用 write_file_unsafe 或调整安全策略。"
            ).to_str()

        p = _ensure_path(path)
        # 原子写入
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            if p.exists():
                p.replace(p.with_suffix(".bak"))  # 保留备份
            shutil.move(tmp, p)
        except Exception:
            os.unlink(tmp)
            raise
        return ToolResult.ok(f"写入成功: {path}").to_str()
    except Exception as e:
        return ToolResult.fail(f"写入失败: {str(e)}").to_str()


# ── edit_file（核心）─────────────────────────────────────────────────────

@tool(
    name="edit_file",
    description="精确编辑文件：old_string → new_string，支持多组替换和全量替换。",
    danger_level=DangerLevel.DANGEROUS,
)
async def edit_file(
    path: str,
    edits: List[Dict[str, Any]],
) -> str:
    """
    Claude Code 风格的精确文件编辑。

    对每个 Edit：
    1. 在文件中查找 old_string（按出现顺序）
    2. 如果 replace_all=false 且出现多次 → 报错，列出所有出现位置
    3. 全部验证通过后，原子性写入

    Args:
        path: 文件路径
        edits: 编辑列表，每个元素：
            - old_string (str): 原始字符串（必须唯一或 replace_all=true）
            - new_string (str): 替换后字符串
            - replace_all (bool): 是否替换所有匹配项，默认 False

    Returns:
        成功或详细错误信息。
    """
    if not edits:
        return ToolResult.fail("edits 列表为空").to_str()

    try:
        p = _safe_path(path)
        original_content = p.read_text(encoding="utf-8")
        content = original_content

        conflicts: List[Dict[str, Any]] = []

        for i, edit in enumerate(edits):
            old_str = edit.get("old_string", "")
            new_str = edit.get("new_string", "")
            replace_all = edit.get("replace_all", False)

            if not old_str:
                return ToolResult.fail(f"第 {i+1} 个 edit 的 old_string 为空").to_str()

            # 查找所有出现
            matches = [m.start() for m in re.finditer(re.escape(old_str), content)]

            if not matches:
                return ToolResult.fail(
                    f"第 {i+1} 个 edit 的 old_string 未在文件中找到：\n"
                    f"---\n{old_str}\n---\n"
                ).to_str()

            if len(matches) > 1 and not replace_all:
                # 列出所有位置（带上下文）
                lines = content.split("\n")
                positions = []
                pos = 0
                for lineno, line in enumerate(lines):
                    for _ in re.finditer(re.escape(old_str), line):
                        positions.append(f"  行 {lineno + 1}: {line.strip()[:80]}")
                        pos += 1
                return ToolResult.fail(
                    f"第 {i+1} 个 edit 的 old_string 在文件中出现 {len(matches)} 次：\n"
                    + "\n".join(positions[:10])
                    + "\n请使用 replace_all=true 替换全部，或提供更长的 old_string 使其唯一。"
                ).to_str()

        # 全部验证通过，依次替换
        for edit in edits:
            old_str = edit["old_string"]
            new_str = edit["new_string"]
            replace_all = edit.get("replace_all", False)

            if replace_all:
                content = content.replace(old_str, new_str)
            else:
                content = content.replace(old_str, new_str, 1)  # 只替换第一个

        if content == original_content:
            return ToolResult.fail("编辑后内容未发生变化").to_str()

        # 原子写入
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            shutil.move(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise

        return ToolResult.ok(
            f"编辑成功：{len(edits)} 处修改，文件 {path}"
        ).to_str()

    except FileNotFoundError:
        return ToolResult.fail(f"文件未找到: {path}").to_str()
    except Exception as e:
        return ToolResult.fail(f"编辑失败: {str(e)}").to_str()


# ── list_directory ────────────────────────────────────────────────────────

@tool(
    name="list_directory",
    description="列出目录内容，支持递归深度控制。",
    danger_level=DangerLevel.SAFE,
    read_only=True,
)
async def list_directory(path: str = ".", depth: int = 2) -> str:
    """
    列出目录树。

    Args:
        path: 目录路径
        depth: 递归深度（1=仅当前目录）
    """
    try:
        p = _safe_path(path)
        if not p.is_dir():
            return ToolResult.fail(f"不是目录: {path}").to_str()

        lines = []
        IGNORE = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}

        def _walk(current: Path, indent: str = "", max_depth: int = 2, current_depth: int = 0):
            if current_depth >= max_depth:
                return
            try:
                for entry in sorted(current.iterdir()):
                    if entry.name in IGNORE or entry.name.startswith("."):
                        continue
                    size = f" ({entry.stat().st_size:,} B)" if entry.is_file() else ""
                    icon = "📁" if entry.is_dir() else "📄"
                    lines.append(f"{indent}{icon} {entry.name}{size}")
                    if entry.is_dir():
                        _walk(entry, indent + "  ", max_depth, current_depth + 1)
            except PermissionError:
                lines.append(f"{indent}⚠️ [无权限]")

        lines.append(f"📂 {p.resolve()}")
        _walk(p, "  ", depth)

        return ToolResult.ok("\n".join(lines)).to_str()

    except Exception as e:
        return ToolResult.fail(f"列出目录失败: {str(e)}").to_str()


# ── search_file / search_content ─────────────────────────────────────────

@tool(
    name="search_file",
    description="按文件名搜索（支持通配符）。",
    danger_level=DangerLevel.SAFE,
    read_only=True,
)
async def search_file(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
) -> str:
    """在文件名中搜索。"""
    try:
        p = _safe_path(path)
        results: List[str] = []
        regex = re.compile(pattern.replace("*", ".*").replace("?", "."))

        def _search(d: Path):
            try:
                for entry in d.iterdir():
                    if entry.name.startswith("."):
                        continue
                    if regex.match(entry.name):
                        results.append(str(entry.relative_to(p)))
                    if entry.is_dir() and recursive:
                        _search(entry)
            except PermissionError:
                pass

        _search(p)
        if not results:
            return ToolResult.ok(f"未找到匹配 '{pattern}' 的文件").to_str()
        return ToolResult.ok(f"找到 {len(results)} 个匹配：\n" + "\n".join(results)).to_str()
    except Exception as e:
        return ToolResult.fail(f"搜索失败: {str(e)}").to_str()


@tool(
    name="search_content",
    description="在文件内容中搜索文本或正则表达式，显示行号。",
    danger_level=DangerLevel.SAFE,
    read_only=True,
)
async def search_content(
    pattern: str,
    path: str = ".",
    file_types: str = "*",
    recursive: bool = True,
    max_results: int = 100,
) -> str:
    """
    内容搜索，类似 grep。

    Args:
        pattern: 搜索模式（正则或文本）
        path: 搜索目录
        file_types: 文件类型过滤，如 "py,md,txt"
        recursive: 是否递归
        max_results: 最大结果数
    """
    try:
        p = _safe_path(path)
        extensions = {f".{ext.strip().lstrip('.')}" for ext in file_types.split(",")} if file_types != "*" else None

        results: List[str] = []
        count = 0
        is_regex = any(c in pattern for c in r"\[](){}|?*+^$")
        prog = re.compile(pattern) if is_regex else None

        def _matches(text: str) -> bool:
            if prog:
                return bool(prog.search(text))
            return pattern.lower() in text.lower()

        def _search(d: Path):
            nonlocal count
            if count >= max_results:
                return
            try:
                for entry in d.iterdir():
                    if entry.name.startswith("."):
                        continue
                    if entry.is_dir() and recursive:
                        _search(entry)
                    elif entry.is_file():
                        if extensions and entry.suffix not in extensions:
                            continue
                        try:
                            lines = entry.read_text(encoding="utf-8", errors="ignore").splitlines()
                            for lineno, line in enumerate(lines, 1):
                                if _matches(line):
                                    rel = entry.relative_to(p)
                                    snippet = line.strip()[:120]
                                    results.append(f"{rel}:{lineno}: {snippet}")
                                    count += 1
                                    if count >= max_results:
                                        return
                        except Exception:
                            pass
            except PermissionError:
                pass

        _search(p)

        if not results:
            return ToolResult.ok(f"未在 {path} 中找到匹配 '{pattern}' 的内容").to_str()

        header = f"找到 {len(results)} 处匹配：\n"
        return ToolResult.ok(header + "\n".join(results)).to_str()

    except Exception as e:
        return ToolResult.fail(f"搜索失败: {str(e)}").to_str()


# ── delete_file ──────────────────────────────────────────────────────────

@tool(
    name="delete_file",
    description="删除文件或目录（极度危险）。",
    danger_level=DangerLevel.DANGEROUS,
)
async def delete_file(path: str, recursive: bool = False) -> str:
    """
    删除文件或目录。

    Args:
        path: 文件/目录路径
        recursive: 是否递归删除目录
    """
    try:
        p = _safe_path(path)
        if not p.exists():
            return ToolResult.fail(f"路径不存在: {path}").to_str()

        if p.is_dir():
            if recursive:
                shutil.rmtree(p)
            else:
                p.rmdir()
        else:
            p.unlink()

        return ToolResult.ok(f"已删除: {path}").to_str()
    except Exception as e:
        return ToolResult.fail(f"删除失败: {str(e)}").to_str()


# ── Shell 执行 ────────────────────────────────────────────────────────────

# 危险命令黑名单
DENY_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"mkfs",
    r"dd\s+if=/dev/zero",
    r":\(\)\{",  # fork bomb
    r">\s*/dev/sd",
    r"chattr\s+-i",
]

ALLOWED_COMMANDS = {"python", "python3", "node", "npm", "git", "ls", "cat", "grep", "find", "echo"}


def _check_command(command: str) -> Tuple[bool, str]:
    """检查命令是否安全。"""
    for pat in DENY_PATTERNS:
        if re.search(pat, command):
            return False, f"命令包含危险模式: {pat}"
    return True, ""



# Shell 黑名单正则
_SHELL_BLACKLIST_PATTERNS = [
    r"^\s*rm\s+-rf\s+/\s*$",
    r"^\s*rm\s+-rf\s+/\S+",
    r"mkfs\s",
    r"dd\s+if=/dev/zero\s+of=/dev/",
    r":\(\){:\|:&};:",
    r"curl.*\|\s*sh",
    r"wget.*\|\s*sh",
    r"shutdown",
    r"reboot",
    r"init\s+6",
    r"poweroff",
]
_SHELL_BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in _SHELL_BLACKLIST_PATTERNS]

def _check_shell_command(command: str):
    for pat in _SHELL_BLACKLIST_RE:
        if pat.search(command):
            return False, f"禁止执行的命令（匹配黑名单）: {command[:80]}"
    return True, ""

@tool(
    name="run_shell",
    description="执行 Shell 命令，返回 stdout/stderr。",
    danger_level=DangerLevel.DANGEROUS,
)
async def run_shell(
    command: str,
    working_dir: str = ".",
    timeout: int = 60,
    allowed_only: bool = False,
) -> str:
    """
    执行 Shell 命令。

    Args:
        command: 要执行的命令
        working_dir: 工作目录
        timeout: 超时秒数
        allowed_only: 是否仅允许白名单命令
    """
    safe, reason = _check_command(command)
    if not safe:
        return ToolResult.fail(f"命令被拦截: {reason}").to_str()

    if allowed_only:
        cmd_name = command.strip().split()[0]
        if cmd_name not in ALLOWED_COMMANDS:
            return ToolResult.fail(f"命令 '{cmd_name}' 不在白名单中").to_str()

    try:
        p = _safe_path(working_dir)
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(p),
            capture_output=True,
            text=True,
            timeout=min(timeout, 300),  # 上限 5 分钟
            env={k: v for k, v in os.environ.items() if k not in (
                "AWS_SECRET", "OPENAI_KEY", "DEEPSEEK_KEY", "GITHUB_TOKEN", "SECRET"
            )},
        )
        output = []
        if result.stdout:
            output.append(f"[stdout]\n{result.stdout[:5000]}")
        if result.stderr:
            output.append(f"[stderr]\n{result.stderr[:5000]}")
        output.append(f"[退出码: {result.returncode}]")

        return ToolResult.ok("\n".join(output)).to_str()

    except subprocess.TimeoutExpired:
        return ToolResult.fail(f"命令超时（>{timeout}s）").to_str()
    except Exception as e:
        return ToolResult.fail(f"执行失败: {str(e)}").to_str()


@tool(
    name="run_test",
    description="运行测试命令，自动解析 pytest/unittest 结果。",
    danger_level=DangerLevel.SENSITIVE,
)
async def run_test(
    test_command: str = "pytest",
    working_dir: str = ".",
    verbose: bool = True,
) -> str:
    """运行测试并格式化输出。"""
    safe, reason = _check_command(test_command)
    if not safe:
        return ToolResult.fail(f"测试命令被拦截: {reason}").to_str()

    try:
        p = _safe_path(working_dir)
        extra = "-v" if verbose else ""
        result = subprocess.run(
            test_command,
            shell=True,
            cwd=str(p),
            capture_output=True,
            text=True,
            timeout=120,
        )
        summary = result.stdout or result.stderr
        passed = len(re.findall(r"PASSED", summary))
        failed = len(re.findall(r"FAILED", summary))
        return ToolResult.ok(
            f"测试完成，退出码 {result.returncode}。"
            f"通过 {passed}，失败 {failed}。"
            f"\n{summary[:3000]}"
        ).to_str()
    except subprocess.TimeoutExpired:
        return ToolResult.fail("测试超时（>120s）").to_str()
    except Exception as e:
        return ToolResult.fail(f"测试运行失败: {str(e)}").to_str()

# ── kill_process ────────────────────────────────────────────────────────

@tool(
    name="kill_process",
    description="终止指定 PID 的进程。",
    danger_level=DangerLevel.SENSITIVE,
)
async def kill_process(pid: int) -> str:
    """终止指定 PID 的进程。"""
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
        return ToolResult.ok(f"进程 {pid} 已终止（SIGTERM）").to_str()
    except ProcessLookupError:
        return ToolResult.fail(f"进程 {pid} 不存在").to_str()
    except PermissionError:
        return ToolResult.fail(f"无权限终止进程 {pid}").to_str()
    except Exception as e:
        return ToolResult.fail(f"终止进程 {pid} 失败: {str(e)}").to_str()


