"""
Git 工具集 — git_diff / git_log / git_status / git_checkout / git_commit / git_push。
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import tool, DangerLevel, ToolResult


def _run_git(args: List[str], cwd: str = ".") -> ToolResult:
    """执行 git 命令的通用封装。"""
    try:
        p = Path(cwd).resolve()
        result = subprocess.run(
            ["git"] + args,
            cwd=str(p),
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip() or result.stderr.strip() or f"[退出码 {result.returncode}]"
        if result.returncode != 0 and "fatal" in result.stderr.lower():
            return ToolResult.fail(f"Git 命令失败：{result.stderr.strip()}")
        return ToolResult.ok(output)
    except subprocess.TimeoutExpired:
        return ToolResult.fail("Git 命令超时")
    except FileNotFoundError:
        return ToolResult.fail("未找到 git 命令，请确保已安装 Git")
    except Exception as e:
        return ToolResult.fail(f"Git 执行错误：{str(e)}")


@tool(name="git_diff", description="查看工作区或暂存区的文件变更差异。", danger_level=DangerLevel.SAFE, read_only=True)
async def git_diff(
    path: str = ".",
    staged: bool = False,
    stat: bool = False,
    ignore_whitespace: bool = False,
) -> str:
    """
    显示文件变更。

    Args:
        path: 文件/目录路径（相对于仓库根目录）
        staged: 是否显示暂存区（--cached）
        stat: 仅显示统计信息
        ignore_whitespace: 忽略空白符差异
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    if stat:
        args.append("--stat")
    if ignore_whitespace:
        args.append("--ignore-all-space")
    if path and path != ".":
        args.append("--")
        args.append(path)

    return _run_git(args).to_str()


@tool(name="git_log", description="查看 Git 提交历史。", danger_level=DangerLevel.SAFE, read_only=True)
async def git_log(
    count: int = 10,
    path: str = "",
    format: str = "%h|%s|%an|%ad",
    since: str = "",
    until: str = "",
) -> str:
    """
    查看提交历史。

    Args:
        count: 显示最近 N 条（默认 10）
        path: 只显示涉及某文件的提交
        format: 格式（%h=hash, %s=subject, %an=author, %ad=date）
        since: 起始日期（YYYY-MM-DD）
        until: 截止日期
    """
    args = ["log", f"-n{count}", f"--format={format}"]
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")
    if path:
        args.append("--")
        args.append(path)

    res = _run_git(args)
    if not res.success:
        return res.to_str()

    # 格式化输出
    lines = res.data.split("\n") if res.data else []
    formatted = []
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 4:
            formatted.append(
                f"  `{parts[0]}`  {parts[1]}\n"
                f"       Author: {parts[2]}  Date: {parts[3]}"
            )
        elif line:
            formatted.append(f"  {line}")

    return ToolResult.ok(
        f"📜 最近 {len(lines)} 条提交记录：\n" + "\n\n".join(formatted)
        if formatted else "无提交记录"
    ).to_str()


@tool(name="git_status", description="查看仓库当前状态（修改/暂存/未跟踪文件）。", danger_level=DangerLevel.SAFE, read_only=True)
async def git_status(path: str = ".") -> str:
    """显示工作区状态。"""
    args = ["status", "--porcelain=v1", "-b"]
    if path and path != ".":
        args.append("--")
        args.append(path)

    res = _run_git(args)
    if not res.success:
        return res.to_str()

    lines = res.data.split("\n") if res.data else []
    modified, staged, untracked, ahead = [], [], [], []

    branch_line = ""
    for line in lines:
        if line.startswith("##"):
            branch_line = line
            continue
        if not line.strip():
            continue
        status_code = line[:2]
        file_path = line[3:]
        if status_code[0] in "MADRC" or status_code[1] in "MADRC":
            staged.append(f"  📝 {file_path} ({status_code})")
        elif status_code[0] == "?":
            untracked.append(f"  🆕 {file_path}")
        else:
            modified.append(f"  ✏️  {file_path} ({status_code})")

    # 解析分支
    branch = ""
    if branch_line:
        parts = branch_line.replace("## ", "").split("...")
        branch = f"📍 分支：`{parts[0]}`"
        if len(parts) > 1:
            branch += f" → `{parts[1]}`"

    parts_out = [branch]
    if staged:
        parts_out.append("\n✅ 暂存区（已 staged）：\n" + "\n".join(staged))
    if modified:
        parts_out.append("\n📝 工作区修改：\n" + "\n".join(modified))
    if untracked:
        parts_out.append("\n🆕 未跟踪文件：\n" + "\n".join(untracked))
    if not staged and not modified and not untracked:
        parts_out.append("\n✅ 工作区干净，无变更")

    return ToolResult.ok("\n".join(parts_out)).to_str()


@tool(name="git_checkout", description="切换 Git 分支或恢复文件。", danger_level=DangerLevel.SENSITIVE)
async def git_checkout(
    branch: str = "",
    file: str = "",
    create: bool = False,
) -> str:
    """
    切换分支或恢复文件。

    Args:
        branch: 目标分支名
        file: 恢复的文件路径（与 branch 二选一）
        create: 是否创建并切换到新分支
    """
    args = ["checkout"]
    if create:
        args.append("-b")
        if not branch:
            return ToolResult.fail("创建分支需要指定 branch 参数").to_str()
    if branch:
        args.append(branch)
    if file:
        args.append("--")
        args.append(file)

    if not branch and not file:
        return ToolResult.fail("至少需要指定 branch 或 file 参数").to_str()

    return _run_git(args).to_str()


@tool(name="git_commit", description="提交暂存区变更（需确认提交信息）。", danger_level=DangerLevel.SENSITIVE)
async def git_commit(
    message: str = "",
    amend: bool = False,
    no_edit: bool = False,
) -> str:
    """
    提交变更。

    Args:
        message: 提交信息（必填）
        amend: 修改上一次提交（不新建）
        no_edit: 复用上次的提交信息（需 amend=true）
    """
    if not message and not (amend and no_edit):
        return ToolResult.fail("提交信息 message 不能为空").to_str()

    args = ["commit"]
    if amend:
        args.append("--amend")
    if no_edit:
        args.append("--no-edit")
    args.append("-m")
    args.append(message)

    return _run_git(args).to_str()


@tool(name="git_push", description="推送本地分支到远程仓库。", danger_level=DangerLevel.SENSITIVE)
async def git_push(
    remote: str = "origin",
    branch: str = "",
    force: bool = False,
    set_upstream: bool = False,
) -> str:
    """
    推送到远程。

    Args:
        remote: 远程仓库名（默认 origin）
        branch: 分支名（默认当前分支）
        force: 强制推送（危险）
        set_upstream: 设置上游分支
    """
    args = ["push"]
    if force:
        args.append("--force-with-lease")  # 比 --force 更安全
    if set_upstream:
        args.append("-u")
    if remote:
        args.append(remote)
    if branch:
        args.append(branch)

    return _run_git(args).to_str()


@tool(name="git_branch", description="列出/创建/删除分支。", danger_level=DangerLevel.SENSITIVE)
async def git_branch(
    list: bool = True,
    delete: str = "",
    rename: str = "",
    current: bool = False,
) -> str:
    """
    分支管理。

    Args:
        list: 列出所有本地分支
        delete: 删除指定分支
        rename: 重命名当前分支（格式：old_name new_name）
        current: 显示当前分支名
    """
    if current:
        return _run_git(["branch", "--show-current"]).to_str()

    args = ["branch"]
    if delete:
        args.append("-d")
        args.append(delete)
    elif list:
        res = _run_git(["branch", "-a"])
        if not res.success:
            return res.to_str()
        lines = res.data.split("\n") if res.data else []
        formatted = []
        for line in lines:
            marker = "👉 " if line.startswith("*") else "  "
            branch_name = line.lstrip("* ").strip()
            remote_marker = "☁️ " if line.startswith("remotes/") else ""
            formatted.append(f"{marker}{remote_marker}{branch_name}")
        return ToolResult.ok("🌿 分支列表：\n" + "\n".join(formatted)).to_str()
    else:
        args.append("-a")
        args.append("--list")

    return _run_git(args).to_str()
