"""
DeepSeek Code Agent CLI 入口 — 支持交互模式、单任务模式、API 服务模式。
"""

from __future__ import annotations

import asyncio
import argparse
import os
import sys
import json
import colorama

from .core.client import DeepSeekClient
from .tools import (
    ToolRegistry,
    read_file, write_file, edit_file, list_directory,
    search_file, search_content, delete_file,
    run_shell, run_test,
    git_diff, git_log, git_status, git_checkout,
    git_commit, git_push, git_branch,
    web_fetch, read_docs,
)
from .memory.manager import MemoryManager
from .agent.loop import AgentLoop, LoopConfig, LoopMode, CLIApprovalCallback


def _colorize(content: str, role: str) -> str:
    colors = {
        "think": colorama.Fore.CYAN,
        "tool": colorama.Fore.YELLOW,
        "tool_result": colorama.Fore.MAGENTA,
        "final": colorama.Fore.GREEN,
        "error": colorama.Fore.RED,
        "plan": colorama.Fore.BLUE,
        "header": colorama.Fore.WHITE + colorama.Style.BRIGHT,
    }
    return f"{colors.get(role, '')}{content}{colorama.Style.RESET_ALL}"


def _register_all_tools(registry: ToolRegistry):
    """注册全部工具（工具已由 @tool 装饰器包装为 Tool 对象）。"""
    for tool_obj in [
        # 文件系统
        read_file, write_file, edit_file, list_directory,
        search_file, search_content, delete_file,
        run_shell, run_test,
        # Git
        git_diff, git_log, git_status, git_checkout,
        git_commit, git_push, git_branch,
        # Web
        web_fetch, read_docs,
    ]:
        registry.register(tool_obj)


async def interactive_mode(
    api_key: str,
    project_path: str | None,
    model: str,
    mode: str,
):
    """交互式聊天模式。"""
    colorama.init(autoreset=True)

    client = DeepSeekClient(api_key=api_key, model=model)
    registry = ToolRegistry()
    _register_all_tools(registry)
    cwd = project_path or os.getcwd()

    memory = MemoryManager(
        project_path=cwd,
        long_term_dir=os.path.expanduser("~/.deepseek_agent_memory"),
    )

    loop_config = LoopConfig(
        mode=LoopMode.PLAN_EXECUTE if mode == "plan" else LoopMode.REACT,
        max_steps=50,
        max_execution_time=600,
    )

    agent = AgentLoop(
        client=client,
        registry=registry,
        memory=memory,
        config=loop_config,
        permission_callback=CLIApprovalCallback(),
    )

    print(_colorize("🧠 DeepSeek Code Agent 已启动（Phase 2）", "header"))
    print(_colorize(f"📁 项目目录: {cwd}", "plan"))
    print(_colorize(f"🔧 模式: {mode}  |  工具数: {len(registry.list_tools())}", "plan"))
    print(_colorize("💡 可用工具: 文件/Git/Web/Shell 全套\n", "tool"))
    print("输入你的问题，Ctrl+C 退出\n")

    try:
        while True:
            try:
                task = input("👤 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not task:
                continue

            print("\n" + "─" * 60)

            async for resp in agent.run(task):
                if resp.thinking:
                    print(_colorize(f"\n🤔 思考\n{resp.thinking[:600]}", "think"))
                elif resp.tool_calls:
                    for tc in resp.tool_calls:
                        args_str = json.dumps(tc.arguments, ensure_ascii=False)[:200]
                        print(_colorize(f"\n🔧 {tc.name}({args_str})", "tool"))
                elif resp.content:
                    if "计划已生成" in resp.content or "📋" in resp.content:
                        print(_colorize(f"\n{resp.content}", "plan"))
                    else:
                        print(_colorize(f"\n{resp.content}", "final"))

            print("\n" + "─" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n已退出。")
    finally:
        await client.close()


async def run_task(
    api_key: str,
    task: str,
    project_path: str | None,
    model: str,
    mode: str,
):
    """单任务模式。"""
    colorama.init(autoreset=True)

    client = DeepSeekClient(api_key=api_key, model=model)
    registry = ToolRegistry()
    _register_all_tools(registry)
    cwd = project_path or os.getcwd()

    memory = MemoryManager(
        project_path=cwd,
        long_term_dir=os.path.expanduser("~/.deepseek_agent_memory"),
    )

    loop_config = LoopConfig(
        mode=LoopMode.PLAN_EXECUTE if mode == "plan" else LoopMode.REACT,
        max_steps=30,
        max_execution_time=300,
    )
    agent = AgentLoop(
        client=client, registry=registry, memory=memory, config=loop_config
    )

    async for resp in agent.run(task):
        if resp.content:
            print(resp.content)

    await client.close()




def _do_setup():
    """下载 DeepSeek tokenizer.json 并完成初始化。"""
    import urllib.request

    tokenizers_dir = os.path.expanduser("~/.deepseek-agent")
    os.makedirs(tokenizers_dir, exist_ok=True)
    target = os.path.join(tokenizers_dir, "tokenizer.json")

    urls = [
        "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-LLM/main/tokenizer_config.json",
        "https://raw.githubusercontent.com/deepseek-ai/deepseek-mcp/main/tokenizer.json",
    ]

    print("Downloading DeepSeek tokenizer...")
    for attempt_url in urls:
        try:
            with urllib.request.urlopen(attempt_url, timeout=15) as resp:
                data = resp.read()
            with open(target, "wb") as f:
                f.write(data)
            size = os.path.getsize(target)
            print(f"Saved {target} ({size} bytes)")
            try:
                from tokenizers import Tokenizer
                enc = Tokenizer.from_file(target)
                test = enc.encode("hello world")
                print(f"Tokenizer verified ({len(test)} tokens)")
            except Exception as ve:
                print(f"Warning: tokenizer verification failed ({ve}), using fallback")
            return
        except Exception as e:
            print(f"Failed to download from {attempt_url}: {e}")
            continue
    print("Setup failed. TokenCounter will use fallback (no functionality loss).")


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek Code Agent — 基于 DeepSeek 模型的代码智能助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  交互模式：   deepseek-agent chat --project ./myproject
  单任务：     deepseek-agent run "修复所有 lint 错误"
  API 服务：   deepseek-agent serve --port 8000
  环境变量：   export DEEPSEEK_API_KEY=sk-xxx
        """,
    )
    parser.add_argument("task", nargs="?", help="单次任务（省略则进入交互模式）")
    parser.add_argument("--api-key", "-k", default=os.environ.get("DEEPSEEK_API_KEY", ""),
                        help="DeepSeek API Key")
    parser.add_argument("--project", "-p", default=".",
                        help="项目根目录（默认当前目录）")
    parser.add_argument("--model", "-m", default="deepseek-chat",
                        choices=["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
                        help="使用的模型")
    parser.add_argument("--mode", default="react",
                        choices=["react", "plan"],
                        help="Agent 模式：react（推理+行动）或 plan（规划+执行）")
    parser.add_argument("run", nargs="?", help=argparse.SUPPRESS)  # 别名
    parser.add_argument("serve", nargs="?", help=argparse.SUPPRESS)  # 别名
    parser.add_argument("--setup", action="store_true", help="下载 DeepSeek tokenizer.json 并完成初始化")
    parser.add_argument("--port", type=int, default=8000, help="API 服务端口（serve 模式）")
    parser.add_argument("--host", default="0.0.0.0", help="API 服务地址")

    args = parser.parse_args()

    # setup 子命令：下载 tokenizer.json
    if args.setup:
        _do_setup()
        return

    if not args.api_key:
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量或通过 --api-key 参数指定", file=sys.stderr)
        sys.exit(1)

    # serve 模式
    if args.serve:
        from .server import run_server
        print(f"🚀 启动 API 服务：http://{args.host}:{args.port}")
        run_server(host=args.host, port=args.port)
        return

    # 单任务模式（run 子命令或位置参数）
    is_run = args.run or (args.task and args.task == "run")
    task_arg = args.task if not is_run else None

    if is_run or args.task:
        # 找实际任务文本
        actual_task = args.task
        if is_run:
            # 从 sys.argv 重新解析（太复杂，用简化策略）
            actual_task = " ".join(sys.argv[2:] if args.task == "run" else sys.argv[1:])
            actual_task = actual_task.strip().strip('"')

        asyncio.run(run_task(args.api_key, actual_task or args.task, args.project, args.model, args.mode))
    else:
        asyncio.run(interactive_mode(args.api_key, args.project, args.model, args.mode))


if __name__ == "__main__":
    main()
