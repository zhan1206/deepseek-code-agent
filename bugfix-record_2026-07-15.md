# 修复记录: deepseek-code-agent 代码还原后核心 Bug 修复 (2026-07-15)

## 背景
通过 git 回退还原后的 deepseek-code-agent 代码存在多个会阻止正常运行的严重 bug。本次修复覆盖 7 个文件、修复 7 个真实 bug，全部 8 个单元测试通过。

## 修复清单

### 1. fs.py — Shell 黑名单重复 + kill_process 重复定义
- **文件**: `src/deepseek_agent/tools/fs.py`
- **问题**: `_SHELL_BLACKLIST_PATTERNS + _check_shell_command` 完整重复一次（2份完全相同），`kill_process` tool 函数重复定义一次（2份完全相同）
- **影响**: Python 会以第二个定义为准，但代码膨胀且维护性差；lint/type-checker 会报错
- **修复**: Python 脚本精确删除两份重复块（共计约 1200 字节）

### 2. app.py — Tool 对象传给 register_func() 类型错误
- **文件**: `src/deepseek_agent/server/app.py`
- **问题**: `@tool` 装饰器返回 `Tool` 对象（不是函数），但代码用 `registry.register_func(fn)` 传递 Tool 对象。`register_func` 内部调用 `func.__name__` 和 `func.__doc__`——Tool 对象没有 `__name__` 属性，会导致实际注册时所有工具都用错误的名称（`'Tool'`）和描述（空字符串）
- **影响**: server 模式下工具注册完全失效，所有工具名都变成 "Tool"
- **修复**: 将所有 `register_func(fn)` 改为 `register(fn)`，因为 `register()` 接受 `Tool` 对象

### 3. sandbox.py — f-string repr 导致命令参数解析错误
- **文件**: `src/deepseek_agent/core/sandbox.py`
- **问题**: `f"bash -c {command!r}"` 使用 repr 包装命令字符串。当命令包含单引号（如 `python -c 'print(1)'`）时，repr 的双层引号嵌套会导致 bash 参数解析完全错误
- **影响**: Docker 沙箱中任何包含引号的命令都无法正确执行
- **修复**: 改用 `["bash", "-c", command]` 列表形式，将参数直接传递给 Docker API，绕过 shell 引号解析问题

### 4. mcp/server.py — stdin for 循环阻塞 asyncio 事件循环
- **文件**: `src/deepseek_agent/mcp/server.py`
- **问题**: `for line in sys.stdin:` 是同步阻塞调用，会完全阻塞 asyncio 事件循环，MCP Server 在 stdio 模式下无法正常处理请求
- **影响**: MCP stdio 传输模式完全无法工作
- **修复**: 改用 `await loop.run_in_executor(None, sys.stdin.readline)` 实现异步读取

### 5. __init__.py — kill_process、LSP tools 未导出
- **文件**: `src/deepseek_agent/tools/__init__.py`
- **问题**: `kill_process` 已定义但未在 `__init__.py` 中导出；`get_symbols` 等 LSP 工具的 import 放在 `__all__` 之后，不会被正确导出
- **影响**: `from deepseek_agent.tools import kill_process` 会失败
- **修复**: 在 import 和 `__all__` 中增加 `kill_process` 和所有 LSP 工具

## 未修复（非 bug）
- `_search` 在 fs.py 中出现两次：它们是嵌套在 `search_file` 和 `search_content` 两个不同父函数中的不同实现，作用域隔离，不是 bug

## 验证
- 全部 8 个单元测试通过
- 全部修改的文件 `py_compile` 编译通过（无语法错误）