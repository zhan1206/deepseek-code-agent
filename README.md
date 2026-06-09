# DeepSeek Code Agent

> 基于 DeepSeek 模型的代码智能助手框架

## 快速开始

`ash
export DEEPSEEK_API_KEY=sk-xxx
pip install -e .
deepseek-agent chat --project ./myproject
`

## Phase 进度

| Phase | 状态 | 内容 |
|-------|------|------|
| Phase 1 | DONE | 核心框架（DeepSeekClient + 工具系统 + 记忆 + AgentLoop） |
| Phase 2 | DONE | 代码专用工具（Git x7 + Web + Docker 沙箱 + FastAPI） |
| Phase 3 | DONE | 沙箱集成（Docker 镜像 + SandboxRunner）+ ChromaDB 向量记忆 + 检查点 |
| Phase 4 | DONE | MCP Server/Client + VS Code 插件 |

## 工具集（18+ 个）

FS: read_file / write_file / edit_file / list_directory / search_file / search_content / delete_file / run_shell / run_test
Git: git_diff / git_log / git_status / git_checkout / git_commit / git_push / git_branch
Web: web_fetch（6层SSRF防护）/ read_docs

## 架构

AgentLoop(ReAct+Plan-Execute) -> ToolRegistry -> 18 tools
                              -> MemoryManager(Short+Project+Long+Checkpoint)
                              -> DeepSeekClient(流式|reasoner)

## 沙箱

DockerSandbox: 非root+sandbox用户+资源限制+网络隔离
LocalSandbox: 子进程降级+白名单+超时

## MCP

MCP Server: python -m deepseek_agent.mcp.server (stdio模式)
MCP Client: 接入外部MCP Server

## VS Code

cd vscode && vsce package && code --install-extension deepseek-agent.vsix
