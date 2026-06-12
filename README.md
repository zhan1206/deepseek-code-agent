# DeepSeek Code Agent

基于 DeepSeek 模型的代码智能助手框架。

## 安装

```bash
pip install -e .
pip install -e '.[dev]'    # 开发依赖
pip install -e '.[all]'   # 全部依赖（包含沙箱、向量库）
```

## 快速开始

```bash
# 设置 API Key
export DEEPSEEK_API_KEY=sk-xxx

# 交互模式
deepseek-agent chat --project ./myproject

# 单任务
deepseek-agent run "修复所有 lint 错误"

# API 服务
deepseek-agent serve --port 8000
```

## Phase 进度

- ✅ Phase 1：核心框架（DeepSeekClient + 工具系统 + 记忆 + Agent 主循环）
- ✅ Phase 2：代码专用工具（Git + Web + Docker 沙箱 + FastAPI 服务）
- ⬜ Phase 3：沙箱与高级循环
- ⬜ Phase 4：生态与集成
