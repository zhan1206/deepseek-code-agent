"""Setup script for DeepSeek Code Agent."""
from setuptools import setup, find_packages

with open("README.md", "w", encoding="utf-8") as f:
    f.write("# DeepSeek Code Agent\n\n基于 DeepSeek 模型的代码智能助手框架。\n\n## 安装\n\n```bash\npip install -e .\npip install -e '.[dev]'    # 开发依赖\npip install -e '.[all]'   # 全部依赖（包含沙箱、向量库）\n```\n\n## 快速开始\n\n```bash\n# 设置 API Key\nexport DEEPSEEK_API_KEY=sk-xxx\n\n# 交互模式\ndeepseek-agent chat --project ./myproject\n\n# 单任务\ndeepseek-agent run \"修复所有 lint 错误\"\n\n# API 服务\ndeepseek-agent serve --port 8000\n```\n\n## Phase 进度\n\n- ✅ Phase 1：核心框架（DeepSeekClient + 工具系统 + 记忆 + Agent 主循环）\n- ✅ Phase 2：代码专用工具（Git + Web + Docker 沙箱 + FastAPI 服务）\n- ⬜ Phase 3：沙箱与高级循环\n- ⬜ Phase 4：生态与集成\n")

setup(
    name="deepseek-code-agent",
    version="2.0.0",
    description="基于 DeepSeek 模型的代码智能助手框架",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="DeepSeek Community",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "httpx>=0.27.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "core": [
            "httpx>=0.27.0",
            "colorama>=0.4.6",
        ],
        "dev": [
            "pytest>=8.0",
            "pytest-asyncio>=0.23.0",
            "ruff>=0.4.0",
        ],
        "memory": [
            "chromadb>=0.4.0",
        ],
        "knowledge": [
            "networkx>=3.0",
            "scikit-learn>=1.0",
            "watchdog>=3.0",
        ],
        "security": [
            "bandit>=1.7",
        ],
        "debug": [
            "debugpy>=1.8",
        ],
        "telemetry": [
            "opentelemetry-api>=1.20",
            "opentelemetry-sdk>=1.20",
        ],
        "sandbox": [
            "docker>=7.0",
        ],
        "server": [
            "fastapi>=0.110.0",
            "uvicorn>=0.29.0",
        ],
        "all": [
            "chromadb>=0.4.0",
            "docker>=7.0",
            "fastapi>=0.110.0",
            "uvicorn>=0.29.0",
            "pydantic>=2.0",
            "networkx>=3.0",
            "bandit>=1.7",
            "watchdog>=3.0",
            "debugpy>=1.8",
            "opentelemetry-api>=1.20",
            "opentelemetry-sdk>=1.20",
            "pytest>=8.0",
            "pytest-asyncio>=0.23.0",
            "pytest-benchmark>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepseek-agent=deepseek_agent.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
