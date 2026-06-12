@echo off
REM 启动 DeepSeek Code Agent 后端
SET PYTHON="C:\Users\朱子瞻\AppData\Local\Programs\Python\Python312\python.exe"
SET BACKEND_DIR=D:\deepseek-code-agent\src
SET PYTHONPATH=%BACKEND_DIR%

%PYTHON% -m deepseek_agent.server.app
