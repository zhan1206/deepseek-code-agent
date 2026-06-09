# ============================================================
# DeepSeek Code Agent — Sandbox Docker Image
# ============================================================
# 用法：
#   构建：docker build -t deepseek-sandbox:latest .
#   运行：docker run --rm -v $(pwd):/workspace deepseek-sandbox:latest python test.py
# ============================================================

FROM python:3.12-slim

# ── 安全基础 ───────────────────────────────────────────────
RUN groupadd --gid 1000 sandbox && \
    useradd --uid 1000 --gid sandbox --shell /bin/bash --create-home sandbox

WORKDIR /workspace

# ── 系统依赖（根据常见语言扩展）────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    nodejs \
    npm \
    vim \
    ripgrep \
    fd-find \
    tree \
    jq \
    unzip \
    zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python 工具 ────────────────────────────────────────────
USER sandbox
ENV PATH="/home/sandbox/.local/bin:$PATH"
RUN python -m pip install --upgrade pip && \
    pip install --user \
        httpx \
        pytest \
        ruff

# ── 网络隔离（默认 deny）────────────────────────────────────
# 可通过 --network=host 覆盖
# 取消下行注释以强制无网络：
# RUN echo 'blacklist usb-storage' > /etc/modprobe.d/no-usb.conf || true

# ── 资源限制说明 ────────────────────────────────────────────
# 内存/CPU 由 Docker daemon 在 run 时通过 --memory / --cpus 限制
# 此镜像本身不硬编码限制

# ── 入口脚本 ────────────────────────────────────────────────
COPY --chown=sandbox:sandbox <<'EOF' /home/sandbox/run.sh
#!/bin/bash
# 沙箱入口脚本：默认 sleep infinity（AgentLoop 发送命令时激活）
# 也可直接执行传入的命令
if [ -n "$1" ]; then
    exec "$@"
else
    echo "[Sandbox] Ready. Waiting for commands..."
    sleep infinity
fi
EOF
chmod +x /home/sandbox/run.sh

ENTRYPOINT ["/home/sandbox/run.sh"]
