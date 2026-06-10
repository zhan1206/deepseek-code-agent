"""
Web / 文档工具 — web_fetch / read_docs。
含 SSRF 防护（内网 IP / DNS 重绑定过滤）。
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import urllib.parse
from typing import Any, Dict, Optional, Set

import httpx

from .base import tool, DangerLevel, ToolResult


# ── SSRF 防护 ─────────────────────────────────────────────────────────────

# 保留/内网 IP 段
BLOCKED_IP_RANGES: Set[str] = {
    "127.0.0.0/8",      # Loopback
    "10.0.0.0/8",       # 私有
    "172.16.0.0/12",    # 私有
    "192.168.0.0/16",   # 私有
    "169.254.0.0/16",   # 链路本地
    "0.0.0.0/8",        # 当前网络
    "100.64.0.0/10",    # 运营商级 NAT
    "192.0.0.0/24",     # IETF 协议保留
    "192.0.2.0/24",     # TEST-NET-1
    "198.51.100.0/24",  # TEST-NET-2
    "203.0.113.0/24",   # TEST-NET-3
    "224.0.0.0/4",      # 多播
    "240.0.0.0/4",      # 保留
    "::1/128",          # IPv6 loopback
    "fc00::/7",         # IPv6 私有
    "fe80::/10",        # IPv6 链路本地
}

BLOCKED_HOSTNAMES = {
    "localhost", "127.0.0.1", "0.0.0.0",
    "169.254.169.254",  # 云元数据端点
    "metadata.google.internal",
    "metadata.azure.com",
    "100.100.100.100",  # 腾讯云
}


def _is_ip_blocked(ip_str: str) -> bool:
    """检查 IP 是否属于内网/保留段。"""
    try:
        ip = ipaddress.ip_address(ip_str)
        for net_str in BLOCKED_IP_RANGES:
            net = ipaddress.ip_network(net_str, strict=False)
            if ip in net:
                return True
    except ValueError:
        pass
    return False


def _resolve_and_check(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """
    解析域名并检查 IP 是否安全。

    Returns:
        (is_safe, resolved_ip_or_error)
    """
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False, "无法解析主机名"

        # 阻塞名单
        if hostname.lower() in BLOCKED_HOSTNAMES:
            return False, f"主机名在黑名单中: {hostname}"

        # DNS 解析
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        if not results:
            return False, f"无法解析域名: {hostname}"

        for family, _, _, _, sockaddr in results:
            ip_str = sockaddr[0]
            if _is_ip_blocked(ip_str):
                return False, f"解析到内网 IP: {ip_str}（原始域名: {hostname}）"

        # 取第一个安全 IP
        return True, results[0][4][0]

    except socket.gaierror as e:
        return False, f"DNS 解析失败: {e}"
    except Exception as e:
        return False, f"URL 检查失败: {e}"


def _validate_url(url: str) -> tuple[bool, str]:
    """完整 URL 安全校验。"""
    try:
        parsed = urllib.parse.urlparse(url)

        # 只允许 http/https
        if parsed.scheme not in ("http", "https"):
            return False, f"仅支持 http/https，不支持 {parsed.scheme}"

        if not parsed.netloc:
            return False, "URL 缺少主机名"

        # 端口黑名单
        BLOCKED_PORTS = {22, 23, 25, 445, 3389, 5900}
        if parsed.port and parsed.port in BLOCKED_PORTS:
            return False, f"端口 {parsed.port} 被禁止"

        # 检查解析结果
        return _resolve_and_check(url)

    except Exception as e:
        return False, f"URL 格式错误: {e}"


# ── 内容提取 ──────────────────────────────────────────────────────────────

def _extract_text(html: str, max_length: int = 8000) -> str:
    """从 HTML 中提取纯文本。"""
    # 移除 script/style
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 转 HTML 实体
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#39;", "'").replace("&quot;", '"').replace("&amp;", "&")
    # 移除标签
    text = re.sub(r"<[^>]+>", "", text)
    # 合并空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_length]


# ── 工具实现 ──────────────────────────────────────────────────────────────

@tool(
    name="web_fetch",
    description="获取网页文本内容（需防 SSRF，限制内网 IP）",
    danger_level=DangerLevel.SENSITIVE,
)
async def web_fetch(
    url: str,
    max_length: int = 8000,
    timeout: int = 15,
) -> str:
    """
    获取网页内容并提取纯文本。

    Args:
        url: 目标 URL
        max_length: 最大返回字符数
        timeout: 请求超时（秒）
    """
    # 安全校验
    safe, detail = _validate_url(url)
    if not safe:
        return ToolResult.fail(f"URL 安全检查失败: {detail}").to_str()

    # 额外的 HTTP 校验（检查最终解析 IP）
    try:
        parsed = urllib.parse.urlparse(url)
        safe2, resolved_ip = _resolve_and_check(url, timeout=5.0)
        if not safe2:
            return ToolResult.fail(f"URL 解析到禁止地址: {resolved_ip}").to_str()
    except Exception as e:
        return ToolResult.fail(f"URL 校验异常: {e}").to_str()

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; DeepSeek-Agent/1.0)",
                "Accept": "text/html,application/xhtml+xml,text/plain",
            },
        ) as client:
            resp = await client.get(url)

        if resp.status_code != 200:
            return ToolResult.fail(
                f"HTTP {resp.status_code}: {url}"
            ).to_str()

        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type:
            text = _extract_text(resp.text, max_length)
        else:
            text = resp.text[:max_length]

        return ToolResult.ok(
            f"URL: {url}\n"
            f"状态码: {resp.status_code}\n"
            f"内容类型: {content_type}\n"
            f"长度: {len(text)} 字符\n\n"
            f"{'='*60}\n"
            f"{text}"
        ).to_str()

    except httpx.TimeoutException:
        return ToolResult.fail(f"请求超时（>{timeout}s）").to_str()
    except httpx.InvalidURL as e:
        return ToolResult.fail(f"无效 URL: {e}").to_str()
    except Exception as e:
        return ToolResult.fail(f"请求失败: {str(e)}").to_str()


@tool(
    name="read_docs",
    description="查询离线文档（devdocs.io 类），返回相关章节",
    danger_level=DangerLevel.SAFE,
    read_only=True,
)
async def read_docs(
    library: str,
    query: str = "",
    version: str = "",
) -> str:
    """
    模拟离线文档查询。

    Phase 1 简单实现：返回指向官方文档的 URL 建议。
    Phase 3 可接入 devdocs.io API 或本地文档索引。

    Args:
        library: 库名（如 "python", "numpy", "react"）
        query: 查询主题
        version: 版本号
    """
    DOC_BASE = {
        "python": "https://docs.python.org/3/",
        "numpy": "https://numpy.org/doc/stable/",
        "pandas": "https://pandas.pydata.org/docs/",
        "django": "https://docs.djangoproject.com/en/stable/",
        "flask": "https://flask.palletsprojects.com/en/stable/",
        "fastapi": "https://fastapi.tiangolo.com/",
        "react": "https://react.dev/reference/",
        "typescript": "https://www.typescriptlang.org/docs/",
        "rust": "https://doc.rust-lang.org/book/",
        "go": "https://go.dev/doc/",
        "pytorch": "https://pytorch.org/docs/stable/",
        "tensorflow": "https://www.tensorflow.org/api_docs/python/",
    }

    lib_lower = library.lower()
    base = DOC_BASE.get(lib_lower, f"https://duckduckgo.com/?q={library}+docs")

    if query:
        # 简单 URL 拼接（实际应接入搜索 API）
        return ToolResult.ok(
            f"📖 文档查询结果：\n"
            f"库: {library}\n"
            f"版本: {version or 'latest'}\n"
            f"查询: {query}\n\n"
            f"🔗 建议文档链接：\n"
            f"{base}\n\n"
            f"💡 如需详细内容，请使用 web_fetch 工具获取上述页面。"
        ).to_str()

    return ToolResult.ok(
        f"📚 {library} 文档\n"
        f"版本: {version or 'latest'}\n"
        f"🔗 {base}"
    ).to_str()
