"""
可选隐私过滤器 — 默认关闭，过滤敏感信息再发送至 LLM。

Patterns:
- GitHub Token: ghp_[a-zA-Z0-9]{36}
- Generic API Key: [a-zA-Z0-9]{20,} matching "api[_-]?key|secret|token"
- Bearer Token: Bearer [a-zA-Z0-9_.~-]+
- Private Key: -----BEGIN (RSA | OPENSSH | EC | DSA | PKCS#8) PRIVATE KEY-----
- AWS Key: AKIA[0-9A-Z]{16}
- IP address (optional secondary concern)

默认关闭，避免误屏蔽正常的代码内容。
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

# 隐私模式全局开关（默认 False，工具侧不依赖，CLI/桌面端可按需启用）
_PRIVACY_ENABLED = False


def set_privacy_mode(enabled: bool) -> None:
    """全局启用/禁用隐私过滤。"""
    global _PRIVACY_ENABLED
    _PRIVACY_ENABLED = enabled


def is_privacy_enabled() -> bool:
    return _PRIVACY_ENABLED


# ── 检测正则 ────────────────────────────────────────────────────────────────

_PATTERNS = [
    # GitHub Personal Access Token
    (re.compile(r'ghp_[a-zA-Z0-9]{36}'), '[GITHUB_TOKEN]'),
    # Generic API Key / Secret / Token (≥20 字符，区分大小写避免误匹配 base64）
    (re.compile(r'(?<![a-zA-Z0-9])(api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token)\s*[:=]\s*["\']?([a-zA-Z0-9_~-]{20,})["\']?', re.I), r'\1: [REDACTED]'),
    # Bearer Token
    (re.compile(r'Bearer\s+([a-zA-Z0-9_.~-]{20,})'), 'Bearer [REDACTED]'),
    # RSA / SSH / EC / DSA / PKCS8 Private Key
    (re.compile(r'-----BEGIN\s+(?:RSA|OPENSSH|EC|DSA|PKCS#8)\s+PRIVATE KEY-----'), '[PRIVATE_KEY]'),
    # AWS Access Key ID
    (re.compile(r'AKIA[0-9A-Z]{16}'), '[AWS_KEY_ID]'),
]


def filter_sensitive(text: str) -> str:
    """
    扫描并替换敏感信息。默认关闭，调用 set_privacy_mode(True) 启用。
    返回清理后的文本。原始文本中敏感内容替换为占位符而非删除，以保持上下文完整性。
    """
    if not _PRIVACY_ENABLED:
        return text
    result = text
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def filter_messages(messages: List[dict]) -> List[dict]:
    """
    过滤对话消息列表中的敏感信息（用于 client.chat 调用前预处理）。
    返回新列表，不修改原始 messages。
    """
    if not _PRIVACY_ENABLED:
        return messages
    return [
        {**msg, "content": filter_sensitive(msg.get("content", ""))}
        for msg in messages
    ]
