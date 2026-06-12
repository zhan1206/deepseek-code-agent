"""Template loader and manager.

Template format (YAML front-matter + Markdown body):
---
name: code-review
description: Perform a thorough code review
category: review
tools: [read_file, search_files, git_diff]
context_budget: 4096
chain:
  - step: gather
    prompt: "Read the changed files and gather context"
    tools: [read_file, git_diff]
  - step: analyze
    prompt: "Analyze for bugs, style, and performance issues"
    tools: [search_files]
  - step: report
    prompt: "Write the review summary"
---
You are a code reviewer. Analyze the following changes...

$git_diff
$changed_files
"""

from __future__ import annotations

import os
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_TEMPLATE_DIR = Path.home() / ".deepseek-agent" / "templates"
BUILTIN_TEMPLATE_DIR = Path(__file__).parent / "builtin"


@dataclass
class TemplateStep:
    """A single step in a prompt chain."""
    step: str
    prompt: str
    tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "prompt": self.prompt, "tools": self.tools}


@dataclass
class TemplateChain:
    """Ordered sequence of steps in a template."""
    steps: List[TemplateStep] = field(default_factory=list)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.steps]


@dataclass
class Template:
    """A prompt & task template with version support."""
    name: str
    version: str = "1.0.0"          # SemVer 版本号
    description: str = ""
    category: str = "general"
    tools: List[str] = field(default_factory=list)
    context_budget: int = 4096
    chain: TemplateChain = field(default_factory=TemplateChain)
    body: str = ""  # Markdown body with $variable placeholders
    source: str = ""  # File path the template was loaded from
    variables: List[Dict] = field(default_factory=list)  # 定义模板变量 schema

    def render(self, **kwargs: str) -> str:
        """Render the template body, substituting $variable placeholders."""
        result = self.body
        for key, value in kwargs.items():
            result = result.replace(f"${key}", str(value))
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "tools": self.tools,
            "context_budget": self.context_budget,
            "chain": self.chain.to_dict(),
            "variables": self.variables,
            "body": self.body,
            "source": self.source,
        }


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_template(text: str, source: str = "") -> Template:
    """Parse a YAML front-matter + Markdown template string."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        # No front-matter: treat entire content as body with name from filename
        name = Path(source).stem if source else "unnamed"
        return Template(name=name, body=text.strip(), source=source)

    meta = yaml.safe_load(match.group(1)) or {}
    body = text[match.end():].strip()

    chain = TemplateChain()
    for step_data in meta.get("chain", []):
        chain.steps.append(TemplateStep(
            step=step_data.get("step", ""),
            prompt=step_data.get("prompt", ""),
            tools=step_data.get("tools", []),
        ))

    return Template(
        name=meta.get("name", Path(source).stem if source else "unnamed"),
        version=meta.get("version", "1.0.0"),
        description=meta.get("description", ""),
        category=meta.get("category", "general"),
        tools=meta.get("tools", []),
        context_budget=meta.get("context_budget", 4096),
        chain=chain,
        body=body,
        source=source,
        variables=meta.get("variables", []),
    )


class TemplateLoader:
    """Load and manage templates from the template directory."""

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or DEFAULT_TEMPLATE_DIR
        self._cache: Dict[str, Template] = {}

    def _ensure_dir(self) -> None:
        self.template_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> Dict[str, Template]:
        """Load all templates: built-in first, then user overrides."""
        self._cache.clear()
        # Load built-in templates
        if BUILTIN_TEMPLATE_DIR.is_dir():
            for f in sorted(BUILTIN_TEMPLATE_DIR.rglob("*.md")):
                try:
                    text = f.read_text(encoding="utf-8")
                    tpl = _parse_template(text, source=str(f))
                    tpl.source = "builtin:" + str(f.relative_to(BUILTIN_TEMPLATE_DIR))
                    self._cache[tpl.name] = tpl
                except Exception:
                    continue
        # Load user templates (override built-ins with same name)
        self._ensure_dir()
        for f in sorted(self.template_dir.rglob("*.md")):
            try:
                text = f.read_text(encoding="utf-8")
                tpl = _parse_template(text, source=str(f))
                self._cache[tpl.name] = tpl
            except Exception:
                continue
        return self._cache

    def experiment(self, prompt_id: str, variants: List[str]) -> Template:
        """
        A/B 实验：随机选择一个模板变体用于渲染。

        支持格式：
        - prompt_id = "code-review/v2" → 加载特定版本
        - prompt_id = "code-review" → 随机选择 variants 中的版本

        使用步骤：
        1. 在 templates/ 下创建 code-review/v1.md, code-review/v2.md
        2. 调用 loader.experiment("code-review", ["v1", "v2"])
        3. 渲染并记录 telemetry 指标
        """
        import random
        if not variants:
            return self.get(prompt_id) or Template(name=prompt_id)
        chosen = random.choice(variants)
        versioned_id = f"{prompt_id}/{chosen}"
        tpl = self.get(versioned_id)
        if tpl:
            return tpl
        # 回退到基础模板
        return self.get(prompt_id) or Template(name=prompt_id, version=chosen)

    def get(self, name: str) -> Optional[Template]:
        """Get a template by name (loads all if not cached)."""
        if not self._cache:
            self.load_all()
        return self._cache.get(name)

    def list_templates(self, category: Optional[str] = None) -> List[Template]:
        """List available templates, optionally filtered by category."""
        if not self._cache:
            self.load_all()
        templates = list(self._cache.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def list_categories(self) -> List[str]:
        """List all template categories."""
        if not self._cache:
            self.load_all()
        return sorted(set(t.category for t in self._cache.values()))

    def save(self, template: Template) -> Path:
        """Save a template to the template directory."""
        self._ensure_dir()
        filename = template.name.replace("/", "_").replace("\\", "_") + ".md"
        filepath = self.template_dir / template.category / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build YAML front-matter
        meta: Dict[str, Any] = {
            "name": template.name,
            "version": template.version,
            "description": template.description,
            "category": template.category,
            "tools": template.tools,
            "context_budget": template.context_budget,
            "variables": template.variables,
        }
        if template.chain.steps:
            meta["chain"] = [s.to_dict() for s in template.chain.steps]

        content = f"---\n{yaml.dump(meta, allow_unicode=True, default_flow_style=False).strip()}\n---\n\n{template.body}\n"
        filepath.write_text(content, encoding="utf-8")
        template.source = str(filepath)
        self._cache[template.name] = template
        return filepath

    def delete(self, name: str) -> bool:
        """Delete a template by name."""
        tpl = self.get(name)
        if not tpl:
            return False
        if tpl.source and os.path.isfile(tpl.source):
            os.remove(tpl.source)
        self._cache.pop(name, None)
        return True
