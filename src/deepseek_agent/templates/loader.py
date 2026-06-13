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
import time
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


class ExperimentTracker:
    """A/B 实验结果追踪器，记录每次实验的模板版本和效果指标。"""

    def __init__(self):
        self._results: List[Dict[str, Any]] = []

    def record(self, prompt_id: str, variant: str, score: float, metadata: Optional[Dict] = None) -> None:
        """记录一次实验结果。"""
        self._results.append({
            "prompt_id": prompt_id,
            "variant": variant,
            "score": score,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })

    def summary(self, prompt_id: str) -> Dict[str, Any]:
        """获取指定 prompt 的实验汇总（各版本平均分和样本数）。"""
        relevant = [r for r in self._results if r["prompt_id"] == prompt_id]
        if not relevant:
            return {}
        by_variant: Dict[str, List[float]] = {}
        for r in relevant:
            by_variant.setdefault(r["variant"], []).append(r["score"])
        return {
            v: {"count": len(scores), "avg_score": round(sum(scores) / len(scores), 3)}
            for v, scores in by_variant.items()
        }

    def top_variant(self, prompt_id: str) -> Optional[str]:
        """返回平均分最高的变体。"""
        s = self.summary(prompt_id)
        if not s:
            return None
        return max(s, key=lambda v: s[v]["avg_score"])


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
        self._experiment_tracker = ExperimentTracker()
        self._variant_weights: Dict[str, Dict[str, float]] = {}  # {prompt_id: {variant: weight}}

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

    def set_weights(self, prompt_id: str, weights: Dict[str, float]) -> None:
        """设置 A/B 实验变体权重分布（weights 之和不必为 1，自动归一化）。"""
        self._variant_weights[prompt_id] = weights

    def experiment(self, prompt_id: str, variants: List[str]) -> Template:
        """
        A/B 实验：根据权重随机选择模板变体（默认等权重）。

        支持格式：
        - prompt_id = "code-review/v2" → 加载特定版本
        - prompt_id = "code-review" → 按权重选择 variants 中的版本
        - 未设置权重时使用均匀分布

        使用步骤：
        1. 可选：loader.set_weights("code-review", {"v1": 0.7, "v2": 0.3})
        2. 在 templates/ 下创建 code-review/v1.md, code-review/v2.md
        3. 调用 loader.experiment("code-review", ["v1", "v2"])
        4. 获取结果后调用 loader.record_outcome(prompt_id, chosen_variant, score)
        """
        import random
        if not variants:
            return self.get(prompt_id) or Template(name=prompt_id)

        weights = self._variant_weights.get(prompt_id, {})
        if weights:
            # 加权随机选择
            w_list = [weights.get(v, 0.0) for v in variants]
            total = sum(w_list)
            if total == 0:
                w_list = [1.0] * len(variants)
                total = float(len(variants))
            normalized = [w / total for w in w_list]
            chosen = random.choices(variants, weights=normalized, k=1)[0]
        else:
            chosen = random.choice(variants)

        versioned_id = f"{prompt_id}/{chosen}"
        tpl = self.get(versioned_id)
        if tpl:
            return tpl
        return self.get(prompt_id) or Template(name=prompt_id, version=chosen)

    def record_outcome(self, prompt_id: str, variant: str, score: float, metadata: Optional[Dict] = None) -> None:
        """记录一次 A/B 实验结果。"""
        self._experiment_tracker.record(prompt_id, variant, score, metadata)

    def experiment_summary(self, prompt_id: str) -> Dict[str, Any]:
        """获取指定 prompt 的 A/B 实验汇总。"""
        return self._experiment_tracker.summary(prompt_id)

    def top_variant(self, prompt_id: str) -> Optional[str]:
        """返回平均分最高的实验变体。"""
        return self._experiment_tracker.top_variant(prompt_id)

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
