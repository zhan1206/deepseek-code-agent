"""Prompt & Task Template System (v2.0 §5.4).

YAML + Markdown templates with prompt chains, tool constraints, and context templates.
Template repo: ~/.deepseek-agent/templates/
"""

from .loader import TemplateLoader, Template, TemplateChain

__all__ = ["TemplateLoader", "Template", "TemplateChain"]
