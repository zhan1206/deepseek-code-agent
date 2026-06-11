"""Template management tools (v2.0 §5.4).

These tools are registered lazily to avoid circular imports between
templates.tools → tools.base → tools.__init__ → templates.tools.
"""

from __future__ import annotations

from typing import List, Optional

from .loader import TemplateLoader, Template, TemplateChain


# --- Tool functions (plain callables, decorated at registration time) ---

def _list_templates(category: Optional[str] = None):
    from ..tools.base import ToolResult
    loader = TemplateLoader()
    templates = loader.list_templates(category=category)
    if not templates:
        return ToolResult(success=True, data={"templates": [], "count": 0})
    items = [
        {"name": t.name, "description": t.description, "category": t.category, "steps": len(t.chain.steps)}
        for t in templates
    ]
    return ToolResult(success=True, data={"templates": items, "count": len(items)})


def _get_template(name: str):
    from ..tools.base import ToolResult
    loader = TemplateLoader()
    tpl = loader.get(name)
    if not tpl:
        return ToolResult(success=False, error=f"Template '{name}' not found")
    return ToolResult(success=True, data=tpl.to_dict())


def _render_template(name: str, **variables: str):
    from ..tools.base import ToolResult
    loader = TemplateLoader()
    tpl = loader.get(name)
    if not tpl:
        return ToolResult(success=False, error=f"Template '{name}' not found")
    rendered = tpl.render(**variables)
    return ToolResult(success=True, data={"name": name, "rendered": rendered, "chain": tpl.chain.to_dict()})


def _save_template(
    name: str,
    body: str,
    description: str = "",
    category: str = "general",
    tools: Optional[List[str]] = None,
    context_budget: int = 4096,
):
    from ..tools.base import ToolResult
    tpl = Template(
        name=name,
        description=description,
        category=category,
        tools=tools or [],
        context_budget=context_budget,
        chain=TemplateChain(),
        body=body,
    )
    loader = TemplateLoader()
    path = loader.save(tpl)
    return ToolResult(success=True, data={"name": name, "path": str(path)})


def _delete_template(name: str):
    from ..tools.base import ToolResult
    loader = TemplateLoader()
    deleted = loader.delete(name)
    if not deleted:
        return ToolResult(success=False, error=f"Template '{name}' not found")
    return ToolResult(success=True, data={"deleted": name})


# --- Registry helper ---

def register_template_tools(registry):
    """Register all template tools with a ToolRegistry (call after both packages are loaded)."""
    from ..tools.base import tool, DangerLevel

    list_templates = tool(
        name="list_templates",
        description="List available prompt & task templates, optionally filtered by category",
        danger_level=DangerLevel.SAFE,
    )(_list_templates)

    get_template = tool(
        name="get_template",
        description="Get a specific template by name with full details including prompt chain",
        danger_level=DangerLevel.SAFE,
    )(_get_template)

    render_template = tool(
        name="render_template",
        description="Render a template with variable substitutions, returning the prompt text",
        danger_level=DangerLevel.SAFE,
    )(_render_template)

    save_template = tool(
        name="save_template",
        description="Create or update a prompt & task template",
        danger_level=DangerLevel.MODERATE,
    )(_save_template)

    delete_template = tool(
        name="delete_template",
        description="Delete a prompt & task template",
        danger_level=DangerLevel.SENSITIVE,
    )(_delete_template)

    for fn in [list_templates, get_template, render_template, save_template, delete_template]:
        registry.register(fn)
