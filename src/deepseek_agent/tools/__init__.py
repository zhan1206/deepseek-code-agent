"""Tools package."""
from .base import Tool, ToolRegistry, ToolResult, DangerLevel, tool, generate_schema
from .fs import (
    read_file, write_file, edit_file, list_directory,
    search_file, search_content, delete_file,
    run_shell, run_test, kill_process,
)
from .git import (
    git_diff, git_log, git_status, git_checkout,
    git_commit, git_push, git_branch,
)
from .web import web_fetch, read_docs
from .lsp import get_symbols, find_references, go_to_definition, get_hover_info, get_diagnostics
from .knowledge import find_symbol, get_callers, get_imports, analyze_impact, ingest_project
from .testing import generate_tests, run_test_suite, get_coverage
from .security import security_scan
from .mutation import MutateCode
from .refactor import auto_refactor
from .arch_check import arch_check
from .benchmark import benchmark

__all__ = [
    # Base
    "Tool", "ToolRegistry", "ToolResult", "DangerLevel", "tool", "generate_schema",
    # Filesystem
    "read_file", "write_file", "edit_file", "list_directory",
    "search_file", "search_content", "delete_file",
    "run_shell", "run_test", "kill_process",
    # Git
    "git_diff", "git_log", "git_status", "git_checkout",
    "git_commit", "git_push", "git_branch",
    # Web
    "web_fetch", "read_docs",
    # LSP
    "get_symbols", "find_references", "go_to_definition", "get_hover_info", "get_diagnostics",
    # Knowledge Graph
    "find_symbol", "get_callers", "get_imports", "analyze_impact", "ingest_project",
    # Testing
    "generate_tests", "run_test_suite", "get_coverage",
    # Security
    "security_scan",
    # Mutation
    "MutateCode",
    # Refactoring
    "auto_refactor",
    # Architecture
    "arch_check",
    # Benchmark
    "benchmark",
]
