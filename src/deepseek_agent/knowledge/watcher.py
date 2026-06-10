"""
知识图谱增量更新 — 基于 watchdog 的文件监听 + 增量解析。

使用方式：
  from deepseek_agent.knowledge.watcher import IncrementalWatcher

  watcher = IncrementalWatcher(project_path=".", graph=graph)
  watcher.start()  # 后台监听
  watcher.stop()   # 停止监听
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# 尝试导入 watchdog（可选依赖）
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


# ── 文件快照 ─────────────────────────────────────────────────────────────

class FileSnapshot:
    """文件快照，用于变更检测。"""

    def __init__(self, snapshot_dir: str = "."):
        self.snapshot_dir = Path(snapshot_dir) / ".deepseek-snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, str] = {}  # path → hash

    def get_hash(self, file_path: str) -> Optional[str]:
        """获取文件快照哈希。"""
        return self._cache.get(file_path)

    def update_hash(self, file_path: str, content: str) -> str:
        """更新文件哈希。"""
        h = hashlib.md5(content.encode()).hexdigest()
        self._cache[file_path] = h
        return h

    def has_changed(self, file_path: str, content: str) -> bool:
        """检查文件是否变更。"""
        h = hashlib.md5(content.encode()).hexdigest()
        old = self._cache.get(file_path)
        if old is None:
            return True
        return h != old

    def save(self) -> None:
        """持久化快照。"""
        path = self.snapshot_dir / "file_hashes.json"
        path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def load(self) -> None:
        """加载快照。"""
        path = self.snapshot_dir / "file_hashes.json"
        if path.exists():
            self._cache = json.loads(path.read_text(encoding="utf-8"))

    def remove(self, file_path: str) -> None:
        """移除文件记录。"""
        self._cache.pop(file_path, None)


# ── 增量更新器 ───────────────────────────────────────────────────────────

class IncrementalUpdater:
    """
    增量更新器 — 只重新解析变更的文件。

    与全量 ingest_project 相比，增量更新：
    1. 只解析新增/修改的文件
    2. 删除已移除文件的节点和边
    3. 保留未变更文件的解析结果
    """

    def __init__(self, graph: Any = None):
        self.graph = graph
        self.snapshot = FileSnapshot()
        self._parsers: Dict[str, Callable] = {}

    def register_parser(self, extension: str, parser: Callable) -> None:
        """注册文件扩展名对应的解析器。"""
        self._parsers[extension] = parser

    def update(self, project_path: str, changed_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行增量更新。

        Args:
            project_path: 项目根目录
            changed_files: 指定变更的文件列表，None 则自动检测

        Returns:
            更新统计
        """
        stats = {"added": 0, "modified": 0, "removed": 0, "unchanged": 0}

        # 加载快照
        self.snapshot = FileSnapshot(project_path)
        self.snapshot.load()

        project = Path(project_path)

        # 获取当前所有文件
        current_files: Set[str] = set()
        for ext in self._parsers:
            for f in project.rglob(f"*{ext}"):
                if any(skip in str(f) for skip in [".git", "__pycache__", ".venv", "node_modules"]):
                    continue
                current_files.add(str(f))

        # 如果指定了变更文件，只处理这些
        if changed_files:
            to_process = set(changed_files)
        else:
            to_process = current_files

        # 检测变更
        added_files = current_files - set(self.snapshot._cache.keys())
        removed_files = set(self.snapshot._cache.keys()) - current_files
        modified_files: Set[str] = set()

        for fp in to_process:
            try:
                content = Path(fp).read_text(encoding="utf-8", errors="replace")
                if self.snapshot.has_changed(fp, content):
                    modified_files.add(fp)
                else:
                    stats["unchanged"] += 1
            except Exception:
                continue

        # 处理新增文件
        for fp in added_files:
            self._parse_and_update(fp, "add")
            self.snapshot.update_hash(fp, Path(fp).read_text(encoding="utf-8", errors="replace"))
            stats["added"] += 1

        # 处理修改文件
        for fp in modified_files:
            self._parse_and_update(fp, "modify")
            self.snapshot.update_hash(fp, Path(fp).read_text(encoding="utf-8", errors="replace"))
            stats["modified"] += 1

        # 处理删除文件
        for fp in removed_files:
            self._remove_from_graph(fp)
            self.snapshot.remove(fp)
            stats["removed"] += 1

        # 保存快照
        self.snapshot.save()

        return stats

    def _parse_and_update(self, file_path: str, action: str) -> None:
        """解析文件并更新知识图谱。"""
        ext = Path(file_path).suffix
        parser = self._parsers.get(ext)
        if parser and self.graph:
            try:
                content = Path(file_path).read_text(encoding="utf-8", errors="replace")
                # 先移除旧节点
                if action == "modify":
                    self._remove_from_graph(file_path)
                # 解析并添加新节点
                nodes = parser(file_path, content)
                if hasattr(self.graph, 'add_nodes'):
                    self.graph.add_nodes(nodes)
            except Exception:
                pass

    def _remove_from_graph(self, file_path: str) -> None:
        """从知识图谱中移除文件相关节点。"""
        if self.graph and hasattr(self.graph, 'remove_nodes_by_file'):
            try:
                self.graph.remove_nodes_by_file(file_path)
            except Exception:
                pass


# ── 文件监听器 ───────────────────────────────────────────────────────────

if HAS_WATCHDOG:

    class _KnowledgeEventHandler(FileSystemEventHandler):
        """watchdog 事件处理器。"""

        def __init__(self, updater: IncrementalUpdater, project_path: str, callback: Optional[Callable] = None):
            super().__init__()
            self.updater = updater
            self.project_path = project_path
            self.callback = callback
            self._debounce_timer: Optional[threading.Timer] = None
            self._pending_changes: Set[str] = set()

        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith(".py"):
                self._schedule_update(event.src_path)

        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith(".py"):
                self._schedule_update(event.src_path)

        def on_deleted(self, event):
            if not event.is_directory and event.src_path.endswith(".py"):
                self._schedule_update(event.src_path)

        def _schedule_update(self, file_path: str) -> None:
            """防抖：500ms 内的多次变更合并为一次更新。"""
            self._pending_changes.add(file_path)

            if self._debounce_timer:
                self._debounce_timer.cancel()

            self._debounce_timer = threading.Timer(0.5, self._do_update)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

        def _do_update(self) -> None:
            """执行增量更新。"""
            changes = list(self._pending_changes)
            self._pending_changes.clear()

            result = self.updater.update(self.project_path, changed_files=changes)

            if self.callback:
                self.callback(result)


class IncrementalWatcher:
    """
    增量文件监听器 — watchdog 后台监听 + 增量解析。
    """

    def __init__(self, project_path: str, graph: Any = None, callback: Optional[Callable] = None):
        self.project_path = project_path
        self.updater = IncrementalUpdater(graph)
        self.callback = callback
        self._observer: Optional[Any] = None
        self._running = False

    def start(self) -> bool:
        """启动后台监听。"""
        if not HAS_WATCHDOG:
            return False

        handler = _KnowledgeEventHandler(self.updater, self.project_path, self.callback)
        self._observer = Observer()
        self._observer.schedule(handler, self.project_path, recursive=True)
        self._observer.daemon = True
        self._observer.start()
        self._running = True
        return True

    def stop(self) -> None:
        """停止监听。"""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def force_update(self, changed_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """强制执行一次增量更新。"""
        return self.updater.update(self.project_path, changed_files)


# ── 非 watchdog fallback ─────────────────────────────────────────────────

if not HAS_WATCHDOG:
    class IncrementalWatcher:
        """Fallback：无 watchdog 时的空实现。"""

        def __init__(self, *args, **kwargs):
            self._running = False

        def start(self) -> bool:
            return False

        def stop(self) -> None:
            pass

        @property
        def is_running(self) -> bool:
            return False

        def force_update(self, changed_files=None) -> Dict[str, Any]:
            return {"added": 0, "modified": 0, "removed": 0, "unchanged": 0}
