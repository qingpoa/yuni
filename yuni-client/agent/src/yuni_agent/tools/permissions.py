from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import select

from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import AllowedRoot
from yuni_agent.tools.errors import ToolExecutionError


class PathPolicy:
    """Resolves tool paths against the user's active authorized roots.

    Roots are loaded once per policy instance so that a single tool call (for
    example a ``find_files`` walk over thousands of files) does not query the
    database for every candidate path. Create a new instance per tool call so
    that revoked roots take effect immediately.
    """

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self._roots: list[Path] | None = None

    @property
    def roots(self) -> list[Path]:
        if self._roots is None:
            self._roots = self._load_roots()
        return self._roots

    def refresh(self) -> None:
        self._roots = None

    def _load_roots(self) -> list[Path]:
        with self.runtime.require_database().session() as session:
            values = list(session.scalars(
                select(AllowedRoot.canonical_path).where(AllowedRoot.status == "active")
            ))
        roots: list[Path] = []
        for value in values:
            try:
                roots.append(Path(value).resolve(strict=True))
            except OSError:
                continue
        return roots

    def is_allowed(self, path: Path) -> bool:
        candidate = os.path.normcase(str(path.resolve(strict=False)))
        for root in self.roots:
            root_value = os.path.normcase(str(root))
            try:
                if os.path.commonpath([candidate, root_value]) == root_value:
                    return True
            except ValueError:
                # Different drives on Windows.
                continue
        return False

    @property
    def workspace(self) -> Path:
        return self.runtime.config.workspace_path

    def resolve(self, value: str | Path, *, must_exist: bool) -> Path:
        path = Path(value)
        if not path.is_absolute():
            # "notes.txt" or "a/b.md" means a file in the default workspace. A
            # drive-relative path like "C:x" or rooted "\\x" is still ambiguous.
            if path.drive or path.root:
                raise ToolExecutionError("工具路径必须是绝对路径，或相对默认工作目录的路径")
            path = self.workspace / path
        try:
            resolved = path.resolve(strict=must_exist)
            if not must_exist:
                resolved.parent.resolve(strict=True)
        except OSError as exc:
            raise ToolExecutionError("目标路径不存在或不可访问") from exc
        self.refresh()
        if not self.is_allowed(resolved):
            raise ToolExecutionError("目标路径不在授权目录内")
        return resolved
