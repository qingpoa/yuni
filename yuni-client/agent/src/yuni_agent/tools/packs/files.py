"""Authorized local text file tool pack (`find_files`, `read_text_file`, `create_text_file`, `replace_text`)."""
from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path

from sqlalchemy import select

from yuni_agent.storage.models import AllowedRoot
from yuni_agent.tools.errors import ToolExecutionError
from yuni_agent.tools.models import (
    CreateTextFileArguments,
    FindFilesArguments,
    ReadTextFileArguments,
    ReplaceTextArguments,
    ToolResult,
    ToolRisk,
)
from yuni_agent.tools.registry import RegisteredTool, ToolContext, ToolPreview


def _files_enabled(runtime) -> bool:
    if runtime is None or not getattr(runtime, "storage_ready", False):
        return True
    try:
        with runtime.require_database().session() as session:
            return session.scalar(
                select(AllowedRoot.id).where(AllowedRoot.status == "active").limit(1)
            ) is not None
    except Exception:
        return False

MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_READ_CHARS = 20_000

# find_files bounds: a large authorized root must not freeze a tool call.
FIND_MAX_RESULTS = 50
FIND_MAX_DEPTH = 8
FIND_MAX_VISITED_FILES = 5_000
FIND_MAX_CONTENT_FILE_BYTES = 512 * 1024
FIND_MAX_CONTENT_TOTAL_BYTES = 32 * 1024 * 1024
FIND_SKIPPED_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode",
    "dist", "build", "target", ".next", ".cache",
})


def _max_text_bytes() -> int:
    from yuni_agent.tools import builtin
    return getattr(builtin, "MAX_TEXT_BYTES", MAX_TEXT_BYTES)


def read_text(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > _max_text_bytes():
        raise ToolExecutionError("仅支持不超过 2 MB 的文本文件")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolExecutionError("文件不是 UTF-8 文本") from exc


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def unified_diff(before: str, after: str, name: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile=f"{name}:before", tofile=f"{name}:after", lineterm=""
    )
    return "\n".join(lines)[:8_000]


def _size_ok(text: str, message: str) -> None:
    if len(text.encode("utf-8")) > _max_text_bytes():
        raise ToolExecutionError(message)


def _prepare_find_files(context: ToolContext, arguments: FindFilesArguments) -> ToolPreview:
    path = arguments.path.strip()
    if path:
        context.paths.resolve(path, must_exist=True)
    return ToolPreview(summary="查找授权目录中的文件", target_paths=[path] if path else [])


def _find_files(context: ToolContext, arguments: FindFilesArguments) -> ToolResult:
    """Bounded breadth-first walk: depth, visited files and content bytes are all capped."""
    from yuni_agent.tools import builtin

    max_results = getattr(builtin, "FIND_MAX_RESULTS", FIND_MAX_RESULTS)
    max_depth = getattr(builtin, "FIND_MAX_DEPTH", FIND_MAX_DEPTH)
    max_visited = getattr(builtin, "FIND_MAX_VISITED_FILES", FIND_MAX_VISITED_FILES)
    max_content_file_bytes = getattr(builtin, "FIND_MAX_CONTENT_FILE_BYTES", FIND_MAX_CONTENT_FILE_BYTES)
    max_content_total_bytes = getattr(builtin, "FIND_MAX_CONTENT_TOTAL_BYTES", FIND_MAX_CONTENT_TOTAL_BYTES)
    skipped_dirs = getattr(builtin, "FIND_SKIPPED_DIRS", FIND_SKIPPED_DIRS)

    path_value = arguments.path.strip()
    roots = [context.paths.resolve(path_value, must_exist=True)] if path_value else context.paths.roots
    query = arguments.query.strip().lower()
    matches: list[str] = []
    seen: set[str] = set()
    visited = 0
    content_bytes = 0
    truncated = False

    for root in roots:
        if not root.is_dir():
            continue
        queue: list[tuple[Path, int]] = [(root, 0)]
        while queue and not truncated:
            directory, depth = queue.pop(0)
            try:
                entries = sorted(os.scandir(directory), key=lambda entry: entry.name.lower())
            except OSError:
                continue
            for entry in entries:
                if len(matches) >= max_results or visited >= max_visited:
                    truncated = True
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if depth + 1 < max_depth and entry.name not in skipped_dirs:
                            queue.append((Path(entry.path), depth + 1))
                        continue
                    if not entry.is_file():
                        continue
                    size = entry.stat().st_size
                except OSError:
                    continue
                visited += 1
                path = Path(entry.path)
                if not context.paths.is_allowed(path):
                    continue
                key = os.path.normcase(str(path))
                if key in seen:
                    continue
                matched = not query or query in entry.name.lower()
                if (not matched and query and size <= max_content_file_bytes
                        and content_bytes + size <= max_content_total_bytes):
                    content_bytes += size
                    try:
                        matched = query in path.read_text(encoding="utf-8").lower()
                    except (UnicodeDecodeError, OSError):
                        pass
                if matched:
                    seen.add(key)
                    matches.append(str(path))
        if truncated:
            break

    if not matches:
        content = "没有找到匹配文件。"
    else:
        content = "找到以下文件：\n" + "\n".join(matches)
    if truncated:
        content += "\n（搜索范围较大，结果已截断；可指定更具体的子目录或关键词。）"
    return ToolResult(content=content, details={"count": len(matches), "truncated": truncated, "visited": visited})


def _prepare_read_text_file(context: ToolContext, arguments: ReadTextFileArguments) -> ToolPreview:
    path = context.paths.resolve(arguments.path, must_exist=True)
    return ToolPreview(summary=f"读取 {path.name}", target_paths=[str(path)])


def _read_text_file(context: ToolContext, arguments: ReadTextFileArguments) -> ToolResult:
    from yuni_agent.tools import builtin

    max_read_chars = getattr(builtin, "MAX_READ_CHARS", MAX_READ_CHARS)
    path = context.paths.resolve(arguments.path, must_exist=True)
    content = read_text(path)
    return ToolResult(content=f"文件：{path}\n\n{content[:max_read_chars]}", details={"path": str(path)})


def _prepare_create_text_file(context: ToolContext, arguments: CreateTextFileArguments) -> ToolPreview:
    path = context.paths.resolve(arguments.path, must_exist=False)
    if path.exists():
        raise ToolExecutionError("目标文件已经存在")
    _size_ok(arguments.content, "文件内容超过 2 MB")
    return ToolPreview(summary=f"创建文本文件 {path.name}", target_paths=[str(path)],
                       diff_preview=unified_diff("", arguments.content, path.name))


def _create_text_file(context: ToolContext, arguments: CreateTextFileArguments) -> ToolResult:
    path = context.paths.resolve(arguments.path, must_exist=False)
    if path.exists():
        raise ToolExecutionError("目标文件已经存在")
    atomic_write(path, arguments.content)
    return ToolResult(content=f"已创建文件：{path}", details={"path": str(path)})


def _prepare_replace_text(context: ToolContext, arguments: ReplaceTextArguments) -> ToolPreview:
    path = context.paths.resolve(arguments.path, must_exist=True)
    original = read_text(path)
    if original.count(arguments.old_text) != 1:
        raise ToolExecutionError("待替换文本必须在文件中唯一出现")
    updated = original.replace(arguments.old_text, arguments.new_text, 1)
    _size_ok(updated, "修改后文件超过 2 MB")
    return ToolPreview(summary=f"修改文本文件 {path.name}", target_paths=[str(path)],
                       diff_preview=unified_diff(original, updated, path.name))


def _replace_text(context: ToolContext, arguments: ReplaceTextArguments) -> ToolResult:
    path = context.paths.resolve(arguments.path, must_exist=True)
    original = read_text(path)
    if original.count(arguments.old_text) != 1:
        raise ToolExecutionError("确认后文件内容已变化，请重新发起")
    updated = original.replace(arguments.old_text, arguments.new_text, 1)
    _size_ok(updated, "修改后文件超过 2 MB")
    atomic_write(path, updated)
    return ToolResult(content=f"已修改文件：{path}", details={"path": str(path)})


FILE_TOOLS: list[RegisteredTool] = [
    RegisteredTool(
        name="find_files",
        description="在用户已授权的目录中列出或按文件名、文本内容查找文件。结果最多 50 个，范围过大时会截断。",
        arguments=FindFilesArguments,
        risk=ToolRisk.READ,
        prepare=_prepare_find_files,
        run=_find_files,
        enabled=_files_enabled,
        group="files",
        summary="在已授权本地目录中按文件名或内容查找文件",
        not_for="检索伴侣脑海中的长期记忆或用户偏好请用 query_memory",
    ),
    RegisteredTool(
        name="read_text_file",
        description="读取授权目录内不超过 2MB 的 UTF-8 文本文件。",
        arguments=ReadTextFileArguments,
        risk=ToolRisk.READ,
        prepare=_prepare_read_text_file,
        run=_read_text_file,
        enabled=_files_enabled,
        group="files",
        summary="读取已授权目录内的 UTF-8 文本文件内容",
        not_for="若还不清楚具体文件路径，请先用 find_files 查找",
    ),
    RegisteredTool(
        name="create_text_file",
        description="在授权目录创建新的 UTF-8 文本文件。执行前必须获得用户确认。",
        arguments=CreateTextFileArguments,
        risk=ToolRisk.WRITE,
        prepare=_prepare_create_text_file,
        run=_create_text_file,
        enabled=_files_enabled,
        group="files",
        summary="在已授权目录创建新的 UTF-8 文本文件",
        not_for="修改已有文件中的局部文字请用 replace_text；保存到伴侣长期记忆请用 save_memory",
        redacted_fields=("content",),
    ),
    RegisteredTool(
        name="replace_text",
        description="在授权文本文件中精确替换唯一一段文字。执行前展示 diff 并获得确认。",
        arguments=ReplaceTextArguments,
        risk=ToolRisk.WRITE,
        prepare=_prepare_replace_text,
        run=_replace_text,
        enabled=_files_enabled,
        group="files",
        summary="在已有授权文本文件中精确替换唯一一段文字",
        not_for="替换前须已知原文件准确文字（可先 read_text_file）；新建完整文件请用 create_text_file",
        redacted_fields=("old_text", "new_text"),
    ),
]
