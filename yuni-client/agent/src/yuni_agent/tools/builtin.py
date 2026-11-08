"""Built-in local tools facade.

Concrete tool specs live in ``yuni_agent.tools.packs``; this module re-exports
all public constants, helpers, image client hooks and ``BUILTIN_TOOLS`` so
existing callers and test monkeypatches remain 100% compatible.
"""
from __future__ import annotations

from yuni_agent.model.image import edit_image
from yuni_agent.model.image import generate_image as request_image
from yuni_agent.tools.packs import ALL_PACK_TOOLS
from yuni_agent.tools.packs.files import (
    FIND_MAX_CONTENT_FILE_BYTES,
    FIND_MAX_CONTENT_TOTAL_BYTES,
    FIND_MAX_DEPTH,
    FIND_MAX_RESULTS,
    FIND_MAX_VISITED_FILES,
    FIND_SKIPPED_DIRS,
    MAX_READ_CHARS,
    MAX_TEXT_BYTES,
    atomic_write,
    read_text,
    unified_diff,
)
from yuni_agent.tools.packs.image import GENERATE_IMAGE_DESCRIPTION
from yuni_agent.tools.packs.memory import (
    QUERY_MEMORY_DESCRIPTION,
    SAVE_MEMORY_DESCRIPTION,
    memory_source,
)
from yuni_agent.tools.packs.reminder import parse_remind_at
from yuni_agent.tools.registry import RegisteredTool, ToolRegistry

BUILTIN_TOOLS: list[RegisteredTool] = list(ALL_PACK_TOOLS)


def build_registry() -> ToolRegistry:
    return ToolRegistry(BUILTIN_TOOLS)


__all__ = [
    "BUILTIN_TOOLS",
    "FIND_MAX_CONTENT_FILE_BYTES",
    "FIND_MAX_CONTENT_TOTAL_BYTES",
    "FIND_MAX_DEPTH",
    "FIND_MAX_RESULTS",
    "FIND_MAX_VISITED_FILES",
    "FIND_SKIPPED_DIRS",
    "GENERATE_IMAGE_DESCRIPTION",
    "MAX_READ_CHARS",
    "MAX_TEXT_BYTES",
    "QUERY_MEMORY_DESCRIPTION",
    "SAVE_MEMORY_DESCRIPTION",
    "atomic_write",
    "build_registry",
    "edit_image",
    "memory_source",
    "parse_remind_at",
    "read_text",
    "request_image",
    "unified_diff",
]
