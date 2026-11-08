"""Built-in declarative tool packs grouped by domain."""
from __future__ import annotations

from yuni_agent.tools.packs.files import FILE_TOOLS
from yuni_agent.tools.packs.image import IMAGE_TOOLS
from yuni_agent.tools.packs.memory import MEMORY_TOOLS
from yuni_agent.tools.packs.reminder import REMINDER_TOOLS
from yuni_agent.tools.registry import RegisteredTool

ALL_PACK_TOOLS: list[RegisteredTool] = [
    *MEMORY_TOOLS,
    *FILE_TOOLS,
    *REMINDER_TOOLS,
    *IMAGE_TOOLS,
]

__all__ = [
    "ALL_PACK_TOOLS",
    "FILE_TOOLS",
    "IMAGE_TOOLS",
    "MEMORY_TOOLS",
    "REMINDER_TOOLS",
]
