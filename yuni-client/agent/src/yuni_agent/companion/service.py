from __future__ import annotations

from typing import Protocol


class CompanionService(Protocol):
    async def has_companion(self) -> bool: ...

