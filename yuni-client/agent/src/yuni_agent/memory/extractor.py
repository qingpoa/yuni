from __future__ import annotations

from typing import Protocol

from yuni_agent.memory.models import MemoryCandidate


class MemoryExtractor(Protocol):
    async def extract(
        self,
        conversation_id: int,
        from_message_id: int,
        to_message_id: int,
    ) -> list[MemoryCandidate]: ...

