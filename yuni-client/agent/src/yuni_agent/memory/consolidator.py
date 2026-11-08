from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from yuni_agent.memory.models import MemoryCandidate


class ConsolidationAction(StrEnum):
    CREATE = "create"
    REINFORCE = "reinforce"
    ENRICH = "enrich"
    TRANSITION = "transition"
    CORRECT = "correct"
    COEXIST = "coexist"
    DISPUTE = "dispute"


class MemoryConsolidator(Protocol):
    async def consolidate(self, candidate: MemoryCandidate) -> ConsolidationAction: ...

