from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from yuni_agent.agent.events import AgentEvent


class ChatService(Protocol):
    async def stream_message(
        self,
        content: str,
        client_message_id: str,
    ) -> AsyncIterator[AgentEvent]: ...

