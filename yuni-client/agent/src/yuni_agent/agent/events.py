from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentEventType(StrEnum):
    MESSAGE_CREATED = "message_created"
    DELTA = "delta"
    TOOL_CONFIRMATION_REQUIRED = "tool_confirmation_required"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    REPLY_COMPLETED = "reply_completed"
    COMPLETED = "completed"
    ERROR = "error"


class AgentEvent(BaseModel):
    type: AgentEventType
    data: dict[str, Any] = Field(default_factory=dict)

