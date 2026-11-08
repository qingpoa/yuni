from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ContextRequest(BaseModel):
    conversation_id: int
    current_user_message_id: int
    current_input: str
    history_message_ids: list[int] | None = None
    tool_messages: list[dict[str, Any]] = Field(default_factory=list)
    planner_tools: list[dict[str, Any]] = Field(default_factory=list)


class BuiltContext(BaseModel):
    messages: list[dict[str, Any]]
    tool_messages: list[dict[str, Any]] = Field(default_factory=list)
    estimated_tokens: int
    projected_tokens: int
    compaction_required: bool = False
    summary_used: bool
    memory_ids: list[int] = Field(default_factory=list)
    recent_message_ids: list[int] = Field(default_factory=list)
