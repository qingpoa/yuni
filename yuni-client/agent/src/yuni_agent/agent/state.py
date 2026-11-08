from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class AgentPhase(StrEnum):
    RECEIVED = "received"
    BUILDING_CONTEXT = "building_context"
    GENERATING = "generating"
    WAITING_FOR_TOOL_CONFIRMATION = "waiting_for_tool_confirmation"
    EXECUTING_TOOL = "executing_tool"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentState(BaseModel):
    conversation_id: int
    user_message_id: int
    assistant_message_id: int | None = None
    phase: AgentPhase = AgentPhase.RECEIVED
    messages: list[dict[str, Any]] = Field(default_factory=list)
    pending_tool_call_id: str | None = None
    error_code: int | None = None


class ChatState(TypedDict, total=False):
    conversation_id: int
    user_message_id: int
    current_input: str
    history_message_ids: list[int] | None
    messages: list[dict[str, Any]]
    tool_messages: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tool_note: str | None
    response_text: str
    reply_parts: list[str]
    sticker: dict[str, Any] | None
    voice: dict[str, Any] | None
    message_type: str
    help_resources: list[dict[str, str]]
    reply_plan: dict[str, Any] | None
    memory_ids: list[int]
    assistant_message_id: int
    pending_tool_call: dict[str, Any] | None
    tool_iterations: int
    activated_tools: list[str]
    executed_tool_keys: list[str]
    executed_tool_names: list[str]
    # Image produced by a tool this turn, attached to the next reply.
    pending_image: dict[str, Any] | None
    image: dict[str, Any] | None
    compaction_required: bool
    reply_call_id: str
    reply_count: int
    planner_steps: int
    stop_reason: str | None
    published_replies: list[dict[str, Any]]

