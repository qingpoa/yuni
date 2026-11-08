from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from yuni_agent.api.dependencies import require_session
from yuni_agent.api.routes.chat import _to_data
from yuni_agent.api.routes.reminders import _to_data as reminder_data
from yuni_agent.api.routes.tools import _tool_data
from yuni_agent.runtime import AgentRuntime, RuntimeStatus
from yuni_agent.storage.models import ChatMessage, ChatTurn, ClientEvent, Reminder, ToolCall

router = APIRouter(tags=["events"])


def _frame(event_type: str, payload: dict, event_id: int | None = None) -> str:
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return prefix + f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def event_stream(runtime: AgentRuntime, after_id: int | None):
    with runtime.require_database().session() as session:
        latest = session.scalar(select(func.max(ClientEvent.id))) or 0
        reset = after_id is not None and after_id > latest
        cursor = latest if after_id is None or reset else after_id
        messages = list(session.scalars(select(ChatMessage).where(
            ChatMessage.status != "streaming",
        ).order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(30)))
        messages.reverse()
        snapshot = {
            "cursor": cursor, "reset": reset,
            "messages": [_to_data(item).model_dump(mode="json", by_alias=True) for item in messages],
            "activeUserMessageIds": list(session.scalars(select(ChatTurn.user_message_id).where(
                ChatTurn.status.in_(["queued", "running"]),
            ))),
            "toolCalls": [_tool_data(item).model_dump(mode="json", by_alias=True) for item in session.scalars(
                select(ToolCall).where(ToolCall.status.in_(["awaiting_confirmation", "running"]))
            )],
            "reminders": [reminder_data(item).model_dump(mode="json", by_alias=True) for item in session.scalars(
                select(Reminder).where(Reminder.status == "pending", Reminder.remind_at <= datetime.now(UTC))
            )],
        }
    yield _frame("ready", snapshot)
    while runtime.status != RuntimeStatus.STOPPING:
        runtime.event_changed.clear()
        with runtime.require_database().session() as session:
            rows = list(session.scalars(select(ClientEvent).where(ClientEvent.id > cursor).order_by(ClientEvent.id).limit(100)))
            frames = [(item.id, _frame(item.event_type, item.payload, item.id)) for item in rows]
        if frames:
            for cursor, frame in frames:
                yield frame
            continue
        try:
            await asyncio.wait_for(runtime.event_changed.wait(), timeout=15)
        except TimeoutError:
            yield ": keepalive\n\n"


@router.get("/api/events")
async def subscribe_events(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
    after_id: int | None = Query(default=None, alias="afterId", ge=0),
) -> StreamingResponse:
    return StreamingResponse(event_stream(runtime, after_id), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })
