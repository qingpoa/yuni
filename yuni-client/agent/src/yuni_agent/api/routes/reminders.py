from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select

from yuni_agent.api.dependencies import require_session
from yuni_agent.api.models import CamelModel, iso_utc
from yuni_agent.api.schemas import ApiResponse
from yuni_agent.common.http import fail
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import Reminder

router = APIRouter(prefix="/api/reminders", tags=["reminders"])


class ReminderData(CamelModel):
    id: int
    title: str
    note: str
    remind_at: str


def _to_data(item: Reminder) -> ReminderData:
    return ReminderData(
        id=item.id,
        title=item.title,
        note=item.note,
        remind_at=iso_utc(item.remind_at),
    )


@router.get("/due", response_model=ApiResponse[list[ReminderData]])
def due_reminders(
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[list[ReminderData]]:
    with runtime.require_database().session() as session:
        items = list(
            session.scalars(
                select(Reminder)
                .where(
                    Reminder.status == "pending",
                    Reminder.remind_at <= datetime.now(UTC),
                )
                .order_by(Reminder.remind_at, Reminder.id)
                .limit(20)
            )
        )
        result = [_to_data(item) for item in items]
    return ApiResponse(data=result)


@router.post("/{reminder_id}/delivered", response_model=ApiResponse[None])
def mark_delivered(
    reminder_id: int,
    runtime: Annotated[AgentRuntime, Depends(require_session)],
) -> ApiResponse[None]:
    with runtime.require_database().session() as session:
        item = session.get(Reminder, reminder_id)
        if item is not None and item.status == "delivered":
            return ApiResponse(data=None)
        if item is None or item.status != "pending":
            fail(status.HTTP_404_NOT_FOUND, 40000, "提醒不存在或已处理")
        item.status = "delivered"
        item.delivered_at = datetime.now(UTC)
    return ApiResponse(data=None)
