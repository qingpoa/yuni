from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from yuni_agent.storage.models import ClientEvent, Reminder

if TYPE_CHECKING:
    from yuni_agent.runtime import AgentRuntime

logger = logging.getLogger(__name__)


def emit_event(runtime: AgentRuntime, event_type: str, payload: dict[str, Any], *,
               session: Session | None = None, dedupe_key: str | None = None) -> int:
    """Use the caller's transaction when the event accompanies a state change."""
    if session is None:
        with runtime.require_database().session() as owned:
            return emit_event(runtime, event_type, payload, session=owned, dedupe_key=dedupe_key)
    if dedupe_key is not None:
        existing = session.scalar(select(ClientEvent.id).where(ClientEvent.dedupe_key == dedupe_key))
        if existing is not None:
            return existing
    event = ClientEvent(event_type=event_type, payload=payload, dedupe_key=dedupe_key)
    session.add(event)
    session.flush()
    # All writers run on the Agent loop; the transaction commits before its next await.
    runtime.event_changed.set()
    return event.id


class ReminderEventsWorker:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    def run_once(self) -> None:
        with self.runtime.require_database().session() as session:
            for item in session.scalars(select(Reminder).where(
                Reminder.status == "pending", Reminder.remind_at <= datetime.now(UTC),
            ).order_by(Reminder.remind_at, Reminder.id)):
                moment = item.remind_at.replace(tzinfo=UTC) if item.remind_at.tzinfo is None else item.remind_at
                emit_event(self.runtime, "reminder_due", {"reminder": {
                    "id": item.id, "title": item.title, "note": item.note, "remindAt": moment.isoformat(),
                }}, session=session, dedupe_key=f"reminder:{item.id}:{moment.isoformat()}")

    async def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Reminder event delivery failed")
            await asyncio.sleep(1)
