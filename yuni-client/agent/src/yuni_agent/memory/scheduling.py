from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from yuni_agent.storage.models import Conversation, MemoryJob


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def enqueue_memory_job(
    session: Session, conversation_id: int, user_message_id: int
) -> None:
    conversation = session.get(Conversation, conversation_id)
    assert conversation is not None
    pending = session.scalar(
        select(MemoryJob).where(
            MemoryJob.conversation_id == conversation_id, MemoryJob.status == "pending"
        )
    )
    now = datetime.now(UTC)
    first_unprocessed = (conversation.memory_processed_until_message_id or 0) + 1
    run_after = now + timedelta(minutes=5)
    if pending is None:
        session.add(MemoryJob(
            conversation_id=conversation_id, from_message_id=first_unprocessed,
            to_message_id=user_message_id, run_after=run_after, status="pending",
        ))
    else:
        pending.to_message_id = max(pending.to_message_id, user_message_id)
        if _utc(pending.run_after) > now:
            # Keep eligible jobs ready and cap batching at fifteen minutes.
            pending.run_after = min(run_after, _utc(pending.created_at) + timedelta(minutes=15))
