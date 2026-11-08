from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from yuni_agent.memory.models import EntityInput, MemoryCandidate
from yuni_agent.storage.models import (
    ChatMessage,
    Conversation,
    Memory,
    MemoryEmbedding,
    MemoryEntity,
    MemoryJob,
    MemorySource,
)
from yuni_agent.storage.models import (
    MemoryCandidate as StoredCandidate,
)

LOCAL_TIMEZONE = timezone(timedelta(hours=8))


def utc(value: datetime | None) -> datetime | None:
    return (value if value.tzinfo else value.replace(tzinfo=LOCAL_TIMEZONE)).astimezone(UTC) if value else None


def stored_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    memory_id: int
    status: str
    outcome: str


def resolve_entity(session: Session, companion_id: int, data: EntityInput) -> MemoryEntity:
    rows = list(session.scalars(select(MemoryEntity).where(
        MemoryEntity.companion_id == companion_id, MemoryEntity.status == 'active',
        MemoryEntity.entity_type == data.entity_type,
    )))
    matches = [row for row in rows if (
        row.canonical_name == data.name or data.name in row.aliases
    ) and (not data.relation_to_user or row.relation_to_user == data.relation_to_user)]
    if len(matches) == 1:
        row = matches[0]
        row.aliases = list(dict.fromkeys([*row.aliases, *data.aliases]))
        return row
    row = MemoryEntity(companion_id=companion_id, canonical_name=data.name,
                       entity_type=data.entity_type, relation_to_user=data.relation_to_user,
                       aliases=list(dict.fromkeys(data.aliases)))
    session.add(row)
    session.flush()
    return row


def _sources(session: Session, companion_id: int, candidate: MemoryCandidate, conversation_id: int | None, manual: bool) -> list[ChatMessage]:
    ids = set(candidate.source_message_ids)
    if manual and not ids:
        return []
    conditions = [ChatMessage.id.in_(ids), ChatMessage.status == 'succeeded', Conversation.companion_id == companion_id]
    if conversation_id is not None:
        conditions.append(ChatMessage.conversation_id == conversation_id)
    conditions.append(ChatMessage.role.in_(['user', 'assistant']) if candidate.domain == 'conversation' else ChatMessage.role == 'user')
    rows = list(session.scalars(select(ChatMessage).join(Conversation).where(*conditions)))
    if not ids or {row.id for row in rows} != ids or len({row.conversation_id for row in rows}) > 1:
        raise ValueError('记忆来源必须是本会话的有效消息；现实事实只能引用用户')
    if candidate.evidence_quote and not any(candidate.evidence_quote in row.content for row in rows):
        raise ValueError('记忆证据引文必须逐字出现在来源消息中')
    return rows


def _same_event(row: Memory, candidate: MemoryCandidate) -> bool:
    if row.id == candidate.target_memory_id:
        return True
    left = row.time_meta or {}
    right = candidate.time_meta.model_dump(mode='json', exclude_none=True)
    for prefix in ('actual', 'planned'):
        a, b = left.get(f'{prefix}_start'), right.get(f'{prefix}_start')
        if a and b:
            start_a, start_b = utc(datetime.fromisoformat(a)), utc(datetime.fromisoformat(b))
            end_a = utc(datetime.fromisoformat(left.get(f'{prefix}_end', a))) or start_a
            end_b = utc(datetime.fromisoformat(right.get(f'{prefix}_end', b))) or start_b
            return start_a == start_b or (start_a < end_b and start_b < end_a)
    return row.content == candidate.content


def _attach_sources(session: Session, row: Memory, sources: list[ChatMessage], quote: str) -> None:
    for source in sources:
        if session.get(MemorySource, (row.id, source.id)) is None:
            session.add(MemorySource(memory_id=row.id, message_id=source.id,
                evidence_text=quote if quote and quote in source.content else source.content[:1000],
                message_sent_at=source.created_at))


def conflict_group(session: Session, row: Memory) -> list[Memory]:
    """Return all conflicted branches in the same slot or linked by conflict_with_id."""
    candidates = list(session.scalars(select(Memory).where(
        Memory.companion_id == row.companion_id,
        Memory.domain == row.domain, Memory.scene_key == row.scene_key,
        Memory.subject_entity_id == row.subject_entity_id,
        Memory.subject_key == row.subject_key, Memory.kind == row.kind,
        Memory.qualifier == row.qualifier, Memory.status == 'conflicted',
        Memory.deleted_at.is_(None),
    )))
    ids = {row.id} | {item.id for item in candidates}
    while True:
        expanded = ids | {item.id for item in candidates if item.conflict_with_id in ids}
        expanded |= {item.conflict_with_id for item in candidates
                     if item.id in ids and item.conflict_with_id is not None}
        if expanded == ids:
            return [item for item in candidates if item.id in ids]
        ids = expanded


def save_candidate(session: Session, companion_id: int, candidate: MemoryCandidate, *,
                   conversation_id: int | None = None, manual: bool = False,
                   tombstone: bool = False) -> MemoryWriteResult:
    sources = _sources(session, companion_id, candidate, conversation_id, manual)
    entities = [resolve_entity(session, companion_id, data) for data in candidate.entities]
    subject = None
    if candidate.subject_person != 'user':
        subject = next((row for row in entities if row.canonical_name == candidate.subject_person), None)
        subject = subject or resolve_entity(session, companion_id, EntityInput(name=candidate.subject_person))
    entity_ids = list(dict.fromkeys([row.id for row in entities] + ([subject.id] if subject else [])))
    scope = [Memory.companion_id == companion_id, Memory.domain == candidate.domain,
             Memory.scene_key == candidate.scene_key, Memory.subject_entity_id == (subject.id if subject else None),
             Memory.subject_key == candidate.subject_key, Memory.kind == candidate.kind]
    rows = list(session.scalars(select(Memory).where(*scope).order_by(Memory.id.desc())))
    if not tombstone:
        for row in rows:
            if row.status == 'deleted' and row.qualifier == candidate.qualifier and sources and all(
                source.id <= (row.suppressed_until_message_id or 0) for source in sources
            ) and (
                row.content == candidate.content
                or (candidate.same_conclusion and (candidate.target_memory_id in (None, row.id)))
            ):
                return MemoryWriteResult(row.id, 'deleted', 'suppressed')
    applicable = [row for row in rows if row.status in ('active', 'conflicted') and row.deleted_at is None
                  and row.qualifier == candidate.qualifier
                  and (candidate.kind != 'event' or _same_event(row, candidate))]
    if candidate.target_memory_id and candidate.operation_hint in ('correct', 'transition'):
        target = session.get(Memory, candidate.target_memory_id)
        if manual and (target is None or target.companion_id != companion_id or target.deleted_at is not None):
            raise ValueError('编辑目标不存在')
        if (
            target is not None and target.companion_id == companion_id and target.deleted_at is None
            and (manual or target.status in ('active', 'conflicted'))
        ):
            if target not in applicable:
                applicable.insert(0, target)
            for branch in conflict_group(session, target):
                if branch not in applicable:
                    applicable.append(branch)
    if not applicable and candidate.operation_hint in ('correct', 'transition'):
        applicable = [row for row in rows if row.status in ('active', 'conflicted') and row.deleted_at is None
                      and (candidate.kind != 'event' or _same_event(row, candidate))]
    existing = next((row for row in applicable if row.id == candidate.target_memory_id), None)
    existing = existing or next((row for row in applicable if row.content == candidate.content), None)
    existing = existing or next(iter(applicable), None)
    same = existing is not None and (existing.content == candidate.content or (
        existing.id == candidate.target_memory_id and candidate.same_conclusion))
    if existing and sources and not same and not manual:
        latest_source = session.scalar(select(func.max(MemorySource.message_id)).where(MemorySource.memory_id == existing.id))
        if latest_source is not None and max(source.id for source in sources) <= latest_source:
            return MemoryWriteResult(existing.id, existing.status, 'stale')
    action = candidate.operation_hint
    now = datetime.now(UTC)
    source_time = max((stored_utc(row.created_at) for row in sources), default=now)
    fresh = source_time + timedelta(days=candidate.fresh_days) if candidate.kind == 'ongoing_state' else None
    resolving = existing is not None and existing.status == 'conflicted' and action in ('correct', 'transition')
    if not tombstone and existing and same and action != 'enrich' and not resolving:
        if manual:
            existing.title = candidate.title
            existing.importance = candidate.importance
            existing.kind = candidate.kind
            existing.type = candidate.type.value
            existing.qualifier = candidate.qualifier
            existing.time_text = candidate.time_text
            existing.time_precision = candidate.time_precision
            existing.valid_from = utc(candidate.valid_from)
            existing.valid_until = utc(candidate.valid_until)
            existing.time_meta = candidate.time_meta.model_dump(mode='json', exclude_none=True)
            existing.event_status = candidate.event_status
            existing.outcome = candidate.outcome
            session.execute(delete(MemoryEmbedding).where(MemoryEmbedding.memory_id == existing.id))
        _attach_sources(session, existing, sources, candidate.evidence_quote)
        existing.updated_at = now
        if fresh:
            existing.fresh_until = max(stored_utc(existing.fresh_until), fresh) if existing.fresh_until else fresh
        return MemoryWriteResult(existing.id, existing.status, 'unchanged')
    if not tombstone and existing and existing.status == 'active' and candidate.kind == 'event' and (
        action == 'enrich' or (action == 'transition' and _same_event(existing, candidate))
    ):
        existing.content = candidate.content
        existing.title = candidate.title
        existing.event_status = candidate.event_status or existing.event_status
        existing.outcome = candidate.outcome if candidate.outcome is not None else existing.outcome
        existing.time_meta = {**existing.time_meta, **candidate.time_meta.model_dump(mode='json', exclude_none=True)}
        existing.entity_ids = list(dict.fromkeys([*existing.entity_ids, *entity_ids]))
        existing.updated_at = now
        session.execute(delete(MemoryEmbedding).where(MemoryEmbedding.memory_id == existing.id))
        _attach_sources(session, existing, sources, candidate.evidence_quote)
        return MemoryWriteResult(existing.id, existing.status, 'enriched')
    row = Memory(companion_id=companion_id, type=candidate.type.value, kind=candidate.kind,
        domain=candidate.domain, scene_key=candidate.scene_key, subject_entity_id=subject.id if subject else None,
        subject_key=candidate.subject_key, qualifier=candidate.qualifier, title=candidate.title,
        content=candidate.content, importance=candidate.importance, confidence=candidate.confidence,
        sensitivity=candidate.sensitivity, status='deleted' if tombstone else 'active',
        valid_from=utc(candidate.valid_from), valid_until=utc(candidate.valid_until),
        fresh_until=fresh, event_status=candidate.event_status, outcome=candidate.outcome,
        time_text=candidate.time_text, time_precision=candidate.time_precision,
        time_meta=candidate.time_meta.model_dump(mode='json', exclude_none=True), entity_ids=entity_ids)
    session.add(row)
    session.flush()
    outcome = 'saved'
    if tombstone:
        delete_memory(session, row)
        outcome = 'suppressed'
    elif existing and action in ('transition', 'correct'):
        for old in applicable:
            old.status = 'historical' if action == 'transition' else 'retracted'
            old.superseded_by_id = row.id
            old.conflict_with_id = None
            old.conflict_content = None
            if action == 'transition':
                old.valid_until = utc(candidate.valid_from) or source_time
                if candidate.time_meta.start_window_end:
                    old.time_meta = {**old.time_meta, 'end_window_start': old.valid_until.isoformat()}
                    old.valid_until = utc(candidate.time_meta.start_window_end)
            session.execute(delete(MemoryEmbedding).where(MemoryEmbedding.memory_id == old.id))
        row.valid_from = row.valid_from or source_time
        outcome = action
    elif existing:
        row.status = existing.status = 'conflicted'
        existing.conflict_content = row.content
        row.conflict_content = existing.content
        row.conflict_with_id = existing.id
        existing.conflict_with_id = row.id
        session.execute(delete(MemoryEmbedding).where(MemoryEmbedding.memory_id == existing.id))
        outcome = 'conflicted'
    _attach_sources(session, row, sources, candidate.evidence_quote)
    return MemoryWriteResult(row.id, row.status, outcome)


def message_watermark(session: Session, companion_id: int) -> int:
    return session.scalar(select(func.max(ChatMessage.id)).join(Conversation).where(
        Conversation.companion_id == companion_id)) or 0


def delete_memory(session: Session, row: Memory) -> None:
    row.status = 'deleted'
    row.deleted_at = datetime.now(UTC)
    row.suppressed_until_message_id = message_watermark(session, row.companion_id)
    session.execute(delete(MemoryEmbedding).where(MemoryEmbedding.memory_id == row.id))
    for conversation in session.scalars(select(Conversation).where(Conversation.companion_id == row.companion_id)):
        conversation.summary = ''
        conversation.summary_until_message_id = None


def clear_memories(session: Session, companion_id: int) -> None:
    for row in session.scalars(select(Memory).where(Memory.companion_id == companion_id)):
        delete_memory(session, row)
    for conversation in session.scalars(select(Conversation).where(Conversation.companion_id == companion_id)):
        conversation.memory_processed_until_message_id = session.scalar(select(func.max(ChatMessage.id)).where(
            ChatMessage.conversation_id == conversation.id)) or 0
        conversation.summary = ''
        conversation.summary_until_message_id = None
        jobs = list(session.scalars(select(MemoryJob).where(MemoryJob.conversation_id == conversation.id)))
        for job in jobs:
            if job.status in ('pending', 'running'):
                job.status = 'cancelled'
            for pending in session.scalars(select(StoredCandidate).where(StoredCandidate.job_id == job.id, StoredCandidate.status == 'pending')):
                pending.status = 'rejected'
                pending.decided_at = datetime.now(UTC)
