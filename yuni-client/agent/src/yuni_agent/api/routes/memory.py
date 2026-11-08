from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import Field
from sqlalchemy import and_, delete, or_, select

from yuni_agent.api.dependencies import require_session
from yuni_agent.api.models import CamelModel, iso_utc
from yuni_agent.api.schemas import ApiResponse
from yuni_agent.common.http import fail
from yuni_agent.memory.index import _vector, content_hash
from yuni_agent.memory.models import MemoryCandidate, MemoryType, TimeMeta
from yuni_agent.memory.writer import (
    clear_memories as clear_memory_rows,
)
from yuni_agent.memory.writer import (
    conflict_group,
    delete_memory,
    save_candidate,
)
from yuni_agent.model.embedding import profile_id
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.models import (
    Companion,
    Conversation,
    Memory,
    MemoryEmbedding,
    MemoryEntity,
    MemoryJob,
    MemorySource,
)
from yuni_agent.storage.models import (
    MemoryCandidate as Pending,
)

router = APIRouter(prefix='/api/memories', tags=['memory'])


class MemoryData(CamelModel):
    id: int
    type: str
    title: str
    content: str
    source_message_id: int | None
    source_label: str
    source_excerpt: str | None
    importance: int
    embedding_status: Literal['indexed', 'pending'] = 'pending'
    status: str
    conflict_content: str | None
    created_at: str
    updated_at: str
    kind: str
    domain: str
    scene_key: str | None
    subject_entity_id: int | None
    subject_key: str
    qualifier: str
    event_status: str | None
    outcome: str | None
    time_text: str | None
    time_precision: str
    valid_from: str | None
    valid_until: str | None
    fresh_until: str | None
    time_meta: dict
    entity_ids: list[int]
    superseded_by_id: int | None
    conflict_with_id: int | None
    sources: list[dict]


class MemoryPage(CamelModel):
    items: list[MemoryData]
    next_cursor: str | None


class MemoryInput(MemoryCandidate, CamelModel):
    pass


class MemoryUpdate(CamelModel):
    type: MemoryType
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=2000)
    importance: int = Field(ge=1, le=5)
    mode: Literal['correct', 'transition'] = 'correct'
    kind: Literal['fact', 'event', 'ongoing_state'] | None = None
    qualifier: str | None = Field(default=None, max_length=200)
    event_status: Literal['planned', 'in_progress', 'completed', 'cancelled'] | None = None
    outcome: str | None = Field(default=None, max_length=1000)
    time_text: str | None = Field(default=None, max_length=200)
    time_precision: Literal['exact', 'day', 'window', 'recurring', 'unknown'] | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    time_meta: TimeMeta | None = None


class ConflictInput(CamelModel):
    content: str = Field(min_length=1, max_length=2000)
    action: Literal['keep', 'adopt', 'coexist'] = 'keep'
    qualifier: str = Field(default='', max_length=200)
    other_qualifier: str = Field(default='', max_length=200)


def _encode_cursor(item: Memory) -> str:
    return base64.urlsafe_b64encode(json.dumps([iso_utc(item.updated_at), item.id]).encode()).decode().rstrip('=')


def _decode_cursor(value: str) -> tuple[datetime, int]:
    try:
        moment, item_id = json.loads(base64.urlsafe_b64decode(value+'='*(-len(value)%4)).decode())
        return datetime.fromisoformat(moment.replace('Z','+00:00')), int(item_id)
    except (ValueError, TypeError, json.JSONDecodeError):
        fail(400, 40000, '分页游标无效')


def _companion(session) -> int:
    companion_id = session.scalar(select(Companion.id).limit(1))
    if companion_id is None:
        fail(404, 40401, '请先创建伴侣')
    return companion_id


def _load_memory(session, memory_id: int) -> Memory:
    row = session.get(Memory, memory_id)
    if row is None or row.companion_id != _companion(session) or row.deleted_at is not None or row.status == 'deleted':
        fail(404, 40401, '记忆不存在')
    return row


def _to_data(session, item: Memory, runtime: AgentRuntime) -> MemoryData:
    sources = [{'messageId': row.message_id, 'evidenceText': row.evidence_text,
                'messageSentAt': iso_utc(row.message_sent_at) if row.message_sent_at else None}
               for row in session.scalars(select(MemorySource).where(MemorySource.memory_id == item.id).order_by(MemorySource.message_id))]
    first = sources[0] if sources else {}
    stored, auth = session.get(MemoryEmbedding,item.id), runtime.embedding_auth
    indexed = bool(auth and auth.apikey.get_secret_value() and stored and item.status in ('active','historical')
                   and stored.profile == profile_id(auth) and stored.content_hash == content_hash(item) and _vector(stored) is not None)
    return MemoryData(**{name:getattr(item,name) for name in (
        'id','type','title','content','importance','status','conflict_content','kind','domain','scene_key',
        'subject_entity_id','subject_key','qualifier','event_status','outcome','time_text','time_precision',
        'time_meta','entity_ids','superseded_by_id','conflict_with_id')},
        source_message_id=first.get('messageId'), source_label=first.get('messageSentAt') or '用户整理',
        source_excerpt=first.get('evidenceText'), embedding_status='indexed' if indexed else 'pending',
        sources=sources, **{name:iso_utc(getattr(item,name)) if getattr(item,name) else None for name in (
            'created_at','updated_at','valid_from','valid_until','fresh_until')})


def candidate_data(session, item: Memory) -> dict:
    entities = list(session.scalars(select(MemoryEntity).where(MemoryEntity.id.in_(item.entity_ids))))
    subject = session.get(MemoryEntity,item.subject_entity_id) if item.subject_entity_id else None
    return {**{name:getattr(item,name) for name in (
        'type','subject_key','title','content','importance','sensitivity','kind','domain','scene_key','qualifier',
        'event_status','outcome','time_text','time_precision','valid_from','valid_until','time_meta')},
        'subject_person':subject.canonical_name if subject else 'user',
        'entities':[{'name':row.canonical_name,'entity_type':row.entity_type,'relation_to_user':row.relation_to_user,'aliases':row.aliases} for row in entities]}


@router.get('', response_model=ApiResponse[MemoryPage])
def list_memories(runtime: Annotated[AgentRuntime, Depends(require_session)], type: str = '',
                  kind: str = '', status_filter: str = Query(default='',alias='status'),
                  entity_id: int | None = Query(default=None,alias='entityId'), q: str = '',
                  cursor: str = '', limit: int = Query(default=50,ge=1,le=100)):
    statuses = status_filter.split(',') if status_filter else ['active','conflicted']
    if (type and type not in {item.value for item in MemoryType}) or (kind and kind not in {'fact','event','ongoing_state'}) or not set(statuses) <= {'active','historical','conflicted','retracted'}:
        fail(422,40000,'记忆筛选条件无效')
    with runtime.require_database().session() as session:
        companion_id = session.scalar(select(Companion.id).limit(1))
        if companion_id is None:
            return ApiResponse(data=MemoryPage(items=[],next_cursor=None))
        conditions = [Memory.companion_id == companion_id, Memory.status.in_(statuses), Memory.deleted_at.is_(None)]
        if type: conditions.append(Memory.type == type)
        if kind: conditions.append(Memory.kind == kind)
        if q: conditions.append(or_(Memory.title.contains(q,autoescape=True),Memory.content.contains(q,autoescape=True)))
        if cursor:
            moment, identifier = _decode_cursor(cursor)
            conditions.append(or_(Memory.updated_at < moment,and_(Memory.updated_at == moment,Memory.id < identifier)))
        rows = list(session.scalars(select(Memory).where(*conditions).order_by(Memory.updated_at.desc(),Memory.id.desc())))
        if entity_id is not None:
            rows = [row for row in rows if entity_id in row.entity_ids or row.subject_entity_id == entity_id]
        rows = rows[:limit+1]
        items = [_to_data(session,row,runtime) for row in rows[:limit]]
        return ApiResponse(data=MemoryPage(items=items,next_cursor=_encode_cursor(rows[limit-1]) if len(rows)>limit else None))


@router.get('/entities')
def list_entities(runtime: Annotated[AgentRuntime, Depends(require_session)]):
    with runtime.require_database().session() as session:
        companion_id = session.scalar(select(Companion.id).limit(1))
        entities = list(session.scalars(select(MemoryEntity).where(MemoryEntity.companion_id == companion_id,MemoryEntity.status=='active')))
        memories = list(session.scalars(select(Memory).where(Memory.companion_id==companion_id,Memory.status.in_(['active','historical','conflicted']),Memory.deleted_at.is_(None))))
        return ApiResponse(data=[{'id':row.id,'canonicalName':row.canonical_name,'entityType':row.entity_type,
            'relationToUser':row.relation_to_user,'aliases':row.aliases,
            'memoryCount':sum(row.id in item.entity_ids or item.subject_entity_id==row.id for item in memories)} for row in entities])


def _pending(session, candidate_id: int):
    row = session.get(Pending,candidate_id)
    job = session.get(MemoryJob,row.job_id) if row else None
    conversation = session.get(Conversation,job.conversation_id) if job else None
    if not row or not conversation or conversation.companion_id != _companion(session):
        fail(404,40401,'候选不存在')
    return row, conversation


@router.get('/candidates')
def list_candidates(runtime: Annotated[AgentRuntime, Depends(require_session)]):
    with runtime.require_database().session() as session:
        companion_id=session.scalar(select(Companion.id).limit(1))
        rows=session.scalars(select(Pending).join(MemoryJob).join(Conversation).where(Conversation.companion_id==companion_id,Pending.status=='pending').order_by(Pending.id.desc()))
        return ApiResponse(data=[{'id':row.id,'title':row.title,'content':row.content,'reason':row.reason,'kind':row.structured_data.get('kind','fact')} for row in rows])


@router.post('/candidates/{candidate_id}/{decision}')
def decide_candidate(candidate_id:int, decision:Literal['approve','reject'],runtime:Annotated[AgentRuntime,Depends(require_session)]):
    with runtime.embedding_lock,runtime.require_database().session() as session:
        row,conversation=_pending(session,candidate_id)
        if row.status!='pending':
            fail(409,40901,'候选已处理')
        payload=row.structured_data or {key:getattr(row,key) for key in ('type','subject_key','title','content','importance','confidence','sensitivity','source_message_ids')}
        candidate=MemoryCandidate.model_validate(payload)
        result=save_candidate(session,conversation.companion_id,candidate,conversation_id=conversation.id,tombstone=decision=='reject')
        row.status='approved' if decision=='approve' else 'rejected'
        row.decided_at=datetime.now(UTC)
        runtime.clear_memory_index_files()
        return ApiResponse(data={'memoryId':result.memory_id,'status':row.status})


@router.post('',response_model=ApiResponse[MemoryData])
def create_memory(body:MemoryInput,runtime:Annotated[AgentRuntime,Depends(require_session)]):
    candidate=MemoryCandidate.model_validate({**body.model_dump(),'source_message_ids':[],'evidence_quote':''})
    with runtime.embedding_lock,runtime.require_database().session() as session:
        result=save_candidate(session,_companion(session),candidate,manual=True)
        row=session.get(Memory,result.memory_id)
        session.flush()
        return ApiResponse(data=_to_data(session,row,runtime))


@router.get('/{memory_id}/history')
def memory_history(memory_id:int,runtime:Annotated[AgentRuntime,Depends(require_session)]):
    with runtime.require_database().session() as session:
        row=_load_memory(session,memory_id)
        rows=list(session.scalars(select(Memory).where(Memory.companion_id==row.companion_id,Memory.deleted_at.is_(None))))
        ids={memory_id}
        for _ in range(len(rows)):
            expanded=ids | {item.id for item in rows if item.superseded_by_id in ids} | {item.superseded_by_id for item in rows if item.id in ids and item.superseded_by_id is not None}
            if expanded==ids: break
            ids=expanded
        return ApiResponse(data=[_to_data(session,item,runtime) for item in rows if item.id in ids])


@router.put('/{memory_id}',response_model=ApiResponse[MemoryData])
def update_memory(memory_id:int,body:MemoryUpdate,runtime:Annotated[AgentRuntime,Depends(require_session)]):
    with runtime.embedding_lock,runtime.require_database().session() as session:
        old=_load_memory(session,memory_id)
        if old.status not in ('active', 'conflicted'):
            fail(409,40901,'请编辑当前版本；历史版本可查看或删除')
        updates=body.model_dump(exclude_unset=True,exclude={'mode'})
        if body.mode == 'transition' and 'valid_from' not in body.model_fields_set:
            updates['valid_from'] = None
        candidate=MemoryCandidate.model_validate({**candidate_data(session,old),**updates,
            'operation_hint':body.mode,'target_memory_id':old.id,'source_message_ids':[]})
        result=save_candidate(session,old.companion_id,candidate,manual=True)
        row=session.get(Memory,result.memory_id)
        session.flush()
        runtime.clear_memory_index_files()
        return ApiResponse(data=_to_data(session,row,runtime))


@router.post('/{memory_id}/resolve',response_model=ApiResponse[MemoryData])
def resolve_memory(memory_id:int,body:ConflictInput,runtime:Annotated[AgentRuntime,Depends(require_session)]):
    with runtime.embedding_lock,runtime.require_database().session() as session:
        row=_load_memory(session,memory_id)
        if row.status!='conflicted': fail(409,40901,'该记忆没有待处理冲突')
        branches = conflict_group(session, row)
        other=session.get(Memory,row.conflict_with_id) if row.conflict_with_id else None
        if other and other.companion_id != row.companion_id: fail(409,40901,'冲突关联无效')
        if other is None and row.conflict_content and body.content == row.conflict_content and body.action != 'coexist':
            # Old backups stored only the alternative text, without a branch row.
            candidate = MemoryCandidate.model_validate({**candidate_data(session, row),
                'content': body.content, 'operation_hint': 'correct',
                'target_memory_id': row.id, 'source_message_ids': []})
            result = save_candidate(session, row.companion_id, candidate, manual=True)
            session.flush()
            runtime.clear_memory_index_files()
            return ApiResponse(data=_to_data(session, session.get(Memory, result.memory_id), runtime))
        if body.action=='coexist':
            if len(branches) > 2:
                fail(409,40901,'该主题有多个冲突结论，请先选择一个结论，或编辑为有明确适用条件的记忆')
            if not other or not body.qualifier.strip() or not body.other_qualifier.strip() or body.qualifier.strip()==body.other_qualifier.strip():
                fail(422,40000,'并存需分别填写不同的适用条件')
            row.qualifier,other.qualifier=body.qualifier.strip(),body.other_qualifier.strip()
            other.status='active'
        else:
            selected = next((item for item in branches if item.content == body.content), None)
            if selected is None and branches:
                fail(422,40000,'请选择冲突中的一个结论，或使用编辑进行修正')
            if selected is not None:
                row = selected
            elif body.content != row.content:
                fail(422,40000,'请选择已有结论，或使用编辑进行修正')
            for item in branches:
                if item is not row:
                    item.status = 'retracted'
                    item.superseded_by_id = row.id
        for item in branches or [row]:
            if item:
                if item is row: item.status='active'
                item.conflict_content=None
                item.conflict_with_id=None
                item.updated_at = datetime.now(UTC)
                session.execute(delete(MemoryEmbedding).where(MemoryEmbedding.memory_id==item.id))
        session.flush()
        runtime.clear_memory_index_files()
        return ApiResponse(data=_to_data(session,row,runtime))


@router.delete('/{memory_id}',response_model=ApiResponse[None])
def remove_memory(memory_id:int,runtime:Annotated[AgentRuntime,Depends(require_session)]):
    with runtime.embedding_lock,runtime.require_database().session() as session:
        delete_memory(session,_load_memory(session,memory_id))
        runtime.clear_memory_index_files()
    return ApiResponse(data=None)


@router.delete('',response_model=ApiResponse[None])
def clear_memories(runtime:Annotated[AgentRuntime,Depends(require_session)]):
    with runtime.embedding_lock,runtime.require_database().session() as session:
        companion_id=session.scalar(select(Companion.id).limit(1))
        runtime.data_generation+=1
        if companion_id is not None: clear_memory_rows(session,companion_id)
        runtime.clear_memory_index_files()
    return ApiResponse(data=None)
