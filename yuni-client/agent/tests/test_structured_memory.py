import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from yuni_agent.config import StartupConfig
from yuni_agent.memory.models import MemoryCandidate
from yuni_agent.memory.worker import MemoryMaintenanceWorker
from yuni_agent.memory.writer import clear_memories, delete_memory, save_candidate, stored_utc
from yuni_agent.runtime import AgentRuntime
from yuni_agent.storage.migrations import upgrade_database
from yuni_agent.storage.models import (
    ChatMessage,
    Companion,
    Conversation,
    Memory,
    MemoryEntity,
    MemoryJob,
    MemorySource,
)


@pytest.fixture
def memory_env(tmp_path):
    runtime = AgentRuntime(config=StartupConfig.for_test(tmp_path))
    runtime.initialize_storage()
    with runtime.require_database().session() as session:
        companion = Companion(name='Yuni', user_nickname='你')
        session.add(companion)
        session.flush()
        conversation = Conversation(companion_id=companion.id)
        session.add(conversation)
        session.flush()
        ids = companion.id, conversation.id
    yield runtime, *ids
    runtime.require_database().close()


def add_message(env, content, **kwargs):
    runtime, _, conversation_id = env
    with runtime.require_database().session() as session:
        row = ChatMessage(conversation_id=conversation_id, role='user', status='succeeded', content=content, **kwargs)
        session.add(row)
        session.flush()
        return row.id


def write(env, content, **kwargs):
    runtime, companion_id, conversation_id = env
    message_id = kwargs.pop('message_id', None) or add_message(env, content)
    subject_key = kwargs.pop('subject_key', '居住城市')
    with runtime.require_database().session() as session:
        return save_candidate(session, companion_id, MemoryCandidate(
            type='user_fact', subject_key=subject_key, title='近况', content=content,
            source_message_ids=[message_id], **kwargs), conversation_id=conversation_id)


@pytest.mark.parametrize('operation,status', [('transition','historical'), ('correct','retracted')])
def test_state_change_and_correction_keep_independent_sources(memory_env, operation, status):
    before = write(memory_env, '住杭州', valid_from='2026-09-01T00:00:00+08:00')
    after = write(memory_env, '住上海', operation_hint=operation, valid_from='2026-10-01T00:00:00+08:00')
    runtime, _, _ = memory_env
    with runtime.require_database().session() as session:
        old, new = session.get(Memory, before.memory_id), session.get(Memory, after.memory_id)
        assert old.status == status and new.status == 'active'
        assert old.superseded_by_id == new.id
        if operation == 'transition':
            assert old.valid_until == new.valid_from
        assert len(list(session.scalars(select(MemorySource).where(MemorySource.memory_id == old.id)))) == 1
        assert len(list(session.scalars(select(MemorySource).where(MemorySource.memory_id == new.id)))) == 1


def test_conditions_and_people_are_separate_slots(memory_env):
    write(memory_env, '工作日住上海', qualifier='工作日')
    write(memory_env, '周末住苏州', qualifier='周末', operation_hint='coexist')
    write(memory_env, '同事小王住北京', subject_person='同事小王', entities=[{'name':'同事小王','relation_to_user':'同事'}])
    runtime, _, _ = memory_env
    with runtime.require_database().session() as session:
        rows = list(session.scalars(select(Memory)))
        assert len(rows) == 3 and all(row.status == 'active' for row in rows)
        assert sum(row.subject_entity_id is not None for row in rows) == 1
        assert len(list(session.scalars(select(MemoryEntity)))) == 1


def test_repeating_a_disputed_branch_does_not_resolve_it(memory_env):
    first = write(memory_env, '住上海')
    second = write(memory_env, '住北京')
    repeat = write(memory_env, '住上海')
    assert repeat.memory_id == first.memory_id and repeat.status == 'conflicted'
    with memory_env[0].require_database().session() as session:
        rows = list(session.scalars(select(Memory)))
        assert len(rows) == 2
        assert session.get(Memory, first.memory_id).conflict_with_id == second.memory_id
        assert session.get(Memory, second.memory_id).conflict_with_id == first.memory_id


def test_explicit_correction_of_repeated_conflicted_conclusion_resolves_all_branches(memory_env):
    first = write(memory_env, '住上海')
    second = write(memory_env, '住北京')
    third = write(memory_env, '住杭州')
    resolved = write(memory_env, '住上海', operation_hint='correct', target_memory_id=first.memory_id)
    assert resolved.status == 'active' and resolved.memory_id != first.memory_id
    with memory_env[0].require_database().session() as session:
        for item in (first, second, third):
            row = session.get(Memory, item.memory_id)
            assert row.status == 'retracted' and row.superseded_by_id == resolved.memory_id
            assert row.conflict_with_id is None


@pytest.mark.asyncio
async def test_time_queries_can_retrieve_noncurrent_active_intervals(memory_env):
    from yuni_agent.memory.retriever import SqliteMemoryRetriever
    future = write(memory_env, '计划明年住上海', valid_from='2030-01-01T00:00:00+08:00')
    retriever = SqliteMemoryRetriever(memory_env[0])
    assert await retriever.retrieve(memory_env[1], '居住城市') == []
    results = await retriever.retrieve(memory_env[1], '居住城市', time_from='2030-02-01', time_to='2030-02-01')
    assert [item.id for item in results] == [future.memory_id]


def test_event_progress_updates_in_place_but_new_date_creates_another_event(memory_env):
    first = write(memory_env, '周五面试', kind='event', event_status='planned', time_meta={'planned_start':'2026-10-23T12:00:00+08:00'})
    changed = write(memory_env, '周五面试取消', kind='event', event_status='cancelled', operation_hint='enrich', target_memory_id=first.memory_id)
    another = write(memory_env, '下个月再面试', kind='event', event_status='planned', time_meta={'planned_start':'2026-11-23T12:00:00+08:00'})
    assert first.memory_id == changed.memory_id != another.memory_id
    with memory_env[0].require_database().session() as session:
        assert session.get(Memory, first.memory_id).event_status == 'cancelled'


def test_deleted_watermark_blocks_old_paraphrase_but_allows_opposite_and_new_evidence(memory_env):
    first = write(memory_env, '喜欢不加糖咖啡')
    old_id = add_message(memory_env, '喜欢不加糖的咖啡')
    opposite_id = add_message(memory_env, '不喜欢不加糖咖啡')
    with memory_env[0].require_database().session() as session:
        delete_memory(session, session.get(Memory, first.memory_id))
    suppressed = write(memory_env, '喜欢不加糖的咖啡', message_id=old_id, target_memory_id=first.memory_id, same_conclusion=True)
    assert suppressed.status == 'deleted'
    opposite = write(memory_env, '不喜欢不加糖咖啡', message_id=opposite_id)
    assert opposite.status == 'active'
    new = write(memory_env, '喜欢不加糖咖啡', operation_hint='transition')
    assert new.status == 'active' and new.memory_id != first.memory_id


def test_deleted_fictional_record_does_not_suppress_real_world(memory_env):
    old = write(memory_env, '住上海', domain='fictional', scene_key='剧场A')
    message_id = add_message(memory_env, '住上海')
    with memory_env[0].require_database().session() as session:
        delete_memory(session, session.get(Memory, old.memory_id))
    assert write(memory_env, '住上海', message_id=message_id).status == 'active'


def test_freshness_is_anchored_to_user_message_and_expiry_does_not_mean_recovery(memory_env):
    sent = datetime(2026, 10, 1, 12, tzinfo=UTC)
    message_id = add_message(memory_env, '这两天感冒', created_at=sent)
    saved = write(memory_env, '这两天感冒', message_id=message_id, kind='ongoing_state')
    with memory_env[0].require_database().session() as session:
        row = session.get(Memory, saved.memory_id)
        assert stored_utc(row.fresh_until) == sent + timedelta(days=5)
        assert row.valid_until is None and row.status == 'active'


@pytest.mark.asyncio
async def test_clear_during_extraction_discards_inflight_results_and_advances_all_conversations(memory_env):
    runtime, companion_id, conversation_id = memory_env
    message_id = add_message(memory_env, '住上海')
    with runtime.require_database().session() as session:
        session.add(MemoryJob(conversation_id=conversation_id, from_message_id=message_id, to_message_id=message_id,
                              run_after=datetime.now(UTC)-timedelta(minutes=1)))
        other = Conversation(companion_id=companion_id)
        session.add(other)
        session.flush()
        other_message = ChatMessage(conversation_id=other.id, role='user', status='succeeded', content='其他会话')
        session.add(other_message)
        session.flush()
        other_id, other_message_id = other.id, other_message.id
    started, release = asyncio.Event(), asyncio.Event()
    class Model:
        async def complete_json(self, messages):
            started.set()
            await release.wait()
            return {'items':[{'type':'user_fact','subject_key':'居住城市','title':'居住地','content':'住上海','source_message_ids':[message_id]}]}
    task = asyncio.create_task(MemoryMaintenanceWorker(runtime).run_once(Model()))
    await started.wait()
    with runtime.embedding_lock, runtime.require_database().session() as session:
        runtime.data_generation += 1
        clear_memories(session, companion_id)
    release.set()
    assert await task
    with runtime.require_database().session() as session:
        assert list(session.scalars(select(Memory))) == []
        assert session.get(Conversation, other_id).memory_processed_until_message_id == other_message_id
        assert session.scalar(select(MemoryJob.status)) == 'cancelled'


def test_old_database_upgrade_preserves_sources_and_backfills_time(tmp_path):
    path = tmp_path / '0008.db'
    with sqlite3.connect(path) as db:
        db.executescript('''
        CREATE TABLE companions (id INTEGER PRIMARY KEY);
        CREATE TABLE conversations (id INTEGER PRIMARY KEY, companion_id INTEGER);
        CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, conversation_id INTEGER, created_at DATETIME);
        CREATE TABLE memories (id INTEGER PRIMARY KEY, companion_id INTEGER, type TEXT, status TEXT, deleted_at DATETIME, subject_key TEXT DEFAULT '');
        CREATE TABLE memory_sources (memory_id INTEGER, message_id INTEGER, evidence_text TEXT, PRIMARY KEY(memory_id,message_id));
        CREATE TABLE memory_candidates (id INTEGER PRIMARY KEY);
        CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY);
        INSERT INTO alembic_version VALUES ('0008_chat_batches');
        INSERT INTO companions VALUES(1);
        INSERT INTO conversations VALUES(1,1);
        INSERT INTO chat_messages VALUES(8,1,'2026-10-23 15:58:00');
        INSERT INTO memories(id,companion_id,type,status,deleted_at) VALUES(1,1,'important_event','active',NULL);
        INSERT INTO memory_sources VALUES(1,8,'明天拔牙');
        ''')
    upgrade_database(path)
    upgrade_database(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT kind,domain FROM memories').fetchone() == ('event','real_world')
        assert db.execute('SELECT evidence_text,message_sent_at FROM memory_sources').fetchone() == ('明天拔牙','2026-10-23 15:58:00')
        assert db.execute('PRAGMA foreign_key_check').fetchall() == []


@pytest.mark.asyncio
async def test_midnight_extraction_uses_message_timestamp_and_keeps_sensitive_payload(memory_env):
    runtime, _, conversation_id = memory_env
    message_id = add_message(memory_env, '明天去拔牙', created_at=datetime(2026,10,23,15,58,tzinfo=UTC))
    with runtime.require_database().session() as session:
        session.add(MemoryJob(conversation_id=conversation_id, from_message_id=message_id,
            to_message_id=message_id, run_after=datetime.now(UTC)-timedelta(minutes=1)))
    class Model:
        async def complete_json(self, messages):
            assert '2026-10-23T23:58:00+08:00' in messages[-1]['content']
            return {'items':[{'type':'important_event','kind':'event','subject_key':'拔牙',
                'title':'拔牙计划','content':'10月24日拔牙','event_status':'planned','sensitivity':'sensitive',
                'source_message_ids':[message_id], 'evidence_quote':'明天去拔牙',
                'time_meta':{'planned_start':'2026-10-24T09:00:00+08:00'}}]}
    assert await MemoryMaintenanceWorker(runtime).run_once(Model())
    from yuni_agent.storage.models import MemoryCandidate as Pending
    with runtime.require_database().session() as session:
        candidate = session.scalar(select(Pending))
        assert candidate.structured_data['event_status'] == 'planned'
        assert candidate.structured_data['time_meta']['planned_start'].startswith('2026-10-24')
        assert list(session.scalars(select(Memory))) == []

@pytest.mark.asyncio
async def test_retrieval_separates_current_history_and_half_open_transition(memory_env):
    from yuni_agent.memory.retriever import SqliteMemoryRetriever
    old = write(memory_env, '住杭州', valid_from='2026-09-01T00:00:00+08:00')
    new = write(memory_env, '住上海', operation_hint='transition', valid_from='2026-10-01T00:00:00+08:00')
    retriever = SqliteMemoryRetriever(memory_env[0])
    previous = await retriever.retrieve(memory_env[1], '居住城市', time_from='2026-09-01', time_to='2026-09-30', include_history=True)
    assert [row.id for row in previous] == [old.memory_id]
    boundary = await retriever.retrieve(memory_env[1], '居住城市', time_from='2026-10-01', time_to='2026-10-01', include_history=True)
    assert [row.id for row in boundary] == [new.memory_id]


@pytest.mark.asyncio
async def test_expired_state_is_available_for_explicit_history_without_inventing_recovery(memory_env):
    from yuni_agent.memory.retriever import SqliteMemoryRetriever
    message_id = add_message(memory_env, '感冒头疼', created_at=datetime.now(UTC)-timedelta(days=10))
    state = write(memory_env, '感冒头疼', message_id=message_id, kind='ongoing_state')
    retriever = SqliteMemoryRetriever(memory_env[0])
    assert await retriever.retrieve(memory_env[1], '感冒') == []
    past = await retriever.retrieve(memory_env[1], '感冒', include_history=True)
    assert [row.id for row in past] == [state.memory_id]
    assert past[0].status == 'active' and past[0].valid_until is None


@pytest.mark.asyncio
async def test_entity_alias_and_domain_filter_do_not_leak_fictional_memories(memory_env):
    from yuni_agent.memory.retriever import SqliteMemoryRetriever
    real = write(memory_env, '同事小王在北京', subject_person='同事小王', entities=[{'name':'同事小王','aliases':['王哥'],'relation_to_user':'同事'}])
    write(memory_env, '同事小王在月球', domain='fictional', scene_key='月球剧场', subject_person='同事小王')
    found = await SqliteMemoryRetriever(memory_env[0]).retrieve(memory_env[1], '王哥', entity_name='王哥')
    assert [row.id for row in found] == [real.memory_id]


def test_embedding_text_uses_natural_content_without_enum_labels(memory_env):
    from yuni_agent.memory.index import memory_text
    saved = write(memory_env, '用户工作日喝美式', qualifier='工作日')
    with memory_env[0].require_database().session() as session:
        text = memory_text(session.get(Memory,saved.memory_id))
    assert 'user_fact' not in text and '类型：' not in text
    assert text.count('工作日') == 1


def test_memory_api_versions_candidates_and_backup_round_trip(tmp_path):
    import json

    from fastapi.testclient import TestClient

    from yuni_agent.api.app import create_app
    from yuni_agent.storage.models import MemoryCandidate as Pending
    config = StartupConfig.for_test(tmp_path)
    runtime = AgentRuntime(config=config)
    headers = {'X-Yuni-Session':config.session_token.get_secret_value()}
    with TestClient(create_app(runtime)) as client:
        with runtime.require_database().session() as session:
            companion=Companion(name='Yuni',user_nickname='你')
            session.add(companion)
            session.flush()
            conversation=Conversation(companion_id=companion.id)
            session.add(conversation)
            session.flush()
            source=ChatMessage(conversation_id=conversation.id,role='user',status='succeeded',content='这两天头疼')
            session.add(source)
            session.flush()
            job=MemoryJob(conversation_id=conversation.id,from_message_id=source.id,to_message_id=source.id,run_after=datetime.now(UTC),status='succeeded')
            session.add(job)
            session.flush()
            payload=MemoryCandidate(type='user_fact',subject_key='健康近况',title='头疼',content='这两天头疼',kind='ongoing_state',sensitivity='sensitive',source_message_ids=[source.id])
            pending=Pending(job_id=job.id,type='user_fact',subject_key=payload.subject_key,title=payload.title,content=payload.content,importance=3,confidence=1,sensitivity='sensitive',source_message_ids=[source.id],structured_data=payload.model_dump(mode='json'))
            session.add(pending)
        created=client.post('/api/memories',headers=headers,json={'type':'user_fact','subjectKey':'居住城市','title':'居住地','content':'住杭州','entities':[{'name':'杭州','entity_type':'place'}]})
        assert created.status_code==200,created.text
        old_id=created.json()['data']['id']
        updated=client.put(f'/api/memories/{old_id}',headers=headers,json={'type':'user_fact','title':'居住地','content':'住上海','importance':3,'mode':'transition','validFrom':'2026-10-01T00:00:00+08:00'})
        assert updated.status_code==200,updated.text
        new_id=updated.json()['data']['id']
        assert new_id!=old_id
        history=client.get(f'/api/memories/{new_id}/history',headers=headers).json()['data']
        assert {row['status'] for row in history}=={'active','historical'}
        assert client.get('/api/memories/entities',headers=headers).json()['data'][0]['canonicalName']=='杭州'
        pending=client.get('/api/memories/candidates',headers=headers).json()['data']
        assert len(pending)==1
        rejected=client.post(f"/api/memories/candidates/{pending[0]['id']}/reject",headers=headers)
        assert rejected.status_code==200,rejected.text
        backup=client.get('/api/data/export',headers=headers).json()
        assert len(backup['tables']['memoryEntities'])==1
        assert any(row['status']=='deleted' for row in backup['tables']['memories'])
        imported=client.post('/api/data/import',headers=headers,files={'file':('backup.json',json.dumps(backup).encode(),'application/json')})
        assert imported.status_code==200,imported.text
        assert len(client.get(f'/api/memories/{new_id}/history',headers=headers).json()['data'])==2
        with runtime.require_database().session() as session:
            tombstone=session.scalar(select(Memory).where(Memory.status=='deleted'))
            assert tombstone.suppressed_until_message_id is not None
        legacy=client.get('/api/data/export',headers=headers).json()
        legacy['tables'].pop('memoryEntities')
        for row in legacy['tables']['memories']:
            row.pop('entity_ids',None)
            row.pop('subject_entity_id',None)
            row.pop('kind',None)
            row.pop('domain',None)
        restored=client.post('/api/data/import',headers=headers,files={'file':('legacy.json',json.dumps(legacy).encode(),'application/json')})
        assert restored.status_code==200,restored.text


def test_api_resolves_all_conflict_branches_and_legacy_alternative(tmp_path):
    from fastapi.testclient import TestClient

    from yuni_agent.api.app import create_app
    config = StartupConfig.for_test(tmp_path)
    runtime = AgentRuntime(config=config)
    headers = {'X-Yuni-Session': config.session_token.get_secret_value()}
    with TestClient(create_app(runtime)) as client:
        with runtime.require_database().session() as session:
            companion = Companion(name='Yuni', user_nickname='你')
            session.add(companion)
            session.flush()
            branches = []
            for content in ('住上海', '住北京', '住杭州'):
                result = save_candidate(session, companion.id, MemoryCandidate(
                    type='user_fact', subject_key='居住地', title='居住地', content=content), manual=True)
                branches.append(result.memory_id)
            legacy = Memory(companion_id=companion.id, type='user_preference', subject_key='饮食',
                            title='饮食', content='喜欢甜食', status='conflicted', conflict_content='不喜欢甜食')
            session.add(legacy)
            session.flush()
            legacy_id = legacy.id
        rejected = client.post(f'/api/memories/{branches[-1]}/resolve', headers=headers,
            json={'action': 'coexist', 'content': '住杭州', 'qualifier': '周末', 'otherQualifier': '工作日'})
        assert rejected.status_code == 409
        response = client.post(f'/api/memories/{branches[-1]}/resolve', headers=headers,
                               json={'content': '住上海', 'action': 'adopt'})
        assert response.status_code == 200, response.text
        assert response.json()['data']['id'] == branches[0]
        with runtime.require_database().session() as session:
            assert [session.get(Memory, item).status for item in branches] == ['active', 'retracted', 'retracted']
        response = client.post(f'/api/memories/{legacy_id}/resolve', headers=headers,
                               json={'content': '不喜欢甜食', 'action': 'adopt'})
        assert response.status_code == 200, response.text
        assert response.json()['data']['status'] == 'active'
        with runtime.require_database().session() as session:
            assert session.get(Memory, legacy_id).status == 'retracted'


def test_llm_schema_normalization_and_event_transition_fallback(memory_env):
    from yuni_agent.memory.worker import _parse_candidates
    raw = {
        "memories": [
            {
                "type": "user_fact", "kind": "fact", "subject_key": "居住城市", "title": "搬家",
                "content": "搬到上海", "source_message_ids": [1], "qualifier": None,
                "entities": [{"name": "上海", "entity_type": "location", "aliases": None}],
                "fresh_days": None,
            },
            {
                "type": "user_fact", "kind": "ongoing_state", "subject_key": "健康:感冒", "title": "感冒",
                "content": "嗓子疼", "source_message_ids": [1], "event_status": "ongoing", "fresh_days": 14,
            },
            {
                "type": "important_event", "kind": "event", "domain": "conversation",
                "subject_key": "task.notes_txt", "title": "把牛奶改成酸奶",
                "content": "把 notes.txt 里的牛奶改成酸奶", "importance": 2, "source_message_ids": [1],
            },
        ]
    }
    parsed = _parse_candidates(raw, {1})
    assert len(parsed) == 2
    assert parsed[0].entities[0].entity_type == "place" and parsed[0].qualifier == ""
    assert parsed[1].event_status is None and parsed[1].fresh_days == 14
    first = write(memory_env, "周五面试", kind="event", event_status="planned", time_meta={"planned_start": "2026-10-23T14:00:00+08:00"})
    updated = write(memory_env, "周五面试取消", kind="event", event_status="cancelled", operation_hint="transition", target_memory_id=first.memory_id)
    assert updated.memory_id == first.memory_id and updated.outcome == "enriched"


@pytest.mark.asyncio
async def test_fresh_ongoing_state_proactive_recall_and_four_branch_conflict_resolution(memory_env):
    from yuni_agent.memory.retriever import SqliteMemoryRetriever
    runtime, companion_id, _ = memory_env
    # 1. Fresh ongoing_state is proactively recalled even when current chat input doesn't repeat the illness word
    fresh_state = write(memory_env, "这两天感冒嗓子疼", kind="ongoing_state")
    retriever = SqliteMemoryRetriever(runtime)
    recalled = await retriever.retrieve(companion_id, "早安呀，今天起得真早")
    assert [item.id for item in recalled] == [fresh_state.memory_id]
    # 2. Event outcome and time_text are searchable via BM25 lexical search
    ev = write(memory_env, "周五去字节面试", kind="event", event_status="cancelled", outcome="因岗位冻结取消", time_text="2026年10月下旬")
    by_outcome = await retriever.retrieve(companion_id, "岗位冻结")
    assert ev.memory_id in [item.id for item in by_outcome]
    # 3. Tombstone suppresses same_conclusion=True even if LLM omitted target_memory_id
    tomb = write(memory_env, "用户讨厌吃苦瓜", subject_key="苦瓜偏好")
    old_msg_id = add_message(memory_env, "我不爱吃苦瓜")
    with runtime.require_database().session() as session:
        delete_memory(session, session.get(Memory, tomb.memory_id))
    suppressed = write(memory_env, "不爱吃苦瓜", subject_key="苦瓜偏好", message_id=old_msg_id, same_conclusion=True, target_memory_id=None)
    assert suppressed.outcome == "suppressed" and suppressed.status == "deleted"
    # 4. 4+ sequential conflict branches all resolve together without leaving orphaned pairs
    b1 = write(memory_env, "住上海", subject_key="四分支测试")
    b2 = write(memory_env, "住北京", subject_key="四分支测试")
    b3 = write(memory_env, "住杭州", subject_key="四分支测试")
    b4 = write(memory_env, "住深圳", subject_key="四分支测试")
    resolved = write(memory_env, "住上海", subject_key="四分支测试", operation_hint="correct", target_memory_id=b1.memory_id)
    assert resolved.status == "active"
    with runtime.require_database().session() as session:
        for b in (b1, b2, b3, b4):
            assert session.get(Memory, b.memory_id).status == "retracted"
    # 5. Preference evolution transitions old memory to historical even if polarity was placed in qualifier,
    #    and default retrieval attaches previous_facts on the active successor.
    old_pref = write(memory_env, "用户不喜欢吃香菜。", subject_key="food.cilantro", qualifier="dislike")
    new_pref = write(
        memory_env, "用户现在挺喜欢吃香菜的。", subject_key="food.cilantro",
        qualifier="current_like", operation_hint="transition", target_memory_id=old_pref.memory_id,
    )
    with runtime.require_database().session() as session:
        old_row = session.get(Memory, old_pref.memory_id)
        new_row = session.get(Memory, new_pref.memory_id)
        assert old_row.status == "historical" and old_row.superseded_by_id == new_row.id
        assert new_row.status == "active" and new_row.qualifier == ""
    recalled_pref = await retriever.retrieve(companion_id, "香菜")
    assert recalled_pref[0].id == new_pref.memory_id and old_pref.memory_id not in [item.id for item in recalled_pref]
    assert recalled_pref[0].previous_facts == ["用户不喜欢吃香菜。"]
