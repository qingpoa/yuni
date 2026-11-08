from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from planner_helpers import reply_call
from sqlalchemy import delete, select
from test_business_api import make_client
from test_chat_pipeline import initialized_runtime, seed_conversation

from yuni_agent.agent.planner import ReplyAction
from yuni_agent.context.budget import ContextBudget
from yuni_agent.expressions import config
from yuni_agent.expressions.index import ExpressionIndex, rank_candidates
from yuni_agent.expressions.selector import ExpressionSelector
from yuni_agent.expressions.worker import ExpressionWorker
from yuni_agent.model.auth import ModelAuth
from yuni_agent.model.client import ModelClientError, OpenAICompatibleClient
from yuni_agent.model.embedding import EmbeddingClient, normalized_vector, profile_id
from yuni_agent.storage.models import (
    ChatMessage,
    Companion,
    Conversation,
    Expression,
    ExpressionProgress,
    ExpressionSource,
    ExpressionVector,
)


class FakeEmbedding:
    def __init__(self, model='embed', dimension=2, delay=0):
        self.auth = ModelAuth(model=model, url='https://embedding.example/v1', apikey='fake')
        self.profile = profile_id(self.auth)
        self.dimension = dimension
        self.delay = delay
        self.calls = []
    async def embed(self, texts):
        self.calls.append(texts)
        await asyncio.sleep(self.delay)
        return [normalized_vector(([1, 0] if '开心' in text or '表达选择' in text else [0, 1]) + [0] * (self.dimension - 2)) for text in texts]


def seed_expressions(runtime, companion_id, count=12):
    with runtime.require_database().session() as session:
        for i in range(count):
            session.add(Expression(companion_id=companion_id, situation=f'开心{i}' if i % 2 else f'安静{i}', style=f'表达{i}', count=i + 1))


@pytest.mark.asyncio
async def test_user_only_review_source_frequency_idempotence_and_progress(tmp_path):
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, current_id = seed_conversation(runtime)
    class Learn:
        calls = 0
        async def complete_json(self, messages):
            self.calls += 1
            if self.calls % 2:
                return {'expressions': [
                    {'situation': '惊叹', 'style': '这么强', 'source_id': current_id},
                    {'situation': '惊叹', 'style': '这么强', 'source_id': current_id},
                    {'situation': '角色', 'style': '不能学我', 'source_id': current_id - 1},
                    {'situation': '错来源', 'style': '别学', 'source_id': 999999},
                ]}
            return {'suitable': True}
    class Consistent(Learn):
        async def complete_json(self, messages):
            if '真实聊天提取' in messages[0]['content']:
                return {'expressions': [
                    {'situation': '惊叹', 'style': '这么强', 'source_id': str(current_id)},
                    {'situation': '惊叹', 'style': '这么强', 'source_id': current_id},
                    {'situation': '角色', 'style': '不学角色', 'source_id': current_id - 1},
                    {'situation': '错来源', 'style': '别学', 'source_id': 999999},
                ]}
            return {'suitable': True}
    try:
        worker = ExpressionWorker(runtime)
        assert await worker.run_once(Consistent(), force=True)
        with runtime.require_database().session() as session:
            rows = list(session.scalars(select(Expression)))
            assert len(rows) == 1 and rows[0].companion_id == companion_id and rows[0].count == 1
            assert len(list(session.scalars(select(ExpressionSource)))) == 1
            assert session.get(ExpressionProgress, conversation_id).until_message_id == current_id
            session.execute(delete(ExpressionProgress))
        assert await worker.run_once(Consistent(), force=True)
        with runtime.require_database().session() as session:
            assert session.scalar(select(Expression)).count == 1
        assert not await worker.run_once(Consistent(), force=True)
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize('style', ['也太离谱了吧', 'xxxx也太离谱了吧'])
async def test_learning_and_review_receive_grounded_same_conversation_protocol(tmp_path, style):
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, current_id = seed_conversation(runtime)
    user_text = '这个报价也太离谱了吧。\n"别替我润色"'
    with runtime.require_database().session() as session:
        previous = session.scalar(select(ChatMessage).where(
            ChatMessage.conversation_id == conversation_id,
            ChatMessage.role == 'assistant', ChatMessage.status == 'succeeded'))
        previous.content = '维修店给出的报价比新机还贵。'
        previous_id = previous.id
        session.get(ChatMessage, current_id).content = user_text
        other = Conversation(companion_id=companion_id)
        session.add(other)
        session.flush()
        unrelated = ChatMessage(conversation_id=other.id, role='user',
                                content='另一个会话的无关原文', status='succeeded')
        session.add(unrelated)
        session.flush()
        unrelated_id = unrelated.id
    expected_context = [
        {'source_id': previous_id, 'speaker': 'SELF', 'content': '维修店给出的报价比新机还贵。'},
        {'source_id': current_id, 'speaker': 'USER', 'content': user_text},
    ]
    candidate = {'situation': '嫌报价过高', 'style': style, 'source_id': current_id}

    class Model:
        def __init__(self):
            self.calls = []

        async def complete_json(self, messages):
            self.calls.append(messages)
            assert messages[0]['role'] == 'system'
            assert all(item['role'] == 'user' for item in messages[1:])
            if len(self.calls) == 1:
                assert '真实聊天提取' in messages[0]['content']
                assert [json.loads(item['content']) for item in messages[1:]] == expected_context
                return {'expressions': [candidate,
                    {**candidate, 'source_id': previous_id},
                    {**candidate, 'source_id': unrelated_id},
                    {**candidate, 'source_id': True}]}
            assert len(self.calls) == 2  # Invalid sources never reach review.
            assert len(messages) == 2
            assert json.loads(messages[1]['content']) == {
                **candidate, 'USER': user_text, 'context': expected_context,
            }
            return {'suitable': True}

    model = Model()
    try:
        assert await ExpressionWorker(runtime).learn_batch(model, force=True)
        assert len(model.calls) == 2
        with runtime.require_database().session() as session:
            expression = session.scalar(select(Expression))
            assert expression.style == style and expression.situation == candidate['situation']
            assert [(source.expression_id, source.message_id) for source in session.scalars(select(ExpressionSource))] == [(expression.id, current_id)]
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize('review', [{'suitable': False}, {'suitable': 1}, {'suitable': True, 'extra': 1}, {}])
async def test_review_must_explicitly_approve_and_index_failure_keeps_learning(tmp_path, review):
    runtime = initialized_runtime(tmp_path)
    _, _, current_id = seed_conversation(runtime)
    class Model:
        calls = 0
        async def complete_json(self, messages):
            self.calls += 1
            return {'expressions': [{'situation': '开心', 'style': '这么强', 'source_id': current_id}]} if self.calls == 1 else review
    try:
        assert await ExpressionWorker(runtime).run_once(Model(), force=True)
        with runtime.require_database().session() as session:
            assert session.scalar(select(Expression)) is None
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_index_backfill_restart_profile_dimension_and_real_vector_recall(tmp_path):
    runtime = initialized_runtime(tmp_path)
    companion_id, _, _ = seed_conversation(runtime)
    seed_expressions(runtime, companion_id)
    client = FakeEmbedding()
    index = ExpressionIndex(runtime)
    try:
        assert await index.backfill(client) == 12
        assert await index.backfill(client) == 0
        with runtime.require_database().session() as session:
            items = [dict(id=item.id, companion_id=item.companion_id, situation=item.situation, style=item.style, count=item.count) for item in session.scalars(select(Expression))]
        recalled = await index.recall(items, '表达选择：开心', client)
        assert recalled[0]['situation'].startswith('开心')
        assert await index.recall([{**item, 'companion_id': 999} for item in items], '开心', client) == []
        runtime.close_storage()
        runtime.initialize_storage()
        assert await ExpressionIndex(runtime).backfill(client) == 0
        changed = FakeEmbedding(model='changed')
        assert await index.recall(items, '开心', changed) == []
        assert await index.backfill(changed) == 12
        dimension = FakeEmbedding(model='changed', dimension=3)
        assert await index.recall(items, '开心', dimension) == []
        assert await index.backfill(dimension) == 12
        with runtime.require_database().session() as session:
            assert {row.dimension for row in session.scalars(select(ExpressionVector))} == {3}
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['config', 'clear-key', 'rebuild'])
async def test_inflight_embedding_cannot_write_after_generation_change(tmp_path, operation):
    runtime = initialized_runtime(tmp_path)
    companion_id, _, _ = seed_conversation(runtime)
    seed_expressions(runtime, companion_id)
    runtime.save_embedding_auth('embed', 'https://embedding.example/v1', 'fake')
    started, release = asyncio.Event(), asyncio.Event()
    class Waiting(FakeEmbedding):
        async def embed(self, texts):
            started.set()
            await release.wait()
            return await super().embed(texts)
    try:
        task = asyncio.create_task(ExpressionIndex(runtime).backfill(Waiting()))
        await started.wait()
        if operation == 'config':
            runtime.save_embedding_auth('changed', 'https://embedding.example/v1', 'new-key')
        elif operation == 'clear-key':
            runtime.remove_embedding_auth()
        else:
            runtime.embedding_generation += 1
        release.set()
        assert await task == 0
        with runtime.require_database().session() as session:
            assert session.scalar(select(ExpressionVector)) is None
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_selector_weighted_fallback_always_role_filters(tmp_path, monkeypatch, failure):
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, _ = seed_conversation(runtime)
    seed_expressions(runtime, companion_id)
    seen = []
    class Model:
        async def complete_json(self, messages):
            seen.extend(messages)
            assert '真实用户当前句' in str(messages) and '当前人物标记' in str(messages)
            assert '表达选择意图' in str(messages)
            if failure:
                raise ModelClientError('fake')
            candidates = json.loads(next(item['content'].split('\n候选：')[1] for item in messages if '\n候选：' in item['content']))
            return {'selected_ids': [candidates[0]['id']]}
    try:
        messages = [{'role': 'system', 'content': '当前人物标记'}, {'role': 'user', 'content': '真实用户当前句'}]
        result = await ExpressionSelector(runtime).select({'conversation_id': conversation_id}, ReplyAction.model_validate(reply_call().arguments), messages, Model(), ContextBudget())
        assert bool(result) is not failure and seen
        if failure:
            assert result == ''
        else:
            assert '不是已发生事实' in result
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_selector_skips_under_ten_and_rejects_invalid_or_out_of_scope_ids(tmp_path):
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, _ = seed_conversation(runtime)
    seed_expressions(runtime, companion_id, 9)
    class Model:
        calls = 0
        async def complete_json(self, messages):
            self.calls += 1
            return {'selected_ids': [999999]}
    model = Model()
    try:
        selector = ExpressionSelector(runtime)
        args = ({'conversation_id': conversation_id}, ReplyAction.model_validate(reply_call().arguments), [{'role': 'user', 'content': '用户'}], model, ContextBudget())
        assert await selector.select(*args) == '' and model.calls == 0
        with runtime.require_database().session() as session:
            session.add(Expression(companion_id=companion_id, situation='新情境', style='新表达', count=1))
        assert await selector.select(*args) == '' and model.calls == 1
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_learning_skips_oversized_source_and_rotates_conversations_without_truncation(tmp_path):
    runtime = initialized_runtime(tmp_path)
    companion_id, first, _ = seed_conversation(runtime)
    with runtime.require_database().session() as session:
        original = session.scalar(select(ChatMessage).where(ChatMessage.conversation_id == first).order_by(ChatMessage.id))
        original.content = '超长' * 1000
        original_id = original.id
        second = Conversation(companion_id=companion_id)
        session.add(second)
        session.flush()
        second_id = second.id
        session.add(ChatMessage(conversation_id=second_id, role='user', content='另一个会话', status='succeeded'))
    class Model:
        calls = 0
        async def complete_json(self, messages):
            self.calls += 1
            assert '另一个会话' in str(messages) and '超长' not in str(messages)
            return {'expressions': []}
    try:
        worker = ExpressionWorker(runtime, ContextBudget(total_tokens=600, response_reserve_tokens=0))
        assert await worker.run_once(Model(), force=True)
        with runtime.require_database().session() as session:
            assert session.get(ExpressionProgress, first).last_skipped_message_id == original_id
            assert session.get(ChatMessage, original_id).content == '超长' * 1000
        model = Model()
        assert await worker.run_once(model, force=True) and model.calls == 1
        with runtime.require_database().session() as session:
            assert session.get(ExpressionProgress, second_id) is not None
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_selector_preserves_current_input_and_published_reply_under_budget_pressure(tmp_path):
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, _ = seed_conversation(runtime)
    seed_expressions(runtime, companion_id)
    class Model:
        calls = 0
        async def complete_json(self, messages):
            self.calls += 1
            return {'selected_ids': []}
    model = Model()
    try:
        from planner_helpers import reply_call

        from yuni_agent.agent.planner import ReplyAction
        messages = [{'role': 'user', 'content': '必须保留当前输入' * 1000},
                    {'role': 'assistant', 'content': '本轮已经发送的台词'}]
        result = await ExpressionSelector(runtime).select({'conversation_id': conversation_id},
            ReplyAction.model_validate(reply_call().arguments), messages, model,
            ContextBudget(total_tokens=2000, response_reserve_tokens=0))
        assert result == '' and model.calls == 0
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_optional_calls_allow_normal_delay_but_bound_timeout_and_propagate_cancel(tmp_path, monkeypatch):
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, _ = seed_conversation(runtime)
    seed_expressions(runtime, companion_id)
    monkeypatch.setattr(config, 'SELECTION_TIMEOUT', 0.06)
    class Slow:
        def __init__(self, delay):
            self.delay = delay
            self.completed = False
        async def complete_json(self, messages):
            await asyncio.sleep(self.delay)
            self.completed = True
            return {'selected_ids': []}
    args = ({'conversation_id': conversation_id}, ReplyAction.model_validate(reply_call().arguments), [{'role': 'user', 'content': '当前'}])
    try:
        normal = Slow(0.02)
        assert await ExpressionSelector(runtime).select(*args, normal, ContextBudget()) == '' and normal.completed
        timeout = Slow(1)
        assert await ExpressionSelector(runtime).select(*args, timeout, ContextBudget()) == '' and not timeout.completed
        task = asyncio.create_task(ExpressionSelector(runtime).select(*args, Slow(1), ContextBudget()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        worker = ExpressionWorker(runtime)
        async with runtime.maintenance_lock:
            # Expression work does not wait for the factual-memory/compact lock.
            assert await asyncio.wait_for(worker.run_once(SimpleNamespace(complete_json=lambda messages: None), force=False), timeout=0.5) is False
    finally:
        runtime.close_storage()


def test_embedding_config_is_independent_restored_redacted_and_rebuildable(tmp_path):
    client, runtime, headers = make_client(tmp_path)
    payload = {'provider': 'openai_compatible', 'baseUrl': 'http://embedding.example/v1', 'modelName': 'embed', 'apiKey': 'embedding-secret-fake'}
    with client:
        assert client.put('/api/settings/embedding', json=payload).status_code == 401
        response = client.put('/api/settings/embedding', headers=headers, json=payload)
        assert response.status_code == 200 and response.json()['data']['embedding']['configured']
        assert runtime.model_api_key is None
        assert 'embedding-secret-fake' not in response.text + client.get('/api/data/export', headers=headers).text
        assert client.put('/api/settings/embedding', headers=headers, json={**payload, 'apiKey': '', 'baseUrl': 'http://other.example/v1'}).status_code == 400
        assert client.post('/api/settings/embedding/rebuild', headers=headers).status_code == 200
    client, runtime, headers = make_client(tmp_path)
    with client:
        assert runtime.embedding_auth.apikey.get_secret_value() == 'embedding-secret-fake'
        assert client.get('/api/settings', headers=headers).json()['data']['embedding']['configured']
        assert client.delete('/api/settings/embedding/key', headers=headers).json()['data']['embedding']['configured'] is False
    client, runtime, headers = make_client(tmp_path)
    with client:
        assert not runtime.embedding_auth.apikey.get_secret_value()
        client.delete('/api/data', headers=headers)
        assert not runtime.embedding_auth_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', ['valid', 'nan', 'dimension', 'model', 'count'])
async def test_embedding_http_validates_vectors_and_hides_error_details(tmp_path, monkeypatch, invalid):
    original = httpx.AsyncClient
    async def handle(request):
        await asyncio.sleep(0.02)
        assert request.url.path == '/v1/embeddings'
        assert json.loads(request.content)['model'] == 'embed'
        data = [{'index': 0, 'embedding': [1, 0]}, {'index': 1, 'embedding': [0, 1]}]
        model = 'embed'
        if invalid == 'nan':
            return httpx.Response(500, text='private-key-and-server-body')
        if invalid == 'dimension':
            data[1]['embedding'] = [1, 0, 0]
        if invalid == 'model':
            model = 'other'
        if invalid == 'count':
            data.pop()
        return httpx.Response(200, json={'model': model, 'data': data})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    client = EmbeddingClient(FakeEmbedding().auth)
    if invalid == 'valid':
        assert await client.embed(['a', 'b']) == [[1.0, 0.0], [0.0, 1.0]]
    else:
        with pytest.raises(ModelClientError) as error:
            await client.embed(['a', 'b'])
        assert 'private' not in str(error.value)


@pytest.mark.asyncio
async def test_internal_expression_json_disables_thinking_without_changing_chat(monkeypatch):
    original = httpx.AsyncClient
    bodies = []
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"selected_ids":[]}'}}]})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    client = OpenAICompatibleClient(base_url='https://gateway.example/v1', model='deepseek-flash', api_key='fake')
    assert await client.for_internal_json().complete_json([]) == {'selected_ids': []}
    assert bodies[-1]['thinking'] == {'type': 'disabled'} and bodies[-1]['response_format'] == {'type': 'json_object'}
    assert client._provider_options()['thinking'] == {'type': 'enabled'}


@pytest.mark.asyncio
async def test_production_embedding_http_to_index_selector_and_graph_replyer(tmp_path, monkeypatch, caplog):
    import logging

    from planner_helpers import reply_delivered, reply_event, stop_event

    from yuni_agent.agent.graph import ChatGraphContext, LangGraphChat
    from yuni_agent.context.builder import SqliteContextBuilder
    from yuni_agent.model.reply import REPLY_OUTPUT_INSTRUCTIONS
    caplog.set_level(logging.INFO)
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, current_id = seed_conversation(runtime)
    seed_expressions(runtime, companion_id)
    runtime.save_embedding_auth('embed', 'https://embedding.example/v1', 'fake-key')
    with runtime.require_database().session() as session:
        session.get(Companion, companion_id).custom_description = '真实人物标记'
    original = httpx.AsyncClient
    requests = []
    def handle(request):
        assert request.url.path == '/v1/embeddings'
        body = json.loads(request.content)
        requests.append(body)
        data = [{'index': i, 'embedding': [1, 0] if '开心' in text else [0, 1]} for i, text in enumerate(body['input'])]
        return httpx.Response(200, json={'model': 'embed', 'data': list(reversed(data))})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr('yuni_agent.expressions.selector.fallback_candidates', lambda items: (_ for _ in ()).throw(AssertionError('vector recall must be used')))
    class Model:
        selected = False
        replied = False
        async def stream_chat_with_tools(self, messages, tools):
            yield stop_event() if reply_delivered(messages) else reply_event('用户开心，参考真实上下文', focus='开心')
        async def complete_json(self, messages):
            assert all(message.get('content') != REPLY_OUTPUT_INSTRUCTIONS for message in messages)
            prompt = next(item['content'] for item in messages if '\n候选：' in item['content'])
            candidates = json.loads(prompt.split('\n候选：')[1])
            assert candidates[0]['situation'].startswith('开心') and len(candidates) <= 50
            assert '真实人物标记' in str(messages) and '用户当前开心' in str(messages)
            self.selected = True
            return {'selected_ids': [candidates[0]['id']]}
        async def stream_chat(self, messages):
            assert any(message.get('content') == REPLY_OUTPUT_INSTRUCTIONS for message in messages)
            assert self.selected and '你可以视情况自由使用以下句式' in str(messages)
            assert '当前开心' in str(messages) and '真实人物标记' in str(messages)
            self.replied = True
            yield '这是纯文本回复。全部标点都保留！'
    try:
        assert await ExpressionIndex(runtime).backfill() == 12
        model = Model()
        events = [event async for event in LangGraphChat().stream({'conversation_id': conversation_id, 'user_message_id': current_id, 'assistant_message_id': current_id + 1, 'current_input': '用户当前开心'}, ChatGraphContext(context_builder=SqliteContextBuilder(runtime), model_client=model, expression_selector=ExpressionSelector(runtime)))]
        assert model.replied and events[-1].data['content'] == '这是纯文本回复。全部标点都保留！'
        assert ''.join(events[-1].data['replyParts']) == events[-1].data['content']
        assert len(requests) == 2 and len(requests[0]['input']) == 12 and len(requests[1]['input']) == 1
        assert '[Flow 1/4 Planner] step=1 action=reply' in caplog.text
        assert '[Flow 2/4 Recall] mode=vector_mmr pool=12 recalled=12' in caplog.text
        assert '[Flow 3/4 Select] selected=1' in caplog.text
        assert '[Flow 4/4 Replyer] style=自然 expressions_injected=True' in caplog.text
        assert '[Flow 1/4 Planner] step=2 action=stop' in caplog.text
        assert 'fake-key' not in caplog.text
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_learning_timeout_or_cancellation_never_advances_cursor(tmp_path, monkeypatch, cancel):
    runtime = initialized_runtime(tmp_path)
    _, conversation_id, _ = seed_conversation(runtime)
    started = asyncio.Event()
    class Slow:
        async def complete_json(self, messages):
            started.set()
            await asyncio.sleep(10)
            return {'expressions': []}
    monkeypatch.setattr(config, 'LEARNING_BATCH_TIMEOUT', 0.03)
    try:
        task = asyncio.create_task(ExpressionWorker(runtime).run_once(Slow(), force=True))
        await started.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task is False
        with runtime.require_database().session() as session:
            assert session.get(ExpressionProgress, conversation_id) is None
            assert session.scalar(select(Expression)) is None
        assert not runtime.expression_lock.locked()
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_same_embedding_space_key_update_keeps_index_but_advances_generation(tmp_path):
    runtime = initialized_runtime(tmp_path)
    companion_id, _, _ = seed_conversation(runtime)
    seed_expressions(runtime, companion_id)
    try:
        runtime.save_embedding_auth('embed', 'https://embedding.example/v1', 'fake')
        assert await ExpressionIndex(runtime).backfill(FakeEmbedding()) == 12
        generation = runtime.embedding_generation
        runtime.save_embedding_auth('embed', 'https://embedding.example/v1', 'changed-key')
        assert runtime.embedding_generation > generation
        with runtime.require_database().session() as session:
            assert len(list(session.scalars(select(ExpressionVector)))) == 12
        assert await ExpressionIndex(runtime).backfill(FakeEmbedding()) == 0
    finally:
        runtime.close_storage()


def test_vector_pool_is_hard_capped_and_mmr_keeps_diverse_candidates():
    candidates = [dict(id=i) for i in range(70)]
    vectors = [[1.0, 0.0]] * 60 + [[0.8, 0.6]] * 10
    selected = rank_candidates(candidates, vectors, [1.0, 0.0], limit=999)
    assert len(selected) == 50 and len({item['id'] for item in selected}) == 50


@pytest.mark.asyncio
async def test_approved_expression_survives_index_failure(tmp_path):
    runtime = initialized_runtime(tmp_path)
    _, _, current_id = seed_conversation(runtime)
    class Model:
        async def complete_json(self, messages):
            return {'expressions': [{'situation': '惊叹', 'style': '这么强', 'source_id': current_id}]} if '真实聊天提取' in messages[0]['content'] else {'suitable': True}
    class Broken(FakeEmbedding):
        async def embed(self, texts):
            raise ModelClientError('fake failure')
    try:
        assert await ExpressionWorker(runtime).run_once(Model(), Broken(), force=True)
        with runtime.require_database().session() as session:
            assert session.scalar(select(Expression)).count == 1
            assert session.scalar(select(ExpressionVector)) is None
    finally:
        runtime.close_storage()


@pytest.mark.asyncio
async def test_learning_failure_does_not_block_existing_expression_index(tmp_path):
    runtime = initialized_runtime(tmp_path)
    companion_id, conversation_id, _ = seed_conversation(runtime)
    seed_expressions(runtime, companion_id)

    class BrokenLearning:
        async def complete_json(self, messages):
            raise ModelClientError('fake learning failure')

    try:
        assert await ExpressionWorker(runtime).run_once(BrokenLearning(), FakeEmbedding(), force=True)
        with runtime.require_database().session() as session:
            assert session.get(ExpressionProgress, conversation_id) is None
            assert len(list(session.scalars(select(ExpressionVector)))) == 12
    finally:
        runtime.close_storage()


def test_rank_candidates_preserves_negative_similarity_penalty():
    candidates = [{'id': 1}, {'id': 2}, {'id': 3}]
    vectors = [
        normalized_vector([0.8, 0.6, 0.0]),
        normalized_vector([-0.5, 0.58, 0.6431174076]),
        normalized_vector([-0.8, 0.57, 0.18734994]),
    ]
    query = normalized_vector([0.0, 1.0, 0.0])
    # Reference MMR without zero-clamping negative similarities:
    # item 1 is selected first (relevance 0.60); item 2 has relevance 0.58 and dot(v2, v1) = -0.052,
    # while item 3 has relevance 0.57 and dot(v3, v1) = -0.298, so item 3 wins round 2 (0.5292 > 0.5008).
    ranked = rank_candidates(candidates, vectors, query, limit=2)
    assert {item['id'] for item in ranked} == {1, 3}


def test_offline_preset_expressions_survive_backup_export_import(tmp_path):
    from test_business_api import partner_payload
    client, runtime, headers = make_client(tmp_path)
    with client:
        assert client.post('/api/partner', headers=headers, json=partner_payload()).status_code == 201
        with runtime.require_database().session() as session:
            comp_id = session.scalar(select(Companion.id))
            session.add(Expression(companion_id=comp_id, situation='离线日常情境', style='自然口语回应', count=3))
        exported = client.get('/api/data/export', headers=headers)
        assert exported.status_code == 200
        backup = exported.json()
        assert len(backup['tables']['expressions']) == 1 and backup['tables']['expressionSources'] == []
        res = client.post('/api/data/import', headers=headers, files={'file': ('backup.json', json.dumps(backup).encode(), 'application/json')})
        assert res.status_code == 200
        with runtime.require_database().session() as session:
            restored = session.scalar(select(Expression))
            assert restored is not None and restored.situation == '离线日常情境' and restored.count == 3
        bad = json.loads(json.dumps(backup))
        bad['tables']['expressions'][0]['count'] = 0
        assert client.post('/api/data/import', headers=headers, files={'file': ('bad.json', json.dumps(bad).encode(), 'application/json')}).status_code == 422
