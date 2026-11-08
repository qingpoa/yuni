from __future__ import annotations

import json
import sqlite3

import pytest
from planner_helpers import reply_delivered, reply_event, stop_event
from test_business_api import make_client, parse_sse, partner_payload

from yuni_agent.model.client import ModelClientError
from yuni_agent.model.reply import REPLY_OUTPUT_INSTRUCTIONS, parse_reply
from yuni_agent.storage.migrations import upgrade_database


@pytest.mark.parametrize('text', [
    '我在。', '!?', '第一句很长所以可以分条。第二句同样保留！第三句？还有第四句。第五句。',
    '他说：“先吃饭。然后睡觉！”你觉得呢？', '「里面，仍然不拆。」外面继续。',
    "I don't think so. See you tomorrow! All right?", '价格为1,234.50元，预计12:30到。',
    '虽然今天很累，但是见到你就开心了。', '保留（括号内，内容。）；还有[引用，内容]。',
    '  第一行。\n\n第二行仍在， 空格也在。\t第三行。  ',
    '```python\nprint("a,b")\n```\n代码后的文字。', '“嵌套「引号。仍然在」。直到这里”以后。',
    ' ' + '任意文本，标点！？ 空格\n' * 100 + ' ',
])
def test_text_split_is_lossless_and_at_most_three(text):
    parts = parse_reply(text)
    assert 1 <= len(parts) <= 3 and all(part.strip() for part in parts)
    assert ''.join(parts) == text.strip()


def test_quotes_and_connected_structure_are_protected():
    text = '他说：“第一句。第二句，第三句！”\n以后再说。'
    parts = parse_reply(text)
    assert any('“第一句。第二句，第三句！”' in part for part in parts)
    assert REPLY_OUTPUT_INSTRUCTIONS.startswith('给出日常且口语化的纯文本回复')


@pytest.mark.parametrize('text', ['', ' \n ', 'a' * 8001])
def test_empty_and_oversized_text_fail_without_format_repair(text):
    with pytest.raises(ModelClientError):
        parse_reply(text)


def test_probabilistic_segment_merge_restores_one_to_three_variation(monkeypatch):
    from yuni_agent.model import reply
    sample = '今天真的有点累。洗个澡就去睡啦！晚安。'
    monkeypatch.setattr(reply.random, 'random', lambda: 0.0)
    merged_parts = parse_reply(sample)
    monkeypatch.setattr(reply.random, 'random', lambda: 1.0)
    unmerged_parts = parse_reply(sample)
    assert len(merged_parts) == 2 and ''.join(merged_parts) == sample
    assert len(unmerged_parts) == 3 and ''.join(unmerged_parts) == sample


def test_text_api_history_replay_and_legacy_backup(tmp_path, monkeypatch):
    from pydantic import SecretStr

    from yuni_agent.api.routes import chat
    from yuni_agent.model import reply
    monkeypatch.setattr(reply.random, 'random', lambda: 1.0)
    class Model:
        calls = 0
        output = '第一句话保留原样。第二句话也保留！第三句话呢？最后一句。'
        def __init__(self, **kwargs):
            pass
        async def stream_chat_with_tools(self, messages, tools):
            Model.calls += 1
            yield stop_event() if reply_delivered(messages) else reply_event()
        async def stream_chat(self, messages):
            Model.calls += 1
            assert not any('replies' in str(item) for item in messages)
            yield Model.output
    monkeypatch.setattr(chat, 'OpenAICompatibleClient', Model)
    client, runtime, headers = make_client(tmp_path)
    with client:
        runtime.model_api_key = SecretStr('fake')
        client.put('/api/settings/memory', headers=headers, json={'memoryEnabled': False})
        client.post('/api/partner', headers=headers, json=partner_payload())
        payload = {'content': '你好', 'clientMessageId': 'plain-text-once'}
        response = client.post('/api/chat/messages/stream', headers=headers, json=payload)
        result = parse_sse(response.text)[-1][1]
        assert result['content'] == Model.output == ''.join(result['replyParts'])
        assert len(result['replyParts']) == 3
        assert client.post('/api/chat/messages/stream', headers=headers, json=payload).text == response.text
        assert Model.calls == 3
        history = client.get('/api/chat/messages', headers=headers).json()['data']['items']
        assert history[-1]['content'] == result['content'] and history[-1]['replyParts'] == result['replyParts']
        Model.output = ''
        failed = parse_sse(client.post('/api/chat/messages/stream', headers=headers, json={'content': '下一轮', 'clientMessageId': 'plain-fail-once'}).text)
        assert failed[-1][0] == 'error' and Model.calls == 5
        Model.output = '重试成功。'
        retried = parse_sse(client.post(f'/api/chat/messages/{failed[0][1]["messageId"]}/retry', headers=headers).text)
        assert retried[-1][1]['content'] == '重试成功。' and Model.calls == 8
        backup = client.get('/api/data/export', headers=headers).json()
        for row in backup['tables']['chatMessages']:
            row.pop('reply_parts', None)
        for key in ('expressions', 'expressionSources', 'expressionProgress'):
            backup['tables'].pop(key)
        assert client.post('/api/data/import', headers=headers, files={'file': ('old.json', json.dumps(backup).encode(), 'application/json')}).status_code == 200
        restored = client.get('/api/chat/messages', headers=headers).json()['data']['items']
        assert restored[1]['replyParts'] == [result['content']]


def test_migration_preserves_legacy_reply_content(tmp_path):
    path = tmp_path / 'old.db'
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, content TEXT);INSERT INTO chat_messages VALUES (1, 'old text');CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY);INSERT INTO alembic_version VALUES ('0004_personalization');")
    upgrade_database(path)
    upgrade_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT content, reply_parts FROM chat_messages').fetchone() == ('old text', None)


@pytest.mark.asyncio
@pytest.mark.parametrize('style', ['自然', '简短', '展开'])
async def test_style_and_real_results_preserved_without_replyer_format_retries(style, monkeypatch):
    from planner_helpers import reply_call
    from test_agent_separation import Builder, Model, Tools

    from yuni_agent.agent.chat_agent import REPLY_STYLE_INSTRUCTIONS
    from yuni_agent.agent.graph import ChatGraphContext, LangGraphChat
    from yuni_agent.model.client import ModelToolCall
    class PlainModel(Model):
        async def stream_chat(self, messages):
            self.chat_calls += 1
            assert '真实工具结果' in str(messages)
            assert not any('你的说话风格可以尝试：' in str(item.get('content', '')) for item in messages)
            instruction = REPLY_STYLE_INSTRUCTIONS[style]
            assert instruction in str(messages)
            yield '自然输出，不需要JSON外壳。'
    def unexpected_style_choice(_):
        raise AssertionError('Replyer must not draw a random temporary style')
    monkeypatch.setattr('yuni_agent.model.reply.random.choice', unexpected_style_choice)
    call = reply_call()
    call.arguments['reply_style'] = style
    model, tools = PlainModel([ModelToolCall(id='r', name='read_text_file'), call]), Tools()
    events = [event async for event in LangGraphChat().stream({'conversation_id': 1, 'user_message_id': 2, 'assistant_message_id': 3, 'current_input': '读一下'}, ChatGraphContext(context_builder=Builder(), model_client=model, tool_executor=tools))]
    assert model.chat_calls == 1 and model.plan_calls == 3 and tools.calls == 1
    assert events[-1].data['content'] == '自然输出，不需要JSON外壳。'
